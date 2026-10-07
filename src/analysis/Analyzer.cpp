#include "analysis/Analyzer.h"
#include "analysis/KeyCnn.h"

#include <algorithm>
#include <array>
#include <map>

namespace skale {

namespace {

// En modo menor se admite además la sensible (7.º grado alzado, armonía
// menor): así V y vii° no cuentan como acordes ajenos a la tonalidad.
bool isDiatonic(const ChordLabel& c, const Speller& sp, Key key) {
    if (c.none) return false;
    const int leadingTone = (key.tonic + 11) % 12;
    for (int pc : chordPitchClasses(c.root, c.type)) {
        if (sp.inScale(pc)) continue;
        if (key.mode == Mode::Minor && pc == leadingTone) continue;
        return false;
    }
    return true;
}

}  // namespace

SongAnalysis analyze(const float* mono, std::size_t n, double sampleRate, const AnalysisOptions& options) {
    SongAnalysis out;
    out.durationSeconds = sampleRate > 0 ? double(n) / sampleRate : 0;
    if (n == 0 || sampleRate <= 0) return out;

    ChromaExtractor extractor(sampleRate, options.chroma);
    const std::vector<ChromaFrame> frames = extractor.process(mono, n);

    Chroma36 acc{};
    Chroma36 bassAcc{};
    std::size_t active = 0;
    for (const auto& f : frames) {
        if (f.silent) continue;
        for (std::size_t i = 0; i < 36; ++i) { acc[i] += f.chroma[i]; bassAcc[i] += f.bass[i]; }
        ++active;
    }
    if (active == 0) return out;

    if (options.keepFrames || options.cnn) {
        FrameChroma pair;
        int k = 0;
        for (const auto& f : frames) {
            if (f.silent) continue;
            for (std::size_t i = 0; i < 36; ++i) { pair.chroma[i] += f.chroma[i]; pair.bass[i] += f.bass[i]; pair.mid[i] += f.mid[i]; pair.high[i] += f.high[i]; }
            if (++k == 2) {
                for (std::size_t i = 0; i < 36; ++i) { pair.chroma[i] *= 0.5f; pair.bass[i] *= 0.5f; pair.mid[i] *= 0.5f; pair.high[i] *= 0.5f; }
                out.frames.push_back(pair);
                pair = {};
                k = 0;
            }
        }
    }

    const int tuning = estimateTuning(acc);
    out.tuningCents = tuningToCents(tuning);
    out.chroma = foldChroma(acc, tuning);
    normalize(out.chroma);

    std::vector<ChordFrame> chordFrames;
    chordFrames.reserve(frames.size());
    for (const auto& f : frames) {
        ChordFrame cf;
        cf.time = f.time;
        cf.silent = f.silent;
        if (!f.silent) {
            cf.chroma = foldChroma(f.chroma, tuning);
            normalize(cf.chroma);
        }
        chordFrames.push_back(cf);
    }

    const ChordDetector detector;
    const auto segments = detector.detect(chordFrames, extractor.hopSeconds());

    // Puntuaciones extra por tonalidad: bajo y acordes.
    std::array<float, 24> extra{};
    bool hasExtra = false;
    Chroma12 bass = foldChroma(bassAcc, tuning);
    normalize(bass);
    out.bassChroma = bass;
    if (options.bassWeight > 0) {
        const auto bs = KeyDetector::bassScores(bass, options.bassWeight);
        for (std::size_t i = 0; i < 24; ++i) extra[i] += bs[i];
        hasExtra = true;
    }
    if (options.chordWeight > 0) {
        double chordTime = 0;
        for (const auto& seg : segments) if (!seg.chord.none) chordTime += std::max(0.0, seg.end - seg.start);
        if (chordTime > 0) {
            static const int kMajor[7] = {0, 2, 4, 5, 7, 9, 11};
            static const int kMinor[8] = {0, 2, 3, 5, 7, 8, 10, 11};  // menor armónica
            for (int t = 0; t < 12; ++t) {
                for (int m = 0; m < 2; ++m) {
                    bool inKey[12] = {};
                    if (m == 0) for (int d : kMajor) inKey[(t + d) % 12] = true;
                    else for (int d : kMinor) inKey[(t + d) % 12] = true;
                    double diatonic = 0, tonic = 0;
                    for (const auto& seg : segments) {
                        if (seg.chord.none) continue;
                        const double dur = std::max(0.0, seg.end - seg.start);
                        bool ok = true;
                        for (int pc : chordPitchClasses(seg.chord.root, seg.chord.type)) if (!inKey[pc]) { ok = false; break; }
                        if (ok) diatonic += dur;
                        if (seg.chord.root == t && seg.chord.type == (m == 0 ? ChordType::Major : ChordType::Minor)) tonic += dur;
                    }
                    extra[std::size_t(t * 2 + m)] += float(options.chordWeight * (diatonic + tonic) / chordTime);
                }
            }
            hasExtra = true;
        }
    }

    // Cromagrama de los últimos segundos con sonido (el final de la pieza).
    Chroma12 ending{};
    bool hasEnding = false;
    if (options.endingSeconds > 0) {
        Chroma36 endAcc{}, startAcc{};
        double lastTime = 0, firstTime = -1;
        for (const auto& f : frames) if (!f.silent) { lastTime = f.time; if (firstTime < 0) firstTime = f.time; }
        for (const auto& f : frames) {
            if (f.silent || f.time > firstTime + options.endingSeconds) continue;
            for (std::size_t i = 0; i < 36; ++i) startAcc[i] += f.chroma[i];
        }
        out.startChroma = foldChroma(startAcc, tuning);
        normalize(out.startChroma);
        for (const auto& f : frames) {
            if (f.silent || f.time < lastTime - options.endingSeconds) continue;
            for (std::size_t i = 0; i < 36; ++i) endAcc[i] += f.chroma[i];
        }
        ending = foldChroma(endAcc, tuning);
        normalize(ending);
        out.endingChroma = ending;
        hasEnding = options.endingWeight > 0;
    }

    {   // ventanas deslizantes de 8 s con paso de 4 s (solo con sonido suficiente)
        const double kWin = options.windowSeconds, kHop = options.windowSeconds / 2;
        const double total = out.durationSeconds;
        for (double t0 = 0; t0 + std::min(3.0, kWin) <= total; t0 += kHop) {
            Chroma36 c36{}, b36{};
            std::size_t n = 0;
            for (const auto& f : frames) {
                if (f.silent || f.time < t0 || f.time >= t0 + kWin) continue;
                for (std::size_t i = 0; i < 36; ++i) { c36[i] += f.chroma[i]; b36[i] += f.bass[i]; }
                ++n;
            }
            if (n < 8) continue;
            WindowChroma w;
            w.start = t0;
            w.chroma = foldChroma(c36, tuning); normalize(w.chroma);
            w.bass = foldChroma(b36, tuning); normalize(w.bass);
            out.windows.push_back(w);
        }
    }

    std::vector<Chroma12> winChromas;
    for (const auto& w : out.windows) winChromas.push_back(w.chroma);
    const std::array<float, 24> votes = KeyDetector::windowVotes(winChromas);

    std::vector<KeyCandidate> cnnCandidates;
    if (options.cnn) {
        auto lp = cnnKeyLogProbs(out.frames);
        if (options.learnedModel) {   // combinación con el modelo lineal
            const auto lin = KeyDetector::learnedLogProbs(out.chroma, out.bassChroma, options.endingSeconds > 0 ? &out.endingChroma : nullptr, &votes);
            for (std::size_t i = 0; i < 24; ++i) lp[i] = float(options.cnnWeight) * lp[i] + float(1.0 - options.cnnWeight) * lin[i];
        }
        std::vector<int> order(24);
        for (int i = 0; i < 24; ++i) order[std::size_t(i)] = i;
        std::sort(order.begin(), order.end(), [&](int a2, int b2) { return lp[std::size_t(a2)] > lp[std::size_t(b2)]; });
        double denom = 0;
        for (float v : lp) denom += std::exp(double(v - lp[std::size_t(order[0])]));
        for (int i : order) {
            KeyCandidate c;
            c.key = {i / 2, i % 2 == 0 ? Mode::Major : Mode::Minor};
            c.confidence = float(std::exp(double(lp[std::size_t(i)] - lp[std::size_t(order[0])])) / denom);
            c.correlation = lp[std::size_t(i)];
            cnnCandidates.push_back(c);
        }
    }

    auto candidates = options.cnn ? cnnCandidates : options.learnedModel
        ? KeyDetector::detectLearned(out.chroma, out.bassChroma, options.endingSeconds > 0 ? &out.endingChroma : nullptr, &votes)
        : KeyDetector::detect(out.chroma, options.profile,
                                          hasEnding ? &ending : nullptr, options.endingWeight,
                                          options.endingMargin, hasExtra ? &extra : nullptr);
    if (candidates.empty()) return out;
    if (candidates.size() > options.maxCandidates) candidates.resize(options.maxCandidates);
    out.candidates = candidates;
    out.key = candidates.front().key;
    out.keyName = skale::keyName(out.key, options.solfege);

    const Speller speller(out.key);
    for (const Note& n2 : speller.scale()) out.scaleNotes.push_back(formatNote(n2, options.solfege));
    out.diatonic = diatonicChords(out.key, options.solfege);

    std::map<std::pair<int, int>, ChordUsage> usage;  // (tipo, raíz)
    double chordTime = 0;
    for (const auto& seg : segments) {
        TimelineEntry e;
        e.start = seg.start;
        e.end = std::min(seg.end, out.durationSeconds);
        e.chord = seg.chord;
        e.name = seg.chord.none ? "-" : chordName(seg.chord.root, seg.chord.type, speller, options.solfege);
        e.diatonic = isDiatonic(seg.chord, speller, out.key);
        out.timeline.push_back(e);

        if (seg.chord.none) continue;
        const double dur = std::max(0.0, e.end - e.start);
        auto& u = usage[{int(seg.chord.type), seg.chord.root}];
        u.name = e.name;
        u.chord = seg.chord;
        u.diatonic = e.diatonic;
        u.seconds += dur;
        chordTime += dur;
    }
    for (auto& kv : usage) {
        kv.second.fraction = chordTime > 0 ? kv.second.seconds / chordTime : 0;
        out.chordUsage.push_back(kv.second);
    }
    std::sort(out.chordUsage.begin(), out.chordUsage.end(),
              [](const ChordUsage& a, const ChordUsage& b) { return a.seconds > b.seconds; });

    out.valid = true;
    return out;
}

}  // namespace skale
