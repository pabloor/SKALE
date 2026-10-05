#include "analysis/Analyzer.h"

#include <algorithm>
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

    ChromaExtractor extractor(sampleRate);
    const std::vector<ChromaFrame> frames = extractor.process(mono, n);

    Chroma36 acc{};
    std::size_t active = 0;
    for (const auto& f : frames) {
        if (f.silent) continue;
        for (std::size_t i = 0; i < 36; ++i) acc[i] += f.chroma[i];
        ++active;
    }
    if (active == 0) return out;

    const int tuning = estimateTuning(acc);
    out.tuningCents = tuningToCents(tuning);
    out.chroma = foldChroma(acc, tuning);
    normalize(out.chroma);

    // Cromagrama de los últimos segundos con sonido (el final de la pieza).
    Chroma12 ending{};
    bool hasEnding = false;
    if (options.endingWeight > 0 && options.endingSeconds > 0) {
        Chroma36 endAcc{};
        double lastTime = 0;
        for (const auto& f : frames) if (!f.silent) lastTime = f.time;
        for (const auto& f : frames) {
            if (f.silent || f.time < lastTime - options.endingSeconds) continue;
            for (std::size_t i = 0; i < 36; ++i) endAcc[i] += f.chroma[i];
        }
        ending = foldChroma(endAcc, tuning);
        normalize(ending);
        hasEnding = true;
    }

    auto candidates = KeyDetector::detect(out.chroma, options.profile,
                                          hasEnding ? &ending : nullptr, options.endingWeight);
    if (candidates.empty()) return out;
    if (candidates.size() > options.maxCandidates) candidates.resize(options.maxCandidates);
    out.candidates = candidates;
    out.key = candidates.front().key;
    out.keyName = skale::keyName(out.key, options.solfege);

    const Speller speller(out.key);
    for (const Note& n2 : speller.scale()) out.scaleNotes.push_back(formatNote(n2, options.solfege));
    out.diatonic = diatonicChords(out.key, options.solfege);

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
