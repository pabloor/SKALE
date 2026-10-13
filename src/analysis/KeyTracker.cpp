#include "analysis/KeyTracker.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "analysis/KeyCnn.h"

namespace skale {

namespace {
constexpr double kRecentDecay = 0.6;           // el acorde actual pesa sobre todo el último ~medio segundo
constexpr double kMinListenSeconds = 2.0;      // antes de esto no se da respuesta
}  // namespace

KeyTracker::KeyTracker(double sampleRate, TrackerOptions options)
    : options_(options), extractor_(sampleRate) {
    const double hop = extractor_.hopSeconds();
    decay_ = options_.halfLifeSeconds > 0 ? std::pow(0.5, hop / options_.halfLifeSeconds) : 1.0;
}

void KeyTracker::reset() {
    extractor_.reset();
    acc_.fill(0.f);
    bassAcc_.fill(0.f);
    recent_.fill(0.f);
    listened_ = 0;
    activeFrames_ = 0;
    window_.clear();
    pair_ = {};
    pairCount_ = 0;
    sinceNets_ = 0;
    netSum_.fill(0.0);
    netWeight_ = 0;
    netUpdates_ = 0;
}

void KeyTracker::runNets() {
    if (window_.empty()) return;
    const std::vector<FrameChroma> frames(window_.begin(), window_.end());
    const auto lp = cnnKeyLogProbs(frames);
    lastLp_ = lp;
    ++netUpdates_;
    const double d = options_.netHalfLifeSeconds > 0 ? std::pow(0.5, options_.netEverySeconds / options_.netHalfLifeSeconds) : 0.0;
    for (std::size_t i = 0; i < 24; ++i) netSum_[i] = d * netSum_[i] + double(lp[i]);
    netWeight_ = d * netWeight_ + 1.0;
}

void KeyTracker::process(const float* mono, std::size_t n) {
    const double hop = extractor_.hopSeconds();
    for (const ChromaFrame& f : extractor_.process(mono, n)) {
        for (std::size_t i = 0; i < 36; ++i) {
            acc_[i] = float(double(acc_[i]) * decay_);
            bassAcc_[i] = float(double(bassAcc_[i]) * decay_);
            recent_[i] = float(double(recent_[i]) * kRecentDecay);
        }
        if (f.silent) continue;
        for (std::size_t i = 0; i < 36; ++i) {
            acc_[i] += f.chroma[i];
            bassAcc_[i] += f.bass[i];
            recent_[i] += f.chroma[i];
        }
        listened_ += hop;
        ++activeFrames_;
        if (!options_.nets) continue;
        // Fotogramas emparejados como en el análisis de archivos (media de 2 fotogramas no silenciosos).
        for (std::size_t i = 0; i < 36; ++i) { pair_.chroma[i] += f.chroma[i]; pair_.bass[i] += f.bass[i]; pair_.mid[i] += f.mid[i]; pair_.high[i] += f.high[i]; }
        for (std::size_t i = 0; i < kSpecBins; ++i) pair_.spec[i] += f.spec[i];
        if (++pairCount_ == 2) {
            for (std::size_t i = 0; i < 36; ++i) { pair_.chroma[i] *= 0.5f; pair_.bass[i] *= 0.5f; pair_.mid[i] *= 0.5f; pair_.high[i] *= 0.5f; }
            for (std::size_t i = 0; i < kSpecBins; ++i) pair_.spec[i] *= 0.5f;
            window_.push_back(pair_);
            while (window_.size() > maxPairs()) window_.pop_front();
            pair_ = {};
            pairCount_ = 0;
        }
        sinceNets_ += hop;
        if (sinceNets_ >= options_.netEverySeconds && listened_ >= kMinListenSeconds) {
            sinceNets_ = 0;
            runNets();
        }
    }
}

TrackerSnapshot KeyTracker::snapshot() const {
    TrackerSnapshot out;
    out.listenedSeconds = listened_;
    if (activeFrames_ == 0 || listened_ < kMinListenSeconds) return out;

    const int tuning = estimateTuning(acc_);
    out.chroma = foldChroma(acc_, tuning);
    normalize(out.chroma);

    std::array<float, 24> extra{};
    const std::array<float, 24>* extraPtr = nullptr;
    if (options_.bassWeight > 0) {
        Chroma12 bass = foldChroma(bassAcc_, tuning);
        normalize(bass);
        extra = KeyDetector::bassScores(bass, options_.bassWeight);
        extraPtr = &extra;
    }

    auto candidates = KeyDetector::detect(out.chroma, options_.profile, nullptr, 0, 1e9, extraPtr);
    if (options_.nets && netWeight_ > 0) {   // tonalidad de las redes (las mismas que en el análisis de archivos)
        std::array<double, 24> lp{};
        double mx = -1e300;
        // Mientras la ventana no está llena, la última respuesta ya ha visto todo el audio y se usa sola (la
        // memoria ahí resta 3-4 puntos en GTZAN); después, la memoria, que incluye también esas primeras ventanas
        // (empezarla con la ventana llena resta 3-5 puntos en canciones completas de FMA a partir de 60 s).
        const bool full = netWindowFull();
        for (std::size_t i = 0; i < 24; ++i) { lp[i] = full ? netSum_[i] / netWeight_ : double(lastLp_[i]); mx = std::max(mx, lp[i]); }
        double z = 0;
        for (double v : lp) z += std::exp(v - mx);
        std::vector<int> order(24);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return lp[std::size_t(a)] > lp[std::size_t(b)]; });
        candidates.clear();
        for (int i : order) {
            KeyCandidate c;
            c.key = {i / 2, i % 2 == 0 ? Mode::Major : Mode::Minor};
            c.confidence = float(std::exp(lp[std::size_t(i)] - mx) / z);
            c.correlation = float(lp[std::size_t(i)]);
            candidates.push_back(c);
        }
        out.fromNets = true;
    }
    if (candidates.empty()) return out;
    if (candidates.size() > options_.maxCandidates) candidates.resize(options_.maxCandidates);
    out.candidates = candidates;
    out.key = candidates.front().key;
    out.confidence = candidates.front().confidence;
    out.keyName = skale::keyName(out.key, options_.solfege);

    const Speller speller(out.key);
    for (const Note& n : speller.scale()) out.scaleNotes.push_back(formatNote(n, options_.solfege));
    if (candidates.size() > 1) {
        out.doubtful = out.fromNets && options_.doubtRatio > 0 &&
                       candidates[1].confidence >= float(options_.doubtRatio) * candidates[0].confidence;
        out.secondKey = candidates[1].key;
        out.secondKeyName = skale::keyName(out.secondKey, options_.solfege);
        const int rel = out.key.mode == Mode::Major ? 9 : 3;
        out.secondSameNotes = out.secondKey.mode != out.key.mode && out.secondKey.tonic == (out.key.tonic + rel) % 12;
    }

    Chroma12 now = foldChroma(recent_, tuning);
    normalize(now);
    const ChordLabel chord = chords_.detectSingle(now);
    out.hasChord = !chord.none;
    out.chordName = chord.none ? "-" : chordName(chord.root, chord.type, speller, options_.solfege);

    out.valid = true;
    return out;
}

}  // namespace skale
