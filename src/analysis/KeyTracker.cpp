#include "analysis/KeyTracker.h"

#include <cmath>

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
    if (candidates.empty()) return out;
    if (candidates.size() > options_.maxCandidates) candidates.resize(options_.maxCandidates);
    out.candidates = candidates;
    out.key = candidates.front().key;
    out.confidence = candidates.front().confidence;
    out.keyName = skale::keyName(out.key, options_.solfege);

    const Speller speller(out.key);
    for (const Note& n : speller.scale()) out.scaleNotes.push_back(formatNote(n, options_.solfege));

    Chroma12 now = foldChroma(recent_, tuning);
    normalize(now);
    const ChordLabel chord = chords_.detectSingle(now);
    out.hasChord = !chord.none;
    out.chordName = chord.none ? "-" : chordName(chord.root, chord.type, speller, options_.solfege);

    out.valid = true;
    return out;
}

}  // namespace skale
