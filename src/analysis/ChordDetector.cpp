#include "analysis/ChordDetector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace skale {

namespace {

constexpr int kStates = kChordTypeCount * 12 + 1;
constexpr int kNoneState = kChordTypeCount * 12;

float priorFor(ChordType t) {
    switch (t) {
        case ChordType::Major:
        case ChordType::Minor: return 1.0f;
        case ChordType::Diminished: return 0.97f;
        default: return 0.96f;  // séptimas y suspendidos: solo si encajan claramente mejor
    }
}

ChordLabel labelOf(int state) {
    ChordLabel l;
    if (state == kNoneState) return l;
    l.none = false;
    l.type = ChordType(state / 12);
    l.root = state % 12;
    return l;
}

}  // namespace

ChordDetector::ChordDetector() : ChordDetector(Params{}) {}

ChordDetector::ChordDetector(Params p) : params_(p) {
    templates_.resize(kChordTypeCount * 12);
    prior_.resize(kChordTypeCount * 12);
    for (int t = 0; t < kChordTypeCount; ++t) {
        const auto intervals = chordIntervals(ChordType(t));
        for (int root = 0; root < 12; ++root) {
            std::array<float, 12> tpl{};
            for (int i : intervals) tpl[std::size_t((root + i) % 12)] = 1.f;
            const float norm = std::sqrt(float(intervals.size()));
            for (float& v : tpl) v /= norm;
            templates_[std::size_t(t * 12 + root)] = tpl;
            prior_[std::size_t(t * 12 + root)] = priorFor(ChordType(t));
        }
    }
}

ChordLabel ChordDetector::detectSingle(const Chroma12& chroma) const {
    double norm = 0;
    for (float v : chroma) norm += double(v) * double(v);
    norm = std::sqrt(norm);
    if (norm <= 0.0) return {};

    int best = kNoneState;
    float bestSim = params_.noChordSimilarity;
    for (int s = 0; s < kNoneState; ++s) {
        float dot = 0;
        for (int i = 0; i < 12; ++i) dot += chroma[std::size_t(i)] * templates_[std::size_t(s)][std::size_t(i)];
        const float sim = dot / float(norm) * prior_[std::size_t(s)];
        if (sim > bestSim) { bestSim = sim; best = s; }
    }
    return labelOf(best);
}

std::vector<ChordSegment> ChordDetector::detect(const std::vector<ChordFrame>& frames, double hopSeconds) const {
    const std::size_t T = frames.size();
    if (T == 0) return {};

    std::vector<std::uint8_t> back(T * std::size_t(kStates));
    std::vector<float> dp(std::size_t(kStates), 0.f);
    std::vector<float> next(std::size_t(kStates), 0.f);
    std::vector<float> emit(std::size_t(kStates), 0.f);

    for (std::size_t t = 0; t < T; ++t) {
        const ChordFrame& f = frames[t];
        double norm = 0;
        for (float v : f.chroma) norm += double(v) * double(v);
        norm = std::sqrt(norm);

        if (f.silent || norm <= 0.0) {
            for (int s = 0; s < kNoneState; ++s) emit[std::size_t(s)] = 0.f;
            emit[std::size_t(kNoneState)] = params_.emissionScale;
        } else {
            for (int s = 0; s < kNoneState; ++s) {
                float dot = 0;
                for (int i = 0; i < 12; ++i) dot += f.chroma[std::size_t(i)] * templates_[std::size_t(s)][std::size_t(i)];
                emit[std::size_t(s)] = params_.emissionScale * (dot / float(norm)) * prior_[std::size_t(s)];
            }
            emit[std::size_t(kNoneState)] = params_.emissionScale * params_.noChordSimilarity;
        }

        if (t == 0) {
            for (int s = 0; s < kStates; ++s) dp[std::size_t(s)] = emit[std::size_t(s)];
            continue;
        }

        int bestPrev = 0;
        for (int s = 1; s < kStates; ++s) {
            if (dp[std::size_t(s)] > dp[std::size_t(bestPrev)]) bestPrev = s;
        }
        const float switchScore = dp[std::size_t(bestPrev)] - params_.switchPenalty;
        for (int s = 0; s < kStates; ++s) {
            const bool stay = dp[std::size_t(s)] >= switchScore;
            next[std::size_t(s)] = (stay ? dp[std::size_t(s)] : switchScore) + emit[std::size_t(s)];
            back[t * std::size_t(kStates) + std::size_t(s)] = std::uint8_t(stay ? s : bestPrev);
        }
        dp.swap(next);
    }

    std::vector<int> path(T);
    int cur = 0;
    for (int s = 1; s < kStates; ++s) {
        if (dp[std::size_t(s)] > dp[std::size_t(cur)]) cur = s;
    }
    for (std::size_t t = T; t-- > 0;) {
        path[t] = cur;
        if (t > 0) cur = back[t * std::size_t(kStates) + std::size_t(cur)];
    }

    std::vector<ChordSegment> segments;
    for (std::size_t t = 0; t < T; ++t) {
        const double start = std::max(0.0, frames[t].time - hopSeconds / 2);
        const double end = frames[t].time + hopSeconds / 2;
        const ChordLabel label = labelOf(path[t]);
        if (!segments.empty() && segments.back().chord == label) {
            segments.back().end = end;
        } else {
            if (!segments.empty()) segments.back().end = start;
            segments.push_back({start, end, label});
        }
    }
    return segments;
}

}  // namespace skale
