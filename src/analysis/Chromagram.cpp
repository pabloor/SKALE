#include "analysis/Chromagram.h"

#include <algorithm>
#include <cmath>

namespace skale {

namespace {

constexpr double kMinFreq = 65.0;
constexpr double kMaxFreq = 2100.0;
constexpr double kBassMinFreq = 40.0;
constexpr double kBassMaxFreq = 250.0;
constexpr float kSilenceRms = 5e-4f;
constexpr float kPeakFloor = 0.03f;  // pico mínimo respecto al mayor (-30 dB)

std::size_t nextPow2(std::size_t v) {
    std::size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

}  // namespace

ChromaExtractor::ChromaExtractor(double sampleRate)
    : sampleRate_(sampleRate),
      fftSize_(std::clamp<std::size_t>(nextPow2(std::size_t(std::ceil(sampleRate / 3.0))), 4096, 65536)),
      hop_(fftSize_ / 4),
      minBin_(std::size_t(std::ceil(kMinFreq * double(fftSize_) / sampleRate))),
      maxBin_(std::min<std::size_t>(std::size_t(kMaxFreq * double(fftSize_) / sampleRate), fftSize_ / 2 - 2)),
      fft_(fftSize_),
      window_(fftSize_),
      buf_(fftSize_),
      mag_(fftSize_ / 2) {
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < fftSize_; ++i) {
        window_[i] = float(0.5 - 0.5 * std::cos(2.0 * pi * double(i) / double(fftSize_)));
    }
}

void ChromaExtractor::reset() {
    pending_.clear();
    pendingStart_ = 0;
    nextFrameStart_ = 0;
}

std::vector<ChromaFrame> ChromaExtractor::process(const float* mono, std::size_t n) {
    std::vector<ChromaFrame> out;
    pending_.insert(pending_.end(), mono, mono + n);

    while (nextFrameStart_ + fftSize_ <= pendingStart_ + pending_.size()) {
        ChromaFrame f;
        analyseFrame(pending_.data() + (nextFrameStart_ - pendingStart_), f);
        f.time = (double(nextFrameStart_) + double(fftSize_) / 2.0) / sampleRate_;
        out.push_back(f);
        nextFrameStart_ += hop_;
    }

    const std::size_t drop = nextFrameStart_ - pendingStart_;
    if (drop > 0 && drop <= pending_.size()) {
        pending_.erase(pending_.begin(), pending_.begin() + std::ptrdiff_t(drop));
        pendingStart_ += drop;
    }
    return out;
}

void ChromaExtractor::analyseFrame(const float* x, ChromaFrame& out) {
    out.chroma.fill(0.f);
    out.bass.fill(0.f);
    out.silent = true;

    double energy = 0;
    for (std::size_t i = 0; i < fftSize_; ++i) energy += double(x[i]) * double(x[i]);
    if (std::sqrt(energy / double(fftSize_)) < kSilenceRms) return;

    for (std::size_t i = 0; i < fftSize_; ++i) buf_[i] = {x[i] * window_[i], 0.f};
    fft_.forward(buf_);

    float frameMax = 0.f;
    for (std::size_t i = minBin_ - 1; i <= maxBin_ + 1; ++i) {
        mag_[i] = std::abs(buf_[i]);
        if (i >= minBin_ && i <= maxBin_) frameMax = std::max(frameMax, mag_[i]);
    }
    if (frameMax <= 0.f) return;

    const float floor = frameMax * kPeakFloor;
    const double binHz = sampleRate_ / double(fftSize_);
    double total = 0;

    for (std::size_t i = minBin_; i <= maxBin_; ++i) {
        const float m = mag_[i];
        if (m < floor || m <= mag_[i - 1] || m < mag_[i + 1]) continue;

        // Interpolación parabólica sobre el logaritmo de la magnitud.
        const double a = std::log(double(mag_[i - 1]) + 1e-12);
        const double b = std::log(double(m) + 1e-12);
        const double c = std::log(double(mag_[i + 1]) + 1e-12);
        const double denom = a - 2.0 * b + c;
        double shift = denom != 0.0 ? 0.5 * (a - c) / denom : 0.0;
        shift = std::clamp(shift, -0.5, 0.5);
        const double freq = (double(i) + shift) * binHz;

        // 3 bins por semitono, plegados a una octava (36 bins); La = bin 27.
        double pos = 36.0 * std::log2(freq / 440.0) + 27.0;
        pos = std::fmod(pos, 36.0);
        if (pos < 0) pos += 36.0;
        const int lo = int(std::floor(pos));
        const float frac = float(pos - double(lo));
        const float w = std::sqrt(m);  // compresión de amplitud
        out.chroma[std::size_t(lo % 36)] += w * (1.f - frac);
        out.chroma[std::size_t((lo + 1) % 36)] += w * frac;
        total += double(w);
    }

    if (total <= 0.0) return;
    for (float& v : out.chroma) v = float(double(v) / total);
    out.silent = false;

    // Cromagrama del bajo: mismos picos y mismo reparto en 36 bins, solo 40-250 Hz.
    const std::size_t bLo = std::max<std::size_t>(2, std::size_t(std::ceil(kBassMinFreq / binHz)));
    const std::size_t bHi = std::min<std::size_t>(std::size_t(kBassMaxFreq / binHz), fftSize_ / 2 - 2);
    float bassMax = 0.f;
    for (std::size_t i = bLo; i <= bHi; ++i) bassMax = std::max(bassMax, std::abs(buf_[i]));
    if (bassMax <= 0.f) return;
    const float bassFloor = bassMax * 0.1f;
    double bassTotal = 0;
    for (std::size_t i = bLo; i <= bHi; ++i) {
        const float m = std::abs(buf_[i]);
        const float mPrev = std::abs(buf_[i - 1]);
        const float mNext = std::abs(buf_[i + 1]);
        if (m < bassFloor || m <= mPrev || m < mNext) continue;
        const double a = std::log(double(mPrev) + 1e-12);
        const double b = std::log(double(m) + 1e-12);
        const double c = std::log(double(mNext) + 1e-12);
        const double denom = a - 2.0 * b + c;
        double shift = denom != 0.0 ? 0.5 * (a - c) / denom : 0.0;
        shift = std::clamp(shift, -0.5, 0.5);
        const double freq = (double(i) + shift) * binHz;
        double pos = 36.0 * std::log2(freq / 440.0) + 27.0;
        pos = std::fmod(pos, 36.0);
        if (pos < 0) pos += 36.0;
        const int lo = int(std::floor(pos));
        const float frac = float(pos - double(lo));
        const float w = std::sqrt(m);
        out.bass[std::size_t(lo % 36)] += w * (1.f - frac);
        out.bass[std::size_t((lo + 1) % 36)] += w * frac;
        bassTotal += double(w);
    }
    if (bassTotal > 0.0) for (float& v : out.bass) v = float(double(v) / bassTotal);
}

int estimateTuning(const Chroma36& c) {
    double score[3] = {0, 0, 0};  // índices 0,1,2 ↔ tuning -1,0,+1
    for (int n = 0; n < 12; ++n) {
        for (int t = -1; t <= 1; ++t) {
            score[t + 1] += double(c[std::size_t((3 * n + t + 36) % 36)]);
        }
    }
    int best = 0;
    // Hace falta una ventaja clara para apartarse de A=440.
    double bestScore = score[1] * 1.05;
    if (score[0] > bestScore) { best = -1; bestScore = score[0]; }
    if (score[2] > bestScore) { best = 1; }
    return best;
}

double tuningToCents(int tuning) { return tuning * (100.0 / 3.0); }

Chroma12 foldChroma(const Chroma36& c, int tuning) {
    Chroma12 out{};
    for (int n = 0; n < 12; ++n) {
        const int centre = 3 * n + tuning;
        out[std::size_t(n)] = c[std::size_t((centre + 36) % 36)]
                            + 0.5f * c[std::size_t((centre + 35) % 36)]
                            + 0.5f * c[std::size_t((centre + 37) % 36)];
    }
    return out;
}

void normalize(Chroma12& c) {
    double sum = 0;
    for (float v : c) sum += double(v);
    if (sum <= 0.0) { c.fill(0.f); return; }
    for (float& v : c) v = float(double(v) / sum);
}

}  // namespace skale
