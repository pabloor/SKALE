#include "analysis/Chromagram.h"

#include <algorithm>
#include <cmath>

namespace skale {

namespace {

constexpr double kBassMinFreq = 40.0;
constexpr double kBassMaxFreq = 250.0;
constexpr float kSilenceRms = 5e-4f;

std::size_t nextPow2(std::size_t v) {
    std::size_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

}  // namespace

ChromaExtractor::ChromaExtractor(double sampleRate, ChromaParams params)
    : sampleRate_(sampleRate),
      params_(params),
      fftSize_(std::clamp<std::size_t>(nextPow2(std::size_t(std::ceil(sampleRate / 3.0))), 4096, 65536)),
      hop_(fftSize_ / 4),
      minBin_(std::size_t(std::ceil(params.minFreq * double(fftSize_) / sampleRate))),
      maxBin_(std::min<std::size_t>(std::size_t(params.maxFreq * double(fftSize_) / sampleRate), fftSize_ / 2 - 2)),
      fft_(fftSize_),
      window_(fftSize_),
      buf_(fftSize_) {
    const double pi = std::acos(-1.0);
    for (std::size_t i = 0; i < fftSize_; ++i) {
        window_[i] = float(0.5 - 0.5 * std::cos(2.0 * pi * double(i) / double(fftSize_)));
    }
    // Bins de magnitud necesarios: picos del cromagrama, espectro logarítmico (hasta 55 Hz * 2^(215/36)) y bajo.
    const double binHz = sampleRate_ / double(fftSize_);
    const std::size_t specTop = std::size_t(55.0 * std::pow(2.0, double(kSpecBins - 1) / 36.0) / binHz) + 2;
    const std::size_t bassTop = std::size_t(kBassMaxFreq / binHz) + 2;
    magBins_ = std::min(fftSize_ / 2, std::max({maxBin_ + 2, specTop, bassTop}) + 1);
}

void ChromaExtractor::reset() {
    pending_.clear();
    pendingStart_ = 0;
    nextFrameStart_ = 0;
    hist_.clear();
    histStart_ = 0;
    framesIn_ = 0;
    nextOut_ = 0;
}

std::vector<ChromaFrame> ChromaExtractor::process(const float* mono, std::size_t n) {
    std::vector<ChromaFrame> out;
    pending_.insert(pending_.end(), mono, mono + n);

    while (nextFrameStart_ + fftSize_ <= pendingStart_ + pending_.size()) {
        SpecFrame sf;
        spectrum(pending_.data() + (nextFrameStart_ - pendingStart_), sf);
        sf.time = (double(nextFrameStart_) + double(fftSize_) / 2.0) / sampleRate_;
        nextFrameStart_ += hop_;
        if (params_.hpss) {
            hist_.push_back(std::move(sf));
            ++framesIn_;
            emitHpss(out, false);
        } else {
            ChromaFrame f;
            f.time = sf.time;
            if (!sf.silent) analyseMagnitude(sf.mag.data(), f);
            out.push_back(f);
        }
    }

    const std::size_t drop = nextFrameStart_ - pendingStart_;
    if (drop > 0 && drop <= pending_.size()) {
        pending_.erase(pending_.begin(), pending_.begin() + std::ptrdiff_t(drop));
        pendingStart_ += drop;
    }
    return out;
}

std::vector<ChromaFrame> ChromaExtractor::flush() {
    std::vector<ChromaFrame> out;
    if (params_.hpss) emitHpss(out, true);
    return out;
}

void ChromaExtractor::spectrum(const float* x, SpecFrame& f) {
    f.mag.assign(magBins_, 0.f);
    f.silent = true;
    double energy = 0;
    for (std::size_t i = 0; i < fftSize_; ++i) energy += double(x[i]) * double(x[i]);
    if (std::sqrt(energy / double(fftSize_)) < kSilenceRms) return;
    for (std::size_t i = 0; i < fftSize_; ++i) buf_[i] = {x[i] * window_[i], 0.f};
    fft_.forward(buf_);
    for (std::size_t i = 0; i < magBins_; ++i) f.mag[i] = std::abs(buf_[i]);
    f.silent = false;
}

// Analiza los fotogramas cuya ventana temporal de mediana ya está completa (o todos, al final).
void ChromaExtractor::emitHpss(std::vector<ChromaFrame>& out, bool final) {
    const std::size_t half = std::size_t(std::max(1, params_.hpssTimeFrames) - 1) / 2;
    const std::size_t fhalf = std::size_t(std::max(1, params_.hpssFreqBins) - 1) / 2;
    masked_.resize(magBins_);
    while (nextOut_ < framesIn_ && (final || nextOut_ + half < framesIn_)) {
        const SpecFrame& c = hist_[nextOut_ - histStart_];
        ChromaFrame f;
        f.time = c.time;
        if (!c.silent) {
            const std::size_t t0 = nextOut_ >= half ? nextOut_ - half : 0;
            const std::size_t t1 = std::min(framesIn_ - 1, nextOut_ + half);
            for (std::size_t k = 0; k < magBins_; ++k) {
                column_.clear();
                for (std::size_t t = t0; t <= t1; ++t) column_.push_back(hist_[t - histStart_].mag[k]);
                auto mid = column_.begin() + std::ptrdiff_t(column_.size() / 2);
                std::nth_element(column_.begin(), mid, column_.end());
                const float h = *mid;                                       // armonía: mediana en el tiempo
                const std::size_t k0 = k >= fhalf ? k - fhalf : 0, k1 = std::min(magBins_ - 1, k + fhalf);
                column_.assign(c.mag.begin() + std::ptrdiff_t(k0), c.mag.begin() + std::ptrdiff_t(k1) + 1);
                mid = column_.begin() + std::ptrdiff_t(column_.size() / 2);
                std::nth_element(column_.begin(), mid, column_.end());
                const float p = *mid;                                       // percusión: mediana en frecuencia
                const float h2 = h * h, p2 = p * p;
                masked_[k] = h2 + p2 > 0.f ? c.mag[k] * h2 / (h2 + p2) : 0.f;
            }
            analyseMagnitude(masked_.data(), f);
        }
        out.push_back(f);
        ++nextOut_;
        while (histStart_ + half < nextOut_ && !hist_.empty()) { hist_.pop_front(); ++histStart_; }
    }
}

void ChromaExtractor::analyseMagnitude(const float* mag, ChromaFrame& out) {
    out.chroma.fill(0.f);
    out.bass.fill(0.f);
    out.mid.fill(0.f);
    out.high.fill(0.f);
    out.spec.fill(0.f);
    out.silent = true;

    float frameMax = 0.f;
    for (std::size_t i = minBin_; i <= maxBin_; ++i) frameMax = std::max(frameMax, mag[i]);
    if (frameMax <= 0.f) return;

    const float floor = frameMax * float(params_.peakFloor);
    const double binHz = sampleRate_ / double(fftSize_);
    double total = 0, midTotal = 0, highTotal = 0;

    for (std::size_t i = minBin_; i <= maxBin_; ++i) {
        const float m = mag[i];
        if (m < floor || m <= mag[i - 1] || m < mag[i + 1]) continue;

        // Interpolación parabólica sobre el logaritmo de la magnitud.
        const double a = std::log(double(mag[i - 1]) + 1e-12);
        const double b = std::log(double(m) + 1e-12);
        const double c = std::log(double(mag[i + 1]) + 1e-12);
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
        const float w = float(std::pow(double(m), params_.gamma));  // compresión de amplitud
        out.chroma[std::size_t(lo % 36)] += w * (1.f - frac);
        out.chroma[std::size_t((lo + 1) % 36)] += w * frac;
        total += double(w);
        if (freq >= 250.0 && freq < 1000.0) {
            out.mid[std::size_t(lo % 36)] += w * (1.f - frac);
            out.mid[std::size_t((lo + 1) % 36)] += w * frac;
            midTotal += double(w);
        } else if (freq >= 1000.0) {
            out.high[std::size_t(lo % 36)] += w * (1.f - frac);
            out.high[std::size_t((lo + 1) % 36)] += w * frac;
            highTotal += double(w);
        }
    }

    if (total <= 0.0) return;
    for (float& v : out.chroma) v = float(double(v) / total);
    if (midTotal > 0.0) for (float& v : out.mid) v = float(double(v) / midTotal);
    if (highTotal > 0.0) for (float& v : out.high) v = float(double(v) / highTotal);

    // Espectro logarítmico: magnitud interpolada linealmente en el centro de cada bin.
    float specMax = 0.f;
    for (std::size_t k = 0; k < kSpecBins; ++k) {
        const double x = 55.0 * std::pow(2.0, double(k) / 36.0) / binHz;
        const std::size_t i0 = std::size_t(x);
        if (i0 + 1 >= magBins_) break;
        const double fr = x - double(i0);
        const float m = float((1.0 - fr) * double(mag[i0]) + fr * double(mag[i0 + 1]));
        out.spec[k] = m;
        specMax = std::max(specMax, m);
    }
    if (specMax > 0.f) for (float& v : out.spec) v = std::sqrt(v / specMax);
    out.silent = false;

    // Cromagrama del bajo: mismos picos y mismo reparto en 36 bins, solo 40-250 Hz.
    const std::size_t bLo = std::max<std::size_t>(2, std::size_t(std::ceil(kBassMinFreq / binHz)));
    const std::size_t bHi = std::min<std::size_t>(std::size_t(kBassMaxFreq / binHz), magBins_ - 2);
    float bassMax = 0.f;
    for (std::size_t i = bLo; i <= bHi; ++i) bassMax = std::max(bassMax, mag[i]);
    if (bassMax <= 0.f) return;
    const float bassFloor = bassMax * 0.1f;
    double bassTotal = 0;
    for (std::size_t i = bLo; i <= bHi; ++i) {
        const float m = mag[i];
        const float mPrev = mag[i - 1];
        const float mNext = mag[i + 1];
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
