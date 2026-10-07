#include "analysis/KeyCnn.h"

#include <algorithm>
#include <cmath>

#include "analysis/KeyCnnWeights.h"

namespace skale {

namespace {

constexpr int kT = 128;      // fotogramas por ventana
constexpr int kP = 36;       // bins de tono

struct Tensor {              // (canales, tiempo, tono)
    int c, t;
    std::vector<float> v;
    Tensor(int c_, int t_) : c(c_), t(t_), v(std::size_t(c_) * std::size_t(t_) * kP, 0.f) {}
    float& at(int ci, int ti, int pi) { return v[(std::size_t(ci) * std::size_t(t) + std::size_t(ti)) * kP + std::size_t(pi)]; }
    float at(int ci, int ti, int pi) const { return v[(std::size_t(ci) * std::size_t(t) + std::size_t(ti)) * kP + std::size_t(pi)]; }
};

// Convolución (3 tiempo x 9 tono) con tono circular y tiempo replicado, más ReLU.
Tensor convRelu(const Tensor& in, const float* w, const float* b, int cOut) {
    Tensor out(cOut, in.t);
    for (int co = 0; co < cOut; ++co) {
        for (int ti = 0; ti < in.t; ++ti) {
            for (int pi = 0; pi < kP; ++pi) {
                float acc = b[co];
                for (int ci = 0; ci < in.c; ++ci) {
                    const float* wk = w + ((std::size_t(co) * std::size_t(in.c) + std::size_t(ci)) * 27);
                    for (int dt = 0; dt < 3; ++dt) {
                        const int tt = std::clamp(ti + dt - 1, 0, in.t - 1);
                        for (int dp = 0; dp < 9; ++dp) {
                            const int pp = (pi + dp - 4 + kP) % kP;
                            acc += wk[dt * 9 + dp] * in.at(ci, tt, pp);
                        }
                    }
                }
                out.at(co, ti, pi) = acc > 0.f ? acc : 0.f;
            }
        }
    }
    return out;
}

void logSoftmax(std::array<float, 24>& x) {
    float m = *std::max_element(x.begin(), x.end());
    double s = 0;
    for (float v : x) s += std::exp(double(v - m));
    const float lse = m + float(std::log(s));
    for (float& v : x) v -= lse;
}

// Una ventana de kT fotogramas -> 24 log-probabilidades.
std::array<float, 24> windowLogProbs(const std::vector<float>& x /* [kT][2][36] */, const cnn::Net& net) {
    Tensor in(2, kT);
    for (int t = 0; t < kT; ++t)
        for (int c = 0; c < 2; ++c)
            for (int p = 0; p < kP; ++p) in.at(c, t, p) = x[(std::size_t(t) * 2 + std::size_t(c)) * kP + std::size_t(p)];

    Tensor h = convRelu(in, net.c1w, net.c1b, cnn::kCh);
    h = convRelu(h, net.c2w, net.c2b, cnn::kCh);
    h = convRelu(h, net.c3w, net.c3b, cnn::kCh);

    // agrupación media y máxima en el tiempo: (2*ch, 36)
    std::vector<float> f(std::size_t(2 * cnn::kCh) * kP);
    for (int c = 0; c < cnn::kCh; ++c) {
        for (int p = 0; p < kP; ++p) {
            float sum = 0.f, mx = h.at(c, 0, p);
            for (int t = 0; t < kT; ++t) { const float v = h.at(c, t, p); sum += v; mx = std::max(mx, v); }
            f[std::size_t(c) * kP + std::size_t(p)] = sum / float(kT);
            f[std::size_t(cnn::kCh + c) * kP + std::size_t(p)] = mx;
        }
    }

    // cabeza equivariante: para cada posición p, características giradas (índice canal*36 + desfase)
    float score[kP][2];
    std::vector<float> hid(std::size_t(cnn::kHidden));
    for (int p = 0; p < kP; ++p) {
        for (int j = 0; j < cnn::kHidden; ++j) {
            const float* w = net.h1w + std::size_t(j) * std::size_t(2 * cnn::kCh * kP);
            float acc = net.h1b[j];
            for (int c = 0; c < 2 * cnn::kCh; ++c)
                for (int o = 0; o < kP; ++o) acc += w[c * kP + o] * f[std::size_t(c) * kP + std::size_t((p + o) % kP)];
            hid[std::size_t(j)] = acc > 0.f ? acc : 0.f;
        }
        for (int m = 0; m < 2; ++m) {
            float acc = net.h2b[m];
            for (int j = 0; j < cnn::kHidden; ++j) acc += net.h2w[m * cnn::kHidden + j] * hid[std::size_t(j)];
            score[p][m] = acc;
        }
    }

    std::array<float, 24> logits{};
    for (int t = 0; t < 12; ++t) {
        for (int m = 0; m < 2; ++m) {
            const float a = score[(3 * t + kP - 1) % kP][m], b = score[3 * t][m], c = score[(3 * t + 1) % kP][m];
            const float mx = std::max({a, b, c});
            logits[std::size_t(t * 2 + m)] = mx + std::log(std::exp(a - mx) + std::exp(b - mx) + std::exp(c - mx));
        }
    }
    logSoftmax(logits);
    return logits;
}

}  // namespace

std::array<float, 24> cnnKeyLogProbs(const std::vector<FrameChroma>& frames) {
    std::array<float, 24> out{};
    if (frames.size() < 8) return out;

    // entrada (n, 2, 36) con raíz cuadrada x 3; si hay menos de kT fotogramas, se repite
    std::vector<float> x;
    const std::size_t n0 = frames.size();
    const std::size_t n = std::max<std::size_t>(n0, kT);
    x.reserve(n * 2 * kP);
    for (std::size_t i = 0; i < n; ++i) {
        const FrameChroma& f = frames[i % n0];
        for (std::size_t p = 0; p < kP; ++p) x.push_back(3.f * std::sqrt(std::max(f.chroma[p], 0.f)));
        for (std::size_t p = 0; p < kP; ++p) x.push_back(3.f * std::sqrt(std::max(f.bass[p], 0.f)));
    }

    std::vector<std::size_t> starts;
    for (std::size_t s = 0; s + kT <= n; s += kT / 2) starts.push_back(s);
    if (starts.empty()) starts.push_back(0);

    std::vector<float> win(std::size_t(kT) * 2 * kP);
    for (std::size_t s : starts) {
        std::copy(x.begin() + std::ptrdiff_t(s * 2 * kP), x.begin() + std::ptrdiff_t((s + kT) * 2 * kP), win.begin());
        for (int k = 0; k < cnn::kNetCount; ++k) {
            const auto lp = windowLogProbs(win, cnn::kNets[k]);
            for (std::size_t i = 0; i < 24; ++i) out[i] += lp[i] / float(starts.size()) / float(cnn::kNetCount);
        }
    }
    return out;
}

}  // namespace skale
