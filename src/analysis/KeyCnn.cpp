#include "analysis/KeyCnn.h"

#include <algorithm>
#include <cmath>

#include "analysis/KeyCnnWeights.h"

namespace skale {

namespace {

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
    const float m = *std::max_element(x.begin(), x.end());
    double s = 0;
    for (float v : x) s += std::exp(double(v - m));
    const float lse = m + float(std::log(s));
    for (float& v : x) v -= lse;
}

// Una ventana de net.crop fotogramas, x = [crop][inCh][36] -> 24 log-probabilidades.
std::array<float, 24> windowLogProbs(const float* x, const cnn::Net& net) {
    const int T = net.crop;
    Tensor in(net.inCh, T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < net.inCh; ++c)
            for (int p = 0; p < kP; ++p) in.at(c, t, p) = x[(std::size_t(t) * std::size_t(net.inCh) + std::size_t(c)) * kP + std::size_t(p)];

    Tensor h = convRelu(in, net.c1w, net.c1b, net.ch);
    h = convRelu(h, net.c2w, net.c2b, net.ch);
    h = convRelu(h, net.c3w, net.c3b, net.ch);

    // agrupación media y máxima en el tiempo: (2*ch, 36)
    std::vector<float> f(std::size_t(2 * net.ch) * kP);
    for (int c = 0; c < net.ch; ++c) {
        for (int p = 0; p < kP; ++p) {
            float sum = 0.f, mx = h.at(c, 0, p);
            for (int t = 0; t < T; ++t) { const float v = h.at(c, t, p); sum += v; mx = std::max(mx, v); }
            f[std::size_t(c) * kP + std::size_t(p)] = sum / float(T);
            f[std::size_t(net.ch + c) * kP + std::size_t(p)] = mx;
        }
    }

    // cabeza equivariante: para cada posición p, características giradas (índice canal*36 + desfase)
    float score[kP][2];
    std::vector<float> hid(std::size_t(net.hidden));
    const std::size_t inW = std::size_t(2 * net.ch * kP);
    for (int p = 0; p < kP; ++p) {
        for (int j = 0; j < net.hidden; ++j) {
            const float* w = net.h1w + std::size_t(j) * inW;
            float acc = net.h1b[j];
            for (int c = 0; c < 2 * net.ch; ++c)
                for (int o = 0; o < kP; ++o) acc += w[c * kP + o] * f[std::size_t(c) * kP + std::size_t((p + o) % kP)];
            hid[std::size_t(j)] = acc > 0.f ? acc : 0.f;
        }
        for (int m = 0; m < 2; ++m) {
            float acc = net.h2b[m];
            for (int j = 0; j < net.hidden; ++j) acc += net.h2w[m * net.hidden + j] * hid[std::size_t(j)];
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

// Media de log-probabilidades de una red sobre ventanas con paso crop/2.
std::array<float, 24> netLogProbs(const std::vector<FrameChroma>& frames, const cnn::Net& net) {
    const std::size_t T = std::size_t(net.crop), C = std::size_t(net.inCh);
    // entrada (n, inCh, 36) con raíz cuadrada x 3; si hay menos de T fotogramas, se repite
    const std::size_t n0 = frames.size(), n = std::max(n0, T);
    std::vector<float> x;
    x.reserve(n * C * kP);
    for (std::size_t i = 0; i < n; ++i) {
        const FrameChroma& f = frames[i % n0];
        const Chroma36* chans[4] = {&f.chroma, &f.bass, &f.mid, &f.high};
        for (std::size_t c = 0; c < C; ++c)
            for (std::size_t p = 0; p < kP; ++p) x.push_back(3.f * std::sqrt(std::max((*chans[c])[p], 0.f)));
    }
    std::vector<std::size_t> starts;
    for (std::size_t s = 0; s + T <= n; s += T / 2) starts.push_back(s);
    if (starts.empty()) starts.push_back(0);

    std::array<float, 24> out{};
    for (std::size_t s : starts) {
        const auto lp = windowLogProbs(x.data() + s * C * kP, net);
        for (std::size_t i = 0; i < 24; ++i) out[i] += lp[i] / float(starts.size());
    }
    return out;
}

}  // namespace

std::array<float, 24> cnnKeyLogProbs(const std::vector<FrameChroma>& frames) {
    std::array<float, 24> out{};
    if (frames.size() < 8) return out;
    for (int k = 0; k < cnn::kNetCount; ++k) {
        const auto lp = netLogProbs(frames, cnn::kNets[k]);
        for (std::size_t i = 0; i < 24; ++i) out[i] += lp[i] / float(cnn::kNetCount);
    }
    return out;
}

}  // namespace skale
