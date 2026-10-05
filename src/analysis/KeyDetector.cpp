#include "analysis/KeyDetector.h"

#include <algorithm>
#include <cmath>

namespace skale {

namespace {

constexpr double kKsMajor[12] = {6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88};
constexpr double kKsMinor[12] = {6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17};
constexpr double kTempMajor[12] = {5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0};
constexpr double kTempMinor[12] = {5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0};

// Temperatura del softmax que convierte correlaciones en "confianza".
constexpr double kSoftmaxTemp = 0.07;

double pearson(const double* x, const double* y) {
    double mx = 0, my = 0;
    for (int i = 0; i < 12; ++i) { mx += x[i]; my += y[i]; }
    mx /= 12; my /= 12;
    double sxy = 0, sxx = 0, syy = 0;
    for (int i = 0; i < 12; ++i) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    if (sxx <= 1e-18 || syy <= 1e-18) return 0;
    return sxy / std::sqrt(sxx * syy);
}

}  // namespace

std::vector<KeyCandidate> KeyDetector::detect(const Chroma12& chroma, KeyProfile profile,
                                              const Chroma12* ending, double endingWeight) {
    double total = 0;
    for (float v : chroma) total += double(v);
    if (total <= 0.0) return {};

    const double* major = profile == KeyProfile::Temperley ? kTempMajor : kKsMajor;
    const double* minor = profile == KeyProfile::Temperley ? kTempMinor : kKsMinor;

    double x[12];
    for (int i = 0; i < 12; ++i) x[i] = double(chroma[std::size_t(i)]);

    std::vector<KeyCandidate> out;
    for (int tonic = 0; tonic < 12; ++tonic) {
        for (int m = 0; m < 2; ++m) {
            const double* base = m == 0 ? major : minor;
            double rotated[12];
            for (int i = 0; i < 12; ++i) rotated[i] = base[((i - tonic) % 12 + 12) % 12];
            KeyCandidate c;
            c.key = {tonic, m == 0 ? Mode::Major : Mode::Minor};
            c.correlation = float(pearson(x, rotated));
            if (ending && endingWeight > 0) {
                const int third = m == 0 ? 4 : 3;
                double triad = 0;
                for (int d : {0, third, 7}) triad += double((*ending)[std::size_t((tonic + d) % 12)]);
                c.correlation += float(endingWeight * triad);
            }
            out.push_back(c);
        }
    }

    std::sort(out.begin(), out.end(),
              [](const KeyCandidate& a, const KeyCandidate& b) { return a.correlation > b.correlation; });

    // Si todas las correlaciones son iguales (cromagrama plano) no hay respuesta.
    if (out.front().correlation - out.back().correlation < 1e-6f) return {};

    double denom = 0;
    for (const auto& c : out) denom += std::exp((double(c.correlation) - double(out.front().correlation)) / kSoftmaxTemp);
    for (auto& c : out) {
        c.confidence = float(std::exp((double(c.correlation) - double(out.front().correlation)) / kSoftmaxTemp) / denom);
    }
    return out;
}

}  // namespace skale
