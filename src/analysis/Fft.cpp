#include "analysis/Fft.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace skale {

Fft::Fft(std::size_t size) : n_(size) {
    if (size < 2 || (size & (size - 1)) != 0) {
        throw std::invalid_argument("Fft: el tamaño debe ser potencia de dos");
    }
    unsigned bits = 0;
    while ((std::size_t(1) << bits) < size) ++bits;

    rev_.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
        std::size_t r = 0;
        for (unsigned b = 0; b < bits; ++b) {
            if (i & (std::size_t(1) << b)) r |= std::size_t(1) << (bits - 1 - b);
        }
        rev_[i] = r;
    }

    twiddle_.resize(size / 2);
    const double pi = std::acos(-1.0);
    for (std::size_t k = 0; k < size / 2; ++k) {
        const double a = -2.0 * pi * double(k) / double(size);
        twiddle_[k] = {float(std::cos(a)), float(std::sin(a))};
    }
}

void Fft::forward(std::vector<std::complex<float>>& a) const {
    for (std::size_t i = 0; i < n_; ++i) {
        if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
    }
    for (std::size_t len = 2; len <= n_; len <<= 1) {
        const std::size_t half = len / 2;
        const std::size_t step = n_ / len;
        for (std::size_t i = 0; i < n_; i += len) {
            for (std::size_t j = 0; j < half; ++j) {
                const float wr = twiddle_[j * step].real();
                const float wi = twiddle_[j * step].imag();
                const float xr = a[i + j + half].real();
                const float xi = a[i + j + half].imag();
                const float vr = xr * wr - xi * wi;
                const float vi = xr * wi + xi * wr;
                const float ur = a[i + j].real();
                const float ui = a[i + j].imag();
                a[i + j] = {ur + vr, ui + vi};
                a[i + j + half] = {ur - vr, ui - vi};
            }
        }
    }
}

}  // namespace skale
