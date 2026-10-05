#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace skale {

// FFT compleja radix-2 en el sitio. El tamaño debe ser potencia de dos.
class Fft {
public:
    explicit Fft(std::size_t size);

    std::size_t size() const { return n_; }
    void forward(std::vector<std::complex<float>>& data) const;

private:
    std::size_t n_;
    std::vector<std::size_t> rev_;
    std::vector<std::complex<float>> twiddle_;
};

}  // namespace skale
