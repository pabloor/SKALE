#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "analysis/Fft.h"

namespace skale {

// 36 bins: 3 por semitono. El bin 3*n es la clase de nota n (0 = Do) afinada
// a A=440; los bins vecinos recogen las desafinaciones de ±33 cents.
using Chroma36 = std::array<float, 36>;
using Chroma12 = std::array<float, 12>;

struct ChromaFrame {
    Chroma36 chroma{};   // suma 1, o todo ceros si el fotograma es silencio
    double time = 0;     // segundos del centro de la ventana
    bool silent = true;
};

// Extrae un cromagrama fino de audio mono en streaming: se puede alimentar en
// bloques de cualquier tamaño y devuelve los fotogramas que se van completando.
class ChromaExtractor {
public:
    explicit ChromaExtractor(double sampleRate);

    std::vector<ChromaFrame> process(const float* mono, std::size_t n);
    void reset();

    std::size_t fftSize() const { return fftSize_; }
    double hopSeconds() const { return double(hop_) / sampleRate_; }

private:
    void analyseFrame(const float* x, ChromaFrame& out);

    double sampleRate_;
    std::size_t fftSize_;
    std::size_t hop_;
    std::size_t minBin_;
    std::size_t maxBin_;
    Fft fft_;
    std::vector<float> window_;
    std::vector<float> pending_;
    std::size_t pendingStart_ = 0;     // índice de pending_[0] en el flujo
    std::size_t nextFrameStart_ = 0;
    std::vector<std::complex<float>> buf_;
    std::vector<float> mag_;
};

// Desviación de afinación global estimada del cromagrama acumulado:
// -1 (≈33 cents bajo), 0 (A=440) o +1 (≈33 cents alto).
int estimateTuning(const Chroma36& accumulated);
double tuningToCents(int tuning);

// Pliega 36 bins a 12 clases de nota compensando la afinación. Sin normalizar.
Chroma12 foldChroma(const Chroma36& c, int tuning);

// Normaliza a suma 1; si la suma es 0 deja ceros.
void normalize(Chroma12& c);

}  // namespace skale
