#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <deque>
#include <vector>

#include "analysis/Fft.h"

namespace skale {

// 36 bins: 3 por semitono. El bin 3*n es la clase de nota n (0 = Do) afinada
// a A=440; los bins vecinos recogen las desafinaciones de ±33 cents.
using Chroma36 = std::array<float, 36>;
using Chroma12 = std::array<float, 12>;
// Espectro en escala logarítmica: 6 octavas desde La1 (55 Hz) a 3 bins por semitono (216 bins).
constexpr std::size_t kSpecBins = 216;
using LogSpec = std::array<float, kSpecBins>;

struct ChromaFrame {
    Chroma36 chroma{};   // suma 1, o todo ceros si el fotograma es silencio
    Chroma36 bass{};     // lo mismo pero solo entre 40 y 250 Hz (el bajo); ceros si no hay picos
    Chroma36 mid{};      // 250-1000 Hz; ceros si no hay picos
    Chroma36 high{};     // 1000 Hz hasta maxFreq; ceros si no hay picos
    LogSpec spec{};      // magnitud interpolada en 55 Hz * 2^(k/36), raíz de (m / máximo); ceros si es silencio
    double time = 0;     // segundos del centro de la ventana
    bool silent = true;
};

struct ChromaParams {
    double minFreq = 65.0;      // rango de picos espectrales usado para el cromagrama
    double maxFreq = 2100.0;
    double gamma = 0.5;         // compresión de amplitud: peso = magnitud^gamma
    double peakFloor = 0.03;    // pico mínimo respecto al mayor del fotograma
    // Separación de armonía y percusión (HPSS, filtros de mediana) antes de buscar picos: la magnitud de cada
    // fotograma se multiplica por H²/(H²+P²), con H = mediana en el tiempo (hpssTimeFrames fotogramas
    // centrados) y P = mediana en frecuencia (hpssFreqBins bins). Retrasa la salida (hpssTimeFrames-1)/2 fotogramas.
    bool hpss = false;
    int hpssTimeFrames = 9;
    int hpssFreqBins = 17;
};

// Extrae un cromagrama fino de audio mono en streaming: se puede alimentar en
// bloques de cualquier tamaño y devuelve los fotogramas que se van completando.
class ChromaExtractor {
public:
    explicit ChromaExtractor(double sampleRate, ChromaParams params = {});

    std::vector<ChromaFrame> process(const float* mono, std::size_t n);
    // Al final del audio: con HPSS devuelve los fotogramas que aún esperaban a los siguientes. Sin HPSS, nada.
    std::vector<ChromaFrame> flush();
    void reset();

    std::size_t fftSize() const { return fftSize_; }
    double hopSeconds() const { return double(hop_) / sampleRate_; }

private:
    struct SpecFrame {
        std::vector<float> mag;   // magnitud de los bins 0..magBins_-1 (ceros si es silencio)
        double time = 0;
        bool silent = true;
    };
    void spectrum(const float* x, SpecFrame& f);
    void analyseMagnitude(const float* mag, ChromaFrame& out);
    void emitHpss(std::vector<ChromaFrame>& out, bool final);

    double sampleRate_;
    ChromaParams params_;
    std::size_t fftSize_;
    std::size_t hop_;
    std::size_t minBin_;
    std::size_t maxBin_;
    Fft fft_;
    std::vector<float> window_;
    std::vector<float> pending_;
    std::size_t pendingStart_ = 0;     // índice de pending_[0] en el flujo
    std::size_t nextFrameStart_ = 0;
    std::size_t magBins_;
    std::vector<std::complex<float>> buf_;
    std::deque<SpecFrame> hist_;       // fotogramas guardados para la mediana temporal (HPSS)
    std::size_t histStart_ = 0;        // índice del fotograma hist_[0]
    std::size_t framesIn_ = 0;         // fotogramas calculados
    std::size_t nextOut_ = 0;          // siguiente fotograma por analizar (HPSS)
    std::vector<float> masked_, column_;
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
