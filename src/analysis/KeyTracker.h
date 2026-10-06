#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "analysis/ChordDetector.h"
#include "analysis/Chromagram.h"
#include "analysis/KeyDetector.h"
#include "analysis/Theory.h"

namespace skale {

struct TrackerOptions {
    KeyProfile profile = KeyProfile::Temperley;
    double halfLifeSeconds = 30;   // memoria: el audio de hace `halfLife` s pesa la mitad
    double bassWeight = 1;         // igual que AnalysisOptions::bassWeight
    bool solfege = false;
    std::size_t maxCandidates = 4;
};

struct TrackerSnapshot {
    bool valid = false;            // false hasta que haya audio suficiente
    double listenedSeconds = 0;    // audio no silencioso analizado
    Key key;
    std::string keyName;
    float confidence = 0;
    std::vector<KeyCandidate> candidates;   // la primera es la elegida
    std::vector<std::string> scaleNotes;
    Chroma12 chroma{};             // cromagrama acumulado (suma 1)
    bool hasChord = false;
    std::string chordName;         // acorde que suena ahora, "-" si no hay
};

// Seguimiento de la tonalidad en tiempo real: se alimenta con bloques de
// audio mono de cualquier tamaño y se consulta con snapshot(). No depende de
// JUCE. process() hace la FFT, así que en un plugin se llama desde un hilo
// de análisis, no desde el hilo de audio.
class KeyTracker {
public:
    explicit KeyTracker(double sampleRate, TrackerOptions options = {});

    void process(const float* mono, std::size_t n);
    void reset();

    TrackerSnapshot snapshot() const;

private:
    TrackerOptions options_;
    ChromaExtractor extractor_;
    ChordDetector chords_;
    Chroma36 acc_{};
    Chroma36 bassAcc_{};
    Chroma36 recent_{};            // cromagrama de los últimos fotogramas, para el acorde actual
    double decay_;                 // factor por fotograma
    double listened_ = 0;
    std::size_t activeFrames_ = 0;
};

}  // namespace skale
