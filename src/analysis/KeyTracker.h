#pragma once

#include <array>
#include <cstddef>
#include <deque>
#include <string>
#include <vector>

#include "analysis/Analyzer.h"
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
    // Redes (las mismas que en el análisis de archivos) sobre los últimos netWindowSeconds de audio, cada
    // netEverySeconds; sus log-probabilidades se promedian con memoria (vida media netHalfLifeSeconds) para no
    // perder la tonalidad principal en puentes o modulaciones. false = método clásico (perfil + bajo).
    bool nets = true;
    double netWindowSeconds = 30;
    double netEverySeconds = 3;      // ~28 % de un núcleo del M5 con las 7 redes (~15 % con las 5 anteriores)
    double netHalfLifeSeconds = 120;
    // Estabilidad: la tonalidad mostrada solo cambia si la nueva la supera en esta log-probabilidad media (mientras
    // tanto, la nueva va de 2ª). Con 0,3, la mitad de cambios por minuto con el mismo acierto. 0 = sin histéresis.
    double netHysteresis = 0.3;
    double doubtRatio = 0.5;       // igual que AnalysisOptions::doubtRatio
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
    bool fromNets = false;         // la tonalidad sale de las redes (si no, del método clásico)
    bool doubtful = false;         // la 2ª tonalidad es casi tan probable: se destacan las dos
    Key secondKey;                 // 2ª más probable (siempre que haya candidatos)
    std::string secondKeyName;
    bool secondSameNotes = false;
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
    // Para medir: número de pasadas de las redes y log-probabilidades de la última.
    std::size_t netUpdates() const { return netUpdates_; }
    const std::array<float, 24>& lastNetLogProbs() const { return lastLp_; }
    bool netWindowFull() const { return window_.size() >= maxPairs(); }

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
    // Redes
    void runNets();
    std::array<double, 24> netView() const;   // log-probabilidades que se muestran (última o con memoria)
    int shown_ = -1;                   // tonalidad mostrada (tónica*2 + menor), -1 al principio
    std::size_t maxPairs() const { return std::size_t(options_.netWindowSeconds / (2.0 * extractor_.hopSeconds())) + 1; }
    std::deque<FrameChroma> window_;   // fotogramas emparejados (~0,19 s) de los últimos netWindowSeconds
    FrameChroma pair_{};
    int pairCount_ = 0;
    double sinceNets_ = 0;
    std::array<double, 24> netSum_{};  // log-probabilidades acumuladas con memoria
    double netWeight_ = 0;
    std::size_t netUpdates_ = 0;
    std::array<float, 24> lastLp_{};
};

}  // namespace skale
