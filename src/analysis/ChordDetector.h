#pragma once

#include <array>
#include <vector>

#include "analysis/Chromagram.h"
#include "analysis/Theory.h"

namespace skale {

struct ChordLabel {
    bool none = true;  // sin acorde (silencio o ruido)
    int root = 0;
    ChordType type = ChordType::Major;

    bool operator==(const ChordLabel& o) const {
        return none == o.none && (none || (root == o.root && type == o.type));
    }
};

struct ChordSegment {
    double start = 0;
    double end = 0;
    ChordLabel chord;
};

struct ChordFrame {
    Chroma12 chroma{};  // ya plegado con la afinación global
    double time = 0;
    bool silent = true;
};

// Reconocimiento de acordes por plantillas (similitud coseno) suavizado con
// Viterbi para que un acorde no se parta por notas de paso.
class ChordDetector {
public:
    struct Params {
        float emissionScale = 8.f;   // peso de la similitud frente a la penalización
        float switchPenalty = 6.f;   // coste de cambiar de acorde entre fotogramas
        float noChordSimilarity = 0.3f;
    };

    ChordDetector();
    explicit ChordDetector(Params p);

    std::vector<ChordSegment> detect(const std::vector<ChordFrame>& frames, double hopSeconds) const;

    // Mejor acorde para un único cromagrama, sin suavizado (tiempo real).
    ChordLabel detectSingle(const Chroma12& chroma) const;

private:
    Params params_;
    // 8 tipos × 12 raíces, normalizadas a norma 1, más un factor de preferencia.
    std::vector<std::array<float, 12>> templates_;
    std::vector<float> prior_;
};

}  // namespace skale
