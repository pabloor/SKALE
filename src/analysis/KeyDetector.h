#pragma once

#include <array>
#include <vector>

#include "analysis/Chromagram.h"
#include "analysis/Theory.h"

namespace skale {

enum class KeyProfile { KrumhanslSchmuckler, Temperley };

struct KeyCandidate {
    Key key;
    float correlation = 0;  // Pearson con el perfil, -1..1
    float confidence = 0;   // 0..1, softmax de las 24 correlaciones (heurística)
};

class KeyDetector {
public:
    // Devuelve las 24 tonalidades ordenadas de más a menos probable. Si el
    // cromagrama es plano o está vacío, devuelve una lista vacía.
    //
    // `ending` es el cromagrama (suma 1) de los últimos segundos de la pieza.
    // Si se da y `endingWeight` > 0, se suma a la correlación de cada
    // tonalidad `endingWeight` x energía de su tríada tónica en el final: las
    // piezas suelen acabar en la tónica, lo que desempata mayor/relativo menor.
    // `extra`: puntuación que se suma a la correlación de cada tonalidad, indexada
    // por tonic * 2 + (menor ? 1 : 0) (bajo, acordes...).
    // `endingMargin`: el bonus solo se da a las tonalidades cuya correlación
    // base está a menos de este margen de la mejor (el final solo desempata).
    static std::vector<KeyCandidate> detect(const Chroma12& chroma,
                                            KeyProfile profile = KeyProfile::Temperley,
                                            const Chroma12* ending = nullptr,
                                            double endingWeight = 0,
                                            double endingMargin = 1e9,
                                            const std::array<float, 24>* extra = nullptr);

    // Puntuación extra por tonalidad a partir del cromagrama del bajo (suma 1):
    // weight x (tónica + 0,5 x quinta). Indexada como `extra` de detect().
    // Modelo aprendido (ver KeyModelWeights.h): ordena las 24 tonalidades con una
    // puntuación lineal sobre la correlación de Temperley, el bajo y (si se da)
    // el final. `confidence` es el softmax de esas puntuaciones (logits).
    static std::vector<KeyCandidate> detectLearned(const Chroma12& chroma, const Chroma12& bass,
                                                   const Chroma12* ending);

    static std::array<float, 24> bassScores(const Chroma12& bass, double weight);
};

}  // namespace skale
