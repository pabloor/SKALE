#pragma once

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
    static std::vector<KeyCandidate> detect(const Chroma12& chroma,
                                            KeyProfile profile = KeyProfile::Temperley);
};

}  // namespace skale
