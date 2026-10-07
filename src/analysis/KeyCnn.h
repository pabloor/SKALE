#pragma once

#include <array>
#include <vector>

#include "analysis/Analyzer.h"

namespace skale {

// Red convolucional de tonalidad (tools/train_key_cnn.py) sobre la serie de cromagramas finos.
// Equivariante a la transposición: 24 log-probabilidades (tónica * 2 + menor) promediadas
// sobre ventanas (128 o 256 fotogramas según la red) con paso de media ventana, y entre redes.
// Devuelve ceros si hay menos de 8 fotogramas.
std::array<float, 24> cnnKeyLogProbs(const std::vector<FrameChroma>& frames);

}  // namespace skale
