#pragma once
// Pesos del modelo de tonalidad aprendido. Generado por tools/train_key_model.py: no editar a mano.
// Para cada tonalidad (tónica t, modo m; fila 0 = mayor, 1 = menor) la puntuación es
//   w[m][0] x corr_Temperley(chroma, t, m)
//   + sum_i w[m][1+i]  x bass_rotado[i]
//   + sum_i w[m][13+i] x ending_rotado[i]   (solo modelo "full")
//   + w[m][25] x 3 x voto[t,m]               (solo modelo "full"; voto = fracción de ventanas de 8 s
//                                              cuya mejor tonalidad con Temperley es (t,m))
// con cromagramas multiplicados por 12 y rotados para que el índice 0 sea la tónica.
namespace skale::model {
// Con final de la pieza: [corr, bajo(12), final(12), voto]
constexpr float kFull[2][26] = {
    {1.92800f, 0.84880f, -0.43627f, -0.45250f, -0.72574f, 0.31428f, -0.09512f, -0.39704f, 0.94506f, 0.33224f, 0.08157f, -0.53838f, -0.04321f, 0.85617f, -1.14815f, 0.14220f, 0.23550f, 0.12904f, -0.23117f, -0.22873f, 0.32069f, -0.44303f, -0.16150f, 0.22021f, 0.14225f, 1.09809f},
    {4.16433f, 0.74581f, -1.11291f, 0.02621f, 0.45637f, 0.27414f, -0.03868f, -0.40134f, 0.45748f, 0.47512f, -0.79006f, 0.06926f, 0.00490f, 0.61822f, -0.45160f, 0.29884f, -0.04345f, -0.52967f, -0.26945f, 0.24877f, 0.76656f, -0.57779f, -0.55829f, 0.59182f, 0.07255f, 0.61099f}
};

// Sin final (tiempo real): [corr, bajo(12)]
constexpr float kLive[2][13] = {
    {6.87975f, 0.92732f, -0.62073f, -0.29864f, -0.48199f, 0.19654f, -0.28087f, -0.49983f, 0.95321f, 0.21892f, -0.14916f, -0.51579f, -0.12363f},
    {6.68524f, 0.81724f, -1.36433f, 0.29363f, 0.49660f, -0.02882f, 0.06862f, -0.27964f, 0.62322f, 0.28313f, -0.61704f, 0.29372f, 0.08831f}
};
}  // namespace skale::model
