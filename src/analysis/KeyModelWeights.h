#pragma once
// Pesos del modelo de tonalidad aprendido. Generado por tools/train_key_model.py: no editar a mano.
// Para cada tonalidad (tónica t, modo m; fila 0 = mayor, 1 = menor) la puntuación es
//   w[m][0] x corr_Temperley(chroma, t, m)
//   + sum_i w[m][1+i]  x bass_rotado[i]
//   + sum_i w[m][13+i] x ending_rotado[i]   (solo modelo "full")
// con cromagramas multiplicados por 12 y rotados para que el índice 0 sea la tónica.
namespace skale::model {
// Con final de la pieza: [corr, bajo(12), final(12)]
constexpr float kFull[2][25] = {
    {4.58178f, 0.81771f, -0.16934f, -0.39461f, -0.84711f, 0.66059f, -0.21317f, -0.61462f, 0.85106f, 0.36010f, -0.09374f, -0.52897f, -0.08672f, 0.89628f, -1.70196f, 0.08253f, 0.24633f, 0.28549f, -0.18147f, -0.43361f, 0.39625f, -0.10256f, -0.22005f, 0.04075f, 0.43300f},
    {5.22921f, 0.76404f, -1.24610f, 0.09793f, 0.47144f, 0.20278f, 0.03878f, -0.40715f, 0.39769f, 0.47473f, -0.64773f, 0.02207f, 0.09033f, 0.65372f, -0.36498f, 0.28601f, 0.04806f, -0.41066f, -0.29750f, 0.20740f, 0.76030f, -0.50053f, -0.68178f, 0.58314f, -0.02417f}
};

// Sin final (tiempo real): [corr, bajo(12)]
constexpr float kLive[2][13] = {
    {6.72378f, 0.87423f, -0.69611f, -0.29133f, -0.53081f, 0.43584f, -0.44980f, -0.50309f, 0.91310f, 0.43261f, -0.44558f, -0.34580f, -0.01401f},
    {6.78398f, 0.74542f, -1.38563f, 0.31759f, 0.49603f, 0.02012f, -0.02107f, -0.14753f, 0.49377f, 0.27650f, -0.57093f, 0.26618f, 0.13031f}
};
}  // namespace skale::model
