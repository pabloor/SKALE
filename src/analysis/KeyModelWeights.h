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
    {3.16896f, 0.65068f, -0.21780f, -0.42818f, -0.68702f, 0.25559f, -0.10195f, -0.29135f, 0.67720f, 0.30626f, 0.08630f, -0.47624f, -0.05393f, 0.80351f, -1.44156f, 0.15639f, 0.23475f, 0.17151f, -0.21739f, -0.13692f, 0.37655f, -0.50578f, -0.08416f, 0.18605f, 0.17618f, 0.78507f},
    {5.64774f, 0.77786f, -0.71296f, -0.20474f, 0.34462f, 0.29819f, -0.27575f, 0.06209f, 0.36488f, 0.17277f, -0.55948f, 0.14122f, -0.12826f, 0.51306f, -0.44713f, 0.21527f, -0.02136f, -0.41931f, -0.31973f, 0.46087f, 0.68515f, -0.60470f, -0.38185f, 0.68350f, -0.08290f, 0.11586f}
};

// Sin final (tiempo real): [corr, bajo(12)]
constexpr float kLive[2][13] = {
    {8.24348f, 0.71564f, -0.51741f, -0.39300f, -0.46102f, 0.08864f, -0.36086f, -0.30901f, 0.71627f, 0.25692f, -0.25846f, -0.42428f, -0.17174f},
    {7.09117f, 0.84636f, -0.93311f, 0.03934f, 0.44412f, 0.17968f, -0.23796f, 0.23846f, 0.55008f, 0.06296f, -0.39579f, 0.42017f, -0.09599f}
};
}  // namespace skale::model
