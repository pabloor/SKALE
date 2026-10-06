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
    {4.71516f, 0.84968f, -0.16549f, -0.43004f, -0.69799f, 0.36594f, -0.00694f, -0.63769f, 0.91631f, 0.19714f, 0.18044f, -0.73117f, -0.10705f, 0.92257f, -1.32661f, 0.07746f, 0.24280f, 0.25478f, -0.20638f, -0.43201f, 0.38921f, -0.43739f, -0.04103f, 0.02529f, 0.26425f},
    {5.15994f, 0.80323f, -1.25206f, 0.09616f, 0.48675f, 0.15894f, 0.09097f, -0.52229f, 0.52632f, 0.44008f, -0.67278f, 0.07138f, 0.04015f, 0.67803f, -0.39912f, 0.32771f, 0.05087f, -0.56309f, -0.26427f, 0.24774f, 0.77133f, -0.52742f, -0.61275f, 0.54188f, 0.01615f}
};

// Sin final (tiempo real): [corr, bajo(12)]
constexpr float kLive[2][13] = {
    {6.87975f, 0.92732f, -0.62073f, -0.29864f, -0.48199f, 0.19654f, -0.28087f, -0.49983f, 0.95321f, 0.21892f, -0.14916f, -0.51579f, -0.12363f},
    {6.68524f, 0.81724f, -1.36433f, 0.29363f, 0.49660f, -0.02882f, 0.06862f, -0.27964f, 0.62322f, 0.28313f, -0.61704f, 0.29372f, 0.08831f}
};
}  // namespace skale::model
