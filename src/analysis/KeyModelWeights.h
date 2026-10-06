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
    {2.81707f, 0.79173f, -0.29150f, -0.47432f, -0.79191f, 0.31883f, -0.09149f, -0.47142f, 0.82991f, 0.33134f, 0.14232f, -0.51758f, -0.12947f, 0.90974f, -1.60443f, 0.16202f, 0.23861f, 0.21262f, -0.21304f, -0.26082f, 0.45353f, -0.60114f, -0.08353f, 0.21993f, 0.21265f, 0.89663f},
    {5.10915f, 0.93901f, -0.79363f, -0.23465f, 0.41504f, 0.32106f, -0.21622f, -0.10166f, 0.47675f, 0.23688f, -0.65600f, 0.15138f, -0.18441f, 0.60117f, -0.51361f, 0.25489f, -0.00634f, -0.44613f, -0.26379f, 0.39121f, 0.79967f, -0.62306f, -0.50994f, 0.71012f, -0.04032f, 0.19798f}
};

// Sin final (tiempo real): [corr, bajo(12)]
constexpr float kLive[2][13] = {
    {8.02503f, 0.85672f, -0.61673f, -0.36794f, -0.54431f, 0.14840f, -0.34495f, -0.50029f, 0.88118f, 0.21987f, -0.21497f, -0.46082f, -0.21207f},
    {6.83747f, 1.00207f, -1.03682f, 0.03995f, 0.50427f, 0.16662f, -0.14017f, 0.05583f, 0.65853f, 0.11310f, -0.47520f, 0.42083f, -0.15310f}
};
}  // namespace skale::model
