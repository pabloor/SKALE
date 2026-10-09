# Skale: notas de traspaso para Claude Code

Detector de tonalidad y escala (C++17, sin dependencias) con CLI (`skale-cli`) y plugin JUCE (VST3/AU +
aplicación independiente). El usuario habla español: responde en español. Detalles y tablas en `README.md`.

## Objetivo actual
**Subir la precisión de la tonalidad en archivos.** Todas las comparaciones se hacen sobre la misma
partición de prueba fija (`tools/samples/test_split.json`, 1.504 pistas: FMA 1.096, Beatport 220,
GuitarSet 72, GiantSteps+ 51, resto pequeños). Métrica principal: acierto exacto en FMA (música real
variada) y en el total; también MIREX (`tools/eval_key_cnn.py`).

## Cómo analiza ahora (por defecto, `--model ensemble`)
Log-probabilidades de 24 tonalidades = 0,85 × redes + 0,15 × modelo lineal (`KeyModelWeights.h`;
`options.cnnWeight` en `src/analysis/Analyzer.h`, `--cnn-weight` en la CLI). Redes = 0,5 × media de 3 redes
de cromagrama + 0,5 × media de 2 redes de espectro (`src/analysis/KeyCnn.cpp`, pesos en `KeyCnnWeights.h`).
Las redes son equivariantes a la transposición. Entrada por fotograma (~0,19 s): cromagrama fino de
36 bins/octava de 4 bandas (cromagrama, bajo, medios, agudos) y espectro logarítmico de 216 bins.
Coste: ≈ 7 s para una pieza de 5,7 min en un M5 (3,5 s con el modelo anterior de 3 redes).

## Resultados en la partición fija (acierto exacto, 1.497 pistas disponibles)
| Modelo | FMA | Beatport | GuitarSet | GS+ | Total |
|---|---|---|---|---|---|
| Lineal | 50,0 | 52,7 | 63,9 | 72,5 | 53,3 |
| 3 redes de cromagrama | 60,7 | 69,4 | 81,9 | 86,3 | 64,7 |
| Anterior: 0,7 × 3 redes + 0,3 × lineal (lineal que vio la prueba) | 60,6 | 70,0 | 83,3 | 88,2 | 65,1 |
| Una red de espectro sola (6 semillas/anchuras) | 59,5–61,5 | 63–69 | 75–79 | 88–94 | 63,1–65,2 |
| 3 cromagrama + 2 espectro (`split_spec`, `split_spec_w24`), sin lineal | 62,2 | 68,9 | 83,3 | 92,2 | 66,2 |
| …con otras 4 redes de espectro (semillas nuevas), sin lineal | 62,4 | 69,4 | 83,3 | 90,2 | 66,3 |
| **Actual: 0,85 × (3 cromagrama + 2 espectro) + 0,15 × lineal** | 62,6 | 70,3 | 84,7 | 88,2 | 66,7 |
| …con 0,3 de lineal | 60,9 | 69,9 | 81,9 | 88,2 | 65,3 |

Las filas con lineal se midieron con skale-cli (C++) y un lineal reentrenado sin la partición de prueba;
las demás con `tools/eval_key_cnn.py` (C++ y PyTorch dan lo mismo). La mejora de las redes de espectro
(+1,7 FMA) se confirmó con dos juegos de semillas; el lineal a 0,15 da +0,4 (ruido, pero no resta).

## Siguiente paso
Ideas sin probar: más redes **distintas** (p. ej. espectro con ventana larga o de 4 bandas); reducir coste
(la red de espectro ancha es la más lenta); comprobar el plugin (`build-plugin`) con los nuevos pesos.

## Modelos entrenados (`models/`, formato {'state','cfg'})
- `final_c2.pt`, `final_c4.pt`, `final_c4_long.pt`: las 3 redes de cromagrama de producción (todos los datos).
- `final_spec.pt`, `final_spec_w24.pt`: las 2 redes de espectro de producción (todos los datos, anchura 16 y 24).
  Exportar las 5: `tools/export_key_cnn.py models/final_c2.pt models/final_c4.pt models/final_c4_long.pt
  models/final_spec.pt models/final_spec_w24.pt --spec-weight 0.5`.
- `split_*.pt`: entrenadas sin la partición de prueba, para comparar (`split_c2`, `split_c4`,
  `split_c4_long`, `split_spec`, `split_spec_w24`).

## Datos y entrenamiento
Pasos completos en `docs/entrenar_en_mac.md`: `tools/fetch_datasets.py` (descarga ~7 GB) →
`tools/prepare_training.py` (características con skale-cli) → `tools/train_key_cnn.py`
(`--spec` para la red de espectro, `--split tools/samples/test_split.json`, `--device auto` usa MPS)
→ `tools/eval_key_cnn.py` → `tools/export_key_cnn.py`. En este Mac (M5): `.venv` con PyTorch (MPS, ~5 min
por red), datos en `~/skale-datos` y características en `~/skale-train`; para descargar hace falta
`SSL_CERT_FILE=~/.skale-cacerts.pem` (certificados del llavero; ver `docs/entrenar_en_mac.md`). Configuración usada: `--epochs 30 --steps 100`;
red de espectro ancha `--ch 24`.

## Ya probado sin mejora (no repetir sin una idea nueva)
Parámetros del cromagrama; cromagrama del inicio y cadencias; árboles de gradiente; más datos del
dominio para el lineal; redes de cromagrama más anchas o profundas; aumentos en prueba (TTA);
corrección del sesgo mayor/menor; autoentrenamiento con 20.000 pistas FMA sin etiquetar
(los alumnos igualan al conjunto, no lo superan). Lo que sí funcionó: promediar redes **distintas**.

## Normas
- No descargar música comercial o con derechos; solo conjuntos con licencia abierta o de investigación.
  Pixabay bloquea el acceso automático y no tiene API de música; el usuario paró la descarga de ccMixter.
- Commits en `main`. Los pesos van compilados en el binario (sin archivos externos).
- Licencia de JUCE: AGPL o comercial; para distribuir el plugin cerrado hace falta la comercial.
