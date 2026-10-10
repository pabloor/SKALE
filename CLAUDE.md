# Skale: notas de traspaso para Claude Code

Detector de tonalidad y escala (C++17, sin dependencias) con CLI (`skale-cli`) y plugin JUCE (VST3/AU +
aplicación independiente). El usuario habla español: responde en español. Detalles y tablas en `README.md`.

## Objetivo actual
**Subir la precisión de la tonalidad en archivos.** Todas las comparaciones se hacen sobre la misma
partición de prueba fija (`tools/samples/test_split.json`, 1.504 pistas: FMA 1.096, Beatport 220,
GuitarSet 72, GiantSteps+ 51, resto pequeños). Métrica principal: acierto exacto en FMA (música real
variada) y en el total; también MIREX (`tools/eval_key_cnn.py`).

## Cómo analiza ahora (por defecto, `--model ensemble`)
Log-probabilidades de 24 tonalidades = 0,5 × media de 3 redes de cromagrama entrenadas con ecualización
aleatoria (`--eq 0.5`) + 0,5 × media de 2 redes de espectro (`src/analysis/KeyCnn.cpp`, pesos en
`KeyCnnWeights.h`). Con audio de 60 s o más se mezcla 0,85 × redes + 0,15 × lineal (`KeyModelWeights.h`,
que usa el acorde final); con menos, solo las redes (`cnnWeight`, `cnnWeightLong`, `longSeconds` en
`src/analysis/Analyzer.h`; `--cnn-weight` en la CLI fija ambos). Si la 2ª tonalidad tiene ≥ 0,5 veces la
probabilidad de la 1ª, el resultado es «dudoso» y se dan las dos (`doubtRatio`, `--doubt`). Las redes son equivariantes a la
transposición. Entrada por fotograma (~0,19 s): cromagrama fino de 36 bins/octava de 4 bandas (cromagrama,
bajo, medios, agudos) y espectro logarítmico de 216 bins.

## Resultados en la partición fija (acierto exacto, 1.497 pistas disponibles)
| Modelo | FMA | Beatport | GuitarSet | GS+ | Total |
|---|---|---|---|---|---|
| Lineal | 50,0 | 52,7 | 63,9 | 72,5 | 53,3 |
| 3 redes de cromagrama | 60,7 | 69,4 | 81,9 | 86,3 | 64,7 |
| Anterior: 0,7 × 3 redes + 0,3 × lineal (lineal que vio la prueba) | 60,6 | 70,0 | 83,3 | 88,2 | 65,1 |
| Una red de espectro sola (6 semillas/anchuras) | 59,5–61,5 | 63–69 | 75–79 | 88–94 | 63,1–65,2 |
| 3 cromagrama + 2 espectro (`split_spec`, `split_spec_w24`), sin lineal | 62,2 | 68,9 | 83,3 | 92,2 | 66,2 |
| …con otras 4 redes de espectro (semillas nuevas), sin lineal | 62,4 | 69,4 | 83,3 | 90,2 | 66,3 |
| Anterior: 0,85 × (3 cromagrama + 2 espectro) + 0,15 × lineal | 62,6 | 70,3 | 84,7 | 88,2 | 66,7 |
| 3 cromagrama **con ecualización** (juego A) + 2 espectro, sin lineal | 63,4 | 68,0 | 80,6 | 90,2 | 66,7 |
| …juego B (otras semillas) | 63,2 | 69,9 | 81,9 | 92,2 | 67,0 |
| **Actual: juego A en C++, sin lineal** | **63,4** | 68,0 | 80,6 | 90,2 | **66,7** |
| …con 0,15 / 0,3 de lineal | 62,3 / 61,2 | | | | 66,3 / 65,2 |

Piezas completas (203: Bach, clásica, autor, Jamendo; redes y lineal entrenados sin ellas): solo redes
82,8 %, con 0,15 de lineal **85,7 %** (escala 90,6 %, correcta entre las mostradas 92,1 %). En la partición
fija, con la doble propuesta, la correcta está entre las mostradas el 74,2 % (FMA 71,0 %).

Las filas con lineal se midieron con skale-cli (C++) y un lineal reentrenado sin la partición de prueba;
las demás con `tools/eval_key_cnn.py` (C++ y PyTorch dan lo mismo). La mejora de las redes de espectro
(+1,7 FMA) y la de la ecualización de las redes de cromagrama (+1,0 FMA frente a las mismas redes sin
ecualizar) se confirmaron con dos juegos de semillas. Con la ecualización el lineal resta y se quitó.

## Siguiente paso
Ideas sin probar: una red con cromagrama y espectro a la vez; ecualización con otras intensidades o
también en tiempo (volumen por fotograma). El límite parece ser la cantidad de datos reales etiquetados
(entrenar más sobreajusta; los datos sintéticos de Lakh no ayudan).

## Modelos entrenados (`models/`, formato {'state','cfg'})
- `final_c2.pt`, `final_c4.pt`, `final_c4_long.pt`: las 3 redes de cromagrama anteriores, sin ecualización.
- `final_c2_eq.pt`, `final_c4_eq.pt`, `final_c4_long_eq.pt`: las 3 redes de cromagrama de producción
  (ecualización aleatoria, todos los datos; mismas opciones que las sin `_eq` más `--eq 0.5`).
- `final_spec.pt`, `final_spec_w24.pt`: las 2 redes de espectro de producción (todos los datos, anchura 16 y 24).
  Exportar las 5: `tools/export_key_cnn.py models/final_c2_eq.pt models/final_c4_eq.pt
  models/final_c4_long_eq.pt models/final_spec.pt models/final_spec_w24.pt --spec-weight 0.5`.
- `split_*.pt` (y `split_*_eq.pt`): entrenadas sin la partición de prueba, para comparar (`split_c2`, `split_c4`,
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
(los alumnos igualan al conjunto, no lo superan). Lo que sí funcionó: promediar redes **distintas**. Entrenar más
(100 épocas: sobreajusta); red de espectro con ventana de ~48 s; ecualización en las redes de espectro.
Datos sintéticos de Lakh MIDI (`tools/gen_lakh.py`, 16.941 fragmentos; características en `~/skale-train`
con `frames4/manifest_lakh.json`): ni mezclados (`--group-scale lakh=0.5`) ni como preentrenamiento
(`--init`) mejoran el conjunto con dos semillas. Etiquetar FMA con detectores externos: en la partición fija
nuestro modelo supera a Essentia (5 perfiles, mejor 54,6 % en FMA) y a madmom (CNN de tonalidad, 56,1 % en
FMA; 78,5 % en Beatport, probablemente visto al entrenar); cuando discrepan de nosotros en FMA aciertan menos
que nosotros, así que sus etiquetas no sirven. Pixabay y ccMixter bloquean el acceso automático (403 y `robots.txt`): no usarlos.

## Normas
- No descargar música comercial o con derechos; solo conjuntos con licencia abierta o de investigación.
- Commits en `main`. Los pesos van compilados en el binario (sin archivos externos).
- El proyecto no es de pago (octubre de 2026): se pueden usar datos y modelos de licencia no comercial para
  entrenar. Herramientas externas (madmom, Essentia) solo para contrastar resultados, sin añadir su código.
- Licencia de JUCE: AGPL o comercial; sin la comercial, el plugin distribuido tiene que ser AGPL (código abierto).
