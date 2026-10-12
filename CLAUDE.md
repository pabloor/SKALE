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
aleatoria (`--eq 0.5`) + 0,5 × media de 2 redes de espectro, todas entrenadas también con las canciones
completas de FMAK (`--max-frames 1280`) (`src/analysis/KeyCnn.cpp`, pesos en
`KeyCnnWeights.h`). Con audio de 60 s o más se mezcla 0,85 × redes + 0,15 × lineal (`KeyModelWeights.h`,
que usa el acorde final); con menos, solo las redes (`cnnWeight`, `cnnWeightLong`, `longSeconds` en
`src/analysis/Analyzer.h`; `--cnn-weight` en la CLI fija ambos). Si la 2ª tonalidad tiene ≥ 0,5 veces la
probabilidad de la 1ª, el resultado es «dudoso» y se dan las dos (`doubtRatio`, `--doubt`). Las redes son equivariantes a la
transposición. Entrada por fotograma (~0,19 s): cromagrama fino de 36 bins/octava de 4 bandas (cromagrama,
bajo, medios, agudos) y espectro logarítmico de 216 bins.

## Prueba con música comercial: GTZAN (solo evaluación)
837 fragmentos de 30 s de GTZAN con la tonalidad de Lerch (`tools/fetch_datasets.py --solo gtzan`, lista en
`tools/samples/gtzan_expected.csv`). `train_key_cnn.py` lo excluye siempre (`--eval-only gtzan`, también con
`--all`). Evaluar: `tools/eval_key_cnn.py ... --split tools/samples/test_split_gtzan.json` (partición fija +
todo GTZAN). Modelo actual: 68,0 % con las redes `split_*` (skale-cli: 68,6 %; pop 83, country 87, reggae 88,
rock 74, metal 72, hip hop 56, jazz 46, blues 36: el 97 % del blues está anotado como menor y el modelo suele
decir la misma tónica en mayor). Entrenar con 585 de GTZAN dio +1,6 en las otras 252 (ruido): se decidió
dejarlo como prueba independiente.

## Canciones completas de FMAK
`tools/get_fmak_full.py` baja las 5.489 canciones completas de FMAK (38 GB, de fma_full por rangos HTTP) a
`~/skale-datos/fma_full`: grupo `fma_full` (4.393, entrenamiento) y `fma_full_test` (las 1.096 de la partición
de prueba, solo evaluación; `--eval-only` las excluye). Evaluar con `--split ~/skale-runs/full2/split_eval.json`
(partición fija + GTZAN + `fma_full_test`). Con las mismas 5 redes, entrenar también con ellas: GTZAN
68,0 -> 69,5 (semillas A) y 66,8 -> 70,0 (semillas B), FMA sin cambio; el modelo final de producción dio
68,7 % en GTZAN frente a 68,6 % del anterior (ruido entre semillas de ±1,5). En canciones completas de FMA de
prueba, las redes `split_*` aciertan el 70,1 % (63,4 % con el fragmento de 30 s); el 15 % de lineal desde 60 s
ahí resta 0,4 (en las 203 piezas clásicas y de Jamendo suma 2,9): se mantiene.
Limpieza de etiquetas (validación cruzada en 4 partes, `--exclude`; 169 o 330 sospechosas): GTZAN +1,5/+1,9,
FMA ±0,5; junto con las canciones completas no suma: no se usa.

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
Todas las ideas probadas (ver abajo). Con los datos abiertos disponibles el modelo está en su límite; lo único
con margen es más música real etiquetada a mano: la página `tools/etiquetar.py` está lista (7.389 pistas de
FMA en `~/skale-datos/fma_user`) si alguien con buen oído puede etiquetar.

## Modelos entrenados (`models/`, formato {'state','cfg'})
- **Producción:** `final_c2_full.pt`, `final_c4_full.pt`, `final_c4_long_full.pt`, `final_spec_full.pt`,
  `final_spec_w24_full.pt` (todos los datos más las canciones completas de FMAK; `--all --eval-only gtzan
  --max-frames 1280`). Exportar: `tools/export_key_cnn.py models/final_c2_full.pt models/final_c4_full.pt
  models/final_c4_long_full.pt models/final_spec_full.pt models/final_spec_w24_full.pt --spec-weight 0.5`.
  Sus versiones sin la partición de prueba: `split_*_full.pt`.
- `final_c2.pt`, `final_c4.pt`, `final_c4_long.pt`: las 3 redes de cromagrama anteriores, sin ecualización.
- Anteriores (sin canciones completas): `final_c2_eq.pt`, `final_c4_eq.pt`, `final_c4_long_eq.pt` (ecualización
  aleatoria) y `final_spec.pt`, `final_spec_w24.pt` (espectro, anchura 16 y 24).
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
MERT-v1-95M como profesor (media temporal por capa + clasificador, 30 s centrales): 52,8 % en FMA
frente a 63,4 % nuestro; mezclado con peso 0,15, +0,4 (ruido). No sirve. Datos sintéticos de Lakh MIDI (`tools/gen_lakh.py`, 16.941 fragmentos; características en `~/skale-train`
con `frames4/manifest_lakh.json`): ni mezclados (`--group-scale lakh=0.5`) ni como preentrenamiento
(`--init`) mejoran el conjunto con dos semillas. Etiquetar FMA con detectores externos: en la partición fija
nuestro modelo supera a Essentia (5 perfiles, mejor 54,6 % en FMA) y a madmom (CNN de tonalidad, 56,1 % en
FMA; 78,5 % en Beatport, probablemente visto al entrenar); cuando discrepan de nosotros en FMA aciertan menos
que nosotros, así que sus etiquetas no sirven. KeyFinder (libKeyFinder 2.2.8, el de Mixxx; herramienta de
contraste en `~/skale-runs/keyfinder`, fuera del repo): FMA 49,5 % frente a 63,4 %, partición fija 53,4 %
frente a 66,7 %, GTZAN 50,7 % frente a 68,6 % (solo gana en blues: 43,9 frente a 35,7). Separar armonía y percusión (HPSS, filtros de mediana, `ChromaParams::hpss`, `skale-cli --hpss`,
desactivado por defecto; características en `~/skale-train-hpss`): las 5 redes con HPSS dan FMA 63,3 / GTZAN
68,3 / partición fija 66,5 frente a 63,4 / 68,0 / 66,7; las 10 juntas 63,0 / 69,2 / 66,5. No mejora (la
ventana de ~370 ms ya diluye la percusión) y cuesta +20 % de tiempo. Red con cromagrama y espectro a la vez (`train_key_cnn.py --both`, KeyNetBoth, ~270.000 parámetros,
ramas unidas por clase de nota antes de la cabeza equivariante): sola 60,3–60,4 % en FMA y 62–64 % en GTZAN
(como la mejor red suelta); añadida al conjunto empeora GTZAN (66,4–66,9 frente a 68,0) y no mejora FMA.
Más datos etiquetados, añadidos uno a uno a la configuración actual (5 redes, semillas A, con canciones
completas; FMA 63,6 / GTZAN 69,5 / FMA completas 69,8): loops de Freesound (`tools/get_fsl10k.py`, 886 con
anotadores de acuerdo) 63,0 / 68,9 / 69,7; MusicNet (`tools/get_musicnet_keys.py`, 147 movimientos con la
tonalidad del título comprobada con las notas; el modelo ya acertaba el 92,5 %) 63,0 / 69,2 / 70,1; AAM
(`tools/get_aam.py`, 2.705 tramos sintéticos de una tonalidad) 63,3 / 69,9 / 69,9 y GuitarSet −5,6. Ninguno
mejora. Sumar dos juegos de semillas (10 redes) y otras formas de combinar las ventanas (por confianza, más
peso al final) tampoco: ±0,4 en FMA completas. Preentrenamiento autosupervisado (`train_key_cnn.py --ssl`, IIC entre dos trozos de la misma pista con
desafinación, ecualización y transposición; 24.625 fragmentos de fma_medium, `manifest_unlabeled.json`) y
ajuste con `--init`: cada red pierde 1-5 puntos y la de espectro ancha se rompe (FMA 59,3 / GTZAN 67,3 frente
a 63,6 / 69,5). La transposición ya la cumplen las redes por construcción; agrupar trozos no es la tonalidad. Pixabay y ccMixter bloquean el acceso automático (403 y `robots.txt`): no usarlos.

## Normas
- Música: conjuntos con licencia abierta y conjuntos de investigación publicados (también con audio comercial,
  como GTZAN o Beatport EDM Key), solo para entrenar y evaluar, nunca para redistribuir el audio. No descargar
  de sitios que lo bloquean (Pixabay, ccMixter) ni de YouTube.
- Commits en `main`. Los pesos van compilados en el binario (sin archivos externos).
- El proyecto no es de pago (octubre de 2026): se pueden usar datos y modelos de licencia no comercial para
  entrenar. Herramientas externas (madmom, Essentia) solo para contrastar resultados, sin añadir su código.
- Licencia de JUCE: AGPL o comercial; sin la comercial, el plugin distribuido tiene que ser AGPL (código abierto).
