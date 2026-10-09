# Skale: notas de traspaso para Claude Code

Detector de tonalidad y escala (C++17, sin dependencias) con CLI (`skale-cli`) y plugin JUCE (VST3/AU +
aplicación independiente). El usuario habla español: responde en español. Detalles y tablas en `README.md`.

## Objetivo actual
**Subir la precisión de la tonalidad en archivos.** Todas las comparaciones se hacen sobre la misma
partición de prueba fija (`tools/samples/test_split.json`, 1.504 pistas: FMA 1.096, Beatport 220,
GuitarSet 72, GiantSteps+ 51, resto pequeños). Métrica principal: acierto exacto en FMA (música real
variada) y en el total; también MIREX (`tools/eval_key_cnn.py`).

## Cómo analiza ahora (por defecto, `--model ensemble`)
Log-probabilidades de 24 tonalidades = 0,7 × media de 3 redes convolucionales de cromagrama
(`src/analysis/KeyCnn.cpp`, pesos en `KeyCnnWeights.h`) + 0,3 × modelo lineal (`KeyModelWeights.h`).
Las redes son equivariantes a la transposición. Entrada por fotograma (~0,19 s): cromagrama fino de
36 bins/octava de 4 bandas (cromagrama, bajo, medios, agudos) y espectro logarítmico de 216 bins.

## Resultados en la partición fija (acierto exacto)
| Modelo | FMA | Beatport | GuitarSet | GS+ | Total |
|---|---|---|---|---|---|
| Lineal | 50,0 | 52,7 | 63,9 | 72,5 | 53,3 |
| 3 redes de cromagrama | 60,7 | 69,1 | 81,9 | 86,3 | 64,7 |
| **Actual: 0,7 × 3 redes + 0,3 × lineal** | 60,6 | 70,0 | 83,3 | 88,2 | 65,1 |
| Red de espectro sola (`models/split_spec.pt`) | 61,5 | 67,7 | 77,8 | 90,2 | 65,2 |
| Red de espectro ancha (`models/split_spec_w24.pt`) | 60,2 | 70,5 | 79,2 | 94,1 | 64,7 |
| 0,5 × espectro + 0,5 × 3 redes de cromagrama (sin lineal) | 62,3 | 68,2 | 83,3 | 90,2 | 66,1 |
| 0,5 × espectro ancha + 0,5 × 3 redes de cromagrama | 62,5 | 69,5 | 83,3 | 92,2 | 66,4 |
| media de las 2 de espectro, mezcla 0,6, + 0,3 lineal | 61,9 | 70,5 | 83,3 | 88,2 | 66,1 |

Conclusión provisional: añadir la red de espectro (peso ~0,5 frente a las de cromagrama) da ≈ +1,5–2
puntos en FMA y ≈ +1,3 en total; el lineal ya no aporta cuando está la red de espectro. Con 1.096
pistas de FMA, 1 punto ≈ el ruido estadístico: confirmar con más redes/semillas.

## Siguiente paso
1. Entrenar las redes de espectro finales con todos los datos (`--all`) y exportarlas junto a las 3 de
   cromagrama: `tools/export_key_cnn.py models/final_c2.pt models/final_c4.pt models/final_c4_long.pt
   fin_spec.pt [fin_spec_w24.pt] --spec-weight 0.5`. La inferencia en C++ de la red de espectro ya está
   hecha y verificada contra PyTorch (misma tonalidad y probabilidad con 4 decimales).
2. Decidir si quitar el lineal del conjunto por defecto (`options.cnnWeight` en `src/analysis/Analyzer.h`).
3. Compilar, `./build/skale_tests` (216 comprobaciones), actualizar las tablas del README.

## Modelos entrenados (`models/`, formato {'state','cfg'})
- `final_c2.pt`, `final_c4.pt`, `final_c4_long.pt`: las 3 redes de producción (todos los datos);
  reexportarlas reproduce byte a byte `src/analysis/KeyCnnWeights.h`.
- `split_*.pt`: entrenadas sin la partición de prueba, para comparar (`split_c2`, `split_c4`,
  `split_c4_long`, `split_spec`, `split_spec_w24`).

## Datos y entrenamiento
Pasos completos en `docs/entrenar_en_mac.md`: `tools/fetch_datasets.py` (descarga ~7 GB) →
`tools/prepare_training.py` (características con skale-cli) → `tools/train_key_cnn.py`
(`--spec` para la red de espectro, `--split tools/samples/test_split.json`, `--device auto` usa MPS)
→ `tools/eval_key_cnn.py` → `tools/export_key_cnn.py`. Configuración usada: `--epochs 30 --steps 100`;
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
