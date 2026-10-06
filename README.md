# Skale

Detecta la **escala de una canción**: tonalidad, notas de la escala, acordes diatónicos y los acordes que suenan. Objetivo final: un plugin de audio VST3/AU que funcione en tiempo real y con archivos mp3/wav.

Estado: **fase 1** (núcleo del analizador + herramienta de línea de comandos). El plugin llega en la fase 2.

## Cómo funciona

1. **Cromagrama** (`src/analysis/Chromagram.*`): FFT con ventana Hann, picos espectrales entre 65 Hz y 2,1 kHz con interpolación parabólica, repartidos en 36 bins (3 por semitono). A partir del cromagrama acumulado se estima la afinación global (A=440 ±33 cents) y se pliega a 12 notas.
2. **Tonalidad** (`KeyDetector.*`): correlación de Pearson con los perfiles de Temperley o de Krumhansl-Schmuckler en las 24 tonalidades. La "confianza" es un softmax de esas correlaciones: una heurística, no una probabilidad calibrada.
3. **Teoría** (`Theory.*`): escala con la ortografía correcta (Eb mayor = Eb F G Ab Bb C D), acordes diatónicos con triadas, séptimas y números romanos.
4. **Acordes** (`ChordDetector.*`): plantillas (mayor, menor, 7, maj7, m7, dim, sus2, sus4) con similitud coseno y suavizado Viterbi. Marca los acordes ajenos a la tonalidad.

El extractor funciona en streaming (acepta bloques de cualquier tamaño), pensado para reutilizarlo en el plugin en tiempo real.

## Compilar y probar

    cmake -S . -B build
    cmake --build build -j
    ctest --test-dir build --output-on-failure

Solo hace falta un compilador C++17 y CMake. `third_party/dr_libs` (dr_wav, dr_mp3) lee wav y mp3.

## Línea de comandos

    build/skale-cli cancion.mp3 --timeline
    build/skale-cli cancion.wav --json --solfege
    build/skale-cli cancion.wav --profile ks

Opciones: `--json`, `--timeline` (línea de tiempo de acordes), `--solfege` (Do Re Mi), `--profile ks|temperley`.

## Límites conocidos

- Mayor y su relativo menor son ambiguos si la música no define el modo (Am F C G tiene las mismas notas que C mayor): por eso se devuelven candidatos con su confianza.
- Un único cambio de tonalidad en la canción no se detecta (se promedia toda). Ventana deslizante: fase posterior.
- Los tests usan audio sintético (acordes con armónicos, con y sin desafinación).

## Validación con música real

`tools/validate.py` pasa una carpeta de audio por la CLI y compara con un CSV de tonalidades esperadas (`tools/samples/wtc1_expected.csv`: los 24 preludios del Clave bien temperado, libro 1, en la grabación de Kimiko Ishizaka, dominio público, [archive.org](https://archive.org/details/bach-well-tempered-clavier-book-1)). El audio no está en el repo.

    python3 tools/validate.py CARPETA --csv tools/samples/wtc1_expected.csv --profile temperley

Resultado (24 preludios, piano solo): **Temperley 18/24 (75 %)**, **Krumhansl-Schmuckler 17/24 (71 %)**. Fallos de Temperley: 5 son el mayor relativo en lugar del menor (Mi menor→Sol mayor, etc.) y 1 es una quinta. Los de KS son sobre todo quintas y paralelas. Con la **ponderación del acorde final** (`--ending-weight`, `--ending-seconds`) se suma a la correlación de cada tonalidad el peso por la energía de su tríada tónica en los últimos segundos. Preludios / fugas (24 + 24, estas últimas no se usaron para elegir el peso):

| Configuración | Preludios | Fugas |
|---|---|---|
| Temperley sin final | 18/24 | 16/24 |
| Temperley, 4 s, peso 0,5 | 21/24 | 18/24 |
| Temperley, 8 s, peso 0,5 | 23/24 | 20/24 |
| KS, 4 s, peso 0,5 | 22/24 | 16/24 |

Comprobación en otros dos conjuntos, descargados de archive.org con licencia libre (listas en `tools/samples/`, con su identificador de archive.org en `*_sources.tsv`; el audio no está en el repo):

| Conjunto | Temperley sin final | Temperley, 4 s, 0,5 | KS sin final | KS, 4 s, 0,5 |
|---|---|---|---|---|
| Sonatas de Beethoven, Chopin (12, dominio público) | 7/12 | 9/12 | 6/12 | 8/12 |
| Rock, blues, funk, folk, electrónica y otros (14, Creative Commons) | 5/14 | 6/14 | 4/14 | 4/14 |

En el segundo, la tonalidad es la que dice el título del autor y no está verificada; en las sonatas completas, los movimientos centrales pueden estar en otra tonalidad. Otros dos conjuntos más cercanos a música real:

| Conjunto | Temperley sin final | Temperley, 4 s, 0,5 | KS sin final | KS, 4 s, 0,5 |
|---|---|---|---|---|
| 29 melodías folk del corpus de music21 (dominio público), sintetizadas con soundfont, sin acompañamiento (`tools/render_corpus.py`) | 15/29 | 16/29 | 11/29 | 15/29 |
| 26 temas de Jamendo (CC; rock, pop, punk, ska, electrónica…), etiquetados solo si coinciden Essentia (EDMA y BGate) y librosa (`tools/label_consensus.py`) | 16/26 | 18/26 | 20/26 | 19/26 |

Aquí **KS supera a Temperley** en música real (77 % frente a 62 % en Jamendo) y el peso del final ya no le ayuda. Aviso de circularidad: una de las tres etiquetas de consenso es una variante de KS (librosa), lo que favorece a KS; las etiquetas son estimadas, no verificadas por una persona. Las melodías sintéticas no tienen acompañamiento, así que el relativo menor es casi indistinguible (8 de los 14 fallos de Temperley). Con la misma configuración (Temperley, 4 s, 0,5) el peso del final mejora en los cuatro conjuntos, pero con pocos archivos y poco margen en los no clásicos. Mejora todos los pesos probados con Temperley (+1 a +5 preludios), pero el máximo (8 s, 0,5) es un pico y está ajustado sobre estos mismos datos. Sigue **desactivado por defecto** (peso 0): las canciones con fundido final (fade-out) o que acaban fuera de la tónica pueden empeorar, y falta comprobarlo con pop y rock. Es música clásica con modulaciones, así que no es representativa de pop o rock.

## Hoja de ruta

1. Núcleo del analizador y CLI ✅
2. Plugin JUCE (VST3/AU) con tiempo real y vista mínima
3. Análisis de archivos arrastrados al plugin
4. Línea de tiempo de acordes en la interfaz
5. Pulido de UI y empaquetado

## Licencia

Copyright © 2026 Pablo Olivares Rodriguez. **Todos los derechos reservados.** Ver [LICENSE](LICENSE).
