# Entrenar las redes de tonalidad en un Mac (Apple Silicon)

El entrenamiento usa PyTorch y aprovecha la GPU del Mac (MPS) automáticamente. Una red que en una
CPU de 4 núcleos tarda 40–75 minutos debería tardar pocos minutos en un M-series.

## 1. Herramientas

```sh
xcode-select --install                      # compilador de Apple (si no está ya)
brew install cmake python@3.12              # https://brew.sh si no tienes Homebrew
git clone https://github.com/pabloor/skale.git && cd skale
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch numpy remotezip requests
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

Comprueba que PyTorch ve la GPU: `python -c "import torch; print(torch.backends.mps.is_available())"` → `True`.

## 2. Datos (una sola vez, ~7 GB de descarga, ~12 GB en disco)

```sh
python tools/fetch_datasets.py ~/skale-datos          # se puede interrumpir y relanzar
python tools/prepare_training.py ~/skale-datos ~/skale-train
```

`fetch_datasets.py` baja FMAK (Free Music Archive), Beatport EDM Key, GiantSteps Key+, GuitarSet y los
conjuntos pequeños de archive.org, con los mismos nombres que las listas de `tools/samples/`.
`prepare_training.py` calcula con `skale-cli` los cromagramas (`frames4/`) y los espectros (`spec/`):
son exactamente las mismas características que usa el analizador.

## 3. Entrenar y evaluar

Siempre con la misma partición de prueba (`tools/samples/test_split.json`, 1.504 pistas), para que los
resultados sean comparables con los del README:

```sh
D=~/skale-train
# red de espectro
python tools/train_key_cnn.py $D/frames4/manifest.json $D/frames4 --spec $D/spec --epochs 30 --steps 100 \
    --split tools/samples/test_split.json --out sp1.pt
# redes de cromagrama (2 canales; 4 canales; ventana larga)
python tools/train_key_cnn.py $D/frames4/manifest.json $D/frames4 --channels 4 --use-channels 2 --epochs 30 --steps 100 \
    --split tools/samples/test_split.json --out c2.pt
python tools/train_key_cnn.py $D/frames4/manifest.json $D/frames4 --channels 4 --epochs 30 --steps 100 \
    --split tools/samples/test_split.json --seed 1 --out c4.pt
python tools/train_key_cnn.py $D/frames4/manifest.json $D/frames4 --channels 4 --crop 256 --epochs 30 --steps 100 \
    --split tools/samples/test_split.json --seed 3 --out c4l.pt

# acierto por grupo, solas y combinadas como en el analizador
python tools/eval_key_cnn.py $D/frames4/manifest.json $D/frames4 --spec $D/spec c2.pt c4.pt c4l.pt sp1.pt
```

Referencia (CPU, misma partición): una red de espectro sola da FMA 61,5 % y 65,2 % en total; las tres
de cromagrama juntas, FMA 60,7 % y 64,7 %.

## 4. Modelo final y exportación

Cuando una combinación sea mejor, se reentrenan esas mismas configuraciones con todos los datos
(`--all` en lugar de `--split ...`), se exportan y se compila:

```sh
python tools/export_key_cnn.py fin_c2.pt fin_c4.pt fin_c4l.pt fin_sp1.pt --spec-weight 0.5
cmake --build build -j && ./build/skale_tests
git add src/analysis/KeyCnnWeights.h && git commit -m "Nuevos pesos de las redes" && git push
```

Los pesos van compilados dentro de `skale-cli` y del plugin: no hace falta ningún archivo aparte.

Con Claude Code instalado en el Mac (`claude` dentro de la carpeta del repositorio) puedes pedirle
que haga todos estos pasos.
