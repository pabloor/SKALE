#!/usr/bin/env python3
"""Convierte los conjuntos de tools/fetch_datasets.py en los datos de entrenamiento de las redes:
SALIDA/frames4/*.bin (cromagramas finos, float32 [n][4][36]), SALIDA/spec/*.bin (espectro logarítmico,
float32 [n][216]) y SALIDA/frames4/manifest.json. Usa skale-cli, así que las características son
exactamente las mismas que calcula el analizador.

Uso: tools/prepare_training.py DATOS SALIDA [--cli build/skale-cli] [--hilos N]
Después: tools/train_key_cnn.py SALIDA/frames4/manifest.json SALIDA/frames4 --channels 4 [--spec SALIDA/spec] \\
             --split tools/samples/test_split.json
"""
import argparse, csv, hashlib, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor

# grupo -> [(carpeta dentro de DATOS, lista csv dentro de la carpeta)]
GROUPS = {
    'fma': [('fma', 'expected.csv')],
    'beatport': [('beatport', 'expected.csv')],
    'gsplus': [('gsplus', 'expected.csv')],
    'guitarset': [('guitarset', 'expected.csv')],
    'bach': [('bach', 'expected.csv'), ('bach', 'fugues.csv')],
    'classic': [('classic', 'expected.csv')],
    'author': [('set2', 'expected.csv'), ('set6', 'expected.csv')],
    'jamendo_cons': [('set4', 'expected.csv'), ('set5', 'expected.csv')],
    'folk_synth': [('set3', 'expected.csv')],
    'synth': [('synth', 'expected.csv')],
    'fma_user': [('fma_user', 'expected.csv')],   # etiquetas a mano (tools/etiquetar.py); siempre de entrenamiento
}
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def parse_key(t):
    n, m = t.split()
    return [(PC[n[0]] + n[1:].count('#') - n[1:].count('b')) % 12, m]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('datos'); ap.add_argument('salida')
    ap.add_argument('--cli', default='build/skale-cli'); ap.add_argument('--hilos', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    fr, sp = os.path.join(a.salida, 'frames4'), os.path.join(a.salida, 'spec')
    os.makedirs(fr, exist_ok=True); os.makedirs(sp, exist_ok=True)
    items = []
    for g, lists in GROUPS.items():
        for folder, name in lists:
            path = os.path.join(a.datos, folder, name)
            if not os.path.exists(path): print(f'(sin {folder}/{name}: se omite)'); continue
            for fn, k in csv.reader(open(path)):
                if os.path.exists(os.path.join(a.datos, folder, fn)): items.append((g, os.path.join(a.datos, folder, fn), parse_key(k), fn))
    print(len(items), 'pistas', flush=True)

    def run(it):
        g, f, k, fn = it
        name = f"{g}_{hashlib.md5(fn.encode()).hexdigest()[:12]}.bin"   # mismo nombre que tools/dump_frames.py
        pf, ps = os.path.join(fr, name), os.path.join(sp, name)
        if not (os.path.exists(pf) and os.path.exists(ps)):
            subprocess.run([a.cli, f, '--model', 'classic', '--frames', pf, '--spec', ps], capture_output=True)
        ok = os.path.exists(pf) and os.path.getsize(pf) >= 4 * 4 * 36 * 20 and os.path.exists(ps)
        return dict(group=g, name=fn, label=k, file=name) if ok else None

    done = 0
    with ThreadPoolExecutor(a.hilos) as ex:
        res = []
        for r in ex.map(run, items):
            done += 1; res.append(r)
            if done % 500 == 0: print(f'  {done}/{len(items)}', flush=True)
    res = [r for r in res if r]
    json.dump(res, open(os.path.join(fr, 'manifest.json'), 'w'))
    print(len(res), 'de', len(items), 'pistas preparadas en', a.salida)


if __name__ == '__main__':
    main()
