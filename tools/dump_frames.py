#!/usr/bin/env python3
"""Vuelca con `skale-cli --frames` la serie de cromagramas finos de varios conjuntos etiquetados, para
entrenar la red (tools/train_key_cnn.py). Escribe DIR/*.bin y DIR/manifest.json.

Uso: tools/dump_frames.py DIR [--cli build/skale-cli] GRUPO CARPETA CSV [GRUPO CARPETA CSV ...]
"""
import csv, hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def parse_key(t):
    n, m = t.split()
    return [(PC[n[0]] + n[1:].count('#') - n[1:].count('b')) % 12, m]


def main():
    args = sys.argv[1:]; out = args.pop(0); cli = 'build/skale-cli'
    if args and args[0] == '--cli': cli = args[1]; args = args[2:]
    os.makedirs(out, exist_ok=True); items = []
    for g, folder, csvname in zip(args[0::3], args[1::3], args[2::3]):
        for fn, k in csv.reader(open(csvname)): items.append((g, f'{folder}/{fn}', parse_key(k), fn))

    def run(it):
        g, f, k, fn = it; name = f"{g}_{hashlib.md5(fn.encode()).hexdigest()[:12]}.bin"; path = f'{out}/{name}'
        if not os.path.exists(path): subprocess.run([cli, f, '--frames', path], capture_output=True)
        return dict(group=g, name=fn, label=k, file=name) if os.path.exists(path) and os.path.getsize(path) >= 288 * 20 else None

    with ThreadPoolExecutor(4) as ex: res = [r for r in ex.map(run, items) if r]
    json.dump(res, open(f'{out}/manifest.json', 'w')); print(len(res), 'de', len(items))


if __name__ == '__main__':
    main()
