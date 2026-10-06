#!/usr/bin/env python3
"""Vuelca con `skale-cli --features` las características de varios conjuntos etiquetados.

Uso: tools/extract_features.py salida.json [--cli build/skale-cli] GRUPO CARPETA CSV [GRUPO CARPETA CSV ...]
Cada CSV tiene líneas "archivo,Tonalidad" (p. ej. "pieza.mp3,Eb minor") relativas a CARPETA.
"""
import csv, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

EXTRA = []
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def parse_key(t):
    n, m = t.split()
    return [(PC[n[0]] + n[1:].count('#') - n[1:].count('b')) % 12, m]


def main():
    args = sys.argv[1:]
    out = args.pop(0)
    cli = 'build/skale-cli'
    while args and args[0].startswith('--'):
        if args[0] == '--cli': cli = args[1]
        else: EXTRA.extend([args[0], args[1]])   # p. ej. --window 4
        args = args[2:]
    items = []
    for g, folder, csvname in zip(args[0::3], args[1::3], args[2::3]):
        for fn, k in csv.reader(open(csvname)):
            items.append((g, f'{folder}/{fn}', parse_key(k), fn))

    def run(it):
        g, f, k, fn = it
        o = subprocess.run([cli, f, '--features'] + EXTRA, capture_output=True, text=True)
        try:
            return dict(group=g, name=fn, label=k, **json.loads(o.stdout))
        except Exception:
            return None

    with ThreadPoolExecutor(4) as ex:
        res = [r for r in ex.map(run, items) if r]
    json.dump(res, open(out, 'w'))
    print(len(res), 'archivos')


if __name__ == '__main__':
    main()
