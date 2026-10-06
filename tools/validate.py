#!/usr/bin/env python3
"""Valida la tonalidad detectada por skale-cli contra un CSV de tonalidades esperadas.

Uso: tools/validate.py CARPETA [--cli build/skale-cli] [--csv CARPETA/expected.csv]
                              [--profile ks|temperley]

El CSV tiene una línea por archivo: nombre,Tonalidad  (p. ej. "pieza.mp3,Eb minor").
Las enarmonías cuentan igual (C# major == Db major).
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path

PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def parse_key(text):
    name, mode = text.split()
    pc = PC[name[0]] + name[1:].count('#') - name[1:].count('b')
    return pc % 12, mode.lower()


def classify(exp, got):
    (ep, em), (gp, gm) = exp, got
    if exp == got:
        return 'ok'
    rel = (ep + 9) % 12 if em == 'major' else (ep + 3) % 12
    if gm != em and gp == rel:
        return 'relativa'
    if gm == em and (gp - ep) % 12 in (5, 7):
        return 'quinta'
    if gp == ep and gm != em:
        return 'paralela'
    return 'otro'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--cli', default='build/skale-cli')
    ap.add_argument('--csv')
    ap.add_argument('--profile', default='temperley')
    ap.add_argument('--ending-weight', default=None, help='por defecto el de la CLI')
    ap.add_argument('--ending-seconds', default=None)
    ap.add_argument('--ending-margin', default=None)
    ap.add_argument('--quiet', action='store_true', help='solo la línea de resumen')
    a = ap.parse_args()
    folder = Path(a.folder)
    rows = list(csv.reader(open(a.csv or folder / 'expected.csv')))
    stats, total = {}, 0
    for fn, key in rows:
        extra = [x for opt, v in (('--ending-weight', a.ending_weight), ('--ending-seconds', a.ending_seconds),
                                  ('--ending-margin', a.ending_margin)) if v is not None for x in (opt, v)]
        out = subprocess.run([a.cli, str(folder / fn), '--json', '--profile', a.profile, *extra],
                             capture_output=True, text=True)
        if out.returncode:
            print(f'{fn}: ERROR {out.stderr.strip()}')
            continue
        k = json.loads(out.stdout)['key']
        got = (k['tonic'], k['mode'])
        res = classify(parse_key(key), got)
        stats[res] = stats.get(res, 0) + 1
        total += 1
        if not a.quiet:
            print(f"{'OK ' if res == 'ok' else 'MAL'} {fn:28} esperada {key:10} detectada {k['name']:10} "
                  f"({k['confidence']:.2f}) {'' if res == 'ok' else res}")
    print(f"\nAcierto: {stats.get('ok', 0)}/{total} = {100 * stats.get('ok', 0) / max(total, 1):.0f}%  "
          + ' '.join(f'{k}={v}' for k, v in sorted(stats.items()) if k != 'ok'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
