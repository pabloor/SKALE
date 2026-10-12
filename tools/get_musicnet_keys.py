#!/usr/bin/env python3
"""Etiquetas de tonalidad para MusicNet (Zenodo 5120004, CC BY 4.0: 330 grabaciones de música clásica con todas
las notas alineadas). La tonalidad sale del título de la obra («Piano Sonata in E major») y se acepta para un
movimiento solo si coincide con la que se calcula sobre sus notas (perfiles de Albrecht y Shanahan, duración de
cada nota) y la nota más frecuente del bajo es la tónica: los movimientos centrales suelen estar en otra tonalidad.
Uso: tools/get_musicnet_keys.py CARPETA   (con musicnet_metadata.csv y la carpeta musicnet/ extraída)
Escribe CARPETA/expected.csv (grupo «musicnet»).
"""
import csv, glob, os, re, sys
import numpy as np

AS_MAJ = np.array([0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081])
AS_MIN = np.array([0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052])
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def title_key(s):
    m = re.search(r'\bin ([A-G])(?:[- ]?(flat|sharp|b|#))?\s*(major|minor)?\b', s, re.I)
    if not m: return None
    t = PC[m.group(1).upper()] + {'flat': -1, 'b': -1, 'sharp': 1, '#': 1}.get((m.group(2) or '').lower(), 0)
    return t % 12, (m.group(3) or 'major').lower()


def notes_key(rows, bass=False):
    h = np.zeros(12)
    for r in rows:
        n, d = int(r['note']), float(r['end_time']) - float(r['start_time'])
        if bass and n >= 48: continue
        h[n % 12] += d
    if h.sum() <= 0: return None, 0.0
    sc = []
    for t in range(12):
        x = np.roll(h, -t); sc += [np.corrcoef(x, AS_MAJ)[0, 1], np.corrcoef(x, AS_MIN)[0, 1]]
    sc = np.nan_to_num(np.array(sc)); o = np.argsort(sc)[::-1]
    return (int(o[0]) // 2, 'major' if o[0] % 2 == 0 else 'minor'), float(sc[o[0]] - sc[o[1]])


def main():
    root = sys.argv[1]
    meta = {r['id']: r for r in csv.DictReader(open(os.path.join(root, 'musicnet_metadata.csv')))}
    rows_out, stats = [], {'sin tonalidad en el título': 0, 'notas no coinciden': 0, 'aceptadas': 0}
    for lab in sorted(glob.glob(os.path.join(root, 'musicnet', '*_labels', '*.csv'))):
        rid = os.path.basename(lab)[:-4]; m = meta.get(rid)
        wav = lab.replace('_labels', '_data')[:-4] + '.wav'
        if not m or not os.path.exists(wav): continue
        tk = title_key(m['composition'])
        if not tk: stats['sin tonalidad en el título'] += 1; continue
        notes = list(csv.DictReader(open(lab)))
        nk, margin = notes_key(notes); bk, _ = notes_key(notes, bass=True)
        if nk != tk or margin < 0.02 or (bk is not None and bk[0] != tk[0]):
            stats['notas no coinciden'] += 1; continue
        stats['aceptadas'] += 1
        rows_out.append([os.path.relpath(wav, root), f'{NAMES[tk[0]]} {tk[1]}'])
    with open(os.path.join(root, 'expected.csv'), 'w', newline='') as fh: csv.writer(fh, lineterminator='\n').writerows(rows_out)
    print(stats, '->', len(rows_out), 'movimientos en', os.path.join(root, 'expected.csv'))


if __name__ == '__main__':
    main()
