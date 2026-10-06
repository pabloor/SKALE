#!/usr/bin/env python3
"""Etiqueta la tonalidad de una carpeta de mp3 por consenso de tres estimadores
independientes (Essentia EDMA, Essentia BGate y Krumhansl-Schmuckler sobre
chroma_cqt de librosa) y escribe CARPETA/expected.csv solo con los archivos en
los que coinciden los tres. Necesita: pip install essentia librosa numpy

Las etiquetas son estimadas, no verificadas por una persona.
Uso: tools/label_consensus.py CARPETA [--jobs 4]
"""
import argparse, glob, os, sys
from multiprocessing import Pool

NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
PC = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6,
      'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}


def estimate(f):
    import numpy as np, librosa, essentia.standard as es
    ks = {'major': np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
          'minor': np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])}
    try:
        a = es.MonoLoader(filename=f, sampleRate=44100)()
        est = []
        for prof in ('edma', 'bgate'):
            k, sc, _ = es.KeyExtractor(profileType=prof)(a)
            est.append((PC[k], sc))
        ch = librosa.feature.chroma_cqt(y=a[::2], sr=22050).mean(axis=1)
        best = max((np.corrcoef(ch, np.roll(ks[m], t))[0, 1], t, m) for t in range(12) for m in ks)
        est.append((best[1], best[2]))
        return os.path.basename(f), est
    except Exception as e:
        return os.path.basename(f), repr(e)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--jobs', type=int, default=4)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.folder, '*.mp3')))
    rows = []
    with Pool(a.jobs) as p:
        for name, est in p.imap(estimate, files):
            if isinstance(est, str):
                print(name, 'ERR', est, flush=True)
            elif est[0] == est[1] == est[2]:
                rows.append((name, f'{NAMES[est[0][0]]} {est[0][1]}'))
            else:
                print(name, 'sin consenso', est, flush=True)
    with open(os.path.join(a.folder, 'expected.csv'), 'w') as fh:
        fh.write(''.join(f'{n},{k}\n' for n, k in rows))
    print(f'consenso: {len(rows)}/{len(files)}')


if __name__ == '__main__':
    sys.exit(main())
