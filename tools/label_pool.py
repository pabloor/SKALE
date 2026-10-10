#!/usr/bin/env python3
"""Prepara las pistas de FMA sin etiquetar para tools/etiquetar.py: las analiza con skale-cli (las redes) y
las ordena para etiquetar primero aquellas en las que el modelo duda más, alternando géneros.

Uso: tools/label_pool.py DATOS [--pool fma_pool/fma_small] [--hilos N]
  DATOS/fma_pool/fma_metadata/tracks.csv  metadatos de FMA (título, artista, género, licencia)
  DATOS/fma/expected.csv                  FMAK: estas pistas ya tienen etiqueta y se excluyen
Escribe DATOS/fma_user/pool.json. Las pistas de FMAK que no son de la partición de prueba se guardan aparte
(«control»): etiquetar.py intercala algunas para medir la coincidencia con las etiquetas de expertos.
"""
import argparse, csv, glob, json, os, random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def tracks_meta(path):
    """track_id -> (título, artista, género principal, licencia) del tracks.csv de FMA (cabecera de 3 filas)."""
    meta = {}
    with open(path, newline='', encoding='utf-8') as fh:
        rd = csv.reader(fh); h0, h1, _ = next(rd), next(rd), next(rd)
        col = {(a, b): i for i, (a, b) in enumerate(zip(h0, h1))}
        for row in rd:
            try:
                meta[int(row[0])] = (row[col[('track', 'title')]], row[col[('artist', 'name')]],
                                     row[col[('track', 'genre_top')]], row[col[('track', 'license')]])
            except (ValueError, IndexError, KeyError):
                pass
    return meta


def analyze(cli, path):
    r = subprocess.run([cli, path, '--json', '--model', 'cnn'], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    return dict(tuning=j['tuningCents'], cands=[[j['key']['name'], j['key']['confidence']]] +
                [[a['name'], a['confidence']] for a in j['alternatives']])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('datos'); ap.add_argument('--pool', default='fma_pool/fma_small')
    ap.add_argument('--cli', default=os.path.join(HERE, '..', 'build', 'skale-cli')); ap.add_argument('--hilos', type=int, default=os.cpu_count())
    a = ap.parse_args()
    meta = tracks_meta(os.path.join(a.datos, 'fma_pool', 'fma_metadata', 'tracks.csv'))
    fmak = {fn: k for fn, k in csv.reader(open(os.path.join(a.datos, 'fma', 'expected.csv')))}
    test = {tuple(x) for x in json.load(open(os.path.join(HERE, 'samples', 'test_split.json')))}
    files = sorted(glob.glob(os.path.join(a.datos, a.pool, '**', '*.mp3'), recursive=True))
    new = [f for f in files if os.path.basename(f) not in fmak]
    control = [os.path.join('fma', fn) for fn in fmak if ('fma', fn) not in test]
    random.Random(0).shuffle(control); control = control[:300]
    print(f'{len(files)} pistas en {a.pool}; {len(new)} sin etiqueta; {len(control)} de control (FMAK)', flush=True)

    def job(rel):
        r = analyze(a.cli, os.path.join(a.datos, rel))
        if not r: return None
        tid = int(os.path.splitext(os.path.basename(rel))[0])
        t, ar, g, lic = meta.get(tid, ('', '', '', ''))
        return dict(id=tid, path=rel, title=t, artist=ar, genre=g or 'otro', license=lic, **r)

    rels = [os.path.relpath(f, a.datos) for f in new] + control
    with ThreadPoolExecutor(a.hilos) as ex:
        res = []
        for i, r in enumerate(ex.map(job, rels)):
            res.append(r)
            if (i + 1) % 1000 == 0: print(f'  {i + 1}/{len(rels)}', flush=True)
    items = [r for r in res[:len(new)] if r]
    ctrl = [dict(r, gold=fmak[os.path.basename(r['path'])]) for r in res[len(new):] if r]
    # orden: por géneros alternos y, dentro de cada uno, de menos a más confianza de la 1ª opción
    by = {}
    for it in sorted(items, key=lambda r: r['cands'][0][1]): by.setdefault(it['genre'], []).append(it)
    order = []
    while any(by.values()):
        for g in sorted(by):
            if by[g]: order.append(by[g].pop(0))
    out = os.path.join(a.datos, 'fma_user'); os.makedirs(out, exist_ok=True)
    json.dump(dict(items=order, control=ctrl), open(os.path.join(out, 'pool.json'), 'w'))
    print(len(order), 'pistas para etiquetar y', len(ctrl), 'de control en', os.path.join(out, 'pool.json'))


if __name__ == '__main__':
    main()
