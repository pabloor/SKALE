#!/usr/bin/env python3
"""Freesound Loop Dataset (FSL10K, Zenodo 3967852, CC BY 4.0; audio de Freesound con licencias Creative Commons):
baja solo los loops con tonalidad anotada por expertos (mayor o menor; si hay varios anotadores, tienen que
coincidir), sacándolos del zip de 8,8 GB por rangos HTTP. Escribe CARPETA/expected.csv (grupo «fsl10k»).
Uso: tools/get_fsl10k.py CARPETA      Necesita: pip install remotezip
"""
import csv, collections, glob, io, json, os, sys, time, urllib.request, zipfile
from remotezip import RemoteZip

REC = 'https://zenodo.org/api/records/3967852/files/{}/content'
NAMES = {'c': 'C', 'c#': 'C#', 'db': 'Db', 'd': 'D', 'd#': 'Eb', 'eb': 'Eb', 'e': 'E', 'f': 'F', 'f#': 'F#', 'gb': 'F#',
         'g': 'G', 'g#': 'Ab', 'ab': 'Ab', 'a': 'A', 'a#': 'Bb', 'bb': 'Bb', 'b': 'B'}


def main():
    out = sys.argv[1]; os.makedirs(os.path.join(out, 'audio'), exist_ok=True)
    if not os.path.isdir(os.path.join(out, 'annotations')):
        data = urllib.request.urlopen(REC.format('annotations.zip')).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(out)
    ann = collections.defaultdict(list)
    for f in glob.glob(os.path.join(out, 'annotations', '*', 'sound-*.json')):
        j = json.load(open(f))
        if j.get('discard'): continue
        k, m = str(j.get('key', '')).lower(), str(j.get('mode', '')).lower()
        ann[os.path.basename(f)[6:-5]].append((NAMES[k], 'major' if m == 'maj' else 'minor') if k in NAMES and m in ('maj', 'min') else None)
    labels = {s: v[0] for s, v in ann.items() if v[0] is not None and all(x == v[0] for x in v)}
    print(f'{len(ann)} loops anotados; {len(labels)} con tonalidad mayor/menor y anotadores de acuerdo', flush=True)
    rz = RemoteZip(REC.format('FSL10K.zip'), timeout=60)   # sin tiempo máximo, una conexión parada bloquea
    wav = {os.path.basename(n).split('_')[0]: n for n in rz.namelist() if n.endswith('.wav')}
    rows = []
    for i, (sid, (k, m)) in enumerate(sorted(labels.items())):
        if sid not in wav: continue
        fn = f'audio/{sid}.wav'; p = os.path.join(out, fn)
        for _ in range(5):
            if os.path.exists(p): break
            try:
                with open(p + '.part', 'wb') as fh: fh.write(rz.read(wav[sid]))
                os.replace(p + '.part', p)
            except Exception:
                time.sleep(20); rz = RemoteZip(REC.format('FSL10K.zip'), timeout=60)
        if os.path.exists(p): rows.append([fn, f'{k} {m}'])
        if (i + 1) % 250 == 0: print(f'  {i + 1}/{len(labels)}', flush=True)
    with open(os.path.join(out, 'expected.csv'), 'w', newline='') as fh: csv.writer(fh, lineterminator='\n').writerows(rows)
    print(len(rows), 'loops con tonalidad en', out)


if __name__ == '__main__':
    main()
