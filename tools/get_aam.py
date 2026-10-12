#!/usr/bin/env python3
"""AAM: Artificial Audio Multitracks (Zenodo 5794629, CC BY 4.0): canciones compuestas por algoritmo con muestras de
instrumentos reales y la tonalidad anotada por secciones (las canciones modulan). Baja las mezclas (FLAC) de N
canciones por rangos HTTP, corta los tramos de una sola tonalidad de al menos 30 s (mono, 44,1 kHz) y borra la
mezcla. Escribe CARPETA/expected.csv (grupo «aam»).
Uso: tools/get_aam.py CARPETA [--n 1500] [--hilos 8]      Necesita: pip install remotezip; ffmpeg
"""
import argparse, csv, glob, io, os, random, subprocess, threading, time, queue, urllib.request, zipfile
from remotezip import RemoteZip

REC = 'https://zenodo.org/api/records/5794629/files/{}/content'
BLOCKS = ['0001-1000', '1001-2000', '2001-3000']
NAMES = {'C': 'C', 'C#': 'C#', 'Db': 'Db', 'D': 'D', 'D#': 'Eb', 'Eb': 'Eb', 'E': 'E', 'F': 'F', 'F#': 'F#', 'Gb': 'F#',
         'G': 'G', 'G#': 'Ab', 'Ab': 'Ab', 'A': 'A', 'A#': 'Bb', 'Bb': 'Bb', 'B': 'B', 'Cb': 'B', 'Fb': 'E', 'E#': 'F', 'B#': 'C'}


def runs(path, min_len=30.0):
    rows = [l for l in open(path).read().split('@DATA')[1].strip().splitlines() if l.strip()]
    seg = [(float(r.split(',')[0]), r.split(',')[3].strip("'")) for r in rows]
    out, i = [], 0
    while i < len(seg) - 1:
        j = i
        while j + 1 < len(seg) - 1 and seg[j + 1][1] == seg[i][1]: j += 1
        t0, t1, k = seg[i][0], seg[j + 1][0], seg[i][1]
        if t1 - t0 >= min_len and k[-3:] in ('maj', 'min') and k[:-3] in NAMES:
            out.append((t0, t1, f"{NAMES[k[:-3]]} {'major' if k.endswith('maj') else 'minor'}"))
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--hilos', type=int, default=8); a = ap.parse_args()
    ann = os.path.join(a.out, 'annotations'); os.makedirs(os.path.join(a.out, 'audio'), exist_ok=True)
    if not glob.glob(os.path.join(ann, '*_segments.arff')):
        for b in BLOCKS: zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(REC.format(f'{b}-annotations-v1.1.0.zip')).read())).extractall(ann)
    songs = {}
    for f in sorted(glob.glob(os.path.join(ann, '*_segments.arff'))):
        r = runs(f)
        if r: songs[os.path.basename(f)[:4]] = r
    ids = sorted(songs); random.Random(0).shuffle(ids); ids = sorted(ids[:a.n])
    print(f'{len(songs)} canciones con tramos de una sola tonalidad; se usan {len(ids)}', flush=True)
    todo = queue.Queue(); [todo.put(i) for i in ids]; rows, lock, done = [], threading.Lock(), [0]

    def worker():
        zips = {}
        while True:
            try: sid = todo.get_nowait()
            except queue.Empty: return
            n = int(sid); b = BLOCKS[(n - 1) // 1000]
            outs = [(f'audio/{sid}_{k}.wav', t0, t1, key) for k, (t0, t1, key) in enumerate(songs[sid])]
            if not all(os.path.exists(os.path.join(a.out, o[0])) for o in outs):
                flac = os.path.join(a.out, f'{sid}_mix.flac')
                for _ in range(3):
                    try:
                        zips[b] = zips.get(b) or RemoteZip(REC.format(f'{b}-audio-mixes.zip'), timeout=60)
                        with open(flac, 'wb') as fh: fh.write(zips[b].read(f'{sid}_mix.flac'))
                        break
                    except Exception:
                        zips[b] = None; time.sleep(20)
                for fn, t0, t1, key in outs:
                    subprocess.run(['ffmpeg', '-v', 'quiet', '-y', '-ss', f'{t0:.3f}', '-t', f'{t1 - t0:.3f}', '-i', flac,
                                    '-ac', '1', '-ar', '44100', os.path.join(a.out, fn)])
                if os.path.exists(flac): os.remove(flac)
            with lock:
                rows.extend([fn, key] for fn, t0, t1, key in outs if os.path.exists(os.path.join(a.out, fn)))
                done[0] += 1
                if done[0] % 100 == 0: print(f'  {done[0]}/{len(ids)}', flush=True)

    ths = [threading.Thread(target=worker) for _ in range(a.hilos)]
    [t.start() for t in ths]; [t.join() for t in ths]
    rows.sort()
    with open(os.path.join(a.out, 'expected.csv'), 'w', newline='') as fh: csv.writer(fh, lineterminator='\n').writerows(rows)
    print(len(rows), 'tramos de', len(ids), 'canciones en', a.out)


if __name__ == '__main__':
    main()
