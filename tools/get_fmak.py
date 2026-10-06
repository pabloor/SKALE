#!/usr/bin/env python3
"""Descarga el audio de FMAK (5.489 pistas de Free Music Archive con tonalidad anotada por expertos)
y escribe CARPETA/expected.csv. Los clips de 30 s se sacan del zip de 100 GB de FMA por rangos HTTP,
sin bajarlo entero (~4 GB en total).

Etiquetas: FMAKv2, CC BY 4.0, https://zenodo.org/records/12759100
Audio: FMA (fma_large), licencias Creative Commons por pista, https://github.com/mdeff/fma
Uso: tools/get_fmak.py CARPETA [--hilos 8]      Necesita: pip install remotezip
"""
import csv, os, queue, sys, threading, time, urllib.request
from remotezip import RemoteZip

FMA = 'https://os.unil.cloud.switch.ch/fma/fma_large.zip'
LABELS = 'https://zenodo.org/api/records/12759100/files/fmakv2.csv/content'


def main():
    out = sys.argv[1]; threads = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == '--hilos' else 8
    os.makedirs(out, exist_ok=True)
    rows = list(csv.DictReader(urllib.request.urlopen(LABELS).read().decode().splitlines()))
    todo = queue.Queue()
    for r in rows:
        if not os.path.exists(f"{out}/{int(r['track_id']):06d}.mp3"): todo.put(int(r['track_id']))

    def worker():
        rz = None
        while True:
            try: t = todo.get_nowait()
            except queue.Empty: return
            for _ in range(3):
                try:
                    rz = rz or RemoteZip(FMA)
                    open(f'{out}/{t:06d}.mp3', 'wb').write(rz.read(f'fma_large/{t // 1000:03d}/{t:06d}.mp3')); break
                except Exception:
                    rz = None; time.sleep(2)

    ths = [threading.Thread(target=worker) for _ in range(threads)]
    [t.start() for t in ths]; [t.join() for t in ths]
    with open(f'{out}/expected.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        for r in rows:
            k = r['key_and_mode'].split()
            w.writerow([f"{int(r['track_id']):06d}.mp3", f'{k[0]} {k[1].lower()}'])
    print('listo', len(rows), 'pistas')


if __name__ == '__main__':
    main()
