#!/usr/bin/env python3
"""Descarga las canciones completas de FMAK (las 5.489 pistas de Free Music Archive con tonalidad anotada por
expertos) desde el zip de 944 GB de fma_full, por rangos HTTP y sin bajarlo entero (~25-30 GB).

Las etiquetas de FMAK son de la canción entera; tools/get_fmak.py baja solo el fragmento de 30 s de fma_large.
Escribe en CARPETA:
  expected.csv       pistas que no están en la partición de prueba fija (grupo «fma_full», entrenamiento)
  expected_test.csv  las de la partición de prueba (grupo «fma_full_test», solo evaluación)
Uso: tools/get_fmak_full.py CARPETA_FMA CARPETA [--hilos 8]
  CARPETA_FMA: la de tools/get_fmak.py (de su expected.csv salen las pistas y las tonalidades)
Necesita: pip install remotezip
"""
import csv, json, os, queue, sys, threading, time
from remotezip import RemoteZip

FMA_FULL = 'https://os.unil.cloud.switch.ch/fma/fma_full.zip'
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    src, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == '--hilos' else 8
    os.makedirs(out, exist_ok=True)
    rows = list(csv.reader(open(os.path.join(src, 'expected.csv'))))
    test = {n for g, n in json.load(open(os.path.join(HERE, 'samples', 'test_split.json'))) if g == 'fma'}
    todo = queue.Queue()
    for fn, _ in rows:
        if not os.path.exists(os.path.join(out, fn)): todo.put(fn)
    total = todo.qsize(); done = [0]; lock = threading.Lock()
    print(f'{len(rows)} pistas, {total} por descargar', flush=True)

    def worker():
        rz = None
        while True:
            try: fn = todo.get_nowait()
            except queue.Empty: return
            t = int(fn[:6])
            for _ in range(3):
                try:
                    rz = rz or RemoteZip(FMA_FULL)
                    data = rz.read(f'fma_full/{t // 1000:03d}/{t:06d}.mp3')
                    with open(os.path.join(out, fn + '.part'), 'wb') as fh: fh.write(data)
                    os.replace(os.path.join(out, fn + '.part'), os.path.join(out, fn)); break
                except KeyError:
                    break                     # no está en fma_full
                except Exception:
                    rz = None; time.sleep(2)
            with lock:
                done[0] += 1
                if done[0] % 250 == 0: print(f'  {done[0]}/{total}', flush=True)

    ths = [threading.Thread(target=worker) for _ in range(threads)]
    [t.start() for t in ths]; [t.join() for t in ths]
    have = [(fn, k) for fn, k in rows if os.path.exists(os.path.join(out, fn))]
    with open(os.path.join(out, 'expected.csv'), 'w', newline='') as fh:
        csv.writer(fh, lineterminator='\n').writerows([r for r in have if r[0] not in test])
    with open(os.path.join(out, 'expected_test.csv'), 'w', newline='') as fh:
        csv.writer(fh, lineterminator='\n').writerows([r for r in have if r[0] in test])
    print(len(have), 'de', len(rows), 'canciones completas;', sum(r[0] in test for r in have), 'de la partición de prueba')


if __name__ == '__main__':
    main()
