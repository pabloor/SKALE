#!/usr/bin/env python3
"""Slakh2100 (Zenodo 4599666, CC BY 4.0): 2.100 canciones de Lakh MIDI renderizadas con instrumentos virtuales, con
todas las notas. Lee el tar.gz de 104 GB en streaming (sin guardarlo) y de cada canción guarda solo la mezcla en
mono (FLAC, 44,1 kHz) y el MIDI con todas las fuentes; las pistas por instrumento se descartan al vuelo.
Para cuando tiene N canciones de la partición pedida (en el orden del archivo).
Uso: tools/get_slakh.py CARPETA [--n 300] [--split train]      Necesita: ffmpeg
Escribe CARPETA/audio/TrackXXXXX.flac y CARPETA/midi/TrackXXXXX.mid. Después: tools/slakh_notes.py.
"""
import argparse, io, os, shutil, subprocess, sys, tarfile, threading, time, urllib.request

URL = 'https://zenodo.org/api/records/4599666/files/slakh2100_flac_redux.tar.gz/content'


class RangeReader(io.RawIOBase):
    """Lectura secuencial del archivo remoto con varias conexiones a la vez: trozos de `chunk` bytes pedidos por rangos
    HTTP por adelantado (hasta `ahead` en memoria) y entregados en orden. Con una sola conexión Zenodo da ~2 MB/s."""
    def __init__(self, url, chunk=16 << 20, workers=8, ahead=24):
        self.url, self.chunk = url, chunk
        req = urllib.request.Request(url, method='HEAD')
        self.size = int(urllib.request.urlopen(req, timeout=120).headers['Content-Length'])
        self.n = (self.size + chunk - 1) // chunk
        self.parts, self.cv, self.next_req, self.cur, self.buf = {}, threading.Condition(), 0, 0, b''
        self.ahead = ahead
        for _ in range(workers): threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        while True:
            with self.cv:
                while self.next_req < self.n and self.next_req >= self.cur + self.ahead: self.cv.wait()
                if self.next_req >= self.n: return
                i = self.next_req; self.next_req += 1
            a, b = i * self.chunk, min(self.size, (i + 1) * self.chunk) - 1
            for k in range(20):
                try:
                    r = urllib.request.Request(self.url, headers={'Range': f'bytes={a}-{b}'})
                    data = urllib.request.urlopen(r, timeout=60).read()
                    if len(data) == b - a + 1: break
                except Exception:
                    pass
                time.sleep(min(60, 5 * (k + 1)))
            with self.cv: self.parts[i] = data; self.cv.notify_all()

    def readable(self): return True

    def readinto(self, out):
        while not self.buf:
            if self.cur >= self.n: return 0
            with self.cv:
                while self.cur not in self.parts: self.cv.wait()
                self.buf = self.parts.pop(self.cur); self.cur += 1; self.cv.notify_all()
        k = min(len(out), len(self.buf)); out[:k] = self.buf[:k]; self.buf = self.buf[k:]
        return k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--split', default='train', help='train, validation, test u omitted'); a = ap.parse_args()
    for d in ('audio', 'midi', 'tmp'): os.makedirs(os.path.join(a.out, d), exist_ok=True)
    resp = io.BufferedReader(RangeReader(URL), 1 << 20)
    t0, done, seen = time.time(), set(), 0
    with tarfile.open(fileobj=resp, mode='r|gz') as tf:
        for m in tf:
            seen += m.size
            parts = m.name.split('/')     # slakh2100_flac_redux/SPLIT/TrackXXXXX/...
            if len(parts) < 4 or parts[1] != a.split or not m.isfile(): continue
            track, rest = parts[2], '/'.join(parts[3:])
            if os.path.exists(os.path.join(a.out, 'audio' if rest == 'mix.flac' else 'midi', track + ('.flac' if rest == 'mix.flac' else '.mid'))):
                pass
            elif rest == 'all_src.mid':
                with open(os.path.join(a.out, 'midi', track + '.mid'), 'wb') as fh: shutil.copyfileobj(tf.extractfile(m), fh)
            elif rest == 'mix.flac':
                tmp = os.path.join(a.out, 'tmp', track + '.flac')
                with open(tmp, 'wb') as fh: shutil.copyfileobj(tf.extractfile(m), fh)
                subprocess.run(['ffmpeg', '-v', 'quiet', '-y', '-i', tmp, '-ac', '1', '-ar', '44100',
                                os.path.join(a.out, 'audio', track + '.flac')])
                os.remove(tmp)
            elif rest != 'mix.flac' and rest != 'all_src.mid': continue
            if all(os.path.exists(os.path.join(a.out, d, track + e)) for d, e in (('audio', '.flac'), ('midi', '.mid'))):
                done.add(track)
                if len(done) % 10 == 0:
                    mb = seen / 1e6; s = time.time() - t0
                    print(f'  {len(done)}/{a.n} canciones, {mb / 1000:.1f} GB leídos, {mb / s:.0f} MB/s', flush=True)
                    open(os.path.join(a.out, 'progreso.txt'), 'w').write(f'{len(done)} {a.n}')
                if len(done) >= a.n: break
    shutil.rmtree(os.path.join(a.out, 'tmp'), ignore_errors=True)
    print(len(done), 'canciones en', a.out)


if __name__ == '__main__':
    sys.exit(main())
