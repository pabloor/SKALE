#!/usr/bin/env python3
"""Características y notas de Slakh (tools/get_slakh.py) para la tarea auxiliar de train_key_cnn.py (--aux).
Por canción, con skale-cli: cromagramas (SALIDA/frames4), espectro (SALIDA/spec) e instante de cada fotograma; con
el MIDI (sin batería): por fotograma, la actividad de las 12 clases de nota (duración solapada con ±0,19 s por
velocidad, dividida por la máxima) y la clase de nota más grave (bajo, una de 12; ceros si no hay). Escribe
SALIDA/notes/*.bin (float32 [n][24]) y SALIDA/slakh_manifest.json (grupo «slakh», sin tonalidad).
Uso: tools/slakh_notes.py CARPETA_SLAKH SALIDA [--cli build/skale-cli] [--hilos N]   Necesita: pretty_midi, ffmpeg
"""
import argparse, glob, hashlib, json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pretty_midi

HALF = 0.19     # medio fotograma emparejado (~0,19 s): ventana de solape alrededor de cada instante


def targets(midi, times):
    notes = [(n.start, n.end, n.pitch, n.velocity) for ins in pretty_midi.PrettyMIDI(midi).instruments if not ins.is_drum
             for n in ins.notes]
    if not notes: return None
    s, e, p, v = (np.array(x, dtype=np.float64) for x in zip(*notes))
    out = np.zeros((len(times), 24), np.float32)
    for i, t in enumerate(times):
        ov = np.clip(np.minimum(e, t + HALF) - np.maximum(s, t - HALF), 0, None)
        k = ov > 0
        if not k.any(): continue
        pc = np.bincount((p[k] % 12).astype(int), weights=ov[k] * v[k] / 127, minlength=12)
        out[i, :12] = pc / pc.max()
        out[i, 12 + int(p[k].min()) % 12] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('slakh'); ap.add_argument('salida')
    ap.add_argument('--cli', default='build/skale-cli'); ap.add_argument('--hilos', type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    dirs = {d: os.path.join(a.salida, d) for d in ('frames4', 'spec', 'notes', 'times')}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    def one(audio):
        track = os.path.basename(audio)[:-5]; midi = os.path.join(a.slakh, 'midi', track + '.mid')
        name = f"slakh_{hashlib.md5(('audio/' + track + '.flac').encode()).hexdigest()[:12]}.bin"
        pf, ps, pt, pn = (os.path.join(dirs[d], name) for d in ('frames4', 'spec', 'times', 'notes'))
        if not os.path.exists(midi): return None      # el MIDI de esa canción aún no se ha bajado
        if not os.path.exists(pn):
            wav = pn + '.wav'     # skale-cli lee wav y mp3
            subprocess.run(['ffmpeg', '-v', 'quiet', '-y', '-i', audio, wav])
            subprocess.run([a.cli, wav, '--model', 'classic', '--frames', pf, '--spec', ps, '--frame-times', pt], capture_output=True)
            if os.path.exists(wav): os.remove(wav)
            if not os.path.exists(pt) or not os.path.exists(midi): return None
            tg = targets(midi, np.fromfile(pt, dtype=np.float32))
            if tg is None: return None
            tg.tofile(pn)
        return dict(group='slakh', name='audio/' + track + '.flac', label=None, file=name)

    with ThreadPoolExecutor(a.hilos) as ex:
        rows = [r for r in ex.map(one, sorted(glob.glob(os.path.join(a.slakh, 'audio', '*.flac')))) if r]
    json.dump(rows, open(os.path.join(a.salida, 'slakh_manifest.json'), 'w'))
    print(len(rows), 'canciones de Slakh con notas en', a.salida)


if __name__ == '__main__':
    main()
