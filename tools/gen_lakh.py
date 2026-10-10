#!/usr/bin/env python3
"""Datos sintéticos de tonalidad a partir del Lakh MIDI Dataset (CC BY 4.0, colinraffel.com/projects/lmd).

Fase 1 (etiquetas): para cada MIDI, la tonalidad de la armadura escrita (o su relativa: los MIDI casi
siempre la escriben como mayor) tiene que coincidir con la que se calcula sobre las notas, con la nota más
frecuente del bajo en la tónica (perfiles de Albrecht y Shanahan, histograma de clases de nota ponderado por
duración), en la pieza entera y en un fragmento de 30 s sin cambios de armadura. Así se descartan las
armaduras puestas por defecto y las piezas que modulan.
    tools/gen_lakh.py etiquetas LMD_FULL salida.jsonl [--max 60000] [--hilos N]

Fase 2 (audio): corta el fragmento, lo transpone al azar (todas las tonalidades por igual), sortea banco de
sonidos, tempo (±10 %) y reverberación, lo sintetiza con fluidsynth y calcula las características con
skale-cli (frames4/ y spec/, como tools/prepare_training.py). Solo se guardan las características.
    tools/gen_lakh.py audio etiquetas.jsonl LMD_FULL SALIDA SOUNDFONTS_DIR [--n 20000] [--hilos N]
Escribe SALIDA/frames4/manifest_lakh.json (grupo «lakh»), para unirlo al manifiesto de los datos reales.
Necesita: pip install pretty_midi; brew install fluid-synth.
"""
import argparse, glob, hashlib, json, os, random, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import warnings; warnings.filterwarnings("ignore")   # avisos de pretty_midi sobre MIDI mal formados

# Albrecht y Shanahan (2013), índice 0 = tónica
AS_MAJ = np.array([0.238, 0.006, 0.111, 0.006, 0.137, 0.094, 0.016, 0.214, 0.009, 0.080, 0.008, 0.081])
AS_MIN = np.array([0.220, 0.006, 0.104, 0.123, 0.019, 0.103, 0.012, 0.214, 0.062, 0.022, 0.061, 0.052])
NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
CLIP = 30.0


def symbolic_key(hist):
    """Mejor tonalidad (tónica, 0 mayor / 1 menor) y margen sobre la segunda, por correlación."""
    if hist.sum() <= 0: return None, 0.0
    sc = []
    for t in range(12):
        h = np.roll(hist, -t)
        sc += [np.corrcoef(h, AS_MAJ)[0, 1], np.corrcoef(h, AS_MIN)[0, 1]]
    sc = np.nan_to_num(np.array(sc)); o = np.argsort(sc)[::-1]
    return (int(o[0]) // 2, int(o[0]) % 2), float(sc[o[0]] - sc[o[1]])


def histogram(pm, t0=0.0, t1=1e9, bass=False):
    """Clases de nota ponderadas por duración; con bass=True solo los instrumentos de bajo (programas 32-39)
    o, si no hay, las notas por debajo de Do3."""
    h = np.zeros(12)
    has_bass = any(32 <= i.program <= 39 and not i.is_drum and i.notes for i in pm.instruments)
    for inst in pm.instruments:
        if inst.is_drum or (bass and has_bass and not 32 <= inst.program <= 39): continue
        for n in inst.notes:
            if bass and not has_bass and n.pitch >= 48: continue
            d = min(n.end, t1) - max(n.start, t0)
            if d > 0: h[n.pitch % 12] += d
    return h


def label_one(path):
    import pretty_midi
    try:
        if os.path.getsize(path) > 400_000: return None
        pm = pretty_midi.PrettyMIDI(path)
    except Exception:
        return None
    ks = pm.key_signature_changes
    if not ks or len({k.key_number for k in ks}) != 1: return None
    dur = pm.get_end_time()
    if dur < 60 or dur > 600: return None
    if sum(1 for i in pm.instruments if not i.is_drum and len(i.notes) > 20) < 2: return None   # al menos 2 instrumentos con notas
    # La armadura no distingue bien mayor de su relativa menor (casi todos los MIDI dicen «mayor»):
    # se acepta la tonalidad escrita o su relativa si las notas lo indican y el bajo insiste en la tónica.
    kn = ks[0].key_number; t, m = kn % 12, kn // 12
    rel = ((t + 9) % 12, 1) if m == 0 else ((t + 3) % 12, 0)
    g, gm = symbolic_key(histogram(pm))
    if g not in ((t, m), rel) or gm < 0.02: return None
    want = g
    bh = histogram(pm, bass=True)
    if bh.sum() <= 0 or int(np.argmax(bh)) != want[0]: return None
    # fragmento de 30 s: el más cercano al centro cuya tonalidad local (y su bajo) coincide
    for off in (0, -20, 20, -40, 40):
        t0 = dur / 2 - CLIP / 2 + off
        if t0 < 5 or t0 + CLIP > dur - 5: continue
        h = histogram(pm, t0, t0 + CLIP)
        if h.sum() < 20: continue
        k, mg = symbolic_key(h); bw = histogram(pm, t0, t0 + CLIP, bass=True)
        if k == want and mg > 0.02 and bw.sum() > 0 and int(np.argmax(bw)) == want[0]:
            return dict(path=path, tonic=want[0], mode=want[1], t0=round(t0, 2), margin=round(gm, 3))
    return None


def cmd_labels(a):
    files = sorted(glob.glob(os.path.join(a.lmd, '**', '*.mid'), recursive=True))
    random.Random(0).shuffle(files); files = files[:a.max]
    print(len(files), 'MIDI', flush=True)
    n = ok = 0
    with ProcessPoolExecutor(a.hilos) as ex, open(a.out, 'w') as fh:
        for r in ex.map(label_one, files, chunksize=32):
            n += 1
            if r: ok += 1; r['path'] = os.path.relpath(r['path'], a.lmd); fh.write(json.dumps(r) + '\n')
            if n % 5000 == 0: print(f'  {n}  válidos {ok}', flush=True)
    print(ok, 'fragmentos válidos de', n)


def render_one(job):
    import pretty_midi
    r, lmd, out, sfs, cli, seed = job
    rng = random.Random(seed)
    name = 'lakh_' + hashlib.md5(f"{r['path']}:{seed}".encode()).hexdigest()[:12] + '.bin'
    pf, ps = os.path.join(out, 'frames4', name), os.path.join(out, 'spec', name)
    shift = rng.randint(-6, 5)
    label = [(r['tonic'] + shift) % 12, 'major' if r['mode'] == 0 else 'minor']
    if not (os.path.exists(pf) and os.path.exists(ps)):
        try:
            pm = pretty_midi.PrettyMIDI(os.path.join(lmd, r['path']))
        except Exception:
            return None
        t0, t1, sp = r['t0'], r['t0'] + CLIP, rng.uniform(0.9, 1.1)
        new = pretty_midi.PrettyMIDI(initial_tempo=120)
        for inst in pm.instruments:
            ni = pretty_midi.Instrument(inst.program, is_drum=inst.is_drum)
            for n in inst.notes:
                if n.end <= t0 or n.start >= t1: continue
                p = n.pitch if inst.is_drum else n.pitch + shift
                if not 0 <= p < 128: continue
                ni.notes.append(pretty_midi.Note(n.velocity, p, (max(n.start, t0) - t0) / sp, (min(n.end, t1) - t0) / sp))
            for b in inst.pitch_bends:
                if t0 <= b.time < t1: ni.pitch_bends.append(pretty_midi.PitchBend(b.pitch, (b.time - t0) / sp))
            for c in inst.control_changes:
                if c.number in (7, 10, 11, 64) and c.time < t1:
                    ni.control_changes.append(pretty_midi.ControlChange(c.number, c.value, max(0.0, c.time - t0) / sp))
            if ni.notes: new.instruments.append(ni)
        with tempfile.TemporaryDirectory() as td:
            mid, wav = os.path.join(td, 'a.mid'), os.path.join(td, 'a.wav')
            new.write(mid)
            sf = rng.choice(sfs)
            opts = ['-o', f'synth.reverb.active={rng.choice(["0", "1"])}', '-o', f'synth.chorus.active={rng.choice(["0", "1"])}',
                    '-g', f'{rng.uniform(0.3, 0.8):.2f}']
            subprocess.run(['fluidsynth', '-ni', '-q', '-r', '44100', *opts, '-F', wav, sf, mid], capture_output=True, timeout=300)
            if not os.path.exists(wav): return None
            subprocess.run([cli, wav, '--model', 'classic', '--frames', pf, '--spec', ps], capture_output=True, timeout=300)
    ok = os.path.exists(pf) and os.path.getsize(pf) >= 4 * 4 * 36 * 20 and os.path.exists(ps)
    return dict(group='lakh', name=f"{r['path']}#{seed}", label=label, file=name) if ok else None


def cmd_audio(a):
    rows = [json.loads(l) for l in open(a.labels)]
    random.Random(1).shuffle(rows); rows = rows[:a.n]
    sfs = sorted(glob.glob(os.path.join(a.soundfonts, '*.sf[23]')))
    os.makedirs(os.path.join(a.out, 'frames4'), exist_ok=True); os.makedirs(os.path.join(a.out, 'spec'), exist_ok=True)
    cli = os.path.abspath(a.cli)
    jobs = [(r, a.lmd, a.out, sfs, cli, i) for i, r in enumerate(rows)]
    print(len(jobs), 'fragmentos,', len(sfs), 'bancos de sonidos', flush=True)
    res = []
    with ProcessPoolExecutor(a.hilos) as ex:
        for i, r in enumerate(ex.map(render_one, jobs, chunksize=4)):
            res.append(r)
            if (i + 1) % 1000 == 0: print(f'  {i + 1}/{len(jobs)}', flush=True)
    res = [r for r in res if r]
    json.dump(res, open(os.path.join(a.out, 'frames4', 'manifest_lakh.json'), 'w'))
    print(len(res), 'fragmentos preparados')


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('etiquetas'); p.add_argument('lmd'); p.add_argument('out')
    p.add_argument('--max', type=int, default=60000); p.add_argument('--hilos', type=int, default=os.cpu_count())
    p = sub.add_parser('audio'); p.add_argument('labels'); p.add_argument('lmd'); p.add_argument('out'); p.add_argument('soundfonts')
    p.add_argument('--n', type=int, default=20000); p.add_argument('--hilos', type=int, default=os.cpu_count())
    p.add_argument('--cli', default='build/skale-cli')
    a = ap.parse_args()
    cmd_labels(a) if a.cmd == 'etiquetas' else cmd_audio(a)


if __name__ == '__main__':
    main()
