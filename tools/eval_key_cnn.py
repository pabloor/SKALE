#!/usr/bin/env python3
"""Evalúa redes de tools/train_key_cnn.py, solas y combinadas como en el analizador, sobre la partición
de prueba (tools/samples/test_split.json). Las redes de cromagrama y las de espectro se promedian por
separado y se mezclan con --spec-weight, igual que en src/analysis/KeyCnn.cpp.

Uso: tools/eval_key_cnn.py SALIDA/frames4/manifest.json SALIDA/frames4 --spec SALIDA/spec a.pt b.pt ...
     [--split tools/samples/test_split.json] [--spec-weight 0.5]
Imprime el % de acierto exacto por grupo y la puntuación MIREX (quinta ascendente 0,5, relativa 0,3, paralela 0,2).
"""
import argparse, importlib.util, json, os
import numpy as np, torch

spec = importlib.util.spec_from_file_location('tkc', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_key_cnn.py'))
tkc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tkc)
GROUPS = ['fma', 'beatport', 'guitarset', 'gsplus', 'gtzan']


def mirex(p, y):
    tp, mp, ty, my = p // 2, p % 2, y // 2, y % 2
    s = np.where(p == y, 1.0, 0.0)
    s += np.where((mp == my) & ((tp - ty) % 12 == 7), 0.5, 0)                     # quinta ascendente (como mir_eval)
    s += np.where((mp == 1) & (my == 0) & ((tp - ty) % 12 == 9) | (mp == 0) & (my == 1) & ((tp - ty) % 12 == 3), 0.3, 0)
    s += np.where((tp == ty) & (mp != my), 0.2, 0)
    return s.mean()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(); ap.add_argument('manifest'); ap.add_argument('frames'); ap.add_argument('models', nargs='+')
    ap.add_argument('--spec', default=''); ap.add_argument('--split', default=os.path.join(here, 'samples', 'test_split.json'))
    ap.add_argument('--spec-weight', type=float, default=0.5); ap.add_argument('--device', default='auto')
    ap.add_argument('--threads', type=int, default=0, help='hilos de CPU (0 = todos)')
    a = ap.parse_args()
    dev = tkc.pick_device(a.device); torch.set_num_threads(a.threads or os.cpu_count() or 4)
    names = {tuple(r) for r in json.load(open(a.split))}
    M = [r for r in json.load(open(a.manifest)) if (r['group'], r['name']) in names]
    print(f'{len(M)} de {len(names)} pistas de prueba disponibles', flush=True)
    y = np.array([r['label'][0] * 2 + (0 if r['label'][1] == 'major' else 1) for r in M]); g = np.array([r['group'] for r in M])

    def report(name, lp):
        p = lp.argmax(1)
        cols = ' '.join(f'{k}={100 * (p[g == k] == y[g == k]).mean():.1f}' for k in GROUPS if (g == k).any())
        print(f'{name:28} {cols}  todo={100 * (p == y).mean():.1f}  MIREX={mirex(p, y):.3f}', flush=True)

    chroma, specs = [], []
    for path in a.models:
        ck = torch.load(path, map_location='cpu'); c = ck['cfg']
        if c.get('spec'):
            m = tkc.KeyNetSpec(ch=c['ch'], hidden=c['hidden']); load = lambda r: np.fromfile(os.path.join(a.spec, r['file']), dtype=np.float32).reshape(-1, 1, 216) * 3.0
        else:
            ic = c.get('in_ch', 2); m = tkc.KeyNet(ch=c['ch'], hidden=c['hidden'], in_ch=ic, layers=c.get('layers', 3))
            load = lambda r, ic=ic: np.sqrt(np.maximum(np.fromfile(os.path.join(a.frames, r['file']), dtype=np.float32).reshape(-1, 4, 36)[:, :ic], 0)) * 3.0
        m.load_state_dict(ck['state']); m.to(dev).eval()
        lp = np.array([tkc.predict(m, load(r), c.get('crop', 128)) for r in M])
        (specs if c.get('spec') else chroma).append(lp); report(os.path.basename(path), lp)
    if len(a.models) > 1:
        if chroma: report('media cromagrama', np.mean(chroma, 0))
        if specs: report('media espectro', np.mean(specs, 0))
        if chroma and specs:
            for w in sorted({0.3, 0.5, 0.7, a.spec_weight}):
                report(f'mezcla espectro {w:g}', (1 - w) * np.mean(chroma, 0) + w * np.mean(specs, 0))


if __name__ == '__main__':
    main()
