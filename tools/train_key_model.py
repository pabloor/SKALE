#!/usr/bin/env python3
"""Entrena el modelo de tonalidad (regresión logística condicional sobre las 24
tonalidades) con las características de tools/extract_features.py.

Para cada tonalidad (tónica t, modo m) la puntuación es lineal en
  [corr. de Temperley, bajo rotado (12), final rotado (12), 3 x voto por ventanas]   modelo "full"
  [corr. de Temperley, bajo rotado (12)]                        modelo "live"
con un vector de pesos por modo. Muestras ponderadas para que cada conjunto
pese igual. Imprime la validación dejando un conjunto fuera ("LOGO"), que es la
cifra honesta, y escribe src/analysis/KeyModelWeights.h.

Uso: tools/train_key_model.py features.json[.gz] [--lam 0.1] [--out src/analysis/KeyModelWeights.h]
Necesita numpy y scipy.
"""
import argparse, collections, gzip, json
import numpy as np
from scipy.optimize import minimize

TM = np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0])
Tm = np.array([5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0])


def pear(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return (x * y).sum() / d if d > 0 else 0.0


def label(r): return r['label'][0] * 2 + (0 if r['label'][1] == 'major' else 1)


def votes(r):
    """Fracción de ventanas de 8 s cuya mejor tonalidad (Temperley) es cada una de las 24."""
    v = np.zeros(24)
    for w in r.get('win', []):
        c = np.array(w[0]) * 12
        sc = [pear(c, np.roll(prof, t)) for t in range(12) for prof in (TM, Tm)]
        v[int(np.argmax(sc))] += 1          # empate: la primera, como en C++
    return v / max(len(r.get('win', [])), 1)


def feats(r, use_end):
    c = np.array(r['chroma']) * 12; b = np.array(r['bass']) * 12; e = np.array(r['ending']) * 12
    vt = votes(r) if use_end else None
    rows = []
    for t in range(12):
        for k, prof in enumerate((TM, Tm)):
            v = [[pear(c, np.roll(prof, t))], np.roll(b, -t)]
            if use_end: v += [np.roll(e, -t), [3.0 * vt[t * 2 + k]]]
            rows.append(np.concatenate(v))
    return np.array(rows)


def fit(X, y, w, lam):
    n, _, d = X.shape
    mode = np.arange(24) % 2

    def f(theta):
        W = theta.reshape(2, d)
        S = np.einsum('nkd,kd->nk', X, W[mode]); S -= S.max(1, keepdims=True)
        P = np.exp(S); P /= P.sum(1, keepdims=True)
        nll = -(w * np.log(P[np.arange(n), y] + 1e-12)).sum() + lam * (theta ** 2).sum()
        G = P.copy(); G[np.arange(n), y] -= 1; G *= w[:, None]
        gW = np.zeros((2, d))
        for m in (0, 1):
            idx = np.arange(m, 24, 2); gW[m] = np.einsum('nk,nkd->d', G[:, idx], X[:, idx])
        return nll, gW.ravel() + 2 * lam * theta

    return minimize(f, np.zeros(2 * d), jac=True, method='L-BFGS-B', options={'maxiter': 300}).x.reshape(2, d)


def predict(X, W): return np.argmax(np.einsum('nkd,kd->nk', X, W[np.arange(24) % 2]), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('features'); ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--out', default='src/analysis/KeyModelWeights.h')
    a = ap.parse_args()
    D = json.load(gzip.open(a.features, 'rt') if a.features.endswith('.gz') else open(a.features))
    y = np.array([label(r) for r in D]); grp = np.array([r['group'] for r in D])
    size = collections.Counter(grp)
    w = np.array([1.0 / size[g] for g in grp]); w *= len(D) / w.sum() / len(size)
    out = {}
    for name, use_end in (('full', True), ('live', False)):
        X = np.array([feats(r, use_end) for r in D])
        pred = np.zeros(len(D), int)
        for g in size:
            tr = grp != g; pred[~tr] = predict(X[~tr], fit(X[tr], y[tr], w[tr], a.lam))
        per = {g: (pred[grp == g] == y[grp == g]).mean() for g in size}
        print(f'LOGO {name:5} media por conjunto {100 * np.mean(list(per.values())):.1f}%  total {100 * (pred == y).mean():.1f}%  ' +
              ' '.join(f'{g}={100 * v:.0f}%' for g, v in per.items()))
        out[name] = fit(X, y, w, a.lam)
    def arr(n, W, desc):
        rows = ',\n'.join('    {' + ', '.join(f'{x:.5f}f' for x in row) + '}' for row in W)
        return f'// {desc}\nconstexpr float {n}[2][{len(W[0])}] = {{\n{rows}\n}};\n'
    hdr = '''#pragma once
// Pesos del modelo de tonalidad aprendido. Generado por tools/train_key_model.py: no editar a mano.
// Para cada tonalidad (tónica t, modo m; fila 0 = mayor, 1 = menor) la puntuación es
//   w[m][0] x corr_Temperley(chroma, t, m)
//   + sum_i w[m][1+i]  x bass_rotado[i]
//   + sum_i w[m][13+i] x ending_rotado[i]   (solo modelo "full")
//   + w[m][25] x 3 x voto[t,m]               (solo modelo "full"; voto = fracción de ventanas de 8 s
//                                              cuya mejor tonalidad con Temperley es (t,m))
// con cromagramas multiplicados por 12 y rotados para que el índice 0 sea la tónica.
namespace skale::model {
'''
    open(a.out, 'w').write(hdr + arr('kFull', out['full'], 'Con final de la pieza: [corr, bajo(12), final(12), voto]') + '\n' +
                           arr('kLive', out['live'], 'Sin final (tiempo real): [corr, bajo(12)]') + '}  // namespace skale::model\n')
    print('escrito', a.out)


if __name__ == '__main__':
    main()
