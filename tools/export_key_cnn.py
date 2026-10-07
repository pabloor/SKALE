#!/usr/bin/env python3
"""Exporta un modelo de tools/train_key_cnn.py a src/analysis/KeyCnnWeights.h (BatchNorm plegado en las
convoluciones). Uso: tools/export_key_cnn.py modelo.pt [--out src/analysis/KeyCnnWeights.h]"""
import argparse, importlib.util, os
import numpy as np, torch

spec = importlib.util.spec_from_file_location('tkc', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train_key_cnn.py'))
tkc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tkc)


def fold(conv, bn):
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    return (conv.weight * s[:, None, None, None]).detach().numpy(), ((conv.bias - bn.running_mean) * s + bn.bias).detach().numpy()


def arr(name, a):
    flat = ', '.join(f'{x:.7g}f' for x in a.ravel())
    return f'constexpr float {name}[{a.size}] = {{{flat}}};\n'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='+'); ap.add_argument('--out', default='src/analysis/KeyCnnWeights.h')
    a = ap.parse_args()
    out = ['#pragma once\n// Pesos de las redes de tonalidad (tools/train_key_cnn.py, exportados con tools/export_key_cnn.py).\n',
           '// No editar a mano. BatchNorm plegado en las convoluciones. Se promedian las log-probabilidades de las redes.\n',
           'namespace skale::cnn {\n',
           '// Entrada por fotograma: canales en este orden: cromagrama, bajo, medios, agudos (inCh primeros).\n',
           'struct Net {\n    int inCh, ch, hidden, crop;\n    const float *c1w, *c1b, *c2w, *c2b, *c3w, *c3b, *h1w, *h1b, *h2w, *h2b;\n};\n']
    nets = []
    for n, path in enumerate(a.models):
        ck = torch.load(path)
        state, cfg = (ck['state'], ck['cfg']) if 'state' in ck else (ck, {})
        if cfg.get('layers', 3) != 3: raise SystemExit('el exportador solo admite redes de 3 capas')
        m = tkc.KeyNet(ch=cfg.get('ch', 16), hidden=cfg.get('hidden', 64), in_ch=cfg.get('in_ch', 2))
        m.load_state_dict(state); m.eval()
        for i, (c, b) in enumerate(((m.c1, m.bn1), (m.c2, m.bn2), (m.c3, m.bn3)), 1):
            w, bias = fold(c, b)
            out += [arr(f'k{n}Conv{i}W', w), arr(f'k{n}Conv{i}B', bias)]
        out += [arr(f'k{n}Head1W', m.head[0].weight.detach().numpy()), arr(f'k{n}Head1B', m.head[0].bias.detach().numpy()),
                arr(f'k{n}Head2W', m.head[3].weight.detach().numpy()), arr(f'k{n}Head2B', m.head[3].bias.detach().numpy())]
        nets.append(f'    {{{m.in_ch}, {m.ch}, {m.head[0].out_features}, {cfg.get("crop", 128)}, k{n}Conv1W, k{n}Conv1B, k{n}Conv2W, k{n}Conv2B, '
                    f'k{n}Conv3W, k{n}Conv3B, k{n}Head1W, k{n}Head1B, k{n}Head2W, k{n}Head2B}}')
        print(f'red {n}: {path}  entrada={m.in_ch} canales={m.ch} oculta={m.head[0].out_features} ventana={cfg.get("crop", 128)}')
    out += [f'constexpr int kNetCount = {len(nets)};\nconstexpr Net kNets[kNetCount] = {{\n' + ',\n'.join(nets) + '\n};\n}  // namespace skale::cnn\n']
    open(a.out, 'w').write(''.join(out)); print('escrito', a.out, len(nets), 'redes')


if __name__ == '__main__':
    main()
