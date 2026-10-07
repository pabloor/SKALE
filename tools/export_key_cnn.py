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
           '#include <cstddef>\nnamespace skale::cnn {\n']
    names = []; ch = hid = 0
    for n, path in enumerate(a.models):
        m = tkc.KeyNet(); m.load_state_dict(torch.load(path)); m.eval(); ch = m.ch; hid = m.head[0].out_features
        for i, (c, b) in enumerate(((m.c1, m.bn1), (m.c2, m.bn2), (m.c3, m.bn3)), 1):
            w, bias = fold(c, b)
            out += [arr(f'k{n}Conv{i}W', w), arr(f'k{n}Conv{i}B', bias)]
        out += [arr(f'k{n}Head1W', m.head[0].weight.detach().numpy()), arr(f'k{n}Head1B', m.head[0].bias.detach().numpy()),
                arr(f'k{n}Head2W', m.head[3].weight.detach().numpy()), arr(f'k{n}Head2B', m.head[3].bias.detach().numpy())]
        names.append(n)
    out += [f'constexpr int kCh = {ch};\nconstexpr int kHidden = {hid};\n',
            'struct Net { const float *c1w, *c1b, *c2w, *c2b, *c3w, *c3b, *h1w, *h1b, *h2w, *h2b; };\n',
            f'constexpr int kNetCount = {len(names)};\nconstexpr Net kNets[kNetCount] = {{\n' +
            ',\n'.join(f'    {{k{n}Conv1W, k{n}Conv1B, k{n}Conv2W, k{n}Conv2B, k{n}Conv3W, k{n}Conv3B, k{n}Head1W, k{n}Head1B, k{n}Head2W, k{n}Head2B}}' for n in names) +
            '\n};\n}  // namespace skale::cnn\n']
    open(a.out, 'w').write(''.join(out)); print('escrito', a.out, len(names), 'redes')


if __name__ == '__main__':
    main()
