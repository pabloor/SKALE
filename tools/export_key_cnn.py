#!/usr/bin/env python3
"""Exporta modelos de tools/train_key_cnn.py a src/analysis/KeyCnnWeights.h (BatchNorm plegado en las
convoluciones). Las redes de cromagrama y las de espectro (--spec) se promedian por separado y se mezclan
con --spec-weight. Uso: tools/export_key_cnn.py a.pt b.pt ... [--spec-weight 0.5] [--out ...]"""
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
    ap.add_argument('--spec-weight', type=float, default=0.5, help='peso de la media de las redes de espectro frente a la de cromagrama')
    a = ap.parse_args()
    out = ['#pragma once\n// Pesos de las redes de tonalidad (tools/train_key_cnn.py, exportados con tools/export_key_cnn.py).\n',
           '// No editar a mano. BatchNorm plegado en las convoluciones. Se promedian las log-probabilidades de las redes\n',
           '// de cada tipo y se mezclan: (1 - kSpecWeight) * cromagrama + kSpecWeight * espectro.\n',
           'namespace skale::cnn {\n', 'constexpr int kMaxAttn = 2;\n',
           '// Entrada por fotograma: canales en este orden: cromagrama, bajo, medios, agudos (inCh primeros).\n',
           'struct Net {\n    int inCh, ch, hidden, crop;\n    const float *c1w, *c1b, *c2w, *c2b, *c3w, *c3b, *h1w, *h1b, *h2w, *h2b;\n'
           '    // Atención entre fotogramas (KeyNetAttn; attn = 0 en las demás): capas, canales d y cabezas. Matrices (salida, entrada).\n'
           '    int attn = 0, d = 0, heads = 0;\n    const float *inw = nullptr, *inb = nullptr;\n'
           '    const float *qkvw[kMaxAttn] = {}, *qkvb[kMaxAttn] = {}, *projw[kMaxAttn] = {}, *projb[kMaxAttn] = {};\n'
           '    const float *ff1w[kMaxAttn] = {}, *ff1b[kMaxAttn] = {}, *ff2w[kMaxAttn] = {}, *ff2b[kMaxAttn] = {};\n'
           '    const float *poolw = nullptr, *poolb = nullptr, *outw = nullptr, *outb = nullptr;\n};\n',
           '// Red sobre el espectro logarítmico (216 bins): misma estructura con entrada de 1 canal.\n',
           'using SpecNet = Net;\n']
    nets, specs = [], []
    for n, path in enumerate(a.models):
        ck = torch.load(path)
        state, cfg = (ck['state'], ck['cfg']) if 'state' in ck else (ck, {})
        if cfg.get('layers', 3) != 3: raise SystemExit('el exportador solo admite redes de 3 capas')
        isSpec = cfg.get('spec', False)
        att = cfg.get('attn', 0)
        if att > 2: raise SystemExit('el exportador admite hasta 2 capas de atención')
        m = (tkc.KeyNetSpec(ch=cfg['ch'], hidden=cfg['hidden']) if isSpec else
             tkc.KeyNetAttn(ch=cfg.get('ch', 16), hidden=cfg.get('hidden', 64), in_ch=cfg.get('in_ch', 2), attn=att) if att else
             tkc.KeyNet(ch=cfg.get('ch', 16), hidden=cfg.get('hidden', 64), in_ch=cfg.get('in_ch', 2)))
        m.load_state_dict(state); m.eval()
        for i, (c, b) in enumerate(((m.c1, m.bn1), (m.c2, m.bn2), (m.c3, m.bn3)), 1):
            w, bias = fold(c, b)
            out += [arr(f'k{n}Conv{i}W', w), arr(f'k{n}Conv{i}B', bias)]
        out += [arr(f'k{n}Head1W', m.head[0].weight.detach().numpy()), arr(f'k{n}Head1B', m.head[0].bias.detach().numpy()),
                arr(f'k{n}Head2W', m.head[3].weight.detach().numpy()), arr(f'k{n}Head2B', m.head[3].bias.detach().numpy())]
        extra = ''
        if att:
            W = lambda l: l.weight.detach().numpy(); B = lambda l: l.bias.detach().numpy()
            out += [arr(f'k{n}InW', W(m.inp)), arr(f'k{n}InB', B(m.inp)), arr(f'k{n}PoolW', W(m.pool)), arr(f'k{n}PoolB', B(m.pool)),
                    arr(f'k{n}OutW', W(m.out)), arr(f'k{n}OutB', B(m.out))]
            for l in range(att):
                out += [arr(f'k{n}Qkv{l}W', W(m.qkv[l])), arr(f'k{n}Qkv{l}B', B(m.qkv[l])), arr(f'k{n}Proj{l}W', W(m.proj[l])),
                        arr(f'k{n}Proj{l}B', B(m.proj[l])), arr(f'k{n}Ff1{l}W', W(m.ff[l][0])), arr(f'k{n}Ff1{l}B', B(m.ff[l][0])),
                        arr(f'k{n}Ff2{l}W', W(m.ff[l][2])), arr(f'k{n}Ff2{l}B', B(m.ff[l][2]))]
            lst = lambda nm, wb: '{' + ', '.join(f'k{n}{nm}{l}{wb}' for l in range(att)) + '}'
            extra = (f', {att}, {m.d}, {m.heads}, k{n}InW, k{n}InB, {lst("Qkv", "W")}, {lst("Qkv", "B")}, {lst("Proj", "W")}, {lst("Proj", "B")}, '
                     f'{lst("Ff1", "W")}, {lst("Ff1", "B")}, {lst("Ff2", "W")}, {lst("Ff2", "B")}, k{n}PoolW, k{n}PoolB, k{n}OutW, k{n}OutB')
        (specs if isSpec else nets).append(f'    {{{m.in_ch}, {m.ch}, {m.head[0].out_features}, {cfg.get("crop", 128)}, k{n}Conv1W, k{n}Conv1B, k{n}Conv2W, k{n}Conv2B, '
                    f'k{n}Conv3W, k{n}Conv3B, k{n}Head1W, k{n}Head1B, k{n}Head2W, k{n}Head2B{extra}}}')
        print(f'red {n}{" (espectro)" if isSpec else ""}: {path}  entrada={m.in_ch} canales={m.ch} oculta={m.head[0].out_features} ventana={cfg.get("crop", 128)}')
    if not nets or not specs: a.spec_weight = 1.0 if specs else 0.0
    out += [f'constexpr int kNetCount = {len(nets)};\nconstexpr Net kNets[kNetCount ? kNetCount : 1] = {{\n' + (',\n'.join(nets) or '    {}') + '\n};\n',
            f'constexpr int kSpecNetCount = {len(specs)};\nconstexpr SpecNet kSpecNets[kSpecNetCount ? kSpecNetCount : 1] = {{\n' + (',\n'.join(specs) or '    {}') + '\n};\n',
            f'constexpr float kSpecWeight = {float(a.spec_weight)!r}f;\n}}  // namespace skale::cnn\n']
    open(a.out, 'w').write(''.join(out)); print('escrito', a.out, len(nets), 'redes de cromagrama,', len(specs), 'de espectro, peso espectro', a.spec_weight)


if __name__ == '__main__':
    main()
