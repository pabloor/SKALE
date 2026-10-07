#!/usr/bin/env python3
"""Red convolucional de tonalidad, equivariante a la transposición, sobre la serie de cromagramas
finos de `skale-cli --frames` (36 bins/octava, cromagrama + bajo).

Entrada (B, 2, T, 36). Convoluciones 2D con relleno circular en el eje de tono, agrupación media y
máxima en el tiempo y una cabeza que puntúa cada tónica (36 posiciones x 2 modos) a partir de las
características giradas hasta esa posición. Los 36 logits por modo se reducen a 12 con logsumexp
sobre los 3 sub-bins de cada semitono. Resultado: 24 logits (tónica*2 + modo).

Uso: tools/train_key_cnn.py MANIFEST.json DIR_FRAMES [--epochs 25] [--out modelo.pt]
Necesita torch (CPU basta) y numpy.
"""
import argparse, collections, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F


class KeyNet(nn.Module):
    def __init__(self, ch=16, hidden=64, drop=0.2, in_ch=2, layers=3):
        super().__init__()
        self.ch = ch; self.in_ch = in_ch; self.nl = layers
        self.c1 = nn.Conv2d(in_ch, ch, (3, 9)); self.c2 = nn.Conv2d(ch, ch, (3, 9)); self.c3 = nn.Conv2d(ch, ch, (3, 9))
        self.bn1 = nn.BatchNorm2d(ch); self.bn2 = nn.BatchNorm2d(ch); self.bn3 = nn.BatchNorm2d(ch)
        if layers > 3:   # capas extra (solo para experimentos; el exportador a C++ admite 3)
            self.extra = nn.ModuleList([nn.Conv2d(ch, ch, (3, 9)) for _ in range(layers - 3)])
            self.ebn = nn.ModuleList([nn.BatchNorm2d(ch) for _ in range(layers - 3)])
        self.head = nn.Sequential(nn.Linear(2 * ch * 36, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, 2))
        idx = (torch.arange(36)[:, None] + torch.arange(36)[None, :]) % 36        # idx[p, o] = (p + o) % 36
        self.register_buffer('rot', idx)

    @staticmethod
    def _conv(x, conv, bn):
        x = F.pad(x, (4, 4, 0, 0), mode='circular')       # tono circular
        x = F.pad(x, (0, 0, 1, 1), mode='replicate')      # tiempo
        return F.relu(bn(conv(x)))

    def forward(self, x):                                  # x: (B, 2, T, 36)
        x = self._conv(x, self.c1, self.bn1); x = self._conv(x, self.c2, self.bn2); x = self._conv(x, self.c3, self.bn3)
        if self.nl > 3:
            for c, b in zip(self.extra, self.ebn): x = self._conv(x, c, b)
        f = torch.cat([x.mean(2), x.amax(2)], dim=1)       # (B, 2ch, 36)
        r = f[:, :, self.rot]                              # (B, 2ch, 36, 36): girado a cada posición
        r = r.permute(0, 2, 1, 3).reshape(x.shape[0], 36, -1)
        s = self.head(r)                                   # (B, 36, 2)
        s = torch.stack([torch.logsumexp(torch.stack([s[:, (3 * t - 1) % 36], s[:, 3 * t], s[:, (3 * t + 1) % 36]], 0), 0)
                         for t in range(12)], dim=1)       # (B, 12, 2)
        return s.reshape(x.shape[0], 24)


def load(manifest, d, channels=2):
    M = json.load(open(manifest)); data = []
    for r in M:
        a = np.fromfile(os.path.join(d, r['file']), dtype=np.float32).reshape(-1, channels, 36)
        data.append(dict(group=r['group'], name=r['name'], y=r['label'][0] * 2 + (0 if r['label'][1] == 'major' else 1),
                         x=np.sqrt(np.maximum(a, 0)) * 3.0))         # compresión; (T, 2, 36)
    return data


def crop(x, T, rng):
    n = len(x)
    if n < T: x = np.concatenate([x] * (T // n + 1))[:T]; n = T
    s = rng.integers(0, n - T + 1); return x[s:s + T]


def predict(model, x, T=128):
    model.eval()
    with torch.no_grad():
        n = len(x)
        if n < T: x = np.concatenate([x] * (T // n + 1))[:T]; n = T
        starts = list(range(0, n - T + 1, T // 2)) or [0]
        b = torch.from_numpy(np.stack([x[s:s + T] for s in starts])).permute(0, 2, 1, 3)   # (K,2,T,36)
        return F.log_softmax(model(b), 1).mean(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest'); ap.add_argument('frames')
    ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--out', default='key_cnn.pt')
    ap.add_argument('--steps', type=int, default=60, help='lotes por época'); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--channels', type=int, default=2, help='canales por fotograma en los .bin (2 = cromagrama+bajo; 4 = +medios+agudos)')
    ap.add_argument('--ch', type=int, default=16); ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--crop', type=int, default=128); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--drop', type=float, default=0.2)
    ap.add_argument('--all', action='store_true', help='entrena con todos los datos (modelo final, sin conjunto de prueba)')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--holdout', default='', help='grupos completos fuera del entrenamiento (coma)')
    a = ap.parse_args()
    torch.set_num_threads(4); rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    data = load(a.manifest, a.frames, a.channels)
    hold = set(filter(None, a.holdout.split(',')))
    idx = np.arange(len(data)); te = np.zeros(len(data), bool); srng = np.random.default_rng(12345)   # partición fija e independiente de la semilla
    for g in sorted({d['group'] for d in data}):
        gi = [i for i in idx if data[i]['group'] == g]
        if g in hold: te[gi] = True
        else: te[srng.permutation(gi)[:max(1, len(gi) // 5)]] = True       # 20 % de cada grupo para prueba
    if a.all: te[:] = False
    tr = [i for i in idx if not te[i]]; ts = [i for i in idx if te[i]]
    gsize = collections.Counter(data[i]['group'] for i in tr)
    p = np.array([gsize[data[i]['group']] ** -0.5 for i in tr]); p /= p.sum()      # muestreo ~ 1/sqrt(tamaño del grupo)
    print(f'entrenamiento {len(tr)}  prueba {len(ts)}', flush=True)
    cfg = dict(ch=a.ch, hidden=a.hidden, drop=a.drop, in_ch=a.channels, layers=a.layers, crop=a.crop)
    model = KeyNet(ch=a.ch, hidden=a.hidden, drop=a.drop, in_ch=a.channels, layers=a.layers); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * a.steps)
    def evaluate():
        per = collections.defaultdict(list)
        for i in ts:
            per[data[i]['group']].append(int(np.argmax(predict(model, data[i]['x'], a.crop))) == data[i]['y'])
        return {g: float(np.mean(v)) for g, v in per.items()}, {g: len(v) for g, v in per.items()}
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); loss_sum = 0
        for _ in range(a.steps):
            bi = rng.choice(len(tr), a.batch, p=p)
            xs = np.stack([crop(data[tr[j]]['x'], a.crop, rng) for j in bi]); ys = np.array([data[tr[j]]['y'] for j in bi])
            xs = torch.from_numpy(xs).permute(0, 2, 1, 3)
            # aumento: desafinación aleatoria de ±1 bin (±33 cents) con la misma etiqueta
            xs = torch.roll(xs, int(rng.integers(-1, 2)), dims=3)
            loss = F.cross_entropy(model(xs), torch.from_numpy(ys).long(), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); loss_sum += loss.item()
        msg = f'época {ep + 1}/{a.epochs}  pérdida {loss_sum / a.steps:.3f}  {time.time() - t0:.0f}s'
        if ts and ((ep + 1) % 5 == 0 or ep == a.epochs - 1):
            acc, n = evaluate(); msg += '  | ' + ' '.join(f'{g[:5]}={100 * v:.0f}({n[g]})' for g, v in acc.items())
        print(msg, flush=True)
    torch.save(dict(state=model.state_dict(), cfg=cfg), a.out)
    json.dump(dict(test=[int(i) for i in ts]), open(a.out + '.split.json', 'w'))


if __name__ == '__main__':
    main()
