#!/usr/bin/env python3
"""Red convolucional de tonalidad, equivariante a la transposición, sobre la serie de cromagramas
finos de `skale-cli --frames` (36 bins/octava, cromagrama + bajo).

Entrada (B, 2, T, 36). Convoluciones 2D con relleno circular en el eje de tono, agrupación media y
máxima en el tiempo y una cabeza que puntúa cada tónica (36 posiciones x 2 modos) a partir de las
características giradas hasta esa posición. Los 36 logits por modo se reducen a 12 con logsumexp
sobre los 3 sub-bins de cada semitono. Resultado: 24 logits (tónica*2 + modo).

Uso: tools/train_key_cnn.py MANIFEST.json DIR_FRAMES [--epochs 25] [--out modelo.pt]
Necesita torch y numpy. Usa la GPU si la hay (CUDA o Apple MPS) y, si no, todos los núcleos de la CPU.
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


class KeyNetSpec(nn.Module):
    """Variante sobre el espectro logarítmico (216 bins = 6 octavas x 36 desde La1, 55 Hz).
    Conv a tercio de semitono -> agrupado máximo a semitono (72) -> 2 conv -> plegado de octavas
    a 12 clases de nota -> cabeza equivariante sobre las 12 tónicas."""
    def __init__(self, ch=16, hidden=64, drop=0.2):
        super().__init__()
        self.ch = ch; self.in_ch = 1; self.nl = 3; self.spec = True
        self.c1 = nn.Conv2d(1, ch, (3, 9)); self.c2 = nn.Conv2d(ch, ch, (3, 5)); self.c3 = nn.Conv2d(ch, ch, (3, 5))
        self.bn1 = nn.BatchNorm2d(ch); self.bn2 = nn.BatchNorm2d(ch); self.bn3 = nn.BatchNorm2d(ch)
        self.head = nn.Sequential(nn.Linear(4 * ch * 12, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, 2))
        idx = (torch.arange(12)[:, None] + torch.arange(12)[None, :]) % 12
        self.register_buffer('rot', idx)

    @staticmethod
    def _conv(x, conv, bn, kp):
        x = F.pad(x, (kp, kp, 1, 1))                         # ceros en frecuencia y en tiempo
        return F.relu(bn(conv(x)))

    def forward(self, x):                                    # x: (B, 1, T, 216)
        x = self._conv(x, self.c1, self.bn1, 4)
        x = F.max_pool2d(x, (1, 3))                          # (B, ch, T, 72): un bin por semitono
        x = self._conv(x, self.c2, self.bn2, 2); x = self._conv(x, self.c3, self.bn3, 2)
        f = torch.cat([x.mean(2), x.amax(2)], dim=1)         # (B, 2ch, 72)
        f = f.reshape(f.shape[0], f.shape[1], 6, 12)         # 6 octavas x 12 semitonos (índice 0 = La)
        f = torch.cat([f.mean(2), f.amax(2)], dim=1)         # (B, 4ch, 12)
        f = torch.roll(f, 9, dims=2)                         # índice n = nota n (0 = Do)
        r = f[:, :, self.rot].permute(0, 2, 1, 3).reshape(f.shape[0], 12, -1)
        return self.head(r).reshape(f.shape[0], 24)          # (tónica, modo) -> tónica*2 + modo


def load(manifest, d, channels=2, use=None, spec_dir=None):
    M = json.load(open(manifest)); data = []
    for r in M:
        if spec_dir:
            path = os.path.join(spec_dir, r['file'])
            if not os.path.exists(path): continue
            a = np.fromfile(path, dtype=np.float32).reshape(-1, 1, 216)
            data.append(dict(group=r['group'], name=r['name'], y=r['label'][0] * 2 + (0 if r['label'][1] == 'major' else 1), x=a * 3.0))
            continue
        a = np.fromfile(os.path.join(d, r['file']), dtype=np.float32).reshape(-1, channels, 36)[:, :use or channels]
        data.append(dict(group=r['group'], name=r['name'], y=r['label'][0] * 2 + (0 if r['label'][1] == 'major' else 1),
                         x=np.sqrt(np.maximum(a, 0)) * 3.0))         # compresión; (T, 2, 36)
    return data


def crop(x, T, rng):
    n = len(x)
    if n < T: x = np.concatenate([x] * (T // n + 1))[:T]; n = T
    s = rng.integers(0, n - T + 1); return x[s:s + T]


def random_eq(xs, rng, spec):
    """Ecualización aleatoria (aumento). Espectro: curva suave de ±6 dB en magnitud sobre las 6 octavas, con la
    misma normalización por fotograma que skale-cli (máximo = 3 tras el escalado). Cromagramas: ganancia de
    cada banda (bajo, medios, agudos) entre 0,6 y 1,6 frente al cromagrama completo."""
    if spec:
        f = np.linspace(0, np.pi, xs.shape[3])
        db = sum(rng.uniform(-1, 1) * np.cos(k * f + rng.uniform(0, np.pi)) for k in (1, 2, 3)) * 6 / 3
        g = torch.from_numpy((10 ** (db / 20)) ** 0.5).float()          # el espectro guardado es sqrt(magnitud)
        xs = xs * g
        return xs / xs.amax(3, keepdim=True).clamp_min(1e-6) * 3.0
    g = torch.ones(1, xs.shape[1], 1, 1)
    for c in range(1, xs.shape[1]): g[0, c] = float(rng.uniform(0.6, 1.6))
    return xs * g


def pick_device(name='auto'):
    if name != 'auto': return torch.device(name)
    if torch.cuda.is_available(): return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')


def predict(model, x, T=128):
    model.eval(); dev = next(model.parameters()).device
    with torch.no_grad():
        n = len(x)
        if n < T: x = np.concatenate([x] * (T // n + 1))[:T]; n = T
        starts = list(range(0, n - T + 1, T // 2)) or [0]
        b = torch.from_numpy(np.stack([x[s:s + T] for s in starts])).permute(0, 2, 1, 3)   # (K,2,T,36)
        return F.log_softmax(model(b.to(dev)), 1).mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest'); ap.add_argument('frames')
    ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--out', default='key_cnn.pt')
    ap.add_argument('--steps', type=int, default=60, help='lotes por época'); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--channels', type=int, default=2, help='canales por fotograma en los .bin (2 = cromagrama+bajo; 4 = +medios+agudos)')
    ap.add_argument('--ch', type=int, default=16); ap.add_argument('--hidden', type=int, default=64); ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--crop', type=int, default=128); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--drop', type=float, default=0.2)
    ap.add_argument('--use-channels', type=int, default=0, help='usa solo los N primeros canales (p. ej. 2 de unos .bin de 4)')
    ap.add_argument('--train-only', default='', help='grupos que van siempre a entrenamiento (p. ej. pseudoetiquetas)')
    ap.add_argument('--group-scale', default='', help='factor de muestreo por grupo, p. ej. pseudo=0.5')
    ap.add_argument('--spec', default='', help='carpeta con espectros logarítmicos (skale-cli --spec): usa KeyNetSpec')
    ap.add_argument('--eval-only', default='gtzan', help='grupos que nunca se usan para entrenar, ni con --all (solo evaluación)')
    ap.add_argument('--init', default='', help='parte de los pesos de otro modelo (.pt de la misma arquitectura), p. ej. preentrenado')
    ap.add_argument('--eq', type=float, default=0.0, help='probabilidad de ecualización aleatoria por lote (aumento; 0 = no)')
    ap.add_argument('--all', action='store_true', help='entrena con todos los datos (modelo final, sin conjunto de prueba)')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--holdout', default='', help='grupos completos fuera del entrenamiento (coma)')
    ap.add_argument('--split', default='', help='JSON con la lista [grupo, nombre] de prueba (p. ej. tools/samples/test_split.json); '
                    'si falta, la partición fija aleatoria del 20 %% por grupo')
    ap.add_argument('--device', default='auto', help='auto, cpu, mps o cuda'); ap.add_argument('--threads', type=int, default=0, help='hilos de CPU (0 = todos)')
    a = ap.parse_args()
    torch.set_num_threads(a.threads or os.cpu_count() or 4); rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    dev = pick_device(a.device); print('dispositivo', dev, flush=True)
    in_ch = a.use_channels or a.channels
    data = load(a.manifest, a.frames, a.channels, in_ch, a.spec or None)
    data = [d for d in data if d['group'] not in set(filter(None, a.eval_only.split(',')))]
    hold = set(filter(None, a.holdout.split(',')))
    idx = np.arange(len(data)); te = np.zeros(len(data), bool); srng = np.random.default_rng(12345)   # partición fija e independiente de la semilla
    for g in sorted({d['group'] for d in data}):
        gi = [i for i in idx if data[i]['group'] == g]
        if g in hold: te[gi] = True
        elif g in set(filter(None, a.train_only.split(','))): pass
        else: te[srng.permutation(gi)[:max(1, len(gi) // 5)]] = True       # 20 % de cada grupo para prueba
    if a.split:
        names = {tuple(r) for r in json.load(open(a.split))}
        te = np.array([(d['group'], d['name']) in names for d in data]); te[[i for i in idx if data[i]['group'] in hold]] = True
    if a.all: te[:] = False
    tr = [i for i in idx if not te[i]]; ts = [i for i in idx if te[i]]
    gsize = collections.Counter(data[i]['group'] for i in tr)
    scale = {kv.split('=')[0]: float(kv.split('=')[1]) for kv in filter(None, a.group_scale.split(','))}
    p = np.array([gsize[data[i]['group']] ** -0.5 * scale.get(data[i]['group'], 1.0) for i in tr]); p /= p.sum()      # muestreo ~ 1/sqrt(tamaño del grupo)
    print(f'entrenamiento {len(tr)}  prueba {len(ts)}', flush=True)
    cfg = dict(ch=a.ch, hidden=a.hidden, drop=a.drop, in_ch=in_ch, layers=a.layers, crop=a.crop)
    model = KeyNetSpec(ch=a.ch, hidden=a.hidden, drop=a.drop) if a.spec else KeyNet(ch=a.ch, hidden=a.hidden, drop=a.drop, in_ch=in_ch, layers=a.layers)
    if a.spec: cfg['spec'] = True
    if a.init: model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['state'])
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
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
            if a.spec and rng.random() < 0.5:   # además, transposición de ±1-2 semitonos (en el espectro no es circular)
                k = int(rng.choice([-6, -3, 3, 6])); xs = torch.roll(xs, k, dims=3); ys = ((ys // 2 + k // 3) % 12) * 2 + ys % 2
            if a.eq and rng.random() < a.eq: xs = random_eq(xs, rng, a.spec)
            loss = F.cross_entropy(model(xs.to(dev)), torch.from_numpy(ys).long().to(dev), label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); loss_sum += loss.item()
        msg = f'época {ep + 1}/{a.epochs}  pérdida {loss_sum / a.steps:.3f}  {time.time() - t0:.0f}s'
        if ts and ((ep + 1) % 5 == 0 or ep == a.epochs - 1):
            acc, n = evaluate(); msg += '  | ' + ' '.join(f'{g[:5]}={100 * v:.0f}({n[g]})' for g, v in acc.items())
        print(msg, flush=True)
    torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, cfg=cfg), a.out)
    json.dump(dict(test=[int(i) for i in ts], names=[[data[i]['group'], data[i]['name']] for i in ts]), open(a.out + '.split.json', 'w'))


if __name__ == '__main__':
    main()
