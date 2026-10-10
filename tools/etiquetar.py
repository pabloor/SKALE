#!/usr/bin/env python3
"""Página local para etiquetar a mano la tonalidad de pistas de FMA (las que prepara tools/label_pool.py).

Uso: tools/etiquetar.py DATOS [--puerto 8765]      y abrir http://127.0.0.1:8765
Las etiquetas se guardan en DATOS/fma_user/labels.jsonl (todas las respuestas) y DATOS/fma_user/expected.csv
(solo las tonalidades, con el formato de los demás conjuntos), así que tools/prepare_training.py las usa como
grupo «fma_user». Se puede cerrar y seguir otro día. Solo escucha en 127.0.0.1.

1 de cada 20 pistas es de FMAK (etiquetada por expertos, fuera de la partición de prueba) sin avisar: sirven
para medir cuánto coinciden tus etiquetas con las de los expertos. No se añaden a expected.csv.
"""
import argparse, csv, json, os, random, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
CONTROL_EVERY = 20


def pk(name):
    n, m = name.split()
    return (PC[n[0]] + n[1:].count('#') - n[1:].count('b')) % 12, m


def same_notes(a, b):
    (ta, ma), (tb, mb) = pk(a), pk(b)
    if ma == mb: return ta == tb
    return tb == (ta + (9 if ma == 'major' else 3)) % 12


class Store:
    def __init__(self, datos):
        self.datos = os.path.realpath(datos); self.dir = os.path.join(self.datos, 'fma_user')
        pool = json.load(open(os.path.join(self.dir, 'pool.json')))
        self.items, self.control = pool['items'], pool['control']
        self.log = os.path.join(self.dir, 'labels.jsonl'); self.lock = threading.Lock()
        self.labels = [json.loads(l) for l in open(self.log)] if os.path.exists(self.log) else []

    def done(self): return {l['path'] for l in self.labels}

    def next(self):
        with self.lock:
            done = self.done(); n = sum(1 for l in self.labels if l['choice'] != 'skip')
            want_control = n > 0 and n % CONTROL_EVERY == CONTROL_EVERY - 1
            pools = ([self.control, self.items] if want_control else [self.items])
            for p in pools:
                for it in p:
                    if it['path'] not in done:
                        return self.public(it)
            return None

    @staticmethod
    def public(it):
        top = [c[0] for c in it['cands'][:3]]
        random.Random(it['id']).shuffle(top)          # orden aleatorio (fijo por pista) para no influir
        return dict(path=it['path'], title=it['title'], artist=it['artist'], genre=it['genre'], tuning=it['tuning'],
                    cands=top)

    def add(self, path, choice):
        with self.lock:
            it = next((x for x in self.items + self.control if x['path'] == path), None)
            if it is None: raise ValueError('pista desconocida')
            if choice not in ('none', 'skip'): pk(choice)    # valida el nombre
            rec = dict(path=path, id=it['id'], choice=choice, time=time.strftime('%Y-%m-%d %H:%M:%S'),
                       model=it['cands'][0][0])
            if 'gold' in it: rec['gold'] = it['gold']
            self.labels.append(rec)
            with open(self.log, 'a') as fh: fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self.write_csv()

    def undo(self):
        with self.lock:
            if not self.labels: return
            self.labels.pop()
            with open(self.log, 'w') as fh:
                for r in self.labels: fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            self.write_csv()

    def write_csv(self):
        rows = [[os.path.relpath(os.path.join(self.datos, l['path']), self.dir), l['choice']]
                for l in self.labels if 'gold' not in l and l['choice'] not in ('none', 'skip')]
        tmp = os.path.join(self.dir, 'expected.csv.tmp')
        with open(tmp, 'w', newline='') as fh: csv.writer(fh).writerows(rows)
        os.replace(tmp, os.path.join(self.dir, 'expected.csv'))

    def stats(self):
        lab = [l for l in self.labels if 'gold' not in l]
        keys = [l for l in lab if l['choice'] not in ('none', 'skip')]
        gold = [l for l in self.labels if 'gold' in l and l['choice'] not in ('none', 'skip')]
        return dict(labeled=len(keys), none=sum(l['choice'] == 'none' for l in lab), skipped=sum(l['choice'] == 'skip' for l in lab),
                    total=len(self.items), agree_model=sum(pk(l['choice']) == pk(l['model']) for l in keys),
                    gold=len(gold), gold_exact=sum(pk(l['choice']) == pk(l['gold']) for l in gold),
                    gold_notes=sum(same_notes(l['choice'], l['gold']) for l in gold),
                    today=sum(l['time'][:10] == time.strftime('%Y-%m-%d') for l in self.labels))


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def send(self, code, body, ctype='application/json; charset=utf-8', extra=None):
            if isinstance(body, (dict, list)) or body is None: body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str): body = body.encode()
            self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for k, v in (extra or {}).items(): self.send_header(k, v)
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/': return self.send(200, PAGE, 'text/html; charset=utf-8')
            if u.path == '/api/next': return self.send(200, dict(item=store.next(), stats=store.stats()))
            if u.path == '/audio':
                rel = parse_qs(u.query).get('path', [''])[0]
                f = os.path.realpath(os.path.join(store.datos, rel))
                if not f.startswith(store.datos + os.sep) or not f.endswith('.mp3') or not os.path.isfile(f):
                    return self.send(404, {'error': 'no encontrado'})
                data = open(f, 'rb').read(); n = len(data)
                rng = self.headers.get('Range')
                if rng and rng.startswith('bytes='):          # Safari necesita peticiones parciales para el audio
                    a, _, b = rng[6:].partition('-'); a = int(a or 0); b = int(b) if b else n - 1
                    return self.send(206, data[a:b + 1], 'audio/mpeg', {'Content-Range': f'bytes {a}-{b}/{n}', 'Accept-Ranges': 'bytes'})
                return self.send(200, data, 'audio/mpeg', {'Accept-Ranges': 'bytes'})
            self.send(404, {'error': 'no encontrado'})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                if self.path == '/api/label': store.add(body['path'], body['choice'])
                elif self.path == '/api/undo': store.undo()
                else: return self.send(404, {'error': 'no encontrado'})
                self.send(200, dict(item=store.next(), stats=store.stats()))
            except (ValueError, KeyError) as e:
                self.send(400, {'error': str(e)})
    return H


PAGE = r'''<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Skale: etiquetar</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d2330;--mut:#667085;--line:#e3e6eb;--acc:#1f9d74;--acc2:#e8f6f1;--warn:#b26b00}
@media (prefers-color-scheme:dark){:root{--bg:#15181d;--card:#1d2128;--fg:#e8eaee;--mut:#9aa3b2;--line:#2c323c;--acc:#5bd6a6;--acc2:#1f3a31;--warn:#e6b450}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:18px;margin:0 0 4px}.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px;margin-top:14px}
.meta{font-size:14px}.meta b{font-size:16px}audio{width:100%;margin:12px 0 4px}
.cands{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px}
.cand{border:1px solid var(--line);border-radius:10px;padding:12px;text-align:center}
.cand .k{font-size:22px;font-weight:700;color:var(--acc)}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:8px;padding:7px 12px;cursor:pointer}
button:hover{border-color:var(--acc)}button.pri{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
@media (prefers-color-scheme:dark){button.pri{color:#10241c}}
.cand button{width:100%;margin-top:8px}.row{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--mut)}
button.pri kbd{color:inherit;border-color:currentColor;opacity:.75}
.grid{display:none;grid-template-columns:repeat(6,1fr);gap:6px;margin-top:12px}.grid.on{display:grid}
.grid button{padding:6px 2px}.stats{display:flex;flex-wrap:wrap;gap:18px;font-size:14px}.stats b{font-size:18px;display:block}
.help{font-size:13px;color:var(--mut)}.msg{color:var(--warn);font-size:14px;min-height:20px;margin-top:8px}
@media (max-width:560px){.cands{grid-template-columns:1fr}.grid{grid-template-columns:repeat(4,1fr)}}
</style></head><body><main>
<h1>Skale · etiquetar tonalidades</h1>
<div class="mut">Escucha el fragmento y elige su tonalidad. «Cadencia» toca I–IV–V–I en esa tonalidad sobre la música para comprobarla de oído.</div>
<div class="card stats" id="stats"></div>
<div class="card" id="card">
  <div class="meta" id="meta">Cargando…</div>
  <audio id="au" controls loop preload="auto"></audio>
  <div class="cands" id="cands"></div>
  <div class="row">
    <button id="other">Otra tonalidad <kbd>O</kbd></button>
    <button id="none">Sin tonalidad clara / cambia <kbd>N</kbd></button>
    <button id="skip">Saltar <kbd>S</kbd></button>
    <button id="undo">Deshacer la última <kbd>Z</kbd></button>
  </div>
  <div class="grid" id="grid"></div>
  <div class="msg" id="msg"></div>
</div>
<div class="help card">Teclas: <kbd>1</kbd><kbd>2</kbd><kbd>3</kbd> elegir · <kbd>Q</kbd><kbd>W</kbd><kbd>E</kbd> cadencia de la 1ª/2ª/3ª ·
<kbd>espacio</kbd> reproducir/pausa · <kbd>R</kbd> desde el principio. Mayor y su relativa menor tienen las mismas notas: decide por el centro
(la nota o el acorde donde la música «descansa»). Si de verdad no lo sabes, salta: una etiqueta dudosa enseña cosas falsas.</div>
</main><script>
const PCN={C:0,D:2,E:4,F:5,G:7,A:9,B:11};
const MAJ=['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B'], MIN=['C','C#','D','Eb','E','F','F#','G','G#','A','Bb','B'];
let item=null, actx=null;
const $=id=>document.getElementById(id), au=$('au');
function pc(n){return (PCN[n[0]]+(n.slice(1).split('#').length-1)-(n.slice(1).split('b').length-1)+12)%12}
function esName(k){const [n,m]=k.split(' ');return n+(m==='major'?' mayor':' menor')}
function cadence(key){
  try{actx=actx||new (window.AudioContext||window.webkitAudioContext)()}catch(e){return}
  const [n,m]=key.split(' '), t=pc(n), minor=m==='minor', base=48+t, cents=(item&&item.tuning)||0;
  const third=minor?3:4, chords=[[0,third,7],[5,minor?8:9,12],[7,11,14],[0,third,7]];
  const now=actx.currentTime+0.05;
  chords.forEach((ch,i)=>{[-12].concat(ch).forEach(iv=>{
    const o=actx.createOscillator(), g=actx.createGain(), f=440*Math.pow(2,(base+iv-69)/12+cents/1200);
    o.type='triangle'; o.frequency.value=f; const t0=now+i*0.75;
    g.gain.setValueAtTime(0,t0); g.gain.linearRampToValueAtTime(iv<0?0.10:0.07,t0+0.03); g.gain.exponentialRampToValueAtTime(0.001,t0+0.72);
    o.connect(g).connect(actx.destination); o.start(t0); o.stop(t0+0.75);});});
}
function render(d){
  item=d.item; const s=d.stats;
  $('stats').innerHTML=`<div><b>${s.labeled}</b>etiquetadas</div><div><b>${s.today}</b>hoy</div><div><b>${s.none+s.skipped}</b>sin tonalidad / saltadas</div>`+
    `<div><b>${s.labeled?Math.round(100*s.agree_model/s.labeled):0} %</b>coinciden con el modelo</div>`+
    (s.gold?`<div><b>${Math.round(100*s.gold_exact/s.gold)} %</b>coinciden con expertos (${s.gold} de control; ${Math.round(100*s.gold_notes/s.gold)} % mismas notas)</div>`:'');
  $('grid').classList.remove('on'); $('msg').textContent='';
  if(!item){$('meta').textContent='No quedan pistas por etiquetar.'; $('cands').innerHTML=''; au.removeAttribute('src'); return}
  $('meta').innerHTML=`<b>${item.title||'(sin título)'}</b> — ${item.artist||''} <span class="mut">· ${item.genre}</span>`;
  au.src='/audio?path='+encodeURIComponent(item.path); au.play().catch(()=>{});
  $('cands').innerHTML=item.cands.map((k,i)=>`<div class="cand"><div class="k">${esName(k)}</div>
    <button onclick="cadence('${k}')">Cadencia <kbd>${'QWE'[i]}</kbd></button>
    <button class="pri" onclick="label('${k}')">Es esta <kbd>${i+1}</kbd></button></div>`).join('');
}
async function call(url,body){
  const r=await fetch(url,{method:body?'POST':'GET',headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  const d=await r.json(); if(!r.ok){$('msg').textContent='Error: '+(d.error||r.status); return} render(d);
}
function label(choice){ if(item) call('/api/label',{path:item.path,choice}); }
$('none').onclick=()=>label('none'); $('skip').onclick=()=>label('skip'); $('undo').onclick=()=>call('/api/undo',{});
$('other').onclick=()=>$('grid').classList.toggle('on');
$('grid').innerHTML=MAJ.map(n=>`<button onclick="label('${n} major')">${n}</button>`).join('')+
  MIN.map(n=>`<button onclick="label('${n} minor')">${n}m</button>`).join('');
document.addEventListener('keydown',e=>{
  if(e.metaKey||e.ctrlKey||e.altKey||!item) return; const k=e.key.toLowerCase();
  if('123'.includes(k)&&item.cands[+k-1]) label(item.cands[+k-1]);
  else if('qwe'.includes(k)&&k) {const i='qwe'.indexOf(k); if(item.cands[i]) cadence(item.cands[i]);}
  else if(k==='n') label('none'); else if(k==='s') label('skip'); else if(k==='z') call('/api/undo',{});
  else if(k==='o') $('grid').classList.toggle('on'); else if(k==='r'){au.currentTime=0; au.play();}
  else if(k===' '){e.preventDefault(); au.paused?au.play():au.pause();} else return;
});
call('/api/next');
</script></body></html>'''


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('datos'); ap.add_argument('--puerto', type=int, default=8765)
    a = ap.parse_args()
    store = Store(a.datos)
    print(f'{len(store.items)} pistas, {len(store.labels)} respuestas guardadas. Abre http://127.0.0.1:{a.puerto}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', a.puerto), make_handler(store)).serve_forever()


if __name__ == '__main__':
    main()
