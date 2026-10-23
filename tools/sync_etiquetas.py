#!/usr/bin/env python3
"""Trae las respuestas de la página web de etiquetado (https://pabloor.github.io/etiquetar, repositorio
pabloor/etiquetar) desde la hoja de Google y las guarda en el repositorio:

  tools/samples/fma_user_labels.jsonl   todas las respuestas válidas (una por línea, sin las deshechas)
  tools/samples/fma_user_expected.csv   tonalidad acordada por pista de FMA (formato de los demás conjuntos)

Uso: tools/sync_etiquetas.py URL_EXEC CLAVE      (la URL de la aplicación web de Apps Script y su CLAVE_LECTURA)

Por pista, cuenta la última respuesta de cada navegador. Entra en el CSV si hay al menos una tonalidad y ninguna
respuesta contradictoria (otra tonalidad con notas distintas); «sin tonalidad» o saltar no cuentan. Las pistas de
FMAK (control, etiquetadas por expertos) no entran en el CSV: sirven para medir la coincidencia con los expertos.
Después, `tools/fetch_datasets.py DATOS --solo fma_user` baja el audio y prepare_training.py lo usa como grupo fma_user.
"""
import collections, csv, json, os, sys, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def pk(name):
    n, m = name.split()
    return (PC[n[0]] + n[1:].count('#') - n[1:].count('b')) % 12, m


def same_notes(a, b):
    (ta, ma), (tb, mb) = pk(a), pk(b)
    return ta == tb if ma == mb else tb == (ta + (9 if ma == 'major' else 3)) % 12


def main():
    if len(sys.argv) != 3: raise SystemExit(__doc__)
    url = sys.argv[1] + ('&' if '?' in sys.argv[1] else '?') + urllib.parse.urlencode({'clave': sys.argv[2]})
    data = json.load(urllib.request.urlopen(url, timeout=120))
    if isinstance(data, dict): raise SystemExit('La hoja respondió: ' + data.get('error', str(data)))
    head, rows = data[0], [dict(zip(data[0], r)) for r in data[1:]]
    undone = {r['id_respuesta'] for r in rows if r['accion'] == 'undo'}
    resp = [dict(uid=r['id_respuesta'], id=int(r['pista']), choice=r['eleccion'], model=r['modelo'], who=r['persona'],
                 client=r['navegador'], time=r['hora']) for r in rows if r['accion'] == 'label' and r['id_respuesta'] not in undone]
    resp.sort(key=lambda r: r['time'])
    fmak = {int(fn[:6]): k for fn, k in csv.reader(open(os.path.join(HERE, 'samples', 'fmak_expected.csv')))}

    last = {}                                   # (pista, navegador) -> última respuesta
    for r in resp: last[(r['id'], r['client'])] = r
    by = collections.defaultdict(list)
    for (tid, _), r in last.items(): by[tid].append(r)

    out, conflicts, gold, gold_ok, gold_notes, model_ok = [], 0, 0, 0, 0, 0
    for tid, rs in sorted(by.items()):
        keys = [r['choice'] for r in rs if r['choice'] not in ('none', 'skip')]
        if not keys: continue
        if tid in fmak:                         # control
            for k in keys: gold += 1; gold_ok += pk(k) == pk(fmak[tid]); gold_notes += same_notes(k, fmak[tid])
            continue
        if any(not same_notes(k, keys[0]) for k in keys): conflicts += 1; continue
        c = collections.Counter(keys).most_common()
        if len(c) > 1 and c[0][1] == c[1][1]: conflicts += 1; continue    # empate entre una escala y su relativa
        out.append((f'{tid:06d}.mp3', c[0][0])); model_ok += pk(c[0][0]) == pk(rs[0]['model'])

    with open(os.path.join(HERE, 'samples', 'fma_user_labels.jsonl'), 'w', encoding='utf-8') as fh:
        for r in resp: fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(os.path.join(HERE, 'samples', 'fma_user_expected.csv'), 'w', newline='') as fh: csv.writer(fh).writerows(out)
    people = collections.Counter(r['who'] or '(sin nombre)' for r in last.values())
    print(f'{len(resp)} respuestas de {len({r["client"] for r in resp})} navegadores ({", ".join(f"{p}: {n}" for p, n in people.most_common())})')
    print(f'{len(out)} pistas con tonalidad en fma_user_expected.csv; {conflicts} con respuestas contradictorias (fuera)')
    if out: print(f'coinciden con la 1ª opción del modelo: {100 * model_ok / len(out):.0f} %')
    if gold: print(f'control (FMAK, expertos): {gold} respuestas, tonalidad exacta {100 * gold_ok / gold:.0f} %, mismas notas {100 * gold_notes / gold:.0f} %')


if __name__ == '__main__':
    main()
