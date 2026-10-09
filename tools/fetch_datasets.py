#!/usr/bin/env python3
"""Descarga los conjuntos etiquetados con los que se entrenan y evalúan las redes de tonalidad, con los
mismos nombres de archivo que las listas de tools/samples/*.csv. Se puede interrumpir y relanzar: lo que
ya está descargado no se vuelve a bajar.

Uso: tools/fetch_datasets.py DATOS [--solo fma,beatport,...]     Necesita: pip install remotezip requests

Conjuntos (carpeta dentro de DATOS, tamaño aproximado de la descarga):
  fma        FMAKv2: 5.489 clips de 30 s de Free Music Archive, etiquetas de expertos (~4 GB)
  beatport   Beatport EDM Key Dataset, Zenodo 1101082 (~1,3 GB)
  gsplus     GiantSteps Key+, Zenodo 4153506 (~0,8 GB)
  guitarset  GuitarSet, micrófono mono, Zenodo 3371780 (~0,7 GB)
  bach       Clave bien temperada I, Kimiko Ishizaka (CC0), archive.org (~120 MB)
  classic    sonatas de Beethoven y Chopin (Veesey, CC0), archive.org
  set2, set6 piezas CC con la tonalidad en el título del autor, archive.org (grupo «author»)
  set4, set5 Jamendo CC con etiqueta por consenso de varios detectores, archive.org (grupo «jamendo_cons»)
  synth      108 progresiones sintéticas generadas aquí (tools/gen_synth_training.py)
El grupo «folk_synth» (29 melodías de music21 sintetizadas) no se descarga: necesita music21 y fluidsynth
(tools/render_corpus.py); son pocas pistas y el entrenamiento funciona sin ellas.

Algunas pistas de archive.org pueden haber desaparecido; se avisa y se siguen las demás.
"""
import argparse, csv, json, os, subprocess, sys, time, urllib.parse, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, 'samples')
ZENODO = 'https://zenodo.org/api/records/{rec}/files/{name}/content'
WTC = 'bach-well-tempered-clavier-book-1'


def get(url, path, tries=4):
    if os.path.exists(path) and os.path.getsize(path) > 0: return True
    tmp = path + '.part'
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'skale-fetch'}), timeout=120) as r, open(tmp, 'wb') as fh:
                while chunk := r.read(1 << 20): fh.write(chunk)
            os.replace(tmp, path); return True
        except Exception as e:
            print(f'  reintento {k + 1} ({e}): {url}', flush=True); time.sleep(2 ** (k + 1))
    return False


def zenodo_zip(rec, name, folder, sub='aud'):
    """Descarga un zip de Zenodo, lo extrae en folder/sub y borra el zip."""
    if os.path.isdir(os.path.join(folder, sub)) and os.listdir(os.path.join(folder, sub)): return
    os.makedirs(folder, exist_ok=True); z = os.path.join(folder, name)
    print(f'descargando {name} (Zenodo {rec})...', flush=True)
    if not get(ZENODO.format(rec=rec, name=name), z): raise SystemExit(f'no se pudo descargar {name}')
    with zipfile.ZipFile(z) as zf:
        zf.extractall(os.path.join(folder, sub), [m for m in zf.namelist() if not m.startswith('__MACOSX')])
    os.remove(z)


def ia_meta(ident):
    with urllib.request.urlopen(f'https://archive.org/metadata/{urllib.parse.quote(ident)}', timeout=60) as r: return json.load(r)


def ia_file(ident, name, path):
    return get(f'https://archive.org/download/{ident}/{urllib.parse.quote(name)}', path)


def ia_first_mp3(ident, path, lo, hi):
    """Primer mp3 del elemento con tamaño entre lo y hi bytes (la regla con la que se eligió cada pista)."""
    if os.path.exists(path): return True
    try:
        mp3 = [f for f in ia_meta(ident)['files'] if f['name'].lower().endswith('.mp3') and lo < int(f.get('size', 0) or 0) < hi]
    except Exception:
        mp3 = []
    return bool(mp3) and ia_file(ident, mp3[0]['name'], path)


def copy_csv(name, dst):
    with open(os.path.join(SAMPLES, name)) as a, open(dst, 'w') as b: b.write(a.read())


def fetch_fma(d):
    subprocess.run([sys.executable, os.path.join(HERE, 'get_fmak.py'), d], check=True)


def fetch_beatport(d):
    zenodo_zip(1101082, 'audio.zip', d); copy_csv('beatport_edm_clean_expected.csv', f'{d}/expected.csv')


def fetch_gsplus(d):
    zenodo_zip(4153506, 'audio.zip', d); copy_csv('giantsteps_plus_conf2_expected.csv', f'{d}/expected.csv')


def fetch_guitarset(d):
    zenodo_zip(3371780, 'audio_mono-mic.zip', d); copy_csv('guitarset_expected.csv', f'{d}/expected.csv')


def fetch_bach(d):
    import re
    os.makedirs(d, exist_ok=True)
    for f in ia_meta(WTC)['files']:
        m = re.search(r'- (\d\d) (Prelude|Fugue) No', f['name'])
        if m and f['name'].endswith('.mp3'): ia_file(WTC, f['name'], f'{d}/wtc1_{m.group(1)}_{m.group(2).lower()}.mp3')
    copy_csv('wtc1_expected.csv', f'{d}/expected.csv'); copy_csv('wtc1_fugues_expected.csv', f'{d}/fugues.csv')


def from_tsv(d, tsv, csvname, pick):
    os.makedirs(d, exist_ok=True); missing = []
    for row in csv.reader(open(os.path.join(SAMPLES, tsv)), delimiter='\t'):
        if not pick(row, f'{d}/{row[0]}'): missing.append(row[0])
    copy_csv(csvname, f'{d}/expected.csv')
    if missing: print(f'  {os.path.basename(d)}: {len(missing)} pistas ya no están en archive.org: {", ".join(missing[:8])}', flush=True)


def fetch_classic(d):   # nombre, tonalidad, identificador, archivo
    from_tsv(d, 'classical_sources.tsv', 'classical_expected.csv', lambda r, p: ia_file(r[2], r[3], p))


def fetch_set2(d):      # nombre, tonalidad, identificador, archivo
    from_tsv(d, 'cc_other_sources.tsv', 'cc_other_expected.csv', lambda r, p: ia_file(r[2], r[3], p))


def fetch_set6(d):      # nombre, tonalidad, identificador, título, licencia
    from_tsv(d, 'author_labeled_sources.tsv', 'author_labeled_clean_expected.csv', lambda r, p: ia_first_mp3(r[2], p, 1_000_000, 14_000_000))


def fetch_set4(d):      # nombre, identificador, género, licencia
    from_tsv(d, 'jamendo_consensus_sources.tsv', 'jamendo_consensus_expected.csv', lambda r, p: ia_first_mp3(r[1], p, 1_500_000, 14_000_000))


def fetch_set5(d):
    from_tsv(d, 'jamendo2_consensus_sources.tsv', 'jamendo2_consensus_expected.csv', lambda r, p: ia_first_mp3(r[1], p, 1_500_000, 14_000_000))


def fetch_synth(d):
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f'{d}/expected.csv'): subprocess.run([sys.executable, os.path.join(HERE, 'gen_synth_training.py'), d], check=True)


SETS = dict(synth=fetch_synth, bach=fetch_bach, classic=fetch_classic, set2=fetch_set2, set6=fetch_set6, set4=fetch_set4,
            set5=fetch_set5, guitarset=fetch_guitarset, gsplus=fetch_gsplus, beatport=fetch_beatport, fma=fetch_fma)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('datos'); ap.add_argument('--solo', default='', help='lista de conjuntos separados por comas')
    a = ap.parse_args()
    want = [s for s in a.solo.split(',') if s] or list(SETS)
    for s in want:
        if s not in SETS: raise SystemExit(f'conjunto desconocido: {s} (hay {", ".join(SETS)})')
        print(f'== {s}', flush=True); SETS[s](os.path.join(a.datos, s))
    print('listo. Siguiente paso: tools/prepare_training.py', a.datos, 'SALIDA')


if __name__ == '__main__':
    main()
