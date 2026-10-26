#!/usr/bin/env python3
"""Huella (SHA-256) de los datos crudos, para saber si dos personas trabajan con los mismos archivos.

Los datos estan en .gitignore y las fuentes (football-data.co.uk, Kaggle) se actualizan, asi que una
descarga nueva puede diferir de la usada en el informe. Este script guarda las huellas en
data_manifest.json (que si se versiona) y permite comprobarlas.

Uso:
    python scripts/download/manifest.py          # genera/actualiza data_manifest.json
    python scripts/download/manifest.py --check  # compara los datos locales con el manifiesto
"""
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = [BASE / 'data' / 'raw', BASE / 'data' / 'players' / 'raw']
OUT = Path(__file__).resolve().parent / 'data_manifest.json'


def huellas():
    out = {}
    for carpeta in RAW:
        for f in sorted(carpeta.glob('*.csv')):
            if f.name == 'players_all.csv':   # derivado (lo genera process_data.py), no es un crudo
                continue
            out[str(f.relative_to(BASE / 'data'))] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


if __name__ == '__main__':
    actual = huellas()
    if '--check' in sys.argv:
        esperado = json.load(open(OUT))['archivos']
        faltan = sorted(set(esperado) - set(actual)); nuevos = sorted(set(actual) - set(esperado))
        difieren = sorted(k for k in set(esperado) & set(actual) if esperado[k] != actual[k])
        print(f'faltan: {len(faltan)} | nuevos: {len(nuevos)} | con contenido distinto: {len(difieren)}')
        for k in faltan + nuevos + difieren: print('  ', k)
        sys.exit(1 if (faltan or nuevos or difieren) else 0)
    json.dump({'nota': 'SHA-256 de los CSV crudos usados en el informe', 'archivos': actual}, open(OUT, 'w'), indent=1)
    print(f'{len(actual)} archivos -> {OUT}')
