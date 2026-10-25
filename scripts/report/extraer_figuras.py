"""Extrae de los cuadernos ejecutados las figuras completas que se guardan en docs/figures (y las del test holdout).
Cada entrada: (cuaderno, celda, k) = la k-esima imagen PNG (0-based) de la salida de esa celda. Despues se corre recortar_figuras.py."""
import base64, json
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
NB, FIG = RAIZ / 'scripts' / 'notebooks', RAIZ / 'docs' / 'figures'
FIGURAS = {
    'prep_fuerza':        ('03_preprocesamiento', 22, 0),
    'res_bloques':        ('04_modelo', 8, 0),
    'res_modelos':        ('04_modelo', 26, 0),
    'res_mipca':          ('04_modelo', 32, 0),
    'hp_rejillas':        ('05_hiperparametros', 8, 0),
    'mercado_fiabilidad': ('06_mercado', 14, 0),
    'mercado_sesgo':      ('06_mercado', 17, 0),
    'eval_confusion':     ('07_evaluacion', 11, 0),
    'eval_bloques':       ('07_evaluacion', 20, 0),
    'final_diag':         ('08_modelo_final', 20, 0),
    'final_liga':         ('08_modelo_final', 20, 1),
    'final_backtest':     ('08_modelo_final', 23, 0),
    'final_uso':          ('08_modelo_final', 26, 0),
    'final_test':         ('08_modelo_final', 34, 0),
    'final_test_backtest': ('08_modelo_final', 36, 0),
}
if __name__ == '__main__':
    cache = {}
    for destino, (cuaderno, celda, k) in FIGURAS.items():
        nb = cache.setdefault(cuaderno, json.load(open(NB / f'{cuaderno}.ipynb')))
        imgs = [o['data']['image/png'] for o in nb['cells'][celda].get('outputs', []) if 'data' in o and 'image/png' in o['data']]
        (FIG / f'{destino}.png').write_bytes(base64.b64decode(imgs[k])); print(destino, '<-', cuaderno, celda, k)
