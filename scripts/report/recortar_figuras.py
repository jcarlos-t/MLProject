"""Recorta los paneles clave de las figuras generadas por los cuadernos (docs/figures/*.png) para que el texto sea legible
a ancho de pagina en el informe (IEEE, 2 columnas). Entrada: figuras completas; salida: docs/figures/p_*.png.

Cada figura completa sale del cuaderno indicado; las cajas son (x0, x1) en pixeles del PNG original (todo el alto)."""
from pathlib import Path
from PIL import Image

FIG = Path(__file__).resolve().parents[2] / 'docs' / 'figures'
# nombre de salida: (figura de origen, [(x0, x1), ...]) ; varias cajas se pegan una al lado de la otra
CORTES = {
    'p_modelos_ll':      ('res_modelos',        [(0, 620)]),                 # 04_modelo: log loss por modelo
    'p_modelos_delta':   ('res_modelos',        [(620, 1260)]),              # 04_modelo: diferencia frente a la LR y al mercado
    'p_bloques_ll':      ('res_bloques',        [(0, 600)]),                 # 04_modelo: bloques de variables
    'p_bloques_delta':   ('res_bloques',        [(600, 1180)]),              # 04_modelo: mejora al anadir cada bloque (LR)
    'p_rejillas_svm':    ('hp_rejillas',        [(0, 470)]),                 # 05_hiperparametros: SVM
    'p_rejillas_xgb':    ('hp_rejillas',        [(470, 935)]),               # 05_hiperparametros: XGBoost
    'p_sesgo_liga':      ('mercado_sesgo',      [(640, 1235)]),              # 06_mercado: por liga
    'p_sesgo_tramo':     ('mercado_sesgo',      [(1240, 1990)]),             # 06_mercado: por tramo de probabilidad
    'p_fiab_local':      ('mercado_fiabilidad', [(0, 505)]),                 # 06_mercado: fiabilidad del local
    'p_fiab_pend':       ('mercado_fiabilidad', [(1515, 2089)]),             # 06_mercado: pendientes de calibracion
    'p_revision_info':   ('prep_revision',      [(840, 1450)]),              # 02_feature: informacion frente a estabilidad
    'p_revision_veredicto': ('prep_revision',   [(1463, 1990)]),             # 02_feature: veredicto por bloque
    'p_diag_roc':        ('final_diag',         [(0, 530, 0, 445)]),         # 08_modelo_final: curva ROC del local
    'p_diag_fiab':       ('final_diag',         [(0, 530, 445, 890)]),       # 08_modelo_final: fiabilidad del local
    'p_backtest_b365':   ('final_backtest',     [(0, 790)]),                 # 08_modelo_final: ROI con Bet365
    'p_backtest_pin':    ('final_backtest',     [(795, 1590)]),              # 08_modelo_final: ROI con Pinnacle
    'p_confusion':       ('eval_confusion',     [(519, 890), (962, 1528)]),  # 07_evaluacion: matriz de la LR + precision/recall
    'p_robustez':        ('eval_bloques',       [(0, 800)]),                 # 07_evaluacion: IC por bloques frente a partidos sueltos
    'p_liga':            ('final_liga',         [(0, 885)]),                 # 08_modelo_final: por liga
    'p_mipca':           ('res_mipca',          [(1035, 1790)]),             # 04_modelo: MI vs PCA con la k elegida
    'p_empate':          ('prep_empate',        [(0, 565)]),                 # 02_feature: empates segun goles esperados
}
for destino, (origen, cajas) in CORTES.items():
    im = Image.open(FIG / f'{origen}.png').convert('RGB')
    trozos = [im.crop((c[0], c[2] if len(c) == 4 else 0, c[1], c[3] if len(c) == 4 else im.height)) for c in cajas]
    w = sum(t.width for t in trozos); h = max(t.height for t in trozos); out = Image.new('RGB', (w, h), 'white'); x = 0
    for t in trozos: out.paste(t, (x, 0)); x += t.width
    out.save(FIG / f'{destino}.png'); print(destino, out.size)
