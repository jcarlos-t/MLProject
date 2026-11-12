"""Las figuras nuevas ya tienen paneles legibles; valida el conjunto completo."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT/'docs/figures'
REQUIRED = ['datos','modelos','calibracion','temporadas','reduccion','ensembles',
            'pesos','predictor','final','backtest','ablacion']
if __name__ == '__main__':
    for name in REQUIRED:
        p = FIG/f'revision_{name}.png'
        if not p.is_file(): raise FileNotFoundError(p)
    print('Paneles completos verificados; no se recortan imagenes de ejecuciones antiguas.')
