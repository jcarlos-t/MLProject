"""Regenera las figuras completas desde la ejecucion congelada vigente."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from mlcore.report import context, figures
from mlcore.experiment import current
if __name__ == '__main__':
    figures(context(current()))
    print('Figuras revision_* generadas desde resultados verificados, sin indices de celdas.')
