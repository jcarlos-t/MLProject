"""Descriptive diagnostics; never changes selection, odds or betting thresholds."""
import numpy as np
import pandas as pd
from mlcore.data import read_data, FOLDS, TEST, ROOT
from mlcore.experiment import current, aligned
from mlcore.metrics import metrics


def audit(out=None):
    out = out or current(); d, _ = read_data(out)
    tables = []
    for split, seasons in [('validacion', FOLDS), ('test', TEST)]:
        f = d[d.Season.isin(seasons)].copy()
        odds = f[['B365H','B365D','B365A']].to_numpy(float)
        p = f[['MKT_pH','MKT_pD','MKT_pA']].to_numpy(float)
        valid = np.isfinite(odds).all(1) & (odds > 1).all(1)
        ev = np.where(valid[:,None], p*odds-1, -np.inf)
        side = ev.argmax(1); selected = valid & (ev.max(1) > .02)
        f['selection'] = np.array(['H','D','A'])[side]
        f['odds'] = odds[np.arange(len(f)),side]
        f['profit'] = np.where(f.y.to_numpy()==side, f.odds-1, -1.)
        f['EV'] = ev.max(1); f['split'] = split
        # Best price published by any bookmaker in the source (two column families across seasons).
        best = np.column_stack([pd.to_numeric(f.get('Max'+c), errors='coerce').fillna(pd.to_numeric(f.get('BbMx'+c), errors='coerce')) for c in 'HDA'])
        f['market_max'] = best[np.arange(len(f)), side]
        tables.append(f[selected])
    bets = pd.concat(tables)
    cols = ['split','match_id','Date','Season','Div','HomeTeam','AwayTeam','FTR','selection','odds','EV','profit'] + [pre+c for pre in ('B365','BW','PS','WH') for c in 'HDA']
    bets[cols].to_csv(out/'audit_consensus_bets.csv',index=False)
    annual = bets.groupby(['split','Season']).profit.agg(['size','sum','mean'])
    annual.to_csv(out/'audit_consensus_by_season.csv')
    lines = ['# Diagnósticos de resultados', '', f'Ejecución: `{out.name}`.', '',
        '## Consenso contra Bet365', '',
        'Se recalcula la regla original (EV > 2 %, una unidad). Los siguientes análisis son descriptivos; no cambian precios, umbral ni selección.']
    for split,g in bets.groupby('split'):
        top = g.nlargest(10,'profit'); rest = g.drop(top.index)
        lines += ['', f'- {split}: {len(g)} apuestas, beneficio {g.profit.sum():.2f}, ROI {100*g.profit.mean():.2f} %. Las diez mayores ganancias suman {top.profit.sum():.2f} unidades.',
            f'  Cuotas de esas ganancias: {top.odds.min():.2f}–{top.odds.max():.2f}. Sin esas diez apuestas el ROI es {100*rest.profit.mean():.2f} %.',
            f'  Cuota mediana apostada: {g.odds.median():.2f}; aciertos: {100*(g.profit > 0).mean():.1f} % de las apuestas.',
            f'  Bet365 ofrece el mejor precio publicado en el {100*(g.odds >= .999*g.market_max).mean():.0f} % de estas apuestas y lo supera en {int((g.odds > 1.001*g.market_max).sum())}.']
    wins = bets[bets.split.eq('validacion')].nlargest(5, 'profit')
    lines += ['', 'Las cinco mayores ganancias de validación corresponden a partidos reales con un resultado inesperado:', '']
    lines += [f'- {pd.Timestamp(r.Date).date()} · {r.HomeTeam} – {r.AwayTeam} · resultado {r.FTR} · cuota {r.odds:.2f} (mejor precio publicado {r.market_max:.2f})' for r in wins.itertuples()]
    lines += ['', 'La regla apuesta casi siempre a resultados poco probables en los que Bet365 paga más que las demás casas. La ganancia de validación depende de unas pocas sorpresas y desaparece casi por completo al quitarlas; en test la misma regla pierde. '
        'Las cuotas apostadas no superan, salvo en casos contados, el mejor precio publicado, así que no hay indicios de cuotas erróneas: es variabilidad de apuestas de cuota alta. Los crudos no permiten acreditar que esos precios estuvieran disponibles a la vez, por lo que tampoco es una rentabilidad replicable.', '',
        '## Recalibración de la variante balanceada', '',
        'Comparación en las mismas cinco temporadas: 2019/20–2023/24. No comparar directamente con métricas de seis temporadas.']
    recal = pd.read_csv(out/'oof_recalibration.csv'); weighted = pd.read_csv(out/'oof_market_guided.csv'); rows=[]
    for name in recal.candidate.unique():
        f=d[d.Season.isin(FOLDS[1:])]
        for variant,source,candidate in [('balance',weighted,name.removesuffix('_recal')),('balance_recal',recal,name)]:
            rows.append(dict(candidate=name,variant=variant,**metrics(f.y,aligned(source,candidate,f))))
    table=pd.DataFrame(rows);table.to_csv(out/'recalibration_results.csv',index=False)
    age = pd.read_csv(out/'coverage.csv').set_index('Season').loc[2425, 'antiguedad_H']
    lines += ['', '```text', table[['candidate','variant','log_loss','ECE_masa']].to_string(index=False), '```', '',
        '## Disponibilidad y calibración', '',
        f'Las valoraciones usadas en 2024/25 tienen {age:.0f} días de antigüedad media. No hay una versión con fecha acreditada entre septiembre de 2023 y julio de 2025: la fuente de FC24 solo incluye la versión de lanzamiento, y los archivos de FC25 de esa época no traen fecha de publicación ni la alineación titular con la que se calcula el once. Por eso se conserva la versión anterior verificable y se declara la antigüedad.', '',
        'El calibrador de diferencia de goles aprende con predicciones de una temporada posterior al entrenamiento base. Luego se reajusta el regresor con todo el pasado. No hay etiquetas futuras en ese ajuste, pero puede cambiar la distribución de la entrada al calibrador; debe evaluarse en datos futuros.']
    (ROOT/'docs/DIAGNOSTICOS.md').write_text('\n'.join(lines)+'\n')
    return table

if __name__ == '__main__':
    audit()
