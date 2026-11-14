#!/usr/bin/env python3
"""Temporal follow-up: goal counts, squad similarity and F1 decision rules.

Keeps the primary frozen experiment and its models intact. No test label is used
for fitting, tuning, or selecting a family. The test was already observed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from draw_models import (GoalsModel, DrawDetector, DrawPredictor, boost_draw, roster_rule,
                         goal_band, goal_probabilities, replace_draw, draw_threshold)
from mlcore.data import ROOT, DATA, FOLDS, TEST, read_data, sha, verify_inputs
from mlcore.experiment import current, aligned, ensemble_member_names, json_write
from mlcore.metrics import metrics, block_codes

PROTOCOL = dict(version=1, goal_alpha=[.1, 1., 10.], detector_C=[.01, .1, 1.],
    boost=[1., 1.15, 1.3, 1.5, 1.75, 2., 2.5, 3.],
    roster=[-1., 0., 1., 2., 3., 4., 5., 7.5, 10.],
    bands=[0., .1, .2, .3, .4, .5, .75, 1., 1.5, 2.],
    thresholds=[.2, .225, .25, .275, .3, .325, .35, .4, .45, .5, 1.],
    outer=list(FOLDS), warmup=1718, inner='up to two previous OOF seasons',
    probability_selection='log loss', decision_selection='macro F1; accuracy tie break',
    family_selection='pooled outer macro F1; simpler family within 0.002',
    test_status='previously observed; exploratory follow-up after inspecting draw F1',
    bootstrap=5000, seed=2026)
FAMILIES = ['base', 'boost', 'roster', 'poisson', 'poisson_band_1',
            'poisson_band_tuned', 'detector_prob', 'detector_threshold']
LABEL = dict(base='Original: mayor probabilidad', boost='Mayor peso al empate (decisión)',
    roster='Plantillas similares → empate', poisson='Goles Poisson: mayor probabilidad',
    poisson_band_1='Goles: diferencia esperada ≤ 1', poisson_band_tuned='Goles: banda elegida con pasado',
    detector_prob='Detector de empate: probabilidades', detector_threshold='Detector de empate: umbral F1')


def confusion(y, pred): return np.bincount(3*np.asarray(y, int)+np.asarray(pred, int), minlength=9).reshape(3, 3)


def label_metrics(counts):
    counts = np.asarray(counts, float)
    tp = np.diagonal(counts, axis1=-2, axis2=-1)
    support, predicted = counts.sum(-1), counts.sum(-2)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros_like(tp), where=(support+predicted) > 0)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    return dict(accuracy=tp.sum(-1)/counts.sum(axis=(-1, -2)), F1_macro=f1.mean(-1),
        F1_weighted=(f1*support).sum(-1)/support.sum(-1), F1_H=f1[..., 0],
        F1_D=f1[..., 1], F1_A=f1[..., 2], precision_D=div(tp[..., 1], predicted[..., 1]),
        recall_D=div(tp[..., 1], support[..., 1]), draw_rate=predicted[..., 1]/support.sum(-1))


def scored(frame, p, pred):
    result = metrics(frame.y, p)
    result.update({k: float(v) for k, v in label_metrics(confusion(frame.y, pred)).items()})
    return result


def best_rule(y, grid, predict):
    def score(value):
        m = label_metrics(confusion(y, predict(value)))
        return (float(m['F1_macro']), float(m['accuracy']))
    return max(grid, key=score)


def select_past(t, season):
    seasons = sorted(s for s in t.Season.unique() if s < season)[-2:]
    if not seasons: raise ValueError('Faltan predicciones anteriores para ajustar')
    return t[t.Season.isin(seasons)].copy()


def base_predictions(parent, frame):
    base = pd.read_csv(parent/'oof_base.csv'); val = pd.read_csv(parent/'validation_predictions.csv')
    output = {}
    for branch in ('con', 'sin'):
        pieces = []
        for s in sorted(frame.Season.unique()):
            va = frame[frame.Season.eq(s)]
            if s == 1718:
                names = ['con/LR_REF'] if branch == 'con' else ensemble_member_names('sin')
                p = np.mean([aligned(base, name, va) for name in names], axis=0)
            else: p = aligned(val, 'modelo_'+branch, va)
            pieces.append(pd.DataFrame(p, index=va.match_id, columns=['P_H', 'P_D', 'P_A']))
        output[branch] = pd.concat(pieces).reindex(frame.match_id).to_numpy()
    return output


def raw_oof(df, sets, out):
    path = out/'raw_oof.csv'
    if path.exists(): return pd.read_csv(path, float_precision='round_trip')
    rows = []
    for season in (1718, *FOLDS):
        tr, va = df[df.Season.lt(season)], df[df.Season.eq(season)]
        assert tr.Date.max() < va.Date.min()
        for branch, key in [('con', 'con_cuotas'), ('sin', 'sin_cuotas')]:
            part = va[['match_id', 'Season', 'y']].copy(); part['branch'] = branch
            for i, alpha in enumerate(PROTOCOL['goal_alpha']):
                model = GoalsModel(sets[key], alpha).fit(tr); rates = model.predict_rates(va)
                part[f'H{i}'], part[f'A{i}'] = rates.T
            for i, C in enumerate(PROTOCOL['detector_C']):
                model = DrawDetector(branch, C).fit(tr); part[f'D{i}'] = model.predict_draw(va)
            rows.append(part)
            print(f'Predicciones temporales {branch} {season}: {len(tr)} → {len(va)}', flush=True)
    result = pd.concat(rows, ignore_index=True); result.to_csv(path, index=False)
    return result


def configuration(past, pbase):
    y = past.y.to_numpy()
    goal_scores = [log_loss(y, goal_probabilities(past[[f'H{i}', f'A{i}']].to_numpy()), labels=[0, 1, 2])
                   for i in range(len(PROTOCOL['goal_alpha']))]
    detector_scores = [log_loss((y == 1).astype(int), past[f'D{i}'], labels=[0, 1])
                       for i in range(len(PROTOCOL['detector_C']))]
    gi, di = int(np.argmin(goal_scores)), int(np.argmin(detector_scores))
    rates = past[[f'H{gi}', f'A{gi}']].to_numpy()
    pdet = replace_draw(pbase, past[f'D{di}'].to_numpy())
    return dict(goal_index=gi, detector_index=di,
        boost=best_rule(y, PROTOCOL['boost'], lambda v: boost_draw(pbase, v)),
        roster=best_rule(y, PROTOCOL['roster'], lambda v: roster_rule(past, pbase, v)),
        band=best_rule(y, PROTOCOL['bands'], lambda v: goal_band(rates, v)),
        threshold=best_rule(y, PROTOCOL['thresholds'], lambda v: draw_threshold(pdet, v)),
        seasons=[int(s) for s in sorted(past.Season.unique())],
        goal_scores=goal_scores, detector_scores=detector_scores)


def candidates(frame, pbase, rates, q, config):
    pg = goal_probabilities(rates); pd = replace_draw(pbase, q)
    return dict(base=(pbase, pbase.argmax(1)), boost=(pbase, boost_draw(pbase, config['boost'])),
        roster=(pbase, roster_rule(frame, pbase, config['roster'])), poisson=(pg, pg.argmax(1)),
        poisson_band_1=(pg, goal_band(rates, 1.)),
        poisson_band_tuned=(pg, goal_band(rates, config['band'])),
        detector_prob=(pd, pd.argmax(1)), detector_threshold=(pd, draw_threshold(pd, config['threshold'])))


def export_predictions(frame, branch, family, p, pred, split):
    t = frame[['match_id', 'Season', 'Date', 'y']].copy()
    t['branch'], t['family'], t['split'] = branch, family, split
    t[['P_H', 'P_D', 'P_A']] = p; t['pred'] = pred
    return t


def validation(df, raw, base, frame, out):
    rows, params = [], []
    for branch in ('con', 'sin'):
        t = raw[raw.branch.eq(branch)].set_index('match_id').reindex(frame.match_id).reset_index()
        t = t.merge(frame[['match_id', 'SQ_DIF']], on='match_id', validate='one_to_one')
        for season in FOLDS:
            past = select_past(t, season); ids = frame.match_id
            index = pd.Index(ids).get_indexer(past.match_id); config = configuration(past, base[branch][index])
            config.update(branch=branch, season=int(season)); params.append(config)
            v = frame[frame.Season.eq(season)]; vi = pd.Index(ids).get_indexer(v.match_id)
            r = t.set_index('match_id').reindex(v.match_id)
            rates = r[[f'H{config["goal_index"]}', f'A{config["goal_index"]}']].to_numpy()
            for family, (p, pred) in candidates(v, base[branch][vi], rates, r[f'D{config["detector_index"]}'].to_numpy(), config).items():
                rows.append(export_predictions(v, branch, family, p, pred, 'validacion'))
    json_write(out/'parameters_by_fold.json', params)
    result = pd.concat(rows, ignore_index=True); result.to_csv(out/'validation_predictions.csv', index=False)
    return result


def summary(predictions):
    return pd.DataFrame([dict(split=split, branch=branch, family=family,
        **scored(t, t[['P_H', 'P_D', 'P_A']].to_numpy(), t.pred.to_numpy()))
        for (split, branch, family), t in predictions.groupby(['split', 'branch', 'family'])])


def freeze(parent, out, table, raw, base, frame):
    if (out/'frozen.json').exists(): return json.loads((out/'frozen.json').read_text())
    selected, configs = {}, {}
    for branch in ('con', 'sin'):
        t = table[table.branch.eq(branch)]
        near = t[t.F1_macro.ge(t.F1_macro.max()-.002)]
        selected[branch] = min(near.family, key=FAMILIES.index)
        r = raw[raw.branch.eq(branch)].merge(frame[['match_id', 'SQ_DIF']], on='match_id', validate='one_to_one')
        past = select_past(r, 2425); ix = pd.Index(frame.match_id).get_indexer(past.match_id)
        configs[branch] = configuration(past, base[branch][ix])
    frozen = dict(selected=selected, configs=configs, protocol_sha=sha(out/'protocol.json'),
        validation_predictions_sha=sha(out/'validation_predictions.csv'), test_status=PROTOCOL['test_status'])
    json_write(out/'frozen.json', frozen); return frozen


def test_evaluation(parent, out, df, sets, frozen):
    if (out/'test_predictions.csv').exists():
        marker = json.loads((out/'complete.json').read_text())
        if marker['frozen_sha'] != sha(out/'frozen.json') or marker['predictions_sha'] != sha(out/'test_predictions.csv'):
            raise ValueError('La cache del test no coincide con la selección congelada')
        return pd.read_csv(out/'test_predictions.csv', parse_dates=['Date'], float_precision='round_trip')
    tr, te = df[~df.Season.isin(TEST)], df[df.Season.isin(TEST)]
    old = pd.read_csv(parent/'test_predictions.csv'); rows = []
    for branch, key in [('con', 'con_cuotas'), ('sin', 'sin_cuotas')]:
        config = frozen['configs'][branch]; pbase = aligned(old, 'modelo_'+branch, te)
        goals = GoalsModel(sets[key], PROTOCOL['goal_alpha'][config['goal_index']]).fit(tr)
        detector = DrawDetector(branch, PROTOCOL['detector_C'][config['detector_index']]).fit(tr)
        for family, (p, pred) in candidates(te, pbase, goals.predict_rates(te), detector.predict_draw(te), config).items():
            rows.append(export_predictions(te, branch, family, p, pred, 'test'))
        family = frozen['selected'][branch]
        parameter = dict(boost=config['boost'], roster=config['roster'], poisson_band_1=1.,
                         poisson_band_tuned=config['band'], detector_threshold=config['threshold']).get(family)
        bundle = DrawPredictor(joblib.load(parent/f'modelo_{branch}_cuotas.joblib'), goals, detector, family, parameter)
        joblib.dump(bundle, out/f'predictor_empates_{branch}.joblib')
        selected = rows[-len(FAMILIES)+FAMILIES.index(family)]
        restored = joblib.load(out/f'predictor_empates_{branch}.joblib')
        np.testing.assert_allclose(restored.predict_proba(te), selected[['P_H', 'P_D', 'P_A']].to_numpy(), atol=1e-12)
        np.testing.assert_array_equal(restored.predict(te), selected.pred.to_numpy())
    result = pd.concat(rows, ignore_index=True); result.to_csv(out/'test_predictions.csv', index=False)
    return result


def bootstrap(predictions):
    rows = []
    for (split, branch), g in predictions.groupby(['split', 'branch']):
        base = g[g.family.eq('base')].set_index('match_id'); frame = base.reset_index()
        cod = block_codes(frame); K = int(cod.max()+1)
        def blocks(pred):
            return np.bincount(cod*9+3*frame.y.to_numpy()+np.asarray(pred), minlength=K*9).reshape(K, 3, 3)
        bb = blocks(frame.pred)
        for family in FAMILIES[1:]:
            t = g[g.family.eq(family)].set_index('match_id').reindex(base.index)
            bc = blocks(t.pred); rng = np.random.default_rng(PROTOCOL['seed']); draws = []
            for start in range(0, PROTOCOL['bootstrap'], 250):
                ix = rng.integers(0, K, (min(250, PROTOCOL['bootstrap']-start), K))
                a, b = label_metrics(bc[ix].sum(1)), label_metrics(bb[ix].sum(1))
                draws.append(np.c_[a['F1_macro']-b['F1_macro'], a['accuracy']-b['accuracy']])
            intervals = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
            a, b = label_metrics(bc.sum(0)), label_metrics(bb.sum(0))
            rows.append(dict(split=split, branch=branch, family=family,
                delta_F1=float(a['F1_macro']-b['F1_macro']), F1_lo=intervals[0, 0], F1_hi=intervals[1, 0],
                delta_accuracy=float(a['accuracy']-b['accuracy']), acc_lo=intervals[0, 1], acc_hi=intervals[1, 1]))
    return pd.DataFrame(rows)


def plot_results(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    short = ['Original', 'Peso al empate', 'Plantillas similares', 'Poisson', 'Banda de 1 gol',
             'Banda ajustada', 'Detector probabilístico', 'Detector con umbral']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for row, branch in enumerate(('con', 'sin')):
        t = table[(table.branch.eq(branch)) & table.split.eq('test')].set_index('family').reindex(FAMILIES)
        color = '#0d9488' if branch == 'con' else '#2563eb'
        for col, (metric, title) in enumerate([('F1_macro', 'F1 macro'), ('accuracy', 'Accuracy 1X2')]):
            ax = axes[row, col]; ax.barh(short, t[metric], color=color)
            ax.invert_yaxis(); ax.set_xlim(0, .6); ax.set_title(f'{"Con" if branch == "con" else "Sin"} cuotas · {title}')
            ax.axvline(t.loc['base', metric], linestyle='--', color='#475569', linewidth=1)
            for i, value in enumerate(t[metric]): ax.text(value+.005, i, f'{value:.3f}', va='center', fontsize=8)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Test: mejorar F1 puede reducir el acierto total')
    fig.tight_layout(); fig.savefig(ROOT/'docs/figures/empates_revision.png', dpi=180); plt.close(fig)


def report(out, table, ci, frozen, predictions, df):
    lines = ['# Experimento adicional: detectar empates', '',
        'Se prueban las hipótesis de diferencia de goles prevista y similitud de plantillas en las dos ramas. '
        'Es una extensión exploratoria. Los modelos principales y sus probabilidades se conservan.', '',
        '**Protocolo.** Seis temporadas exteriores (2018/19–2023/24), con modelos ajustados solo con temporadas anteriores. '
        'Los parámetros de Poisson se eligen por log loss 1X2 y los del detector por log loss binario en hasta dos temporadas OOF anteriores. '
        'Las reglas de decisión se eligen por F1 macro usando esas temporadas anteriores. 2017/18 inicializa el primer fold; '
        'sus modelos usan valores iniciales fijados en el protocolo. Imputación y escalado se ajustan dentro de cada entrenamiento. '
        'La familia final se elige por F1 macro de validación, con tolerancia 0,002 a favor de la más sencilla; '
        'este uso de la validación para seleccionar familia conserva sesgo de desarrollo. El test no interviene '
        'en los ajustes, umbrales ni selección.', '',
        '**Qué significa cada alternativa.** Poisson predice por separado los goles esperados de local y visitante y calcula '
        'P(H), P(D), P(A) mediante la diferencia de dos Poisson independientes. Su hipótesis de independencia es una limitación. '
        'Una diferencia **esperada** de goles ≤1 no equivale a empate: se prueba como regla, sin usar jamás los goles reales como entrada. '
        'La regla de plantilla utiliza la diferencia absoluta de rating agregado y conserva la decisión original si falta el rating. '
        'El detector aprende empate/no empate con diferencias absolutas y firmadas de fuerza, similitud de habilidades, '
        'goles e historial previos; la rama con cuotas añade probabilidades del mercado. Las valoraciones siguen siendo snapshots del juego.', '',
        '**Probabilidades frente a decisiones.** Aumentar el peso del empate o aplicar un umbral puede mejorar F1 sin cambiar '
        'log loss: en esas variantes se conservan las probabilidades y se cambia solamente la clase elegida. '
        'El detector reemplaza P(D) y distribuye 1−P(D) entre H/A conservando la razón del predictor original. '
        'Se reportan F1 macro, F1 del empate, precisión y recall del empate, accuracy, frecuencia de predicción y log loss por separado.', '']
    va = df[df.Season.isin(FOLDS)]
    close, far = va[va.SQ_DIF.abs().le(2)], va[va.SQ_DIF.abs().gt(2)]
    lines += ['**Señal de similitud de plantillas.** En validación, la frecuencia real de empate es '
        f'{100*close.y.eq(1).mean():.1f}% entre {len(close):,} partidos con diferencia de rating ≤2 y '
        f'{100*far.y.eq(1).mean():.1f}% entre {len(far):,} con diferencia >2; se excluyen los ratings faltantes de este diagnóstico. '
        'Es una asociación descriptiva, no causal. La similitud sí contiene señal, pero el empate sigue siendo minoría incluso '
        'entre equipos de calidad similar. No basta para decidir empate en todos esos partidos.', '',
        '**Reglas congeladas para el test.** '+
        f'La rama con cuotas multiplica por {frozen["configs"]["con"]["boost"]:g} el valor de P(D) únicamente al elegir la etiqueta. '+
        f'La rama sin cuotas usa el detector y elige empate cuando P(D)≥{frozen["configs"]["sin"]["threshold"]:.3f}; '+
        'si no, elige H/A según sus probabilidades relativas. Estos valores se eligieron con 2022/23–2023/24. '
        'Las reglas deportivas de comparación usan diferencia de rating ≤2 y bandas de goles esperados '
        f'de {frozen["configs"]["con"]["band"]:g} (con cuotas) y {frozen["configs"]["sin"]["band"]:g} (sin cuotas).', '']
    for branch in ('con', 'sin'):
        lines += [f'## Rama {"con" if branch == "con" else "sin"} cuotas', '',
                  f'Familia elegida en validación: **{LABEL[frozen["selected"][branch]]}**.', '']
        for split in ('validacion', 'test'):
            lines += [f'### {split}', '', '| Método | F1 macro ↑ | F1 empate ↑ | Precisión empate | Recall empate | Accuracy ↑ | Predice empate | Log loss ↓ |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
            t = table[(table.branch.eq(branch)) & table.split.eq(split)].set_index('family')
            for family in FAMILIES:
                r = t.loc[family]
                lines.append(f'| {LABEL[family]} | {r.F1_macro:.4f} | {r.F1_D:.4f} | {r.precision_D:.4f} | {r.recall_D:.4f} | {100*r.accuracy:.2f}% | {100*r.draw_rate:.2f}% | {r.log_loss:.6f} |')
            lines += ['']
        chosen = frozen['selected'][branch]
        if chosen != 'base':
            for split in ('validacion', 'test'):
                r = ci[(ci.branch.eq(branch)) & ci.family.eq(chosen) & ci.split.eq(split)].iloc[0]
                lines += [f'{split}: frente al original, Δ F1 macro **{r.delta_F1:+.4f}**, IC 95% [{r.F1_lo:+.4f}, {r.F1_hi:+.4f}]; '
                    f'Δ accuracy **{100*r.delta_accuracy:+.2f} puntos porcentuales**, IC [{100*r.acc_lo:+.2f}, {100*r.acc_hi:+.2f}].', '']
    lines += ['## Alcance de la comparación', '',
        'Los intervalos usan 5.000 bootstrap por bloques de temporada/año ISO/semana. Son intervalos exploratorios '
        'condicionados al procedimiento; no corrigen la selección de la familia ni toda la multiplicidad de la exploración. '
        'No se mezclan las predicciones de decisión con el cálculo de log loss ni se sustituyen probabilidades por etiquetas 0/1.', '',
        f'Ejecución: `{out.name}`. Resultados, configuraciones por fold, selección congelada, predicciones y modelos guardados en '
        f'`{out.relative_to(ROOT)}`. El predictor exportado ofrece `predict_proba(frame)` y `predict(frame)`; las dos salidas '
        'pueden diferir de argmax cuando el objetivo de decisión es F1.', '',
        'Reproducción: `.venv/bin/python scripts/experiment_draws.py`.', '',
        '![Comparación de F1 y accuracy en ambas ramas](figures/empates_revision.png)', '']
    (ROOT/'docs/EXPERIMENTO_EMPATES.md').write_text('\n'.join(lines))


def main():
    verify_inputs(); parent = current(); df, sets = read_data(parent)
    identity = dict(protocol=PROTOCOL, parent_run=parent.name,
        sources={p: sha(Path(__file__).parent/p) for p in ('draw_models.py', 'experiment_draws.py')},
        inputs={p: sha(parent/p) for p in ('protocol.json', 'dataset.csv', 'oof_base.csv',
            'validation_predictions.csv', 'test_predictions.csv', 'modelo_con_cuotas.joblib', 'modelo_sin_cuotas.joblib')})
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    out = DATA/'draw_runs'/digest; out.mkdir(parents=True, exist_ok=True)
    json_write(out/'protocol.json', identity)
    raw = raw_oof(df, sets, out); frame = df[df.Season.isin([1718, *FOLDS])].copy()
    base = base_predictions(parent, frame)
    val = validation(df, raw, base, frame, out); table = summary(val)
    frozen = freeze(parent, out, table, raw, base, frame)
    assert frozen['validation_predictions_sha'] == sha(out/'validation_predictions.csv')
    test = test_evaluation(parent, out, df, sets, frozen)
    all_predictions = pd.concat([val, test], ignore_index=True); all_predictions.Date = pd.to_datetime(all_predictions.Date)
    table = summary(all_predictions); table.to_csv(out/'metrics.csv', index=False)
    ci = bootstrap(all_predictions); ci.to_csv(out/'paired_intervals.csv', index=False)
    all_predictions.groupby(['split', 'branch', 'family', 'Season']).apply(
        lambda t: pd.Series(scored(t, t[['P_H', 'P_D', 'P_A']].to_numpy(), t.pred)),
        include_groups=False).to_csv(out/'by_season.csv')
    table.to_csv(ROOT/'docs/resultados_empates.csv', index=False)
    plot_results(table)
    report(out, table, ci, frozen, all_predictions, df)
    json_write(out/'complete.json', dict(frozen_sha=sha(out/'frozen.json'), metrics_sha=sha(out/'metrics.csv'),
                                       predictions_sha=sha(out/'test_predictions.csv')))
    (DATA/'draw_actual.txt').write_text(str(out.relative_to(ROOT)))
    print(table[['split', 'branch', 'family', 'F1_macro', 'F1_D', 'accuracy', 'log_loss']].to_string(index=False), flush=True)
    print('Selección:', frozen['selected'], '\nResultados:', out, flush=True)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The y_prob values')
        main()
