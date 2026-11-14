"""Ordered stages: prepare -> nested CV -> feedback -> freeze -> final evaluation."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import DATA, ROOT, TEST, FOLDS, prepare, read_data, sha, verify_inputs
from .models import GRIDS, PRIORITY, Predictor, Spec, normalize, specifications
from .metrics import metrics, losses, compare, bh, backtest

PROTOCOL = dict(version='3.0', seed=42, outer_folds=list(FOLDS), warmup_meta=1718,
    test_seasons=list(TEST), test_status='previamente observado; reevaluacion posterior a auditoria',
    tuning='two previous seasons; preprocessing fitted inside each training split',
    tolerance=.0005, svm_cap=9000, tau=.02, bootstrap_reps=5000,
    selection='minimum pooled OOF log loss, then family complexity within 0.0005',
    pin_bad_from='2025-07-23', grids=GRIDS)


def json_write(path, value):
    def cast(x):
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=cast))


def initialize():
    manifest, _ = verify_inputs()
    identity = dict(protocol=PROTOCOL, raw=manifest,
        source={p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))
                if p.name not in ('report.py', 'notebooks.py')},
        snapshot_manifest=sha(ROOT/'scripts/download/snapshot_availability.json'),
        identity_map=sha(DATA/'players/club_map.csv'))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    out = DATA/'runs'/digest[:16]
    out.mkdir(parents=True, exist_ok=True)
    if not (out/'protocol.json').exists():
        json_write(out/'protocol.json', dict(run_id=digest[:16], identity=identity))
    (DATA/'revision_actual.txt').write_text(str(out.relative_to(ROOT)))
    return out


def current():
    return ROOT/(DATA/'revision_actual.txt').read_text().strip()


def task_key(spec, parameter, season):
    return hashlib.sha256(json.dumps([dataclasses.asdict(spec), parameter, int(season)], sort_keys=True).encode()).hexdigest()[:24]


def prediction_task(df, spec, parameter, season, out):
    path = out/'cv_cache'/(task_key(spec, parameter, season)+'.npz')
    if path.exists(): return
    tr, va = df[df.Season.lt(season)], df[df.Season.eq(season)]
    assert tr.Date.max() < va.Date.min()
    t = time.monotonic()
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.filterwarnings('ignore', message='The y_prob values')
        model = Predictor(spec, parameter).fit(tr)
        p = model.predict_proba(va)
    np.savez_compressed(path, p=p, ids=va.match_id.to_numpy(str), y=va.y.to_numpy(),
                        seconds=time.monotonic()-t)


def cached_prediction(df, spec, parameter, season, out):
    path = out/'cv_cache'/(task_key(spec, parameter, season)+'.npz')
    a = np.load(path, allow_pickle=False)
    va = df[df.Season.eq(season)]
    if not np.array_equal(a['ids'], va.match_id.to_numpy(str)) or not np.array_equal(a['y'], va.y):
        raise RuntimeError('Predicciones de otra cohorte')
    return normalize(a['p'])


def choose_parameter(df, spec, season, out):
    past = sorted(s for s in df.Season.unique() if 1617 <= s < season and s not in TEST)
    inner = past[-2:]
    if not inner: raise ValueError('No hay ventana de ajuste temporal')
    scores = []
    for parameter in GRIDS[spec.family]:
        ll = np.mean([log_loss(df.loc[df.Season.eq(s), 'y'], cached_prediction(df, spec, parameter, s, out), labels=[0, 1, 2]) for s in inner])
        scores.append(float(ll))
    # Grids go from the most to the least regularised configuration, so the
    # first one within the tolerance is the simplest of the near-ties.
    best = next(i for i, v in enumerate(scores) if v <= min(scores)+PROTOCOL['tolerance'])
    return GRIDS[spec.family][best], dict(inner=[int(s) for s in inner], scores=scores, chosen=best)


def nested_cv(out, jobs=4):
    df, sets = read_data(out)
    tv = df[~df.Season.isin(TEST)]
    specs = specifications(sets)
    (out/'cv_cache').mkdir(exist_ok=True)
    json_write(out/'specifications.json', [dataclasses.asdict(s) for s in specs])
    # Computing each temporal split once shares work between inner and outer folds.
    # The cache is scoped by source+protocol+raw hash; no model is loaded across runs.
    seasons = [1617, 1718]+list(FOLDS)
    for season in seasons:
        tasks = [(spec, par) for spec in specs for par in GRIDS[spec.family]
                 if not (out/'cv_cache'/(task_key(spec, par, season)+'.npz')).exists()]
        print(f'CV {season}: {len(tasks)} ajustes; {len(specs)} candidatos simetricos', flush=True)
        Parallel(n_jobs=jobs, backend='loky', inner_max_num_threads=1)(
            delayed(prediction_task)(tv, spec, par, season, out) for spec, par in tasks)
    rows, params = [], []
    for spec in specs:
        for season in [1718]+list(FOLDS):
            parameter, trace = choose_parameter(tv, spec, season, out)
            p = cached_prediction(tv, spec, parameter, season, out)
            va = tv[tv.Season.eq(season)]
            rows.append(prediction_frame(va, p, spec.id))
            params.append(dict(candidate=spec.id, season=season, parameter=parameter, **trace))
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(out/'oof_base.csv', index=False)
    json_write(out/'parameters_by_fold.json', params)
    result = summarize(predictions[predictions.Season.isin(FOLDS)], df)
    result.to_csv(out/'model_comparison.csv', index=False)
    print(result[['candidate', 'log_loss', 'AUC_macro', 'ECE_masa']].to_string(index=False), flush=True)
    return result


def prediction_frame(frame, p, candidate):
    return pd.DataFrame(dict(match_id=frame.match_id.to_numpy(), row_id=frame.row_id.to_numpy(),
        Season=frame.Season.to_numpy(), y=frame.y.to_numpy(int), candidate=candidate,
        P_H=p[:, 0], P_D=p[:, 1], P_A=p[:, 2]))


def aligned(predictions, candidate, frame):
    p = predictions[predictions.candidate.eq(candidate)].set_index('match_id').reindex(frame.match_id)
    if not np.array_equal(p.y.to_numpy(), frame.y.to_numpy()):
        raise ValueError(f'Predicciones desalineadas o incompletas: {candidate}')
    return normalize(p[['P_H', 'P_D', 'P_A']].to_numpy())


def summarize(predictions, df):
    rows = []
    for name, g in predictions.groupby('candidate', sort=False):
        rows.append(dict(candidate=name, **metrics(g.y.to_numpy(int), g[['P_H', 'P_D', 'P_A']].to_numpy())))
    return pd.DataFrame(rows).sort_values('log_loss').reset_index(drop=True)


def references(frame, training):
    prior = np.bincount(training.y.to_numpy(int), minlength=3)/len(training)
    return {'prior': np.tile(prior, (len(frame), 1)),
        'consenso': normalize(frame[['MKT_pH', 'MKT_pD', 'MKT_pA']].to_numpy()),
        'Pinnacle_respaldo': normalize(frame[['REF_pH', 'REF_pD', 'REF_pA']].to_numpy())}


def ensemble_weights(probs, y):
    # Convex optimization on the simplex, no softmax local-optimum artifact.
    z = np.stack(probs)
    def f(w): return log_loss(y, normalize(np.tensordot(w, z, axes=1)), labels=[0, 1, 2])
    result = minimize(f, np.ones(len(probs))/len(probs), method='SLSQP',
                      bounds=[(0, 1)]*len(probs), constraints={'type': 'eq', 'fun': lambda w: w.sum()-1},
                      options={'maxiter': 300, 'ftol': 1e-9})
    if not result.success: raise RuntimeError('No convergio el ensemble: '+result.message)
    return result.x


def logits(p):
    z = np.log(np.clip(p, 1e-12, 1)); z -= z.mean(1, keepdims=True)
    return z[:, :2]


def ensemble_member_names(branch):
    # Include the parsimonious LR views; excluding the strongest market-only LR
    # would unfairly handicap the market ensemble. Both branches have eight members.
    views = ['LR_REF', 'LR_consenso'] if branch == 'con' else ['LR_historia', 'LR_plantilla']
    return [branch+'/'+name for name in views+['LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM']]


def feedback(out):
    df, sets = read_data(out)
    base = pd.read_csv(out/'oof_base.csv')
    specs = specifications(sets)
    results = []
    weights_log = []
    for branch in ('con', 'sin'):
        members = ensemble_member_names(branch)
        for season in FOLDS:
            va = df[df.Season.eq(season)]
            past = df[df.Season.ge(1718) & df.Season.lt(season)]
            pp = [aligned(base, k, past) for k in members]
            pv = [aligned(base, k, va) for k in members]
            w = ensemble_weights(pp, past.y.to_numpy(int))
            weights_log.append(dict(branch=branch, season=season, **dict(zip(members, w))))
            stack = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=3000))
            stack.fit(np.hstack([logits(p) for p in pp]), past.y)
            pred = {'mean': np.mean(pv, axis=0), 'weighted': np.tensordot(w, np.stack(pv), axes=1),
                    'stack': stack.predict_proba(np.hstack([logits(p) for p in pv]))}
            for name, p in pred.items(): results.append(prediction_frame(va, normalize(p), branch+'/'+name))
    ensembles = pd.concat(results, ignore_index=True)
    ensembles.to_csv(out/'oof_ensembles.csv', index=False)
    pd.DataFrame(weights_log).to_csv(out/'ensemble_weights.csv', index=False)
    # Market-guided data-centric experiments use training outcomes only.
    schemes = ['none', 'w2', 'w3', 'w5', 'balance', 'sqrt_inverse', 'inverse', 'undersample', 'oversample']
    weighted_rows, calibration_rows = [], []
    for branch in ('con', 'sin'):
        for family in ('LR', 'HGB'):
            spec = next(s for s in specs if s.branch == branch and
                        s.name == ('LR_REF' if branch == 'con' and family == 'LR' else family))
            for season in FOLDS:
                tr, va = df[df.Season.lt(season)], df[df.Season.eq(season)]
                par, _ = choose_parameter(df, spec, season, out)
                p_market = tr[['MKT_pH', 'MKT_pD', 'MKT_pA']].to_numpy()
                failed = p_market.argmax(1) != tr.y.to_numpy()
                inverse = 1/np.clip(p_market[np.arange(len(tr)), tr.y.to_numpy()], .02, 1)
                for scheme in schemes:
                    if scheme == 'none':
                        p = aligned(base, spec.id, va)
                    else:
                        weight = None; sample = tr
                        if scheme.startswith('w'): weight = np.where(failed, float(scheme[1:]), 1.)
                        if scheme == 'balance': weight = np.where(failed, (~failed).sum()/failed.sum(), 1.)
                        if scheme == 'inverse': weight = inverse
                        if scheme == 'sqrt_inverse': weight = np.sqrt(inverse)
                        if scheme in ('undersample', 'oversample'):
                            seeds = []
                            for seed in (0, 1, 2):
                                rng = np.random.default_rng(seed+int(season))
                                hit, fail = np.flatnonzero(~failed), np.flatnonzero(failed)
                                ix = (np.r_[rng.choice(hit, len(fail), replace=False), fail]
                                      if scheme == 'undersample' else np.r_[hit, rng.choice(fail, len(hit), replace=True)])
                                # Sort again after resampling; duplicate sampled rows are intentional.
                                sample = tr.iloc[np.sort(ix)]
                                seeds.append(Predictor(spec, par).fit(sample).predict_proba(va))
                            p = np.mean(seeds, axis=0)
                        else:
                            weight = weight/weight.mean()
                            p = Predictor(spec, par).fit_weighted(sample, weight).predict_proba(va)
                    name = f'{branch}/{family}/{scheme}'
                    weighted_rows.append(prediction_frame(va, p, name))
                print(f'Feedback {branch}/{family} {season}', flush=True)
            # Post-calibration of the balanced variant, learned on previous OOF seasons.
    weighted = pd.concat(weighted_rows, ignore_index=True)
    weighted.to_csv(out/'oof_market_guided.csv', index=False)
    # A calibrator must have an earlier season of the same procedure.
    for branch in ('con', 'sin'):
        for family in ('LR', 'HGB'):
            name = f'{branch}/{family}/balance'
            for season in FOLDS[1:]:
                prev = FOLDS[FOLDS.index(season)-1]
                ca, va = df[df.Season.eq(prev)], df[df.Season.eq(season)]
                pca, pva = aligned(weighted, name, ca), aligned(weighted, name, va)
                cal = LogisticRegression(C=1., max_iter=3000).fit(logits(pca), ca.y)
                calibration_rows.append(prediction_frame(va, cal.predict_proba(logits(pva)), name+'_recal'))
    pd.concat(calibration_rows).to_csv(out/'oof_recalibration.csv', index=False)
    summarize(weighted, df).to_csv(out/'market_guided_results.csv', index=False)
    all_oof = pd.concat([base[base.Season.isin(FOLDS)], ensembles], ignore_index=True)
    all_oof.to_csv(out/'oof_candidates.csv', index=False)
    summary = summarize(all_oof, df)
    summary.to_csv(out/'model_comparison.csv', index=False)
    return summary


def complexity(name, specs):
    found = next((s for s in specs if s.id == name), None)
    family = found.family if found else name.split('/')[-1]
    return PRIORITY[family], len(found.columns) if found else 1000, name


def freeze(out):
    if (out/'frozen.json').exists(): return json.loads((out/'frozen.json').read_text())
    df, sets = read_data(out)
    specs = specifications(sets)
    summary = pd.read_csv(out/'model_comparison.csv')
    chosen = {}
    for branch in ('con', 'sin'):
        t = summary[summary.candidate.str.startswith(branch+'/')]
        near = t[t.log_loss.le(t.log_loss.min()+PROTOCOL['tolerance'])]
        selected = min(near.candidate, key=lambda name: complexity(name, specs))
        spec = next((s for s in specs if s.id == selected), None)
        parameter = choose_parameter(df, spec, 2425, out)[0] if spec else None
        chosen[branch] = dict(candidate=selected, spec=dataclasses.asdict(spec) if spec else None,
                              parameter=parameter, validation_log_loss=float(t.set_index('candidate').loc[selected, 'log_loss']))
    frozen = dict(selected=chosen, protocol_sha=sha(out/'protocol.json'),
        oof_sha=sha(out/'oof_candidates.csv'), comparison_sha=sha(out/'model_comparison.csv'),
        test_status=PROTOCOL['test_status'], date=str(pd.Timestamp.now()),
        note='Seleccion cerrada solo con train/val antes de generar las predicciones del test.')
    json_write(out/'frozen.json', frozen)
    return frozen


class EnsemblePredictor:
    def __init__(self, branch, kind, models, weights=None, stack=None):
        self.branch, self.kind, self.models, self.weights, self.stack = branch, kind, models, weights, stack
        self.columns = sorted(set(c for m in models for c in m.spec.columns))
    def predict_proba(self, frame):
        pp = [m.predict_proba(frame) for m in self.models]
        if self.kind == 'mean': return normalize(np.mean(pp, axis=0))
        if self.kind == 'weighted': return normalize(np.tensordot(self.weights, np.stack(pp), axes=1))
        return normalize(self.stack.predict_proba(np.hstack([logits(p) for p in pp])))


def fit_final(df, sets, selection, out):
    tr = df[~df.Season.isin(TEST)]
    if selection['spec'] is not None:
        s = selection['spec']; s['columns'] = tuple(s['columns'])
        return Predictor(Spec(**s), selection['parameter']).fit(tr)
    branch, kind = selection['candidate'].split('/')
    registry = {s.id: s for s in specifications(sets)}
    members = [registry[name] for name in ensemble_member_names(branch)]
    base = pd.read_csv(out/'oof_base.csv')
    past = tr[tr.Season.ge(1718)]
    pp = [aligned(base, s.id, past) for s in members]
    models = [Predictor(s, choose_parameter(df, s, 2425, out)[0]).fit(tr) for s in members]
    w = ensemble_weights(pp, past.y.to_numpy()) if kind == 'weighted' else None
    meta = None
    if kind == 'stack':
        meta = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=3000))
        meta.fit(np.hstack([logits(p) for p in pp]), past.y)
    return EnsemblePredictor(branch, kind, models, w, meta)


EVALUATION_OUTPUTS = ('final_metrics.csv', 'primary_comparisons.csv', 'by_season.csv', 'by_league.csv',
    'backtest.csv', 'similarity.csv', 'validation_predictions.csv',
    'modelo_con_cuotas.joblib', 'modelo_sin_cuotas.joblib', 'modelo_con_cuotas.json', 'modelo_sin_cuotas.json')


def verify_evaluation_outputs(out, marker):
    """A cached evaluation is reused only if every file it produced is byte-identical."""
    recorded = marker.get('outputs_sha')
    if not isinstance(recorded, dict) or 'final_metrics.csv' not in recorded:
        raise RuntimeError('Evaluacion sin huellas de sus resultados')
    for name in EVALUATION_OUTPUTS:
        path = out/name
        if path.exists() != (name in recorded) or (path.exists() and sha(path) != recorded[name]):
            raise RuntimeError(f'Resultado modificado despues de evaluar: {name}')


def evaluate(out):
    frozen = freeze(out)
    if sha(out/'protocol.json') != frozen['protocol_sha']:
        raise RuntimeError('Protocolo modificado despues de congelar')
    if sha(out/'oof_candidates.csv') != frozen['oof_sha'] or sha(out/'model_comparison.csv') != frozen['comparison_sha']:
        raise RuntimeError('Seleccion modificada despues de congelar')
    if (out/'evaluation_complete.json').exists():
        marker = json.loads((out/'evaluation_complete.json').read_text())
        if marker['frozen_sha'] != sha(out/'frozen.json'): raise RuntimeError('Test de otra seleccion')
        if marker['test_predictions_sha'] != sha(out/'test_predictions.csv'):
            raise RuntimeError('Predicciones de test modificadas despues de evaluar')
        verify_evaluation_outputs(out, marker)
        return pd.read_csv(out/'final_metrics.csv')
    df, sets = read_data(out)
    tv, te = df[~df.Season.isin(TEST)], df[df.Season.isin(TEST)]
    val = tv[tv.Season.isin(FOLDS)]
    assert tv.Date.max() < te.Date.min()
    oof = pd.read_csv(out/'oof_candidates.csv')
    pp_val, pp_test, table, prediction_rows = {}, references(te, tv), [], []
    for season in FOLDS:
        va = df[df.Season.eq(season)]
        for name, p in references(va, df[df.Season.lt(season)]).items():
            prediction_rows.append(prediction_frame(va, p, name))
    refs = pd.concat(prediction_rows)
    for name in ('prior', 'consenso', 'Pinnacle_respaldo'): pp_val[name] = aligned(refs, name, val)
    for branch, selected in frozen['selected'].items():
        model = fit_final(df, sets, selected, out)
        joblib.dump(model, out/f'modelo_{branch}_cuotas.joblib')
        name = 'modelo_'+branch
        pp_test[name] = model.predict_proba(te)
        pp_val[name] = aligned(oof, selected['candidate'], val)
        columns = list(model.spec.columns) if isinstance(model, Predictor) else model.columns
        json_write(out/f'modelo_{branch}_cuotas.json', dict(candidate=selected['candidate'],
            parameters=selected['parameter'], variables=columns, classes=['H', 'D', 'A'],
            train_ids_sha=hashlib.sha256('\n'.join(tv.match_id).encode()).hexdigest(),
            train_n=len(tv), train_until=str(tv.Date.max().date()), run_id=out.name,
            test_status=PROTOCOL['test_status'], frozen_sha=sha(out/'frozen.json')))
    tests = []
    for name, p in pp_test.items(): tests.append(prediction_frame(te, p, name))
    pd.concat(tests).to_csv(out/'test_predictions.csv', index=False)
    vals = []
    for name, p in pp_val.items(): vals.append(prediction_frame(val, p, name))
    pd.concat(vals).to_csv(out/'validation_predictions.csv', index=False)
    comparisons, bt, by_season, by_league = [], [], [], []
    for split, frame, probs in [('validacion', val, pp_val), ('test', te, pp_test)]:
        for name, p in probs.items():
            table.append(dict(split=split, candidate=name, **metrics(frame.y, p)))
            for season in frame.Season.unique():
                mask = frame.Season.eq(season).to_numpy()
                by_season.append(dict(split=split, candidate=name, Season=int(season), **metrics(frame.y.to_numpy()[mask], p[mask])))
            for league in sorted(frame.Div.unique()):
                mask = frame.Div.eq(league).to_numpy()
                by_league.append(dict(split=split, candidate=name, Div=league, **metrics(frame.y.to_numpy()[mask], p[mask])))
            for prefix in ('B365', 'PS'):
                bt.append(dict(split=split, candidate=name, **backtest(frame, p, prefix)))
        # A single declared primary testing family per split, including direct references.
        pairs = [('modelo_con', 'consenso'), ('modelo_con', 'Pinnacle_respaldo'),
                 ('modelo_sin', 'consenso'), ('modelo_sin', 'Pinnacle_respaldo'), ('modelo_sin', 'modelo_con')]
        family = []
        for a, b in pairs:
            family.append(dict(split=split, a=a, b=b, **compare(frame, losses(frame.y, probs[a]), losses(frame.y, probs[b]))))
        adj = bh([r['p'] for r in family])
        for r, p in zip(family, adj): r['p_BH'] = p
        comparisons.extend(family)
    pd.DataFrame(table).to_csv(out/'final_metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out/'primary_comparisons.csv', index=False)
    pd.DataFrame(by_season).to_csv(out/'by_season.csv', index=False)
    pd.DataFrame(by_league).to_csv(out/'by_league.csv', index=False)
    pd.DataFrame(bt).to_csv(out/'backtest.csv', index=False)
    # Similarity to the two references is explicitly labeled, never conflated.
    similarity = []
    for split, frame, probs in [('validacion', val, pp_val), ('test', te, pp_test)]:
        nm = probs['modelo_sin']; ll0 = log_loss(frame.y, probs['prior']); lln = log_loss(frame.y, nm)
        for name in ('consenso', 'Pinnacle_respaldo'):
            market = probs[name]
            similarity.append(dict(split=split, reference=name,
                corr_logit=float(np.corrcoef(np.log(nm[:, 0]/nm[:, 2]), np.log(market[:, 0]/market[:, 2]))[0, 1]),
                corr_pD=float(np.corrcoef(nm[:, 1], market[:, 1])[0, 1]),
                mismo_favorito=float((nm.argmax(1) == market.argmax(1)).mean()),
                fraccion_ganancia_prior=float((ll0-lln)/(ll0-log_loss(frame.y, market)))))
    pd.DataFrame(similarity).to_csv(out/'similarity.csv', index=False)
    json_write(out/'evaluation_complete.json', dict(frozen_sha=sha(out/'frozen.json'),
        test_predictions_sha=sha(out/'test_predictions.csv'), note=PROTOCOL['test_status'],
        outputs_sha={name: sha(out/name) for name in EVALUATION_OUTPUTS}))
    return pd.DataFrame(table)


def run(stage='all', jobs=4):
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    out = initialize()
    print('Ejecucion:', out.name, flush=True)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        if stage in ('prepare', 'all'):
            if not (out/'dataset.csv').exists(): prepare(out)
        if stage in ('cv', 'all'): nested_cv(out, jobs)
        if stage in ('feedback', 'all'): feedback(out)
        if stage in ('freeze', 'all'): freeze(out)
        if stage in ('evaluate', 'all'): print(evaluate(out).to_string(index=False), flush=True)
    return out
