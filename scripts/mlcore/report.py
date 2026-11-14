"""Generate figures, both deliverables and a single metrics source from a frozen run."""
from __future__ import annotations

import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

from .data import ROOT, FOLDS, TEST, read_data
from .experiment import current, aligned, json_write


def validation_diagnostics(out):
    """Descriptive checks on development folds; never selects with the test."""
    from .experiment import choose_parameter, references
    from .models import specifications, Predictor
    from .metrics import metrics, compare, losses, bh
    df, sets = read_data(out); frame = df[df.Season.isin(FOLDS)]
    base = pd.read_csv(out/'oof_base.csv')
    mk = frame[['MKT_pH', 'MKT_pD', 'MKT_pA']].to_numpy()
    rows = []
    for branch in ('con', 'sin'):
        for family in ('LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM'):
            p = aligned(base, branch+'/'+family, frame)
            rows.append(dict(candidate=branch+'/'+family, **metrics(frame.y, p),
                             **compare(frame, losses(frame.y, p), losses(frame.y, mk))))
    t = pd.DataFrame(rows); t['p_BH'] = bh(t.p)
    t.to_csv(out/'validation_diagnostics.csv', index=False)
    importance = []
    for branch in ('con', 'sin'):
        spec = next(s for s in specifications(sets) if s.id == branch+'/LR')
        groups = {'Historia': sets['history'], 'Plantilla': sets['roster'], 'Habilidades': sets['skills']}
        if branch == 'con': groups['Mercado'] = [c for c in spec.columns if c not in sets['sin_cuotas']]
        for season_ in FOLDS:
            tr, va = df[df.Season.lt(season_)], df[df.Season.eq(season_)].copy()
            model = Predictor(spec, choose_parameter(df, spec, season_, out)[0]).fit(tr)
            ll0 = metrics(va.y, model.predict_proba(va))['log_loss']
            for group, columns in groups.items():
                differences = []
                for seed in (0, 1, 2):
                    copy = va.copy(); ix = np.random.default_rng(seed).permutation(len(va))
                    copy[columns] = va[columns].to_numpy()[ix]
                    differences.append(metrics(va.y, model.predict_proba(copy))['log_loss']-ll0)
                importance.append(dict(branch=branch, Season=season_, grupo=group, delta=float(np.mean(differences))))
    pd.DataFrame(importance).to_csv(out/'importance_groups.csv', index=False)
    return t

def training_gap(out):
    """Training versus validation loss on two development folds; descriptive, never selects."""
    path = out/'train_validation_gap.csv'
    if path.exists(): return pd.read_csv(path).set_index('modelo')
    import warnings
    from sklearn.metrics import log_loss
    from .experiment import choose_parameter
    from .models import specifications, Predictor
    df, sets = read_data(out); specs = {s.id: s for s in specifications(sets)}; rows = []
    names = ['con/LR_REF']+[b+'/'+f for b in ('con', 'sin') for f in ('LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM')]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for season_ in (2122, 2324):
            tr, va = df[df.Season.lt(season_)], df[df.Season.eq(season_)]
            for name in names:
                model = Predictor(specs[name], choose_parameter(df, specs[name], season_, out)[0]).fit(tr)
                used = tr.iloc[-9000:] if name.endswith('SVM') else tr  # the SVM only sees its most recent rows
                rows.append(dict(modelo=name, temporada=season_,
                    entrenamiento=log_loss(used.y, model.predict_proba(used), labels=[0, 1, 2]),
                    validacion=log_loss(va.y, model.predict_proba(va), labels=[0, 1, 2])))
    t = pd.DataFrame(rows).groupby('modelo')[['entrenamiento', 'validacion']].mean()
    t['brecha'] = t.validacion-t.entrenamiento
    # The two-variable price LR cannot overfit: its gap only reflects season difficulty.
    t['brecha_extra'] = t.brecha-t.loc['con/LR_REF', 'brecha']
    t.to_csv(path)
    return t


COL = {'modelo_con': '#0d9488', 'modelo_sin': '#2563eb', 'consenso': '#475569',
       'Pinnacle_respaldo': '#9b596c', 'prior': '#a1a1aa'}
LABEL = {'modelo_con': 'Modelo con cuotas', 'modelo_sin': 'Modelo sin cuotas',
         'consenso': 'Consenso directo', 'Pinnacle_respaldo': 'Pinnacle fiable / respaldo', 'prior': 'Prior'}


def f(value, digits=4): return f'{float(value):.{digits}f}'
def pct(value): return f'{100*float(value):.1f} %'
def season(s): return f'{2000+int(s)//100}/{int(s)%100:02d}'
def model_label(name):
    return {'LR_REF': 'LR referencia', 'LR_consenso': 'LR consenso', 'LR_historia': 'LR historia',
            'LR_plantilla': 'LR historia y plantilla', 'GD_LIN': 'GD lineal', 'GD_RF': 'GD RF',
            'GD_HGB': 'GD HGB', 'GD_SVR': 'GD SVR', 'GD_ALL': 'GD promedio',
            'mean': 'Promedio', 'weighted': 'Ponderado', 'stack': 'Stacking'}.get(name, name.replace('_', ' '))


def context(out):
    df, sets = read_data(out)
    frozen = json.loads((out/'frozen.json').read_text())
    final = pd.read_csv(out/'final_metrics.csv').set_index(['split', 'candidate'])
    return dict(out=out, df=df, sets=sets, frozen=frozen, final=final,
                comparison=pd.read_csv(out/'model_comparison.csv'),
                primary=pd.read_csv(out/'primary_comparisons.csv'),
                feedback=pd.read_csv(out/'market_guided_results.csv'),
                similarity=pd.read_csv(out/'similarity.csv'),
                coverage=pd.read_csv(out/'coverage.csv'),
                backtest=pd.read_csv(out/'backtest.csv'),
                by_season=pd.read_csv(out/'by_season.csv'))


def development_comparisons(out):
    from .metrics import losses, compare, bh
    df, _ = read_data(out); va = df[df.Season.isin(FOLDS)]
    weighted = pd.read_csv(out/'oof_market_guided.csv')
    rows = []
    for branch in ('con', 'sin'):
        for family in ('LR', 'HGB'):
            prefix=f'{branch}/{family}/'
            baseline=aligned(weighted,prefix+'none',va)
            for candidate in weighted.candidate.unique():
                if candidate.startswith(prefix) and not candidate.endswith('/none'):
                    p=aligned(weighted,candidate,va)
                    rows.append(dict(candidate=candidate,reference=prefix+'none',
                                     **compare(va,losses(va.y,p),losses(va.y,baseline))))
    result=pd.DataFrame(rows);result['p_BH']=bh(result.p)
    result.to_csv(out/'market_guided_comparisons.csv',index=False)
    base=pd.read_csv(out/'oof_candidates.csv');rows=[]
    for branch in ('con','sin'):
        reference=branch+('/LR_REF' if branch=='con' else '/LR_plantilla')
        baseline=aligned(base,reference,va)
        for name in ('mean','weighted','stack'):
            candidate=branch+'/'+name;p=aligned(base,candidate,va)
            rows.append(dict(candidate=candidate,reference=reference,
                             **compare(va,losses(va.y,p),losses(va.y,baseline))))
    result=pd.DataFrame(rows);result['p_BH']=bh(result.p)
    result.to_csv(out/'ensemble_comparisons.csv',index=False)


def figures(c):
    figdir = ROOT/'docs/figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 220, 'font.family': 'DejaVu Sans'})
    def save(fig, name):
        fig.tight_layout(); fig.savefig(figdir/(name+'.png'), bbox_inches='tight'); plt.close(fig)
    d, result, final = c['df'], c['comparison'], c['final']
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
    axes[0].bar(['H', 'D', 'A'], d.y.value_counts(normalize=True).sort_index(), color=['#0d9488', '#94a3b8', '#2563eb'])
    axes[0].set(title='Distribución del resultado', ylabel='Proporción')
    cv = c['coverage']; x = np.arange(len(cv))
    axes[1].plot(x, cv.cobertura_plantilla, 'o-', color=COL['modelo_sin']); axes[1].set_ylim(.8, 1.01)
    axes[1].set(title='Plantilla disponible antes del partido', ylabel='Cobertura de ambos equipos')
    axes[2].plot(x, cv.pinnacle_fiable, 'o-', color=COL['modelo_con']); axes[2].set(title='Pinnacle: cobertura fiable', ylim=(-.03, 1.03))
    for ax in axes[1:]: ax.set_xticks(x[::2], [season(s) for s in cv.Season.to_numpy()[::2]], rotation=45, fontsize=7)
    save(fig, 'revision_datos')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, branch, key in zip(axes, ('con', 'sin'), ('modelo_con', 'modelo_sin')):
        t = result[result.candidate.str.startswith(branch+'/') & ~result.candidate.str.contains('MI|PCA')].sort_values('log_loss', ascending=False)
        ax.barh(t.candidate.str.split('/').str[-1], t.log_loss, color=COL[key])
        lo, hi = t.log_loss.min(), t.log_loss.max(); ax.set_xlim(lo-.004, hi+.004)
        for i, v in enumerate(t.log_loss): ax.text(v+.0002, i, f(v), va='center', fontsize=7)
        ax.axvline(final.loc[('validacion', 'consenso'), 'log_loss'], ls='--', color=COL['consenso'], label='Consenso directo')
        ax.set(title=LABEL[key]+' · validación', xlabel='Log loss (menor es mejor)'); ax.legend(fontsize=7)
    save(fig, 'revision_modelos')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, key in zip(axes, ('modelo_con', 'modelo_sin')):
        for name in (key, 'consenso', 'prior'):
            t = c['by_season'][c['by_season'].candidate.eq(name)].sort_values('Season')
            ax.plot(np.arange(len(t)), t.log_loss, 'o-', color=COL[name], label=LABEL[name], markersize=3)
        labels = list(FOLDS)+list(TEST)
        ax.set_xticks(range(8), [season(s) for s in labels], rotation=45, fontsize=7)
        ax.axvspan(5.5, 7.5, color='#fef3c7', alpha=.7); ax.set(title=LABEL[key], ylabel='Log loss'); ax.legend(fontsize=6)
    save(fig, 'revision_temporadas')
    val = d[d.Season.isin(FOLDS)]; te = d[d.Season.isin(TEST)]
    preds = pd.read_csv(c['out']/'validation_predictions.csv'); testpred = pd.read_csv(c['out']/'test_predictions.csv')
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
    for row, key in enumerate(('modelo_con', 'modelo_sin')):
        for j, outcome in enumerate(('Local', 'Empate', 'Visitante')):
            ax = axes[row, j]
            for name in (key, 'consenso'):
                p = aligned(preds, name, val)[:, j]; o = val.y.to_numpy() == j
                bins = np.digitize(p, np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1])))
                ax.plot([p[bins == k].mean() for k in np.unique(bins)], [o[bins == k].mean() for k in np.unique(bins)], 'o-', ms=3, color=COL[name], label=LABEL[name])
            ax.plot([0, 1], [0, 1], ':', color='#cbd5e1'); ax.set(title=outcome, xlabel='Probabilidad', ylabel='Frecuencia observada')
            if j == 0: ax.legend(fontsize=6)
    save(fig, 'revision_calibracion')
    # Presentation panel: show only its own branch at a readable size.
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    for j, ax in enumerate(axes):
        for name in ('modelo_con', 'consenso'):
            p = aligned(preds, name, val)[:, j]; o = val.y.to_numpy() == j
            bins = np.digitize(p, np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1])))
            ax.plot([p[bins == k].mean() for k in np.unique(bins)],
                    [o[bins == k].mean() for k in np.unique(bins)], 'o-', color=COL[name], label=LABEL[name])
        ax.plot([0, 1], [0, 1], ':', color='#cbd5e1')
        ax.set(title=['Local','Empate','Visitante'][j], xlabel='Probabilidad', ylabel='Frecuencia observada')
        ax.legend(fontsize=7)
    save(fig, 'revision_calibracion_con')
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, branch, key in zip(axes, ('con', 'sin'), ('modelo_con', 'modelo_sin')):
        names = [branch+'/LR'+suffix for suffix in ('', '_MI5', '_MI10', '_PCA5', '_PCA95')]
        t = result.set_index('candidate').loc[names]
        delta = t.log_loss-t.iloc[0].log_loss
        ax.bar(['Todas', 'MI-5', 'MI-10', 'PCA-5', 'PCA-95%'], delta, color=COL[key]); ax.axhline(0, color='#475569', lw=.7)
        ax.set(title=LABEL[key]+' · LR', ylabel='Δ log loss frente a todas')
    save(fig, 'revision_reduccion')
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, branch, key in zip(axes, ('con', 'sin'), ('modelo_con', 'modelo_sin')):
        t = result.set_index('candidate').loc[[branch+'/'+x for x in ('LR', 'mean', 'weighted', 'stack')]]
        ax.bar(['LR', 'Promedio', 'Ponderado', 'Stacking'], t.log_loss-t.iloc[0].log_loss, color=COL[key]); ax.axhline(0, color='#475569', lw=.7)
        ax.set(title=LABEL[key], ylabel='Δ log loss frente a LR')
    save(fig, 'revision_ensembles')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    schemes = ['none', 'balance', 'undersample', 'oversample', 'sqrt_inverse', 'w2', 'inverse', 'w3', 'w5']
    for ax, branch, key in zip(axes, ('con', 'sin'), ('modelo_con', 'modelo_sin')):
        for family, style in [('LR', '-'), ('HGB', '--')]:
            t = c['feedback'].set_index('candidate').loc[[f'{branch}/{family}/{s}' for s in schemes]]
            ax.plot(range(len(t)), t.log_loss-t.iloc[0].log_loss, style, marker='o', ms=3, label=family, color=COL[key])
        ax.set_xticks(range(len(schemes)), ['Base', '50/50', 'Sub', 'Sobre', '√1/p', 'w2', '1/p', 'w3', 'w5'], rotation=45)
        ax.set(title=LABEL[key]+' · pesos guiados por mercado', ylabel='Δ log loss'); ax.legend(fontsize=7)
    save(fig, 'revision_pesos')
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    nm, market = aligned(preds, 'modelo_sin', val), aligned(preds, 'consenso', val)
    for j, ax in enumerate(axes):
        ax.hexbin(market[:, j], nm[:, j], gridsize=30, mincnt=1, bins='log', cmap='Blues')
        ax.plot([0, 1], [0, 1], '--', color='#94a3b8'); ax.set(title=['Local', 'Empate', 'Visitante'][j], xlabel='Probabilidad del consenso', ylabel='Probabilidad sin cuotas')
    save(fig, 'revision_predictor')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, split in zip(axes, ('validacion', 'test_revisado')):
        t = final.loc[split].loc[['consenso', 'Pinnacle_respaldo', 'modelo_con', 'modelo_sin']]
        ax.barh([LABEL[k] for k in t.index], t.log_loss, color=[COL[k] for k in t.index])
        ax.set_xlim(t.log_loss.min()-.004, t.log_loss.max()+.004)
        for i, v in enumerate(t.log_loss): ax.text(v+.0002, i, f(v), va='center', fontsize=8)
        ax.set(title='Validación' if split == 'validacion' else 'Test 2024/25–2025/26', xlabel='Log loss')
    save(fig, 'revision_final')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, key in zip(axes, ('modelo_con', 'modelo_sin')):
        t = c['backtest'][c['backtest'].candidate.eq(key)].copy()
        labels = [f'{r.casa} · '+('validación' if r.split == 'validacion' else 'test')+f' (n={r.apuestas})' for r in t.itertuples()]
        for i, r in enumerate(t.itertuples()):
            if np.isfinite(r.ROI):
                ax.plot([r.lo, r.hi], [i, i], color=COL[key]); ax.plot(r.ROI, i, 'o', color=COL[key])
        ax.set_yticks(range(len(t)), labels, fontsize=7); ax.axvline(0, color='#475569', lw=.8)
        ax.set(title=LABEL[key], xlabel='ROI por unidad e IC 95% por bloques')
    save(fig, 'revision_backtest')
    fig, ax = plt.subplots(figsize=(5.5, 3))
    t = result.set_index('candidate').loc[['sin/LR_historia', 'sin/LR_plantilla', 'sin/LR']]
    ax.barh(['Historia', 'Historia + plantilla', 'Historia + plantilla + habilidades'], t.log_loss, color=COL['modelo_sin'])
    ax.set_xlim(t.log_loss.min()-.004, t.log_loss.max()+.004)
    for i, v in enumerate(t.log_loss): ax.text(v+.0002, i, f(v), va='center', fontsize=8)
    ax.set(title='Ablación del predictor sin cuotas · LR', xlabel='Log loss de validación')
    save(fig, 'revision_ablacion')
    imp_path = c['out']/'importance_groups.csv'
    if imp_path.exists():
        importance = pd.read_csv(imp_path)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        for ax, branch, key in zip(axes, ('con', 'sin'), ('modelo_con', 'modelo_sin')):
            t = importance[importance.branch.eq(branch)].groupby('grupo').delta.agg(['mean', 'std'])
            ax.barh(t.index, t['mean'], xerr=2.571*t['std']/np.sqrt(6), color=COL[key], capsize=2)
            ax.set(title=LABEL[key]+' · LR', xlabel='Aumento del log loss al permutar')
        save(fig, 'revision_importancia')


def latex_escape(s):
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def paper_figures(c):
    """Compact single-column figures for the report (the deck keeps its own)."""
    figdir = ROOT/'docs/figures'
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 250, 'font.family': 'DejaVu Sans'})
    def save(fig, name):
        fig.tight_layout(); fig.savefig(figdir/(name+'.png'), bbox_inches='tight'); plt.close(fig)
    d, result, final = c['df'], c['comparison'].set_index('candidate'), c['final']
    val = d[d.Season.isin(FOLDS)]
    preds = pd.read_csv(c['out']/'validation_predictions.csv')
    # Data overview
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3), gridspec_kw={'width_ratios': [1, 1.6]})
    share = d.y.value_counts(normalize=True).sort_index()
    axes[0].bar(['Local', 'Empate', 'Visitante'], share, color=['#0d9488', '#94a3b8', '#2563eb'])
    for i, v in enumerate(share): axes[0].text(i, v+.01, f'{100*v:.1f} %', ha='center')
    axes[0].set(title='Resultado de los partidos', ylabel='Proporción', ylim=(0, .52))
    cv = c['coverage']; x = np.arange(len(cv))
    axes[1].plot(x, cv.cobertura_plantilla, 'o-', ms=3, color=COL['modelo_sin'], label='Plantilla disponible')
    axes[1].plot(x, cv.pinnacle_fiable, 's--', ms=3, color=COL['modelo_con'], label='Pinnacle fiable')
    axes[1].set_xticks(x[::2], [season(s) for s in cv.Season.to_numpy()[::2]], rotation=40, fontsize=7)
    axes[1].set(title='Cobertura por temporada', ylabel='Fracción de partidos', ylim=(-.03, 1.05)); axes[1].legend(fontsize=7, loc='lower left')
    save(fig, 'inf_datos')
    # Calibration: three probability sources, one panel per outcome
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    for j, ax in enumerate(axes):
        o_all = val.y.to_numpy() == j
        for name, mk in (('consenso', 's'), ('modelo_con', 'o'), ('modelo_sin', '^')):
            p = aligned(preds, name, val)[:, j]
            bins = np.digitize(p, np.unique(np.quantile(p, np.linspace(0, 1, 11)[1:-1])))
            ax.plot([p[bins == k].mean() for k in np.unique(bins)], [o_all[bins == k].mean() for k in np.unique(bins)],
                    marker=mk, ms=3, lw=1, color=COL[name], label=LABEL[name])
        ax.plot([0, .85], [0, .85], ':', color='#94a3b8')
        ax.set(title=['Local', 'Empate', 'Visitante'][j], xlabel='Probabilidad predicha')
    axes[0].set_ylabel('Frecuencia real'); axes[0].legend(fontsize=6)
    save(fig, 'inf_calibracion')
    # Ablation, compact
    fig, ax = plt.subplots(figsize=(7, 1.7))
    t = result.loc[['sin/LR_historia', 'sin/LR_plantilla', 'sin/LR'], 'log_loss']
    ax.barh(['Historia', '+ plantilla', '+ plantilla + habilidades'], t, color=COL['modelo_sin'], height=.6)
    ax.set_xlim(t.min()-.003, t.max()+.002); ax.invert_yaxis()
    for i, v in enumerate(t): ax.text(v+.0001, i, f(v), va='center', fontsize=7)
    ax.set(xlabel='Log loss de validación (menor es mejor)')
    save(fig, 'inf_ablacion')
    # Sports-only predictor vs market
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.4), sharey=True)
    nm, market = aligned(preds, 'modelo_sin', val), aligned(preds, 'consenso', val)
    for j, ax in enumerate(axes):
        ax.hexbin(market[:, j], nm[:, j], gridsize=28, mincnt=1, bins='log', cmap='Blues')
        ax.plot([0, 1], [0, 1], '--', color='#64748b', lw=.8)
        ax.set(title=['Local', 'Empate', 'Visitante'][j], xlabel='Probabilidad del mercado')
    axes[0].set_ylabel('Probabilidad sin cuotas')
    save(fig, 'inf_predictor')
    # Improvements: MI/PCA and ensembles relative to LR with all variables
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharey=False)
    labels = ['MI-5', 'MI-10', 'PCA-5', 'PCA-95%', 'Promedio', 'Ponderado', 'Stacking']
    for ax, branch, key in zip(axes, ('con', 'sin'), ('modelo_con', 'modelo_sin')):
        names = [branch+'/LR'+s for s in ('_MI5', '_MI10', '_PCA5', '_PCA95')]+[branch+'/'+s for s in ('mean', 'weighted', 'stack')]
        delta = result.loc[names, 'log_loss']-result.loc[branch+'/LR', 'log_loss']
        ax.bar(labels, delta, color=[COL[key] if v < 0 else '#cbd5e1' for v in delta], edgecolor=COL[key])
        ax.axhline(0, color='#475569', lw=.7); ax.tick_params(axis='x', rotation=45)
        ax.set(title=LABEL[key], ylabel='Cambio de log loss')
    save(fig, 'inf_mejoras')
    # Sample weights
    fig, ax = plt.subplots(figsize=(7, 2.3))
    schemes = ['none', 'balance', 'undersample', 'oversample', 'sqrt_inverse', 'w2', 'inverse', 'w3', 'w5']
    names = ['Sin pesos', '50/50', 'Submuestreo', 'Sobremuestreo', '√(1/p)', 'Peso 2', '1/p', 'Peso 3', 'Peso 5']
    for branch, key, mk in (('con', 'modelo_con', 'o'), ('sin', 'modelo_sin', '^')):
        t = c['feedback'].set_index('candidate').loc[[f'{branch}/LR/{s}' for s in schemes]]
        ax.plot(range(len(t)), t.log_loss-t.iloc[0].log_loss, marker=mk, ms=4, color=COL[key], label=LABEL[key]+' (LR)')
    ax.set_xticks(range(len(schemes)), names, rotation=30, fontsize=7); ax.axhline(0, color='#475569', lw=.7)
    ax.set(ylabel='Cambio de log loss'); ax.legend(fontsize=7)
    save(fig, 'inf_pesos')
    # Season by season
    fig, ax = plt.subplots(figsize=(7, 2.5))
    for name in ('consenso', 'modelo_con', 'modelo_sin'):
        t = c['by_season'][c['by_season'].candidate.eq(name)].sort_values('Season')
        ax.plot(np.arange(len(t)), t.log_loss, 'o-', ms=3, color=COL[name], label=LABEL[name])
    ax.axvspan(5.5, 7.5, color='#fef3c7', alpha=.8); ax.text(6.5, ax.get_ylim()[1], 'test', ha='center', va='top', fontsize=7)
    ax.set_xticks(range(8), [season(s) for s in list(FOLDS)+list(TEST)], fontsize=7)
    ax.set(ylabel='Log loss'); ax.legend(fontsize=7)
    save(fig, 'inf_temporadas')
    # Betting simulation
    fig, ax = plt.subplots(figsize=(7, 2.6))
    rows = []
    for key in ('modelo_con', 'modelo_sin'):
        for split in ('validacion', 'test_revisado'):
            for casa in ('B365', 'PS'):
                r = c['backtest'].query('candidate == @key and split == @split and casa == @casa').iloc[0]
                rows.append((key, f"{'Con' if key == 'modelo_con' else 'Sin'} cuotas · {'Bet365' if casa == 'B365' else 'Pinnacle'} · {'validación' if split == 'validacion' else 'test'}", r))
    for i, (key, lab, r) in enumerate(rows):
        ax.plot([100*r.lo, 100*r.hi], [i, i], color=COL[key]); ax.plot(100*r.ROI, i, 'o', color=COL[key])
    ax.set_yticks(range(len(rows)), [x[1] for x in rows], fontsize=7); ax.invert_yaxis()
    ax.axvline(0, color='#475569', lw=.8); ax.set(xlabel='Retorno por unidad apostada (%) e intervalo del 95 %')
    save(fig, 'inf_backtest')


def latex_table(headers, rows, label, caption, wide=False, align=None):
    head = ' & '.join(headers)+r' \\'
    body = '\n'.join(' & '.join(map(str, row))+r' \\' for row in rows)
    tab = r'\begin{tabular}{'+(align or 'l'+'r'*(len(headers)-1))+'}'
    inner = [tab, r'\toprule', head, r'\midrule', body, r'\bottomrule', r'\end{tabular}']
    if wide: inner = [r'\resizebox{\columnwidth}{!}{'+inner[0]]+inner[1:-1]+[r'\end{tabular}}']
    return '\n'.join([r'\begin{table}[!htb]', r'\caption{'+caption+'}', r'\centering\footnotesize', *inner,
        r'\label{'+label+'}', r'\end{table}'])+'\n\n'


def latex_figure(name, label, caption, wide=False):
    env = 'figure*' if wide else 'figure'
    return '\n'.join([r'\begin{'+env+'}[!htb]', r'\centering',
        r'\includegraphics[width='+(r'0.96\textwidth' if wide else r'\columnwidth')+']{'+name+'.png}',
        r'\caption{'+caption+'}', r'\label{'+label+'}', r'\end{'+env+'}'])+'\n\n'


def dec(x, digits=4):
    """Spanish decimal comma for prose."""
    return f'{float(x):.{digits}f}'.replace('.', '{,}').replace('-', '$-$')


def eda_figures(c):
    """Exploratory figures for the report, recomputed from the current data."""
    figdir = ROOT/'docs/figures'
    def save(fig, name):
        fig.tight_layout(); fig.savefig(figdir/(name+'.png'), bbox_inches='tight'); plt.close(fig)
    raw = pd.read_csv(ROOT/'scripts/data/matches.csv', low_memory=False)
    raw = raw[raw.Season >= 1415]
    d = c['df']; stats = {}
    # Missing values
    groups = {'Bet365': 'B365H', 'Bet&Win': 'BWH', 'Pinnacle': 'PSH', 'William Hill': 'WHH', 'Interwetten': 'IWH',
              'VC Bet': 'VCH', 'Ladbrokes': 'LBH', 'Máx./prom. mercado': 'MaxH', 'Tiros': 'HS', 'Faltas': 'HF',
              'Córners': 'HC', 'Tarjetas': 'HY', 'Árbitro': 'Referee', 'Hora': 'Time'}
    na = pd.Series({k: 100*raw[v].isna().mean() for k, v in groups.items()}).sort_values()
    fig, ax = plt.subplots(figsize=(7, 2.7))
    colors = ['#0d9488' if k in ('Bet365', 'Bet&Win', 'Pinnacle', 'William Hill') else '#94a3b8' for k in na.index]
    ax.barh(na.index, na, color=colors)
    for i, v in enumerate(na): ax.text(v+.8, i, f'{v:.1f} %', va='center', fontsize=6.5)
    ax.axvline(20, ls='--', color='#b91c1c', lw=.8); ax.text(21, 0, '20 %', color='#b91c1c', fontsize=7)
    ax.set(xlabel='Partidos sin dato (%)', xlim=(0, 100)); ax.tick_params(axis='y', labelsize=7)
    save(fig, 'inf_nulos')
    stats['na_stats'] = 100*raw.HS.isna().mean()
    stats['na_leagues'] = int((raw.groupby('Div').HS.apply(lambda x: x.isna().mean()) > .1).sum())
    stats['na_pin'] = 100*raw[raw.Season.eq(2526)].PSH.isna().mean()
    stats['na_wh'] = 100*raw[raw.Season.eq(2526)].WHH.isna().mean()
    p = 1/raw[['B365H', 'PSH', 'BWH', 'WHH']]; cm = p.corr().to_numpy()
    stats['corr_casas'] = cm[np.triu_indices(4, 1)].min()
    # Extremes
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3), gridspec_kw={'width_ratios': [1.3, 1]})
    cols = [('FTHG', 'Goles local'), ('FTAG', 'Goles visit.'), ('HST', 'Tiros a puerta')]
    axes[0].boxplot([raw[c_].dropna() for c_, _ in cols], tick_labels=[l for _, l in cols], flierprops=dict(ms=2, alpha=.3))
    axes[0].set(title='Goles y tiros por partido')
    axes[1].boxplot([raw.B365H.dropna(), raw.B365D.dropna(), raw.B365A.dropna()], tick_labels=['Local', 'Empate', 'Visit.'], flierprops=dict(ms=2, alpha=.3))
    axes[1].set(yscale='log', title='Cuotas de Bet365 (escala log)')
    save(fig, 'inf_extremos')
    def out_share(x):
        x = x.dropna(); q1, q3 = x.quantile([.25, .75]); i = q3-q1
        return 100*((x > q3+1.5*i) | (x < q1-1.5*i)).mean()
    stats['out_goals'] = max(out_share(raw.FTAG), out_share(raw.HS)); stats['out_odds'] = out_share(raw.B365A)
    stats['max_goals'] = int(raw[['FTHG', 'FTAG']].max().max()); stats['max_odds'] = float(raw[['B365H', 'B365A']].max().max())
    # Odds versus result
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.3))
    res = [('H', 'Ganó local'), ('D', 'Empate'), ('A', 'Ganó visit.')]
    for ax, col, title in zip(axes, ('MKT_pH', 'XI_DIF'), ('Prob. de victoria local según casas', 'Diferencia de valoración del once')):
        ax.boxplot([d.loc[d.FTR.eq(r), col].dropna() for r, _ in res], tick_labels=[l for _, l in res], showfliers=False,
                   medianprops=dict(color='#b91c1c'))
        ax.set(title=title)
    axes[1].axhline(0, color='#94a3b8', lw=.7)
    save(fig, 'inf_cuotas_resultado')
    med = d.groupby('FTR')[['MKT_pH', 'XI_DIF']].median()
    stats.update(pH_H=100*med.loc['H', 'MKT_pH'], pH_A=100*med.loc['A', 'MKT_pH'], xi_H=med.loc['H', 'XI_DIF'], xi_A=med.loc['A', 'XI_DIF'])
    # Correlation heatmap
    names = {'ELO_DIF': 'Elo', 'EW_Pts10_DIF': 'Puntos recientes', 'EW_GD10_DIF': 'Dif. goles reciente', 'EW_SOTD10_DIF': 'Tiros a puerta',
             'XI_DIF': 'Once titular', 'SQ_DIF': 'Plantilla', 'MKT_logit': 'Favoritismo (cuotas)', 'MKT_pD': 'Prob. empate (cuotas)', 'MKT_margin': 'Margen casa'}
    cr = d[list(names)].corr(); stats['corr'] = cr
    fig, ax = plt.subplots(figsize=(6, 4.3))
    im = ax.imshow(cr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names.values(), rotation=45, ha='right', fontsize=7); ax.set_yticks(range(len(names)), names.values(), fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)): ax.text(j, i, f'{cr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if abs(cr.iloc[i, j]) > .6 else '#0f172a')
    ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, 'inf_correlacion')
    # Confusion matrices (validation)
    val = d[d.Season.isin(FOLDS)]; preds = pd.read_csv(c['out']/'validation_predictions.csv')
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7))
    for ax, key in zip(axes, ('modelo_con', 'modelo_sin')):
        pred = aligned(preds, key, val).argmax(1); y = val.y.to_numpy(int)
        cmx = np.bincount(3*y+pred, minlength=9).reshape(3, 3); rown = cmx/cmx.sum(1, keepdims=True)
        ax.imshow(rown, cmap='Blues', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3): ax.text(j, i, f'{100*rown[i, j]:.0f} %', ha='center', va='center', color='white' if rown[i, j] > .5 else '#0f172a', fontsize=8)
        lab = ['Local', 'Empate', 'Visitante']
        ax.set_xticks(range(3), lab); ax.set_yticks(range(3), lab); ax.set(xlabel='Predicción', ylabel='Resultado real', title=LABEL[key])
        ax.spines[:].set_visible(False)
        stats[key+'_recall'] = rown.diagonal(); stats[key+'_draws_pred'] = 100*(pred == 1).mean()
    save(fig, 'inf_confusion')
    # Draws among balanced matches
    bal = val.XI_DIF.abs() <= 2
    stats['draw_bal'] = 100*val.loc[bal & val.XI_DIF.notna(), 'y'].eq(1).mean()
    stats['draw_unbal'] = 100*val.loc[~bal & val.XI_DIF.notna(), 'y'].eq(1).mean()
    # Permutation importance by group
    imp_path = c['out']/'importance_groups.csv'
    importance = pd.read_csv(imp_path)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.2), gridspec_kw={'width_ratios': [1, 1]})
    for ax, branch, key in zip(axes, ('con', 'sin'), ('modelo_con', 'modelo_sin')):
        t = importance[importance.branch.eq(branch)].groupby('grupo').delta.agg(['mean', 'std']).sort_values('mean')
        ax.barh(t.index, t['mean'], xerr=2.571*t['std']/np.sqrt(6), color=COL[key], capsize=2)
        ax.set(title=LABEL[key], xlabel='Aumento del log loss al desordenar')
    save(fig, 'inf_importancia')
    stats['imp'] = importance.groupby(['branch', 'grupo']).delta.mean()
    # Draw-oriented decision rules (validation only)
    pointer = ROOT/'scripts/data/draw_actual.txt'
    stats['draws'] = pd.read_csv(ROOT/pointer.read_text().strip()/'metrics.csv') if pointer.exists() else None
    return stats


def paper(c):
    # Keep authors and the IEEE preamble. Replace the body so old claims cannot survive.
    templates = ROOT/'scripts/report/templates'
    preamble = (templates/'preamble.tex').read_text()
    preamble = preamble.replace('Predicción probabilística del resultado 1X2 en partidos de fútbol mediante Machine Learning',
        'Predicción probabilística de fútbol con y sin cuotas: comparación temporal de modelos y mercado')
    paper_figures(c); E = eda_figures(c)
    m = c['final']; v, t = 'validacion', 'test_revisado'
    result = c['comparison'].set_index('candidate')
    sim = c['similarity'].query('split == "validacion" and reference == "consenso"').iloc[0]
    pr = c['primary'].set_index(['split', 'a', 'b'])
    bs = c['by_season'].pivot(index='Season', columns='candidate', values='log_loss')
    bt = c['backtest'].set_index(['split', 'candidate', 'casa'])
    ens = pd.read_csv(c['out']/'ensemble_comparisons.csv').set_index('candidate')
    weights = pd.read_csv(c['out']/'ensemble_weights.csv')
    share = c['df'].y.value_counts(normalize=True).sort_index()
    ll = lambda split, k: m.loc[(split, k), 'log_loss']
    d1, d2 = pr.loc[(v, 'modelo_con', 'consenso')], pr.loc[(v, 'modelo_con', 'Pinnacle_respaldo')]
    d3, d4 = pr.loc[(t, 'modelo_con', 'consenso')], pr.loc[(t, 'modelo_con', 'Pinnacle_respaldo')]
    bcon, bsin = bt.loc[(t, 'modelo_con', 'B365')], bt.loc[(t, 'modelo_sin', 'B365')]
    w_ref = weights[weights.branch.eq('con')]['con/LR_REF']
    V = {
        'n': f"{len(c['df']):,}".replace(',', '.'), 'ntest': f"{int(c['df'].Season.isin(TEST).sum()):,}".replace(',', '.'),
        'vcon': dec(ll(v, 'modelo_con')), 'vsin': dec(ll(v, 'modelo_sin')), 'vm': dec(ll(v, 'consenso')),
        'vpin': dec(ll(v, 'Pinnacle_respaldo')), 'vprior': dec(ll(v, 'prior')),
        'tcon': dec(ll(t, 'modelo_con')), 'tsin': dec(ll(t, 'modelo_sin')), 'tm': dec(ll(t, 'consenso')), 'tprior': dec(ll(t, 'prior')),
        'aucm': dec(m.loc[(v, 'consenso'), 'AUC_macro'], 3), 'auccon': dec(m.loc[(v, 'modelo_con'), 'AUC_macro'], 3),
        'ecem': dec(m.loc[(v, 'consenso'), 'ECE_masa']), 'ececon': dec(m.loc[(v, 'modelo_con'), 'ECE_masa']),
        'acccon': dec(100*m.loc[(t, 'modelo_con'), 'accuracy'], 1), 'accm': dec(100*m.loc[(t, 'consenso'), 'accuracy'], 1),
        'accsin': dec(100*m.loc[(t, 'modelo_sin'), 'accuracy'], 1),
        'H': dec(100*share[0], 1), 'D': dec(100*share[1], 1), 'A': dec(100*share[2], 1),
        'd1': dec(-d1.delta), 'd1lo': dec(d1.lo), 'd1hi': dec(d1.hi), 'd1s': d1.temporadas_mejora.split('/')[0],
        'd2': dec(-d2.delta), 'd3': dec(-d3.delta), 'd4': dec(-d4.delta),
        'gap': dec(ll(v, 'modelo_sin')-ll(v, 'consenso')),
        'corr': dec(sim.corr_logit, 2), 'fav': dec(100*sim.mismo_favorito, 1), 'frac': dec(100*sim.fraccion_ganancia_prior, 1),
        'abl0': dec(result.loc['sin/LR_historia', 'log_loss']), 'abl1': dec(result.loc['sin/LR_plantilla', 'log_loss']),
        'abl2': dec(result.loc['sin/LR', 'log_loss']),
        'ens': dec(-ens.loc['sin/mean', 'delta']), 'enslo': dec(ens.loc['sin/mean', 'lo']), 'enshi': dec(ens.loc['sin/mean', 'hi']),
        'wmin': f'{100*w_ref.min():.0f}', 'wmax': f'{100*w_ref.max():.0f}',
        'wins': str(int((bs.modelo_con < bs.consenso).sum())), 'nseas': str(len(bs)),
        'roicon': dec(100*bcon.ROI, 1), 'roilo': f'{100*bcon.lo:.0f}', 'roihi': f'{100*bcon.hi:.0f}', 'ncon': str(int(bcon.apuestas)),
        'roisin': dec(100*bsin.ROI, 1),
        'cov_bw': f"{100*c['df'].loc[c['df'].Season.eq(2425), ['BWH', 'BWD', 'BWA']].notna().all(axis=1).mean():.0f}",
        'cov_wh': f"{100*c['df'].loc[c['df'].Season.eq(2425), ['WHH', 'WHD', 'WHA']].notna().all(axis=1).mean():.0f}",
        'w2': dec(c['feedback'].set_index('candidate').loc['con/LR/w2', 'log_loss']-ll(v, 'modelo_con'), 3),
        'repo': r'\url{https://github.com/jcarlos-t/MLProject}',
        'na_stats': dec(E['na_stats'], 1), 'na_leagues': str(E['na_leagues']), 'na_pin': f"{E['na_pin']:.0f}", 'na_wh': f"{E['na_wh']:.0f}",
        'corr_casas': dec(E['corr_casas'], 3), 'out_goals': dec(E['out_goals'], 1), 'out_odds': dec(E['out_odds'], 1),
        'max_goals': str(E['max_goals']), 'max_odds': f"{E['max_odds']:.0f}",
        'pH_H': f"{E['pH_H']:.0f}", 'pH_A': f"{E['pH_A']:.0f}", 'xi_H': dec(E['xi_H'], 1), 'xi_A': dec(E['xi_A'], 1),
        'c_elo_mkt': dec(E['corr'].loc['ELO_DIF', 'MKT_logit'], 2), 'c_xi_mkt': dec(E['corr'].loc['XI_DIF', 'MKT_logit'], 2),
        'c_elo_pts': dec(E['corr'].loc['ELO_DIF', 'EW_Pts10_DIF'], 2), 'c_xi_sq': dec(E['corr'].loc['XI_DIF', 'SQ_DIF'], 2),
        'c_pd': dec(E['corr'].loc['MKT_logit', 'MKT_pD'], 2),
        'rec_H': f"{100*E['modelo_con_recall'][0]:.0f}", 'rec_A': f"{100*E['modelo_con_recall'][2]:.0f}",
        'rec_D': dec(100*E['modelo_con_recall'][1], 1), 'pred_D': dec(E['modelo_con_draws_pred'], 1),
        'draw_bal': dec(E['draw_bal'], 1), 'draw_unbal': dec(E['draw_unbal'], 1),
        'imp_mkt': dec(E['imp'][('con', 'Mercado')], 3), 'imp_hist_con': dec(E['imp'][('con', 'Historia')], 3),
        'imp_hist': dec(E['imp'][('sin', 'Historia')], 3), 'imp_sk': dec(E['imp'][('sin', 'Habilidades')], 3), 'imp_sq': dec(E['imp'][('sin', 'Plantilla')], 3),
        'nhist': str(len(c['sets']['history'])), 'nroster': str(len(c['sets']['roster'])), 'nskills': str(len(c['sets']['skills'])),
    }
    gap = training_gap(c['out']).brecha_extra
    V.update(gap_svm=dec(gap['con/SVM'], 3), gap_mlp=dec(gap['con/MLP'], 3), gap_rf=dec(gap['con/RF'], 3))
    dm = E['draws'].query('split == "validacion"').set_index(['branch', 'family'])
    V.update(f1b=dec(dm.loc[('con', 'base'), 'F1_macro'], 3), f1w=dec(dm.loc[('con', 'boost'), 'F1_macro'], 3),
             accb=dec(100*dm.loc[('con', 'base'), 'accuracy'], 1), accw=dec(100*dm.loc[('con', 'boost'), 'accuracy'], 1),
             recw=f"{100*dm.loc[('con', 'boost'), 'recall_D']:.0f}",
             f1sb=dec(dm.loc[('sin', 'base'), 'F1_macro'], 3), f1sw=dec(dm.loc[('sin', 'detector_threshold'), 'F1_macro'], 3),
             accsb=dec(100*dm.loc[('sin', 'base'), 'accuracy'], 1), accsw=dec(100*dm.loc[('sin', 'detector_threshold'), 'accuracy'], 1))
    dt = E['draws'].query('split == "test_revisado"').set_index(['branch', 'family'])
    V.update(draw_ll_base=dec(dm.loc[('sin','base'),'log_loss']),
        draw_ll_detector=dec(dm.loc[('sin','detector_threshold'),'log_loss']),
        draw_test_base=dec(dt.loc[('sin','base'),'log_loss']),
        draw_test_detector=dec(dt.loc[('sin','detector_threshold'),'log_loss']))
    body = r"""\begin{abstract}
¿Se puede predecir el resultado de un partido de fútbol mejor que las casas de apuestas? Este trabajo estima la probabilidad de victoria local, empate y victoria visitante en @@n@@ partidos de nueve ligas europeas, entre 2014/15 y 2025/26, con dos enfoques de igual importancia. El primero usa las cuotas de las casas como dato de entrada y pregunta si el aprendizaje automático puede mejorarlas. El segundo no mira las cuotas: construye la probabilidad solo con información deportiva (historial de resultados y valoraciones de jugadores del videojuego FIFA/FC) y mide qué tan cerca llega del mercado. Se comparan regresión logística, árboles potenciados, bosques aleatorios, redes neuronales, SVM y combinaciones de modelos, entrenando siempre con el pasado y evaluando en temporadas posteriores. La métrica principal es la pérdida logarítmica (log loss), que premia asignar probabilidades altas a lo que realmente ocurre. Con cuotas, una regresión logística sobre el precio mejora ligeramente al consenso de las casas (@@vcon@@ frente a @@vm@@ en validación y @@tcon@@ frente a @@tm@@ en test). Sin cuotas, un promedio de modelos obtiene @@vsin@@ y @@tsin@@: recupera el @@frac@@\,\% de la ventaja que tiene el mercado sobre una predicción ingenua, sin llegar a superarlo. Ninguno de los dos enfoques demuestra ganancias estables en apuestas.
\end{abstract}
\begin{IEEEkeywords}
fútbol, predicción probabilística, cuotas de apuestas, aprendizaje automático, validación temporal, calibración.
\end{IEEEkeywords}

\section{Introducción y formulación del problema}
\label{sec:intro}
Antes de cada partido, las casas de apuestas publican cuotas. Una cuota es, en el fondo, una probabilidad: si la victoria de un equipo paga 2,00 por cada sol apostado, la casa estima que ese equipo gana algo menos de la mitad de las veces. Como las casas arriesgan su propio dinero, sus cuotas suelen ser muy precisas y son una referencia difícil de superar.

Este proyecto estudia el problema desde dos lados. En el \textbf{enfoque con cuotas}, el modelo recibe el precio del mercado y la pregunta es si el aprendizaje automático puede corregirlo. En el \textbf{enfoque sin cuotas}, el modelo solo ve datos deportivos (cómo le ha ido a cada equipo y qué tan buenos son sus jugadores) y la pregunta es cuánto se acerca al mercado sin mirarlo; en la práctica funciona como un generador de cuotas propio. Ambos enfoques se evalúan con los mismos partidos, las mismas métricas y el mismo protocolo.

Nos apoyamos en dos trabajos previos. Tammouch et al.~\cite{tammouch2024} predicen partidos de la liga inglesa a partir de las cuotas y comparan métodos para seleccionar variables. Stübinger et al.~\cite{stubinger2020} usan las habilidades individuales de los jugadores y predicen la diferencia de goles. Tomamos ambas ideas, las extendemos a nueve ligas y las evaluamos sin usar nunca información del futuro. El código está disponible en @@repo@@.

\subsection{Objetivo y tarea de aprendizaje automático}
El objetivo es estimar, antes de cada partido, la probabilidad de cada uno de sus tres resultados posibles. Es una tarea de \textbf{aprendizaje supervisado} de \textbf{clasificación multiclase} con tres clases (victoria local, empate y victoria visitante), en la que interesa la probabilidad de cada clase y no solo la clase más probable. Cada observación es un partido; las entradas son variables conocidas antes de que empiece y la etiqueta es su resultado final.

\subsection{Preguntas de investigación}
Las dos primeras preguntas corresponden a los dos enfoques; las otras dos analizan ambos a la vez:
\begin{itemize}
\item \textbf{P1 (con cuotas):} ¿mejora el aprendizaje automático la probabilidad que ya dan las casas, tanto el consenso de varias casas como Pinnacle?
\item \textbf{P2 (sin cuotas):} ¿qué parte de la calidad del mercado se puede reproducir solo con datos deportivos, y cuánto aporta conocer la plantilla además del historial?
\item \textbf{P3:} ¿ayudan las técnicas para mejorar un modelo: reducir variables (MI o PCA), combinar modelos (ensembles) o dar más peso a los partidos sorpresivos?
\item \textbf{P4:} ¿se mantienen los resultados temporada tras temporada y permiten ganar dinero apostando?
\end{itemize}
Nuestras hipótesis son que el mercado será muy difícil de superar, que un modelo deportivo podrá acercarse sin igualarlo y que predecir mejor no garantiza ganar dinero.

\subsection{Cómo se mide una buena probabilidad}
El modelo no entrega un ganador, sino tres probabilidades que suman uno: victoria local (H), empate (D) y victoria visitante (A). Para saber si son buenas usamos como métrica principal la pérdida logarítmica:
\begin{equation}
\mathrm{LL}=-\frac{1}{N}\sum_{i=1}^{N}\log \hat p_{i,y_i},
\label{eq:ll}
\end{equation}
donde $\hat p_{i,y_i}$ es la probabilidad que el modelo asignó al resultado que realmente ocurrió en el partido $i$. Si el modelo dio 60\,\% al resultado correcto, la penalización es 0,51; si dio 25\,\%, es 1,39; y si dio solo 5\,\%, sube a 3,00. Así se castiga mucho estar muy seguro y equivocarse. Un valor menor es mejor. Como punto de partida, un modelo ingenuo que siempre predice las frecuencias históricas (al que llamamos \emph{prior}) obtiene @@vprior@@. Las diferencias entre buenos modelos son pequeñas, del orden de milésimas, porque el fútbol tiene mucho azar.

Como apoyo se usan otras medidas. El \textbf{AUC} indica si el modelo ordena bien los partidos, por ejemplo si da más probabilidad de victoria local a los que luego terminan así; 0,5 es azar y 1 es perfecto. El \textbf{ECE} (error de calibración) compara lo que se predice con lo que pasa: si de todos los partidos donde se dijo 30\,\% de empate terminan empatados cerca del 30\,\%, el modelo está bien calibrado. También se reporta el \textbf{accuracy}, el porcentaje de partidos donde el resultado más probable fue el correcto, y el RPS~\cite{epstein1969}, que tiene en cuenta que el empate está entre las dos victorias. Preferimos el log loss al accuracy porque el accuracy ignora la confianza: un modelo que da 40\,\% al favorito y otro que le da 90\,\% obtienen el mismo acierto, aunque el segundo sea mucho más arriesgado.

\section{Datos y análisis exploratorio}
\label{sec:datos}
\subsection{Fuentes y variables}
Los resultados y las cuotas provienen de Football-Data~\cite{footballdata} y cubren las primeras divisiones de Inglaterra, Alemania, España, Italia, Francia, Países Bajos, Portugal, Turquía y Bélgica. Hay 36.150 partidos de 296 equipos; 31 no tienen ninguna cuota y solo se usan para calcular el historial de los equipos, de modo que quedan @@n@@ partidos para comparar modelos. La temporada 2013/14 sirve únicamente para iniciar el historial. Las valoraciones de jugadores salen de cuatro bases públicas del videojuego FIFA/FC~\cite{kagglefifa}, que puntúan de 1 a 99 a cada jugador en ritmo, tiro, pase, defensa y otras habilidades.

El Cuadro~\ref{tab:vars} resume qué información hay por partido. Es importante distinguir dos tipos: lo que se sabe \emph{antes} del partido (cuotas, valoraciones, historial) y lo que solo se sabe \emph{después} (goles, tiros, tarjetas). Estas últimas nunca se usan para predecir el mismo partido; solo sirven para construir el historial de los partidos siguientes. Casi todas las variables son numéricas: continuas, como cuotas, valoraciones y promedios, o discretas, como goles, tiros y tarjetas. Las categóricas son la liga, la temporada, la fecha y los nombres de los equipos, que sirven para ordenar y agrupar los partidos, pero no entran directamente al modelo. La variable objetivo es categórica, con tres valores: H, D y A.
"""
    body += latex_table(['Bloque', 'N.º', 'Ejemplos'], [
        ['Identificación', '5', 'liga, temporada, fecha, equipos'],
        ['Resultado (objetivo)', '3', 'H/D/A y goles de cada equipo'],
        ['Estadísticas del partido', '12', 'tiros, córners, faltas, tarjetas (solo para el historial)'],
        ['Cuotas', 'Hasta 4 casas', 'Bet365, Bet\\&Win, Pinnacle y William Hill'],
        ['Historia (construida)', '@@nhist@@', 'Elo, forma, medias recientes, descanso'],
        ['Plantilla', '@@nroster@@', 'valoración media del once y de la plantilla'],
        ['Habilidades', '@@nskills@@', 'definición, regate, pase, defensa, portero']],
        'tab:vars', 'Información disponible por partido.', align=r'lcp{0.42\columnwidth}')
    body += r"""\subsection{Distribución del resultado}
La Fig.~\ref{fig:datos} resume los datos. La victoria local es el resultado más común (@@H@@\,\%), seguida de la victoria visitante (@@A@@\,\%) y el empate (@@D@@\,\%). Las clases no están muy desbalanceadas, por lo que no se rebalancean: estas proporciones son justamente el \emph{prior}, la predicción más ingenua posible. A la derecha se ve que casi todos los partidos tienen valoraciones de ambos equipos y que las cuotas de Pinnacle se dejan de usar en la última temporada.
"""
    body += latex_figure('inf_datos', 'fig:datos', 'Izquierda: frecuencia de cada resultado. Derecha: fracción de partidos con valoraciones de plantilla disponibles antes del partido y con cuotas fiables de Pinnacle.')
    body += r"""\subsection{Calidad de los datos}
\textbf{Datos faltantes.} La Fig.~\ref{fig:nulos} muestra el porcentaje de partidos sin dato en cada columna. Las cuatro casas consideradas (en verde) tienen cobertura variable; otras casas, como Ladbrokes o Interwetten, faltan en muchos partidos y se descartan. Los faltantes no son aleatorios: dependen de la casa y de la temporada. Pinnacle falta en el @@na_pin@@\,\% de 2025/26 y William Hill en el @@na_wh@@\,\%, respectivamente. William Hill dejó de publicarse; Pinnacle se excluye desde el 23/07/2025 por la advertencia de calidad de la fuente, incluso cuando tiene precios. Por eso las cuotas \emph{no se imputan}: inventar un precio que nunca existió sería engañoso. Las estadísticas del partido faltan en el @@na_stats@@\,\% de los casos, concentradas en @@na_leagues@@ ligas cuyas primeras temporadas no las publican; el modelo las recibe como promedios del pasado, con su indicador de dato faltante. El árbitro y la hora se descartan porque faltan en la mayoría de partidos.
"""
    body += latex_figure('inf_nulos', 'fig:nulos', 'Porcentaje de partidos sin dato por columna (2014/15--2025/26). En verde, las casas usadas para el consenso; la línea roja marca el 20\\,\\%.')
    body += r"""\textbf{Valores extremos.} La Fig.~\ref{fig:extremos} muestra la dispersión de goles, tiros y cuotas. Con la regla del rango intercuartílico (IQR), que marca como atípico lo que está muy lejos de la mitad central de los datos, aparecen pocos casos en goles y tiros (a lo sumo el @@out_goals@@\,\%) y más en las cuotas (@@out_odds@@\,\% en la del visitante), con valores de hasta @@max_goals@@ goles y cuotas de @@max_odds@@. Estos extremos pueden corresponder a goleadas y partidos muy desiguales; su magnitud por sí sola no demuestra que sean errores. Por eso no se eliminan ni se recortan.
"""
    body += latex_figure('inf_extremos', 'fig:extremos', 'Diagramas de caja de goles, tiros a puerta y cuotas de Bet365. Los puntos son valores extremos según la regla IQR.')
    body += r"""\subsection{Relaciones entre variables}
Las cuotas anticipan el resultado. En la Fig.~\ref{fig:cuotasres} (izquierda), la probabilidad de victoria local según las casas es mayor cuando el local termina ganando: la mediana es @@pH_H@@\,\% en esos partidos y @@pH_A@@\,\% cuando gana el visitante. La valoración del once titular (derecha) muestra el mismo patrón: cuando gana el local, su once es en la mediana @@xi_H@@ puntos mejor que el rival; cuando gana el visitante, es @@xi_A@@. Ambas fuentes de información apuntan en la misma dirección, aunque con mucha superposición entre las cajas: el fútbol tiene mucho azar.
"""
    body += latex_figure('inf_cuotas_resultado', 'fig:cuotasres', 'Probabilidad de victoria local implícita en las cuotas (izquierda) y diferencia de valoración del once titular (derecha) según el resultado real. Los valores positivos favorecen al local.')
    body += r"""La Fig.~\ref{fig:corr} muestra la correlación entre las variables principales, todas expresadas como diferencia entre local y visitante. Hay tres observaciones. Primero, las variables deportivas están muy relacionadas entre sí: el Elo y los puntos recientes tienen correlación @@c_elo_pts@@, y el once titular y la plantilla completa, @@c_xi_sq@@; por eso muchas son redundantes y conviene regularizar o reducir variables. Segundo, el favoritismo según las cuotas se correlaciona fuerte con el Elo (@@c_elo_mkt@@) y con el once titular (@@c_xi_mkt@@): el mercado ya incorpora la información deportiva, lo que anticipa que añadirla al precio aportará poco. Tercero, el margen de la casa muestra baja correlación lineal con estas variables; esto no descarta asociaciones no lineales ni demuestra ausencia de información sobre el resultado. Además, las probabilidades de las cuatro casas casi coinciden (correlación mínima de @@corr_casas@@), por lo que conviene resumirlas en un solo consenso.
"""
    body += latex_figure('inf_correlacion', 'fig:corr', 'Correlación de Pearson entre variables clave (diferencia local menos visitante). Rojo indica relación positiva y azul negativa.')
    body += r"""\subsection{Sesgos y fuga de datos}
Hay \emph{fuga de datos} cuando el modelo usa, sin querer, información que no estaría disponible al momento de predecir; el resultado es un modelo que parece muy bueno en los experimentos y falla en la práctica. Identificamos estos riesgos y la forma de evitarlos:
\begin{itemize}
\item \textbf{Estadísticas del propio partido.} Goles, tiros o tarjetas solo se conocen al final; se usan únicamente como promedios de partidos anteriores.
\item \textbf{Valoraciones con fecha.} Cada partido usa la versión del videojuego publicada \emph{antes} de esa fecha. Si se usara la versión de fin de temporada, el modelo vería información del futuro, por ejemplo la mejora de un jugador que ocurrió después. El costo es que algunas valoraciones son antiguas: en 2024/25 tienen en promedio más de un año.
\item \textbf{Cuotas.} Las cuotas de cierre se publican justo antes del inicio y no se usan como variables. Se usan cuotas previas, capturadas unos días antes y no todas a la misma hora.
\item \textbf{Partición y preprocesamiento.} La partición es cronológica, el test no interviene en ninguna decisión y el preprocesamiento se aprende solo con datos de entrenamiento (Sección~\ref{sec:prep}).
\end{itemize}
También hay sesgos que no se pueden eliminar, solo medir y documentar:
\begin{itemize}
\item \textbf{Cobertura desigual.} Cuatro ligas no tienen estadísticas de partido en sus primeras temporadas, y la fuente advierte que las cuotas de Pinnacle dejan de ser fiables desde el 23 de julio de 2025; desde esa fecha se usan las demás casas.
\item \textbf{Valoraciones subjetivas.} Las valoraciones del videojuego son la opinión de sus analistas, no una medición del rendimiento real, y pueden ser menos precisas en clubes menos conocidos.
\item \textbf{Cambios en el tiempo.} El fútbol cambia entre temporadas; por ejemplo, 2020/21 se jugó mayormente sin público. Por eso se evalúa siempre en temporadas posteriores al entrenamiento.
\end{itemize}
Por último, no se eliminan extremos ni se imputan cuotas, por las razones descritas arriba.

\section{Preprocesamiento e ingeniería de variables}
\label{sec:prep}
\subsection{Variables construidas sin mirar el futuro}
Todas las variables se calculan en orden cronológico: para predecir un partido solo se usan partidos de fechas anteriores. Para cada equipo se calculan cinco grupos de variables:
\begin{itemize}
\item \textbf{Fuerza Elo:} un puntaje que sube al ganar y baja al perder, más o menos según la fuerza del rival.
\item \textbf{Forma:} puntos y goles de los últimos cinco partidos, y promedios que dan más peso a los partidos recientes.
\item \textbf{Contexto:} días de descanso, cantidad de partidos recientes y tendencia del equipo a empatar.
\item \textbf{Plantilla:} valoración media del once titular según el videojuego.
\item \textbf{Habilidades:} ataque, pase, defensa y portero de los mejores jugadores del equipo.
\end{itemize}
El modelo recibe la diferencia entre local y visitante en cada variable, porque lo que importa es qué equipo es mejor y por cuánto.

\subsection{De cuota a probabilidad}
Una cuota $o$ se convierte en probabilidad con $1/o$. Por ejemplo, cuotas de 2,10, 3,40 y 3,60 dan 0,476, 0,294 y 0,278, que suman 1,048: ese 4,8\,\% extra es el margen de ganancia de la casa. Para eliminarlo se divide cada valor entre la suma,
\begin{equation}
p_j=\frac{1/o_j}{\sum_k 1/o_k},
\label{eq:odds}
\end{equation}
y quedan 45,4\,\%, 28,1\,\% y 26,5\,\%. El \emph{consenso} es el promedio de estas probabilidades entre las casas con cuotas completas.

\subsection{Limpieza dentro de cada entrenamiento}
Los valores faltantes se rellenan con la mediana y se marca que faltaban; las variables se llevan a la misma escala y se elimina una de cada par de variables casi idénticas (correlación mayor a 0,95). Todas estas operaciones se aprenden solo con los datos de entrenamiento y después se aplican a los datos de evaluación. Si se calcularan con todos los datos, la mediana o la escala ya incluirían información de los partidos que se quieren predecir, un error conocido como fuga de datos~\cite{zheng2018}. Por la misma razón, el modelo sin cuotas nunca recibe precios, ni como variable ni de forma indirecta.

\subsection{Codificación}
No hay variables categóricas que codificar como entrada. La liga y los equipos no entran directamente, sino a través de variables numéricas como la diferencia de Elo. Así se evita crear cientos de columnas \emph{one-hot} para 296 equipos y el modelo puede predecir partidos de equipos recién ascendidos, de los que tiene poca historia; esos equipos se marcan con una variable binaria (0/1). La variable objetivo se codifica como 0 (local), 1 (empate) y 2 (visitante).

\section{Metodología y modelos}
\label{sec:modelos}
\subsection{Baselines}
Para saber si un modelo aporta algo hay que compararlo con algo más simple. Usamos dos \emph{baselines}. El \textbf{baseline ingenuo} es el prior: asigna a todos los partidos las frecuencias de local, empate y visitante del entrenamiento, sin mirar quién juega. Cualquier modelo útil debe superarlo. El \textbf{baseline fuerte} es el mercado: la probabilidad directa del consenso de casas y la de Pinnacle, una casa conocida por cobrar márgenes bajos, ambas sin ningún aprendizaje. Es la referencia que de verdad importa: el modelo con cuotas debe superarla para justificar el aprendizaje, y el modelo sin cuotas se mide por cuánto se le acerca.

\subsection{Enfoque con cuotas}
Además de los baselines, se entrenan modelos que reciben el precio: una regresión logística (LR) con solo dos números del precio de referencia (qué tan favorito es el local frente al visitante y la probabilidad de empate), una LR con el consenso y su margen, y modelos que combinan el precio con las variables deportivas. Compararlos con el baseline del mercado permite separar dos efectos: elegir una buena casa y aprender a corregirla.

\subsection{Enfoque sin cuotas}
El segundo enfoque solo recibe las variables deportivas. Se prueban dos estrategias: clasificar directamente el resultado (H, D o A) y, siguiendo a Stübinger et al.~\cite{stubinger2020}, predecir la diferencia de goles para luego convertirla en las tres probabilidades.

\subsection{Modelos comparados}
En ambos enfoques se prueban los mismos seis modelos. La \textbf{LR} combina las variables de forma lineal y es fácil de interpretar. \textbf{HGB} y \textbf{XGBoost} suman muchos árboles de decisión pequeños, cada uno corrigiendo los errores del anterior. \textbf{Random Forest} promedia muchos árboles entrenados por separado. La \textbf{red neuronal} (MLP) aprende relaciones no lineales con capas de neuronas. La \textbf{SVM} busca la frontera que mejor separa las clases.

Además se prueban tres formas de combinar modelos (\emph{ensembles}): el promedio simple de sus probabilidades, un promedio ponderado con pesos aprendidos en temporadas anteriores y \emph{stacking}, donde una LR aprende a combinar las predicciones de los demás. Para reducir variables se prueban la selección por información mutua (MI), que conserva las 5 o 10 variables más relacionadas con el resultado, y el análisis de componentes principales (PCA), que resume muchas variables en pocas combinaciones~\cite{tammouch2024,james2023}.

El Cuadro~\ref{tab:hiper} muestra los hiperparámetros que se buscan en cada modelo. Las rejillas son pequeñas a propósito (tres opciones por modelo) y se fijaron antes de ver resultados. Todas controlan el mismo riesgo, el sobreajuste, que es grande en datos con tanto azar. En la LR, $C$ regula cuánto se castigan los coeficientes grandes: un valor pequeño da un modelo más simple. En HGB y XGBoost se limitan los árboles a 1--3 niveles y se usa una tasa de aprendizaje baja (0,05), para que cada árbol corrija poco a poco. En Random Forest, exigir muchos partidos por hoja evita reglas basadas en pocos casos. En la red neuronal se usan pocas neuronas y una penalización \emph{alpha} alta, y se promedian tres inicializaciones. En la SVM, valores bajos de $C$ y \emph{gamma} producen fronteras suaves; además se entrena con los 9.000 partidos más recientes, porque su costo crece rápido con el tamaño, y sus salidas se convierten en probabilidades con una calibración aprendida en el último 20\,\% del entrenamiento.
"""
    body += latex_table(['Modelo', 'Hiperparámetro', 'Valores probados'], [
        ['LR', '$C$', '0,001; 0,05; 1'],
        ['HGB', 'profundidad / iteraciones', '1/80; 2/100; 3/150'],
        ['XGBoost', 'profundidad / árboles', '1/100; 2/150; 3/200'],
        ['Random Forest', 'mín. partidos por hoja', '30; 100; 200'],
        ['Red neuronal', 'neuronas / alpha', '16/1; 32/1; 16/10'],
        ['SVM', '$C$ / gamma', '0,1/0,001; 1/0,001; 1/0,01']],
        'tab:hiper', 'Rejillas de hiperparámetros. HGB y XGBoost usan tasa de aprendizaje 0,05; Random Forest, 150 árboles.', align='lll', wide=True)
    body += r"""
\subsection{Más peso a los partidos sorpresivos}
También probamos un enfoque centrado en los datos: dar más importancia durante el entrenamiento a los partidos donde el favorito de las casas no ganó, con peso 2, 3 o 5, o con un peso que crece cuanto más inesperado fue el resultado. Otra variante rebalancea la muestra para que los aciertos y los fallos del favorito pesen igual (50/50). La idea es que el modelo aprenda de las sorpresas, que es justamente donde el mercado se equivoca.

\section{Diseño experimental y evaluación}
\label{sec:diseno}
Como el fútbol cambia con el tiempo, los datos no se mezclan al azar: siempre se entrena con el pasado y se evalúa en el futuro~\cite{james2023}. Las temporadas 2014/15 a 2023/24 se usan para desarrollar los modelos y las temporadas 2024/25 y 2025/26 (@@ntest@@ partidos) quedan reservadas como \textbf{test}.

Dentro del desarrollo se usa una \textbf{validación cruzada temporal} con ventana que crece (\emph{expanding window}). No se usa la validación cruzada aleatoria (\emph{k-fold}) porque mezclaría partidos futuros en el entrenamiento. Para evaluar 2018/19, el modelo se entrena con 2014/15 a 2017/18; para evaluar 2019/20, con 2014/15 a 2018/19; y así hasta 2023/24. Estas seis temporadas forman la \textbf{validación}. Los hiperparámetros, como cuánta regularización usa la LR o cuántos árboles tiene un bosque, se eligen en cada paso por búsqueda en rejilla (Cuadro~\ref{tab:hiper}) usando las dos temporadas inmediatamente anteriores, también sin mirar el futuro.

En cada enfoque se elige el modelo con menor log loss en validación. Si dos modelos difieren en menos de 0,0005, se prefiere el más simple. Con esa elección ya fija, los modelos se reentrenan con todas las temporadas de desarrollo y predicen el test sin ningún ajuste adicional.

\textbf{Métricas.} El log loss es la métrica de selección porque mide directamente la calidad de la probabilidad y es una regla de puntuación \emph{propia}: un modelo obtiene su mejor valor esperado solo si dice las probabilidades que realmente cree, de modo que no se puede mejorar exagerando ni suavizando. El accuracy y el F1 solo miran la clase elegida e ignoran la confianza; el AUC mide la ordenación y el ECE la calibración. Por eso se reportan como complemento, no para elegir modelos.

Para saber si una diferencia es real o fruto del azar se usa \emph{bootstrap}: se toman 5.000 muestras al azar de semanas completas de partidos y en cada una se recalcula la diferencia de log loss. Con ellas se obtiene un intervalo de confianza del 95\,\%; si no incluye el cero, la diferencia es consistente. Se remuestrean semanas y no partidos sueltos porque los partidos de una misma jornada comparten condiciones. Los contrastes de validación usan las mismas predicciones con las que se eligió el modelo, así que pueden ser algo optimistas y se leen como diagnósticos de desarrollo. Como se hacen varias comparaciones a la vez, los valores $p$ se corrigen con el método de Benjamini y Hochberg~\cite{benjamini1995}.

\section{Resultados y análisis}
\label{sec:resultados}
\subsection{Con cuotas: ¿se puede mejorar al mercado? (P1)}
El Cuadro~\ref{tab:familias} muestra el log loss de validación de todos los modelos. Con cuotas, el mejor es la LR que solo recibe los dos números del precio (@@vcon@@). Los modelos que además reciben variables deportivas no la superan, y tampoco la red neuronal ni la SVM, que es la peor de todas. Para saber si la causa es el sobreajuste comparamos el error de entrenamiento con el de validación en dos temporadas, tomando como referencia la LR de solo precio. La SVM casi no muestra diferencia (@@gap_svm@@): su error ya es alto en el entrenamiento, así que su problema no es el sobreajuste. La red neuronal y el bosque aleatorio sí se ajustan más al entrenamiento (@@gap_mlp@@ y @@gap_rf@@), aunque el bosque valida bien.
"""
    rows = [['LR solo precio', dec(result.loc['con/LR_REF', 'log_loss']), '--'],
            ['LR consenso y margen', dec(result.loc['con/LR_consenso', 'log_loss']), '--']]
    for name, label in [('LR', 'LR (todas las variables)'), ('HGB', 'HGB'), ('XGB', 'XGBoost'), ('RF', 'Random Forest'), ('MLP', 'Red neuronal'), ('SVM', 'SVM'),
                        ('mean', 'Ens.: promedio'), ('weighted', 'Ens.: ponderado'), ('stack', 'Ens.: stacking')]:
        rows.append([label, dec(result.loc['con/'+name, 'log_loss']), dec(result.loc['sin/'+name, 'log_loss'])])
    rows.append(['Diferencia de goles', '--', dec(result.loc['sin/GD_ALL', 'log_loss'])])
    body += latex_table(['Modelo', 'Con cuotas', 'Sin cuotas'], rows, 'tab:familias',
        'Log loss de validación (seis temporadas) de cada modelo. Menor es mejor.')
    body += r"""El Cuadro~\ref{tab:final} compara los modelos elegidos con los dos baselines, el prior y el mercado, en validación y en test. La LR con cuotas obtiene el menor log loss en ambos periodos. La mejora frente al consenso parece pequeña, pero es consistente: el Cuadro~\ref{tab:contrastes} muestra que en validación su intervalo de confianza no incluye el cero y que el modelo mejora en @@d1s@@ de las 6 temporadas; en test ocurre lo mismo. Frente a la referencia Pinnacle con respaldo del consenso, la mejora es menor y en validación no es concluyente.
"""
    rows = [[LABEL[k], dec(ll(v, k)), dec(ll(t, k)), dec(m.loc[(t, k), 'AUC_macro'], 3), dec(100*m.loc[(t, k), 'accuracy'], 1)]
            for k in ['prior', 'consenso', 'Pinnacle_respaldo', 'modelo_con', 'modelo_sin']]
    body += latex_table(['', 'LL val.', 'LL test', 'AUC', r'Acc.\,\%'], rows, 'tab:final',
        'Modelos elegidos frente a las referencias. AUC y accuracy en test.')
    rows = []
    for a, b, lab in [('modelo_con', 'consenso', 'Con cuotas vs. consenso'), ('modelo_con', 'Pinnacle_respaldo', 'Con cuotas vs. ref. mixta'),
                      ('modelo_sin', 'consenso', 'Sin cuotas vs. consenso')]:
        for i, (split, name) in enumerate(((v, 'Val.'), (t, 'Test'))):
            x = pr.loc[(split, a, b)]
            rows.append([lab if i == 0 else '', name, dec(x.delta), f'[{dec(x.lo)}; {dec(x.hi)}]'])
    body += latex_table(['Comparación', '', r'$\Delta$ LL', r'IC 95\,\%'], rows, 'tab:contrastes',
        'Diferencia de log loss e intervalo del 95\\,\\%. Un valor negativo favorece al primer modelo. La referencia mixta usa Pinnacle fiable y, en su ausencia, consenso; en 2025/26 usa solo consenso.', align='llrr')
    body += r"""¿De dónde viene la mejora? El AUC del modelo (@@auccon@@) es prácticamente igual al del consenso (@@aucm@@): su capacidad agregada de ordenación es similar, y por eso el accuracy también es casi igual. Lo que cambia es la calibración: el ECE baja de @@ecem@@ a @@ececon@@. La Fig.~\ref{fig:cal} lo muestra: los puntos de un modelo bien calibrado caen sobre la diagonal, es decir, cuando dice 40\,\% el resultado ocurre cerca del 40\,\% de las veces. En resumen, el modelo con cuotas no descubre información nueva; ajusta ligeramente las probabilidades del mercado para que coincidan mejor con las frecuencias reales.
"""
    body += latex_figure('inf_calibracion', 'fig:cal', 'Calibración en validación: frecuencia real frente a probabilidad predicha, en diez grupos de partidos. La línea punteada es la calibración perfecta.')
    body += r"""\subsection{Sin cuotas: ¿cuánto se acerca al mercado? (P2)}
Sin cuotas, el modelo elegido es el promedio de ocho modelos, con log loss @@vsin@@ en validación y @@tsin@@ en test (Cuadros~\ref{tab:familias} y~\ref{tab:final}). Es mejor que cualquier modelo individual, pero queda claramente detrás del mercado, @@gap@@ por encima en validación. El promedio de regresores de diferencia de goles supera a varios clasificadores individuales, pero no a los mejores clasificadores ni al ensemble elegido.

La Fig.~\ref{fig:ablacion} muestra qué información aporta cada grupo de variables. Con solo el historial de resultados, la LR obtiene @@abl0@@; al añadir la valoración de la plantilla baja a @@abl1@@. Añadir las habilidades individuales no mejora más (@@abl2@@): este resultado es compatible con redundancia respecto a la valoración general, sin demostrar que esa sea la única causa.
"""
    body += latex_figure('inf_ablacion', 'fig:ablacion', 'Log loss de validación de la LR sin cuotas al añadir grupos de variables. Menor es mejor.')
    body += r"""Para ver si el modelo sin cuotas funciona como un generador de cuotas, la Fig.~\ref{fig:predictor} compara sus probabilidades con las del mercado partido a partido. Los puntos se concentran cerca de la diagonal: la correlación es @@corr@@ y ambos eligen al mismo favorito en el @@fav@@\,\% de los partidos. Otra forma de verlo es medir la distancia: el prior obtiene @@vprior@@ y el mercado @@vm@@; el modelo sin cuotas, con @@vsin@@, recorre el @@frac@@\,\% de ese camino. Es decir, con información pública y sin mirar precios se reproduce la mayor parte del trabajo de las casas. El tramo restante probablemente depende de información que no tenemos, como lesiones, alineaciones confirmadas y noticias de último momento. Estas probabilidades se pueden convertir en cuotas propias con $1/p$, sin el margen de la casa.
"""
    body += latex_figure('inf_predictor', 'fig:predictor', 'Probabilidad del modelo sin cuotas frente a la del mercado en cada partido de validación. Los colores más oscuros indican más partidos; la diagonal es la igualdad.')
    body += r"""\subsection{¿Qué variables pesan más?}
Para saber qué información usa cada modelo, desordenamos al azar un grupo de variables (por ejemplo, todas las de historia) y medimos cuánto empeora el log loss. Si empeora mucho, el modelo dependía de ese grupo. La Fig.~\ref{fig:importancia} muestra el resultado para la LR con todas las variables. Con cuotas, el precio domina por completo: desordenarlo sube el log loss en @@imp_mkt@@, mientras que la historia, el siguiente grupo, solo lo sube en @@imp_hist_con@@. Sin cuotas, lo más importante es la historia del equipo (@@imp_hist@@), seguida de las habilidades (@@imp_sk@@). La plantilla aparece baja (@@imp_sq@@) no porque no sirva (la ablación de la Fig.~\ref{fig:ablacion} muestra que sí ayuda), sino porque las habilidades repiten casi la misma información: al desordenar la plantilla, el modelo la recupera de las habilidades.
"""
    body += latex_figure('inf_importancia', 'fig:importancia', 'Importancia por permutación de grupos en la LR con todas las variables: aumento medio del log loss al desordenar cada grupo, con intervalo del 95\\,\\% entre seis temporadas.')
    body += r"""\subsection{¿Ayudan las técnicas para mejorar un modelo? (P3)}
La Fig.~\ref{fig:mejoras} compara cada técnica con la LR que usa todas las variables; las barras negativas indican mejora. Con cuotas, la selección MI mejora un poco porque descarta variables deportivas redundantes, pero no alcanza a la LR de solo precio; PCA empeora porque mezcla las variables y diluye la señal del precio. Sin cuotas, reducir variables empeora en casi todos los casos: cada grupo de variables deportivas aporta algo.

Los ensembles dependen del enfoque. Sin cuotas, el promedio mejora a la mejor LR en @@ens@@ (intervalo [@@enslo@@; @@enshi@@]), un resultado compatible con la compensación de errores entre modelos. Con cuotas no ayudan: el ponderado asigna a la LR de solo precio entre el @@wmin@@\,\% y el @@wmax@@\,\% del peso, así que termina siendo casi la misma LR. El stacking obtiene mayor pérdida en ambos enfoques; este resultado no identifica por sí solo la causa.
"""
    body += latex_figure('inf_mejoras', 'fig:mejoras', 'Cambio de log loss de validación frente a la LR con todas las variables. Valores negativos (barras de color) indican mejora.')
    body += r"""La Fig.~\ref{fig:pesos} muestra el efecto de dar más peso a los partidos sorpresivos. Las variantes probadas empeoran el log loss agregado; esto no implica empeoramiento en cada temporada: con peso 2, el modelo con cuotas pierde @@w2@@. El modelo aprende a predecir más empates y sorpresas, pero estas siguen siendo minoría; al exagerar su importancia, se vuelve menos confiable en la gran mayoría de partidos normales.
"""
    body += latex_figure('inf_pesos', 'fig:pesos', 'Cambio de log loss de validación al dar más peso o remuestrear los partidos donde el favorito del mercado no ganó.')
    body += r"""\subsection{¿Por qué casi nunca se predice empate?}
La Fig.~\ref{fig:confusion} muestra la matriz de confusión: para cada resultado real, qué porcentaje de veces el resultado más probable según el modelo fue local, empate o visitante. El modelo con cuotas acierta el @@rec_H@@\,\% de las victorias locales y el @@rec_A@@\,\% de las visitantes, pero solo el @@rec_D@@\,\% de los empates: elige empate en apenas el @@pred_D@@\,\% de los partidos, aunque ocurren en uno de cada cuatro. No es un error del modelo. Incluso entre equipos de valoración parecida (diferencia de 2 puntos o menos) el empate ocurre el @@draw_bal@@\,\% de las veces, frente al @@draw_unbal@@\,\% en el resto: es más frecuente, pero casi nunca es el resultado \emph{más} probable. Un modelo bien calibrado le asigna entre 25 y 30\,\% al empate, y eso basta para que alguna de las victorias sea más probable.
"""
    body += latex_figure('inf_confusion', 'fig:confusion', 'Matriz de confusión en validación (filas normalizadas): porcentaje de cada resultado real según la clase más probable del modelo.')
    body += r"""Probamos forzar más empates con reglas de decisión: por ejemplo, multiplicar la probabilidad de empate por 1,5 solo al elegir la clase. Con cuotas, el F1 macro (el promedio del balance entre precisión y recall de cada clase) sube de @@f1b@@ a @@f1w@@ y el recall del empate llega al @@recw@@\,\%, pero el accuracy baja de @@accb@@\,\% a @@accw@@\,\%. Sin cuotas pasa lo mismo: el F1 sube de @@f1sb@@ a @@f1sw@@ y el accuracy baja de @@accsb@@\,\% a @@accsw@@\,\%. Con cuotas, el multiplicador solo cambia la decisión y conserva el log loss. Sin cuotas, el detector reemplaza la probabilidad de empate: el log loss pasa de @@draw_ll_base@@ a @@draw_ll_detector@@ en validación y de @@draw_test_base@@ a @@draw_test_detector@@ en test. La mejora de F1 no implica mejores probabilidades. Se ganan aciertos de empate a cambio de perder más aciertos de victorias. Como nuestro objetivo es estimar probabilidades y no etiquetas, mantenemos el log loss como métrica principal.

\subsection{Estabilidad y rentabilidad (P4)}
La Fig.~\ref{fig:temporadas} muestra el log loss por temporada. Las tres curvas suben y bajan juntas: hay temporadas más difíciles para todos. La más clara es 2020/21, jugada mayormente sin público, en la que la victoria local bajó al 40\,\% frente al 45\,\% habitual. El modelo con cuotas queda por debajo del consenso en @@wins@@ de @@nseas@@ temporadas, y el modelo sin cuotas siempre queda por encima del mercado. Esta regularidad indica que las conclusiones no dependen de una sola temporada~\cite{siddiqi2017}.
"""
    body += latex_figure('inf_temporadas', 'fig:temporadas', 'Log loss por temporada; la zona sombreada es el test. Menor es mejor.')
    body += r"""Por último, simulamos apuestas con una regla fija, decidida antes de ver los resultados: apostar una unidad al resultado cuya ganancia esperada según el modelo supere el 2\,\% ($p_j\,o_j-1>0{,}02$). La Fig.~\ref{fig:bt} muestra el retorno por unidad apostada (ROI) con su intervalo. El modelo con cuotas gana @@roicon@@\,\% en test con Bet365 (@@ncon@@ apuestas), pero su intervalo va de @@roilo@@\,\% a +@@roihi@@\,\%: incluye pérdidas, así que no se puede afirmar que gane dinero de forma estable. El modelo sin cuotas pierde (@@roisin@@\,\%), porque cuando difiere del mercado suele ser el mercado quien tiene razón. El consenso directo contra Bet365 obtiene un ROI alto en validación y negativo en test; el diagnóstico por apuesta muestra ganancias concentradas en resultados de alta cuota. Sin capturas sincronizadas no se puede confirmar su ejecución. Predecir un poco mejor no basta para superar el margen que cobra la casa.
"""
    body += latex_figure('inf_backtest', 'fig:bt', 'Retorno por unidad apostada (punto) e intervalo del 95\\,\\% (línea) con la regla fija. Cero significa no ganar ni perder.')
    body += r"""
\section{Discusión, limitaciones y ética}
\label{sec:discusion}
En estas ligas y periodos, el mercado constituye una referencia predictiva difícil de superar. El experimento no establece su eficiencia general. Con cuotas, el aprendizaje automático solo logra recalibrar el precio; sin cuotas, los datos deportivos públicos se acercan, pero no lo alcanzan. Tiene sentido: las casas combinan toda la información disponible, incluida la que nosotros no tenemos, y ajustan sus precios con el dinero de miles de apostadores.

Otra lección es que los modelos simples resistieron. Con cuotas, una LR de dos variables ganó a la red neuronal y a la SVM; sin cuotas, un promedio simple ganó al stacking. Con estas rejillas, la complejidad adicional no produjo mejores predicciones. El sobreajuste explica solo una parte: aparece en la red neuronal y en el bosque aleatorio, pero no en la SVM. También comprobamos que un accuracy parecido puede esconder diferencias de calibración: por eso conviene evaluar probabilidades con log loss y no solo con aciertos.

\textbf{Limitaciones.} No tenemos alineaciones reales ni lesiones: las plantillas son las del videojuego y algunas valoraciones tienen más de un año. El historial solo incluye partidos de liga, no copas ni torneos europeos. Las cuotas no tienen la hora exacta de captura, por lo que las apuestas simuladas podrían no haber estado disponibles a esos precios. Las búsquedas de hiperparámetros son acotadas. La disponibilidad de cuotas cambia en el test: en 2024/25 Bet\&Win y William Hill cubren el @@cov_bw@@\,\% y el @@cov_wh@@\,\% de los partidos, y en 2025/26 el consenso depende solo de Bet365 y Bet\&Win. En esa última temporada no se usa Pinnacle, de modo que la LR recibe el consenso en lugar del precio con el que se entrenó y la referencia de Pinnacle coincide con el consenso. En los modelos de diferencia de goles, la conversión a probabilidades se aprende con una temporada apartada y luego el modelo se reentrena con todos los datos, lo que puede desajustar un poco esa conversión. Además, el periodo de test se había evaluado con una versión anterior del sistema, por lo que conviene confirmar los resultados con partidos de 2026/27. Por último, los resultados no se extrapolan a otras ligas ni deportes.

\textbf{Ética.} Los datos se usan con fines académicos y no se redistribuyen. No se usan datos personales sensibles. Los equipos con menos historial o sin valoraciones tienen predicciones de menor calidad, por eso publicamos la cobertura en lugar de ocultarla. Las apuestas pueden generar adicción y pérdidas económicas: este análisis es experimental y no es una recomendación para apostar.

\section{Conclusiones}
\label{sec:conclusiones}
Construimos dos predictores con el mismo protocolo: uno que usa las cuotas y otro que solo usa datos deportivos. Las respuestas a las preguntas de investigación son:

\textbf{P1 (con cuotas).} Sí, aunque por poco. Una LR que recalibra el precio reduce el log loss del consenso en @@d1@@ en validación (intervalo [@@d1lo@@; @@d1hi@@], mejora en @@d1s@@ de 6 temporadas) y en @@d3@@ en test. Frente a Pinnacle con respaldo del consenso la mejora es de @@d2@@ en validación, no concluyente, y de @@d4@@ en test. La mejora observada se acompaña de menor error de calibración, con AUC similar.

\textbf{P2 (sin cuotas).} Un promedio de modelos deportivos reproduce el @@frac@@\,\% de la mejora del mercado sobre el prior, con correlación @@corr@@ con sus probabilidades y el mismo favorito en el @@fav@@\,\% de los partidos, pero no lo supera. La plantilla aporta frente al historial solo; las habilidades individuales no añaden más.

\textbf{P3.} MI y PCA no mejoran al mejor modelo de ningún enfoque. Combinar modelos ayuda sin cuotas, pero no con cuotas. Dar más peso a los partidos sorpresivos empeora el log loss agregado de las variantes probadas, aunque no necesariamente en cada temporada.

\textbf{P4.} Los resultados son estables: el modelo con cuotas supera al consenso en @@wins@@ de @@nseas@@ temporadas y el modelo sin cuotas queda detrás del mercado en todas. Predecir mejor no garantiza ganar dinero: el ROI del modelo con cuotas en test (@@roicon@@\,\%) tiene un intervalo que incluye pérdidas, y el del modelo sin cuotas es negativo.

Como trabajo futuro, proponemos incorporar alineaciones y lesiones confirmadas, datos de goles esperados (xG) y valoraciones más frecuentes; probar modelos de conteo de goles~\cite{dixon1997,karlis2003}; y evaluar el protocolo ya congelado en la temporada 2026/27.
"""
    for k, val_ in V.items(): body = body.replace('@@'+k+'@@', val_)
    assert '@@' not in body, body[body.index('@@')-50:body.index('@@')+50]
    bibliography = (templates/'bibliography.tex').read_text()
    preamble = preamble.replace(r'\usepackage[section]{placeins}', r'\usepackage{placeins}')
    for section in ('Resultados y análisis', 'Discusión, limitaciones y ética'):
        body = body.replace(r'\section{'+section+'}', r'\FloatBarrier'+'\n'+r'\section{'+section+'}')
    (ROOT/'docs/informe.tex').write_text(preamble+body+'\n'+r'\FloatBarrier'+'\n'+bibliography)


def presentation(c):
    m = c['final']; val, test = 'validacion', 'test_revisado'
    chosen = c['frozen']['selected']
    sim = c['similarity'].query('split == "validacion" and reference == "consenso"').iloc[0]
    bt = c['backtest'].set_index(['split', 'candidate', 'casa'])
    pr = c['primary'].set_index(['split', 'a', 'b'])
    bs = c['by_season'].pivot(index='Season', columns='candidate', values='log_loss')
    def num(x, d=4): return f'{float(x):.{d}f}'.replace('.', ',')
    def metric(split, name, col='log_loss', d=4): return num(m.loc[(split, name), col], d)
    def acc(split, name): return num(100*m.loc[(split, name), 'accuracy'], 1)+' %'
    def roi(split, name, casa):
        r = bt.loc[(split, name, casa)]
        return f'{100*r.ROI:+.1f}'.replace('.', ',').replace('-', '−')+' %', f'[{100*r.lo:+.0f} %; {100*r.hi:+.0f} %]'.replace('-', '−'), int(r.apuestas)
    def img(name, alt, cls=''): return f'<img class="fig {cls}" src="img/{name}.png" alt="{html.escape(alt)}">'
    def kpi(value, label, cls=''): return f'<div class="kpi {cls}"><b>{value}</b><span>{label}</span></div>'
    def tag(text, cls=''): return f'<span class="tag {cls}">{text}</span>'
    SECTIONS = ['Problema', 'Datos y método', 'Con cuotas', 'Sin cuotas', 'Resultados', 'Cierre']
    slides = []
    def slide(section, title, body, key='', note='', cls='', bg=''):
        n = len(slides)+1
        nav = ''.join(f'<li class="{"on" if s == section else ""} {"s-con" if s == "Con cuotas" else "s-sin" if s == "Sin cuotas" else ""}">{s}</li>' for s in SECTIONS)
        k = f'<p class="key"><span class="kl">En una frase</span>{key}</p>' if key else ''
        slides.append(f'<section class="slide {cls}" data-n="{n}">{bg}<header><ol class="track">{nav}</ol></header>'
                      f'<h2>{title}</h2><div class="content">{body}</div>{k}'
                      f'<footer><span>{note}</span><span class="pg">{n:02d} / 13</span></footer>'
                      f'<div class="progress" style="--p:{n/13:.4f}"></div></section>')
    con_name = model_label(chosen['con']['candidate'].split('/')[-1])
    gain_con = 100*(m.loc[(test, 'consenso'), 'log_loss']-m.loc[(test, 'modelo_con'), 'log_loss'])/m.loc[(test, 'consenso'), 'log_loss']
    r_con, ci_con, n_con = roi(test, 'modelo_con', 'B365'); r_sin, ci_sin, n_sin = roi(test, 'modelo_sin', 'B365')
    wins = int((bs.modelo_con < bs.consenso).sum())
    seasons_val = pr.loc[(val, 'modelo_con', 'consenso')].temporadas_mejora.split('/')[0]
    frac = 100*sim.fraccion_ganancia_prior

    # 1 · Portada: the slide itself is a mown pitch; no example numbers (slide 2 has them)
    pitch = ('<svg class="pitch-bg" viewBox="0 0 1280 720" preserveAspectRatio="none" aria-hidden="true"><g class="lines">'
             '<rect x="36" y="36" width="1208" height="648"/><line x1="640" y1="36" x2="640" y2="684"/><circle cx="640" cy="360" r="92"/>'
             '<rect x="36" y="196" width="170" height="328"/><rect x="36" y="284" width="62" height="152"/><path d="M206 296 A80 80 0 0 1 206 424"/>'
             '<rect x="1074" y="196" width="170" height="328"/><rect x="1182" y="284" width="62" height="152"/><path d="M1074 296 A80 80 0 0 0 1074 424"/>'
             '<path d="M36 56 A20 20 0 0 0 56 36"/><path d="M1224 36 A20 20 0 0 0 1244 56"/><path d="M36 664 A20 20 0 0 1 56 684"/><path d="M1224 684 A20 20 0 0 1 1244 664"/></g>'
             '<circle class="spot" cx="640" cy="360" r="5"/><circle class="spot" cx="150" cy="360" r="5"/><circle class="spot" cx="1130" cy="360" r="5"/>'
             '<circle class="ball" cx="1130" cy="360" r="11"/></svg>')
    slide('Problema', '¿Puede un modelo predecir el fútbol mejor que las casas de apuestas?',
          '<p class="lead">Estimamos la probabilidad de victoria local, empate y victoria visitante en nueve ligas europeas, con y sin mirar las cuotas.</p>'
          '<div class="kpis cover-kpis">'+kpi(f'{len(c["df"]):,}'.replace(',', '.'), 'partidos')+kpi('9', 'ligas')
          +kpi('12', 'temporadas')+kpi('2', 'enfoques')+'</div>'
          '<p class="authors">José Luis Alva · Juan Carlos Ticlia · Elmer Villegas · Joseph Cose</p>',
          note='Machine Learning · UTEC · Sección 3', cls='title-slide', bg=pitch)
    # 2 · Problema y preguntas
    slide('Problema', 'No queremos adivinar un ganador: queremos probabilidades',
          '<div class="match"><span class="team">Local</span><div class="probbar"><div class="h" style="flex:48">Local 48 %</div>'
          '<div class="d" style="flex:27">Empate 27 %</div><div class="a" style="flex:25">Visitante 25 %</div></div><span class="team">Visitante</span></div>'
          '<div class="qgrid">'
          '<div class="card con">'+tag('P1 · con cuotas', 'con')+'<p>¿El aprendizaje automático <b>mejora la probabilidad</b> que ya dan las casas de apuestas?</p></div>'
          '<div class="card sin">'+tag('P2 · sin cuotas', 'sin')+'<p>Solo con datos deportivos, <b>¿cuánto nos acercamos al mercado?</b> ¿Ayuda conocer la plantilla?</p></div>'
          '<div class="card">'+tag('P3 · técnicas')+'<p>¿Ayudan <b>MI, PCA, ensembles</b> o dar más peso a los partidos sorpresivos?</p></div>'
          '<div class="card">'+tag('P4 · estabilidad y dinero')+'<p>¿Los resultados se mantienen <b>temporada a temporada</b> y permiten ganar apostando?</p></div></div>',
          'Tarea: clasificación de 3 clases donde importa la probabilidad de cada clase.', 'Ejemplo ilustrativo de un partido')
    # 3 · Métrica
    slide('Problema', 'Medimos la calidad de la probabilidad, no solo los aciertos',
          '<div class="two"><div class="card plain"><p class="big">Log loss = −ln(probabilidad que dimos a lo que pasó)</p>'
          '<div class="penalty">'
          '<div><span>60 %</span><i style="width:17%"></i><b>0,51</b></div>'
          '<div><span>25 %</span><i style="width:46%"></i><b>1,39</b></div>'
          '<div class="bad"><span>5 %</span><i style="width:100%"></i><b>3,00</b></div></div>'
          '<p class="hint">Probabilidad dada al resultado real → castigo. Estar muy seguro y fallar cuesta mucho.</p></div>'
          '<div class="stack"><div class="card"><h4>AUC</h4><p>¿Ordena bien qué resultado es más probable? 0,5 es azar.</p></div>'
          '<div class="card"><h4>ECE · calibración</h4><p>Si decimos 30 %, ¿ocurre cerca del 30 % de las veces?</p></div>'
          '<div class="card"><h4>Baselines</h4><p><b>Prior</b> (frecuencias históricas): '+metric(val, 'prior')+' · <b>Mercado</b>: '+metric(val, 'consenso')+'</p></div></div></div>',
          'Menor log loss es mejor; el accuracy ignora qué tan seguro estaba el modelo.', 'Valores de baseline: validación 2018/19 a 2023/24')
    # 4 · Datos
    slide('Datos y método', 'Tres fuentes alineadas por fecha para no ver el futuro',
          '<div class="kpis small">'+kpi('Resultados', 'historia de cada equipo: Elo, forma, goles')+kpi('Cuotas', 'Bet365, Bet&amp;Win, Pinnacle, William Hill')
          +kpi('Valoraciones FIFA/FC', 'plantilla y habilidades del once')+'</div>'
          +img('revision_datos', 'Distribución del resultado y cobertura de datos'),
          'Cada partido usa valoraciones publicadas <b>antes</b> de su fecha. En 2025/26 el consenso usa solo Bet365 y Bet&amp;Win: Pinnacle se excluye desde el 23/07/2025 y William Hill ya no se publica.',
          '36.150 partidos originales · 31 sin cuotas · 2013/14 solo como calentamiento')
    # 5 · Protocolo: expanding window grid
    seasons = list(range(14, 26))
    head = '<div class="wl"></div>'+''.join(f'<div class="wh">{s}/{s+1}</div>' for s in seasons)
    rows = ''
    for v in range(18, 24):
        cells = ''.join(f'<div class="w {"tr" if s < v else "va" if s == v else ""}" style="--r:{v-18}"></div>' for s in seasons)
        rows += f'<div class="wl">Validar {v}/{v+1}</div>'+cells
    cells = ''.join(f'<div class="w {"tr" if s < 24 else "te"}" style="--r:6"></div>' for s in seasons)
    rows += '<div class="wl"><b>Modelo final</b></div>'+cells
    slide('Datos y método', 'Entrenamos con el pasado y evaluamos en temporadas posteriores',
          '<div class="window">'+head+rows+'</div>'
          '<div class="legend"><span class="tr">entrenamiento</span><span class="va">validación</span><span class="te">test (5.999 partidos)</span></div>'
          '<div class="flow"><div>Variables solo con partidos anteriores</div><i>→</i><div>Imputar y escalar dentro de cada entrenamiento</div>'
          '<i>→</i><div>Elegir el modelo por log loss en validación</div><i>→</i><div>Congelar y predecir el test</div></div>',
          'Validación cruzada temporal: la ventana de entrenamiento crece y nunca mira el futuro.',
          'Hiperparámetros elegidos con las dos temporadas anteriores a cada validación')
    # 6 · Con cuotas: idea
    slide('Con cuotas', 'Una cuota ya es una probabilidad, con el margen de la casa',
          '<div class="two"><div class="card con">'+tag('Enfoque 1 · con cuotas', 'con')+
          '<table class="mini"><thead><tr><th></th><th>Cuota</th><th>1/cuota</th><th>Probabilidad</th></tr></thead><tbody>'
          '<tr><td>Local</td><td>2,10</td><td>0,476</td><td><b>45,4 %</b></td></tr><tr><td>Empate</td><td>3,40</td><td>0,294</td><td><b>28,1 %</b></td></tr>'
          '<tr><td>Visitante</td><td>3,60</td><td>0,278</td><td><b>26,5 %</b></td></tr><tr class="sum"><td>Suma</td><td></td><td>1,048</td><td>100 %</td></tr></tbody></table>'
          '<p class="hint">El 4,8 % que sobra es la ganancia de la casa; se reparte para que sume 100 %.</p></div>'
          '<div class="stack"><div class="card"><h4>Baseline del mercado</h4><p>Consenso y Pinnacle con respaldo del consenso, sin aprendizaje.</p></div>'
          '<div class="card"><h4>Modelos probados</h4><p>LR, HGB, XGBoost, Random Forest, red neuronal y SVM, con precio y datos deportivos.</p></div>'
          f'<div class="card con"><h4>Elegido: {con_name}</h4><p>Regresión logística con solo dos números del precio.</p></div></div></div>',
          'El modelo con cuotas no adivina desde cero: aprende a corregir el precio del mercado.', 'Ejemplo de cuotas ilustrativo')
    # 7 · Con cuotas: resultados
    slide('Con cuotas', 'El modelo afina el precio del mercado, no lo reemplaza',
          '<div class="kpis">'+kpi(metric(val, 'modelo_con'), 'log loss del modelo', 'con')+kpi(metric(val, 'consenso'), 'log loss del consenso')
          +kpi(f'{seasons_val} de 6', 'temporadas en que mejora', 'con')+kpi(metric(val, 'modelo_con', 'AUC_macro', 3)+' = '+metric(val, 'consenso', 'AUC_macro', 3), 'AUC: ordena igual que el mercado')+'</div>'
          +img('revision_calibracion_con', 'Calibración del modelo con cuotas frente al consenso'),
          f'La mejora viene de la <b>calibración</b> (ECE {metric(val,"consenso","ECE_masa")} → {metric(val,"modelo_con","ECE_masa")}): sus probabilidades coinciden mejor con lo que pasa.',
          'Validación 2018/19 a 2023/24 · puntos sobre la diagonal = calibración perfecta')
    # 8 · Sin cuotas: idea
    slide('Sin cuotas', 'Construimos un precio propio solo con datos deportivos',
          '<div class="two"><div class="stack">'
          '<div class="card sin">'+tag('Enfoque 2 · sin cuotas', 'sin')+'<h4>Historia</h4><p>Elo, forma de 5 partidos, medias de goles y puntos, descanso.</p></div>'
          '<div class="card sin"><h4>Plantilla</h4><p>Valoración media del once según el videojuego FIFA/FC.</p></div>'
          '<div class="card sin"><h4>Habilidades</h4><p>Ataque, pase, defensa y portero de los mejores jugadores.</p></div></div>'
          '<div>'+img('revision_ablacion', 'Ablación del predictor sin cuotas', 'fill')+'</div></div>',
          'La plantilla mejora a la historia sola; las habilidades individuales ya no suman más.',
          'Log loss de validación de la LR al añadir grupos de variables (menor es mejor)')
    # 9 · Sin cuotas: similarity to the market
    slide('Sin cuotas', f'Sin mirar cuotas logramos el {frac:.0f}\u00a0% de la ventaja del mercado',
          '<div class="kpis">'+kpi(f'{frac:.0f}\u00a0%', 'de la mejora del mercado sobre el prior', 'sin')
          +kpi(f'{sim.corr_logit:.2f}'.replace('.', ','), 'correlación con las probabilidades del mercado', 'sin')
          +kpi(f'{100*sim.mismo_favorito:.0f}\u00a0%', 'de partidos con el mismo favorito', 'sin')
          +kpi(metric(val, 'modelo_sin'), 'log loss (prior '+metric(val, 'prior')+' · mercado '+metric(val, 'consenso')+')')+'</div>'
          +img('revision_predictor', 'Probabilidades propias frente al consenso'),
          'Con datos públicos se reproduce casi todo el trabajo de las casas; falta lo que no vemos: lesiones y alineaciones.',
          'Elegido: promedio de 8 modelos · cada punto es un partido de validación; la diagonal es coincidir con el mercado')
    # 10 · Test
    rows = ''.join(f'<tr class="r-{k}"><td>{LABEL[k]}</td><td>{metric(val,k)}</td><td>{metric(test,k)}</td><td>{metric(test,k,"AUC_macro",3)}</td><td>{acc(test,k)}</td></tr>'
                   for k in ('prior', 'consenso', 'Pinnacle_respaldo', 'modelo_con', 'modelo_sin'))
    slide('Resultados', 'En el test se mantiene el mismo orden que en validación',
          '<div class="two"><div style="flex:1.5"><table><thead><tr><th></th><th>LL validación</th><th>LL test</th><th>AUC</th><th>Accuracy</th></tr></thead><tbody>'
          +rows+'</tbody></table></div><div class="stack">'
          f'<div class="card con"><h4>Con cuotas</h4><p>−{num(gain_con, 2)} % de log loss frente al consenso; el intervalo del 95 % excluye el cero.</p></div>'
          '<div class="card sin"><h4>Sin cuotas</h4><p>Muy por delante del prior, todavía detrás del mercado.</p></div></div></div>',
          'El accuracy casi no cambia; la diferencia está en la calidad de las probabilidades.',
          'Test: 5.999 partidos de 2024/25 y 2025/26, con modelos congelados en validación')
    # 11 · Extra experiments
    slide('Resultados', 'Más complejidad no siempre ayuda: lo probamos y lo medimos',
          '<div class="two">'+img('revision_ensembles', 'Ensembles en ambos enfoques')+img('revision_reduccion', 'MI y PCA en ambos enfoques')+'</div>'
          '<div class="three"><div class="card"><h4>Ensembles</h4><p>Sin cuotas el promedio ayuda y es el elegido; con cuotas ninguno supera a la LR de solo precio.</p></div>'
          '<div class="card"><h4>MI y PCA</h4><p>Reducir variables no supera al mejor modelo en ningún enfoque.</p></div>'
          '<div class="card"><h4>Más peso a sorpresas</h4><p>Pesos y remuestreo 50/50 empeoran el log loss agregado; no necesariamente cada temporada.</p></div></div>',
          'Los modelos simples obtuvieron los mejores resultados: con estas rejillas, más complejidad no mejoró las predicciones.',
          'Barras: cambio de log loss frente a la LR con todas las variables (negativo es mejor)')
    # 12 · Money
    slide('Resultados', 'Predecir mejor no garantiza ganar dinero',
          '<div class="kpis">'+kpi(r_con, f'ROI con cuotas, Bet365 ({n_con} apuestas)', 'con')+kpi(ci_con, 'intervalo del 95 % con cuotas')
          +kpi(r_sin, f'ROI sin cuotas, Bet365 ({n_sin} apuestas)', 'sin')+'</div>'
          +img('revision_backtest', 'Backtest de ambos modelos'),
          'El intervalo incluye pérdidas: superar el margen de la casa exige más que predecir un poco mejor.',
          'Regla fija: apostar 1 unidad si la ganancia esperada supera 2 % · ejercicio académico, no recomendación')
    # 13 · Answers
    slide('Cierre', 'Respuestas a nuestras preguntas',
          '<div class="qgrid answers">'
          f'<div class="card con">{tag("P1 · con cuotas", "con")}<p><b>Sí, pero por poco.</b> Log loss {metric(val,"consenso")} → {metric(val,"modelo_con")}, mejor en {seasons_val} de 6 temporadas. Solo <b>recalibra</b> el precio.</p></div>'
          f'<div class="card sin">{tag("P2 · sin cuotas", "sin")}<p>Logramos el <b>{frac:.0f} %</b> de la ventaja del mercado sobre el prior, sin superarlo. La plantilla ayuda; las habilidades no suman más.</p></div>'
          f'<div class="card">{tag("P3 · técnicas")}<p>MI y PCA no mejoran. El <b>promedio de modelos</b> ayuda sin cuotas; dar más peso a las sorpresas empeora.</p></div>'
          f'<div class="card">{tag("P4 · estabilidad y dinero")}<p>Con cuotas supera al consenso en <b>{wins} de {len(bs)}</b> temporadas, pero el ROI de {r_con} tiene un intervalo con pérdidas.</p></div></div>'
          '<p class="limits">Límites: valoraciones del videojuego y no alineaciones reales, solo liga, cuotas sin hora de captura y cambio de Pinnacle al consenso de dos casas.</p>',
          'El mercado es una referencia fuerte en este experimento; el aprendizaje mejora ligeramente su calibración.',
          '¡Gracias! · Preguntas · github.com/jcarlos-t/MLProject')
    style = '''
/* Layout: 16:9 slides, a section tracker on top, one assertion title, one visual, one takeaway line.
   Teal = with odds, blue = without odds, amber = test. */
:root{--paper:#fbfcfb;--ink:#10201c;--mut:#51625d;--line:#dbe4e0;--soft:#eef3f1;--con:#0f8a7e;--con-bg:#e3f3f0;--sin:#2f5bd3;--sin-bg:#e7edfb;--test:#c27a12;--test-bg:#fbf0dc;--bad:#b4332b;
--display:"Space Grotesk","Segoe UI",system-ui,sans-serif;--body:Inter,"Segoe UI",system-ui,Arial,sans-serif}
*{box-sizing:border-box}html,body{height:100%}body{margin:0;background:#cfd8d4;color:var(--ink);font-family:var(--body);overflow:hidden}
#stage{width:1280px;height:720px;position:absolute;left:50%;top:50%;transform-origin:center}
.slide{position:absolute;inset:0;background:var(--paper);padding:26px 60px 50px;display:none;flex-direction:column;gap:14px}.slide.active{display:flex}
header{display:flex}.track{display:flex;gap:6px;list-style:none;margin:0;padding:0}
.track li{font-size:12.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);padding:5px 11px;border-radius:999px;background:var(--soft);font-weight:600}
.track li.on{background:var(--ink);color:var(--paper)}.track li.on.s-con{background:var(--con)}.track li.on.s-sin{background:var(--sin)}
h2{font-family:var(--display);font-size:38px;line-height:1.12;margin:6px 0 0;font-weight:700;max-width:1120px;letter-spacing:-.015em;text-wrap:balance}
h3{font-family:var(--display);font-size:24px;margin:0}h4{font-family:var(--display);font-size:19px;margin:0 0 4px}
.content{flex:1;display:flex;flex-direction:column;gap:14px;min-height:0;justify-content:center}
.fig{max-height:285px;max-width:100%;object-fit:contain;align-self:center;mix-blend-mode:multiply}
.fig.fill{width:100%;max-height:270px}
p,li{font-size:20px;line-height:1.4;margin:0}.hint{font-size:16px;color:var(--mut);margin-top:8px}.big{font-family:var(--display);font-size:23px;font-weight:700;margin-bottom:14px}
.lead{font-size:26px;color:var(--mut);max-width:980px;line-height:1.35}
.key{font-size:19px;padding:12px 18px;background:var(--soft);border-radius:10px}
.key .kl{display:inline-block;margin-right:14px;font-size:12px;letter-spacing:.1em;text-transform:uppercase;font-weight:700;color:var(--con)}
.two{display:flex;gap:26px;align-items:center}.two>*{flex:1;min-width:0}.two>.fig{max-height:250px}
.three{display:flex;gap:16px}.three>*{flex:1}.stack{display:flex;flex-direction:column;gap:12px}
.card{background:var(--soft);border-radius:12px;padding:14px 18px}.card p{font-size:17.5px}.card.con{background:var(--con-bg)}.card.sin{background:var(--sin-bg)}.card.plain{background:none;padding:0}
.tag{display:inline-block;font-size:12.5px;font-weight:700;letter-spacing:.04em;padding:3px 10px;border-radius:999px;color:var(--paper);background:var(--mut);margin-bottom:8px}.tag.con{background:var(--con)}.tag.sin{background:var(--sin)}
.kpis{display:flex;gap:16px}.kpi{flex:1;background:var(--soft);border-radius:12px;padding:14px 18px;display:flex;flex-direction:column;gap:4px}
.kpi b{font-family:var(--display);font-size:36px;line-height:1.05;font-weight:700;font-variant-numeric:tabular-nums}.kpi span{font-size:15px;color:var(--mut)}
.kpi.con{background:var(--con-bg)}.kpi.con b{color:var(--con)}.kpi.sin{background:var(--sin-bg)}.kpi.sin b{color:var(--sin)}.kpis.small .kpi b{font-size:23px}
.title-slide{--pitch-a:#1c6b4c;--pitch-b:#1f7553;--chalk:#f4fbf7;background:repeating-linear-gradient(90deg,var(--pitch-a) 0 106.6px,var(--pitch-b) 106.6px 213.3px);color:var(--chalk)}
.title-slide>:is(header,h2,.content){position:relative}.pitch-bg{position:absolute;inset:0;width:100%;height:100%}
.pitch-bg .lines *{fill:none;stroke:var(--chalk);stroke-opacity:.24;stroke-width:3}.pitch-bg .spot{fill:var(--chalk);fill-opacity:.45}.pitch-bg .ball{fill:var(--chalk);stroke:var(--ink);stroke-width:2}
.title-slide .track li{background:rgba(255,255,255,.12);color:var(--chalk)}.title-slide .track li.on{background:var(--chalk);color:var(--pitch-a)}
.title-slide h2{font-size:52px;max-width:760px;margin-top:22px;color:var(--chalk)}.title-slide .content{justify-content:center;max-width:640px;gap:22px}
.title-slide .lead{color:var(--chalk);opacity:.9}.title-slide .kpi{background:rgba(255,255,255,.12)}.title-slide .kpi b{color:var(--chalk)}.title-slide .kpi span{color:var(--chalk);opacity:.8}
.cover-kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.cover-kpis .kpi b{font-size:30px}
.authors{font-size:17px;color:var(--mut)}.title-slide .authors{color:var(--chalk);opacity:.85}.title-slide footer{color:var(--chalk);opacity:.8}.title-slide .progress{background:rgba(255,255,255,.2)}
.match{display:flex;align-items:center;gap:16px}.team{font-family:var(--display);font-size:22px;font-weight:700}
.probbar{flex:1;display:flex;height:52px;border-radius:10px;overflow:hidden;font-size:18px;font-weight:700;color:var(--paper)}.probbar>div{display:flex;align-items:center;justify-content:center}
.probbar .h{background:var(--con)}.probbar .d{background:#8a97a3}.probbar .a{background:var(--sin)}
.qgrid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.qgrid p{font-size:18.5px}.answers{gap:18px}.answers .card{padding:20px 24px}.answers p{font-size:21px;line-height:1.45}
.penalty{display:flex;flex-direction:column;gap:10px}.penalty div{display:grid;grid-template-columns:60px 1fr 52px;gap:12px;align-items:center;font-size:19px}
.penalty i{display:block;height:22px;background:var(--con);border-radius:4px}.penalty b{text-align:right;font-variant-numeric:tabular-nums}.penalty .bad i{background:var(--bad)}.penalty .bad b{color:var(--bad)}
table{border-collapse:collapse;width:100%;font-size:18px;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:9px 12px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
thead th{font-size:14px;color:var(--mut);font-weight:600}table.mini{font-size:18px;margin-top:4px}tr.sum td{color:var(--mut)}
tr.r-modelo_con td{color:var(--con);font-weight:700}tr.r-modelo_sin td{color:var(--sin);font-weight:700}tr.r-prior td{color:var(--mut)}
.window{display:grid;grid-template-columns:150px repeat(12,1fr);gap:4px;align-items:center}
.wh{font-size:12.5px;color:var(--mut);text-align:center}.wl{font-size:14.5px;color:var(--ink)}.w{height:22px;border-radius:4px;background:transparent;border:1px dashed var(--line)}
.w.tr,.legend .tr::before{background:#b9c6c1;border:0}.w.va,.legend .va::before{background:var(--con);border:0}.w.te,.legend .te::before{background:var(--test);border:0}
.legend{display:flex;gap:22px;font-size:15px;color:var(--mut)}.legend span::before{content:"";display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:7px;vertical-align:-2px}
.flow{display:flex;align-items:stretch;gap:8px}.flow div{flex:1;background:var(--soft);border-radius:10px;padding:12px 14px;font-size:16.5px;font-weight:600;display:flex;align-items:center}.flow i{align-self:center;font-style:normal;font-size:22px;color:var(--mut)}
.limits{font-size:15.5px;color:var(--mut)}
footer{position:absolute;bottom:16px;left:60px;right:60px;display:flex;justify-content:space-between;font-size:13px;color:var(--mut);gap:24px}.pg{font-variant-numeric:tabular-nums;font-weight:600}
.progress{position:absolute;left:0;right:0;bottom:0;height:5px;background:var(--line)}.progress::after{content:"";position:absolute;left:0;top:0;bottom:0;width:calc(var(--p)*100%);background:linear-gradient(90deg,var(--con),var(--sin))}
/* Motion: each slide enters, its blocks rise in sequence and bars grow. Off in print and for reduced motion. */
@keyframes enter{from{opacity:0;transform:translateX(26px)}to{opacity:1;transform:none}}
@keyframes rise{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes wipe{from{clip-path:inset(0 100% 0 0)}to{clip-path:inset(0 0 0 0)}}
@keyframes pop{from{opacity:0;transform:scale(.4)}to{opacity:1;transform:none}}
@keyframes kick{0%{transform:translate(-490px,0)}100%{transform:translate(0,0)}}
.slide.active{animation:enter .45s cubic-bezier(.2,.7,.2,1) both}
.slide.active h2{animation:rise .5s .05s both}
.slide.active .content>*{animation:rise .55s cubic-bezier(.2,.7,.2,1) both;animation-delay:.18s}
.slide.active .content>*:nth-child(2){animation-delay:.32s}.slide.active .content>*:nth-child(3){animation-delay:.46s}.slide.active .content>*:nth-child(4){animation-delay:.6s}
.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*{animation:rise .5s cubic-bezier(.2,.7,.2,1) both}
.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*:nth-child(1){animation-delay:.3s}.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*:nth-child(2){animation-delay:.45s}
.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*:nth-child(3){animation-delay:.6s}.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*:nth-child(4){animation-delay:.75s}
.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*:nth-child(5){animation-delay:.9s}.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*:nth-child(6){animation-delay:1.05s}
.slide.active :is(.kpis,.qgrid,.stack,.three,.flow)>*:nth-child(7){animation-delay:1.2s}
.slide.active .key{animation:rise .5s 1.1s both}
.slide.active .probbar{animation:wipe .9s .35s cubic-bezier(.4,0,.2,1) both}
.penalty i{transform-origin:left}.slide.active .penalty i{animation:grow .7s cubic-bezier(.2,.7,.2,1) both}
.slide.active .penalty div:nth-child(1) i{animation-delay:.5s}.slide.active .penalty div:nth-child(2) i{animation-delay:.8s}.slide.active .penalty div:nth-child(3) i{animation-delay:1.1s}
.slide.active .w.va,.slide.active .w.te{animation:pop .35s cubic-bezier(.3,1.4,.5,1) both;animation-delay:calc(.45s + var(--r)*.18s)}
.slide.active .pitch-bg .ball{animation:kick 1.4s .4s cubic-bezier(.25,.7,.3,1) both}
.progress::after{transition:width .5s ease}
@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
.controls{position:fixed;right:max(16px,env(safe-area-inset-right,0px));bottom:12px;z-index:10;display:flex;gap:6px}
.controls button{padding:7px 13px;border:1px solid #9fb0aa;background:var(--paper);color:var(--ink);border-radius:8px;cursor:pointer;font:600 14px var(--body)}
.controls button:focus-visible{outline:3px solid var(--sin);outline-offset:2px}
@media print{*,*::before,*::after{animation:none!important;transition:none!important}@page{size:1280px 720px;margin:0}html,body{width:1280px;height:auto;overflow:visible;background:var(--paper)}#stage{position:static;transform:none!important;height:auto}
.slide{position:relative;display:flex!important;width:1280px;height:720px;break-after:page;page-break-after:always;-webkit-print-color-adjust:exact;print-color-adjust:exact}.slide:last-child{break-after:auto;page-break-after:auto}.controls{display:none}}
'''
    script = '''let i=Math.max(0,Math.min(12,(parseInt(location.hash.slice(1))||1)-1));const slides=[...document.querySelectorAll('.slide')],stage=document.querySelector('#stage');
function fit(){const s=Math.min(innerWidth/1280,innerHeight/720);stage.style.transform=`translate(-50%,-50%) scale(${s})`}
function show(){slides.forEach((s,k)=>s.classList.toggle('active',k===i));history.replaceState(null,'','#'+(i+1))}
function go(d){i=Math.max(0,Math.min(12,i+d));show()}
addEventListener('resize',fit);
addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();go(1)}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(-1)}else if(e.key==='Home'){i=0;show()}else if(e.key==='End'){i=12;show()}});
document.querySelector('#prev').addEventListener('click',()=>go(-1));document.querySelector('#next').addEventListener('click',()=>go(1));document.querySelector('#print').addEventListener('click',()=>print());
fit();show();'''
    path = ROOT/'docs/presentacion'; (path/'img').mkdir(exist_ok=True)
    for p in (ROOT/'docs/figures').glob('revision_*.png'): shutil.copy2(p, path/'img'/p.name)
    fonts = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Space+Grotesk:wght@500;700&display=swap">'
    controls = '<nav class="controls" aria-label="Navegación"><button id="prev" aria-label="Diapositiva anterior">←</button><button id="next" aria-label="Diapositiva siguiente">→</button><button id="print">Imprimir / PDF</button></nav>'
    (path/'presentacion.html').write_text('<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fútbol con y sin cuotas</title>'
        +fonts+'<style>'+style+'</style></head><body><main id="stage">'+''.join(slides)+'</main>'+controls+'<script>'+script+'</script></body></html>')


def ensure_draw_results(out):
    from .data import sha
    import subprocess, sys
    pointer = ROOT/'scripts/data/draw_actual.txt'
    valid = False
    if pointer.exists():
        run = ROOT/pointer.read_text().strip()
        try:
            identity = json.loads((run/'protocol.json').read_text())
            marker = json.loads((run/'complete.json').read_text())
            valid = (identity['parent_run'] == out.name
                and all(sha(ROOT/'scripts'/name) == digest for name, digest in identity['sources'].items())
                and all(sha(out/name) == digest for name, digest in identity['inputs'].items())
                and marker['metrics_sha'] == sha(run/'metrics.csv')
                and marker['predictions_sha'] == sha(run/'test_predictions.csv')
                and marker['frozen_sha'] == sha(run/'frozen.json'))
        except (OSError, ValueError, KeyError):
            pass
    if not valid:
        if out.resolve() != current().resolve():
            raise ValueError('El experimento de empates debe corresponder a la ejecución solicitada')
        subprocess.run([sys.executable, str(ROOT/'scripts/experiment_draws.py')], check=True)


def render(out=None):
    out = out or current(); c = context(out)
    if not (out/'importance_groups.csv').exists() or not (out/'validation_diagnostics.csv').exists():
        validation_diagnostics(out)
    ensure_draw_results(out)
    from audit_results import audit
    audit(out)
    if not (out/'market_guided_comparisons.csv').exists(): development_comparisons(out)
    figures(c); paper(c); presentation(c)
    # Both deliverables draw from exactly this exported table.
    c['final'].reset_index().to_csv(ROOT/'docs/resultados_revision.csv', index=False)
    print('Figuras, informe y 13 diapositivas actualizados desde', out.name)


if __name__ == '__main__': render()
