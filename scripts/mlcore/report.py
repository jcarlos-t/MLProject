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
        ax.set(title='Validación' if split == 'validacion' else 'Test revisado, ya observado', xlabel='Log loss')
    save(fig, 'revision_final')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, key in zip(axes, ('modelo_con', 'modelo_sin')):
        t = c['backtest'][c['backtest'].candidate.eq(key)].copy()
        labels = [f'{r.casa} · '+('validación' if r.split == 'validacion' else 'test revisado')+f' (n={r.apuestas})' for r in t.itertuples()]
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


def latex_table(headers, rows, label, caption):
    head = ' & '.join(headers)+r' \\'
    body = '\n'.join(' & '.join(map(str, row))+r' \\' for row in rows)
    return '\n'.join([r'\begin{table}[!htb]', r'\caption{'+caption+'}', r'\centering\footnotesize',
        r'\resizebox{\columnwidth}{!}{\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',
        r'\toprule', head, r'\midrule', body, r'\bottomrule\end{tabular}}',
        r'\label{'+label+'}', r'\end{table}'])+'\n\n'


def latex_figure(name, label, caption, wide=False):
    env = 'figure*' if wide else 'figure'
    return '\n'.join([r'\begin{'+env+'}[!htb]', r'\centering',
        r'\includegraphics[width='+(r'0.96\textwidth' if wide else r'\columnwidth')+']{'+name+'.png}',
        r'\caption{'+caption+'}', r'\label{'+label+'}', r'\end{'+env+'}'])+'\n\n'


def paper(c):
    # Keep authors and the IEEE preamble. Replace the body so old claims cannot survive.
    archive = ROOT/'scripts/data/archive/pre_revision_20261112/docs/informe.tex'
    old = archive.read_text(); preamble = old.split(r'\begin{abstract}')[0]
    preamble = preamble.replace('Predicción probabilística del resultado 1X2 en partidos de fútbol mediante Machine Learning',
        'Predicción probabilística de fútbol con y sin cuotas: comparación temporal de modelos y mercado')
    m = c['final']; v, t = 'validacion', 'test_revisado'
    vcon, vsin, vm = [m.loc[(v, k)] for k in ('modelo_con', 'modelo_sin', 'consenso')]
    tcon, tsin, tm = [m.loc[(t, k)] for k in ('modelo_con', 'modelo_sin', 'consenso')]
    chosen = c['frozen']['selected']; result = c['comparison'].set_index('candidate')
    sim = c['similarity'].query('split == "validacion" and reference == "consenso"').iloc[0]
    n, ntest = len(c['df']), int(c['df'].Season.isin(TEST).sum())
    body = rf"""\begin{{abstract}}
Se estudian dos problemas complementarios con la misma evaluación: cuánto añade el aprendizaje a las probabilidades implícitas de las casas de apuestas y cuánto puede predecirse sin consultar sus precios. Se construye una base de {n:,} partidos modelables de nueve ligas europeas, de 2014/15 a 2025/26, con resultados históricos y ratings FIFA/FC. Se corrige la disponibilidad temporal de los ratings, el preprocesamiento fuera de las particiones, la calibración del SVM y la propagación de las configuraciones revisadas. Se comparan clasificación directa, regresión de diferencia de goles, selección MI/PCA, ensembles y reponderación. En seis temporadas de validación, los modelos elegidos con y sin cuotas obtienen log loss {f(vcon.log_loss)} y {f(vsin.log_loss)}, frente a {f(vm.log_loss)} del consenso. En la evaluación revisada de {ntest:,} partidos obtienen {f(tcon.log_loss)} y {f(tsin.log_loss)}, frente a {f(tm.log_loss)}. El predictor sin cuotas reproduce el {100*sim.fraccion_ganancia_prior:.1f}\% de la reducción de pérdida del mercado respecto al prior en validación. El test ya había sido observado antes de la auditoría: esta reevaluación no constituye evidencia prospectiva independiente. Se distinguen calidad probabilística, parecido al mercado y rentabilidad; las conclusiones se limitan a los métodos y datos evaluados.
\end{{abstract}}
\begin{{IEEEkeywords}}
fútbol, clasificación probabilística, cuotas, ratings, validación temporal, calibración, data leakage.
\end{{IEEEkeywords}}

\section{{Introducción y formulación del problema}}
\label{{sec:intro}}
\subsection{{Dos enfoques de igual importancia}}
El precio de una apuesta resume la expectativa de una casa sobre un resultado. Un sistema predictivo puede utilizar esa expectativa como entrada o construir una propia a partir de los equipos. Son objetivos distintos: el primero pregunta si se mejora una referencia informativa; el segundo busca generar una distribución comparable sin observar el precio. Se presentan en ese orden y se evalúan con los mismos partidos y métricas. El código reproducible está en \url{{https://github.com/jcarlos-t/MLProject}}.

El trabajo combina la selección de variables de Tammouch et al.~\cite{{tammouch2024}} con el uso de habilidades y regresión de diferencia de goles de Stübinger et al.~\cite{{stubinger2020}}. No se presupone que un ensemble ni una red neuronal deban ganar. Tampoco se interpreta una accuracy alta como evidencia suficiente de una buena probabilidad: interesa penalizar la confianza equivocada y comprobar la calibración.

\subsection{{Tarea, métricas y preguntas}}
Cada partido tiene tres resultados: victoria local H, empate D y victoria visitante A. Se estima un vector de probabilidades que suma uno. La métrica primaria es la pérdida logarítmica,
\begin{{equation}}
\mathrm{{LL}}=-N^{{-1}}\sum_i\log \hat p_{{i,y_i}},
\label{{eq:ll}}
\end{{equation}}
que penaliza especialmente asignar poca probabilidad al resultado ocurrido. Como medidas complementarias se utilizan Brier, RPS~\cite{{epstein1969}}, AUC macro uno-contra-el-resto, accuracy 1X2 y error de calibración ECE. El AUC evalúa ordenación; el ECE compara frecuencias y probabilidades en diez grupos de igual masa. También se publica ECE de igual ancho, sin intercambiar ambas definiciones.

Se conservan cuatro preguntas: \textbf{{P1}}, estabilidad entre temporadas; \textbf{{P2}}, aporte de plantilla frente a historia deportiva; \textbf{{P3}}, comparación MI frente a PCA; y \textbf{{P4}}, rentabilidad de probabilidades y ensembles. Las hipótesis son que el mercado constituye una referencia difícil, que la plantilla puede complementar la historia y que acertar más no garantiza rentabilidad. Se comprueban en ambas ramas, sin convertir la falta de significación en una prueba de eficiencia universal.

\section{{Datos y análisis exploratorio}}
\label{{sec:datos}}
\subsection{{Fuentes y cobertura}}
Football-Data~\cite{{footballdata}} proporciona 36.150 partidos originales de nueve primeras divisiones europeas: Inglaterra, Alemania, España, Italia, Francia, Países Bajos, Portugal, Turquía y Bélgica. Se conservan {n:,} con al menos una casa válida; los otros 31 alimentan la historia, pero no la comparación pareada. Los resultados de 2013/14 inicializan el historial y no se modelan. Los ratings FIFA/FC proceden de cuatro conjuntos públicos~\cite{{kagglefifa}}.

La Fig.~\ref{{fig:datos}} muestra el balance de clases y la cobertura. Las estadísticas de tiros, córners, faltas y tarjetas tienen faltantes estructurales; se mantienen hasta el ajuste del imputador. La disponibilidad de plantilla ya no se fuerza al 100\%: se exige que el snapshot preceda al partido. No se eliminan automáticamente las colas de cuotas o goles por ser extremas, porque pueden corresponder a partidos reales.
"""
    body += latex_figure('revision_datos', 'fig:datos', 'Distribución y cobertura. La cobertura de plantilla exige un snapshot disponible antes del partido; Pinnacle deja de considerarse fiable desde julio de 2025.', True)
    body += r"""
\subsection{Disponibilidad temporal de los ratings}
La auditoría detectó que asignar un único snapshot a todos los partidos de su temporada introducía datos posteriores. Se acreditaron las fechas de doce versiones: las antiguas se enlazaron por identificador de actualización SoFIFA y coincidencia de ratings con los metadatos de la misma fuente; las recientes conservan una fecha explícita. Cada equipo utiliza su snapshot verificable más reciente estrictamente anterior a la fecha del partido; si falta, retrocede a una versión anterior del club y, en ausencia de ella, conserva NaN.

Esto evita atribuir el snapshot de julio de 2025 a partidos de 2024/25. Como consecuencia, durante esa temporada se usan en ocasiones ratings anteriores: la antigüedad es una limitación medible, no una imputación encubierta. Los titulares son los asignados por el juego al club, no alineaciones conocidas de cada encuentro. Los nombres de los clubes se armonizan mediante una tabla de identidad, que no aporta resultados deportivos.

\subsection{Qué representan las cuotas}
Las columnas sin sufijo de cierre son capturas \emph{pre-cierre}, tomadas después de abrir el mercado; no se las denomina apertura real. La fuente no garantiza que todos los precios sean simultáneos. Además, advierte de la falta de fiabilidad de Pinnacle desde el 23 de julio de 2025. Desde esa fecha se excluye esa casa de la referencia y del consenso; se utilizan las otras casas disponibles. El backtest de Pinnacle se restringe al periodo considerado fiable. La existencia de un número no se confunde con su vigencia.

\section{Preprocesamiento e ingeniería de características}
\label{sec:prep}
\subsection{Una historia que solo avanza}
Las variables se calculan cronológicamente. Se leen los estados previos de todos los partidos de una fecha y solo después se actualizan con sus resultados. La forma resume los cinco encuentros previos, con respaldo en medias anteriores de la liga cuando la ventana está incompleta. Se añaden Elo, medias exponenciales de puntos y diferencia de goles, tendencia al empate, goles totales previos, descanso y congestión. Esta última solo observa liga: no mide copas ni competiciones continentales.

El prior suavizado de puntos se calcula con resultados anteriores a la fecha. La normalización de plantilla utiliza clubes de snapshots ya disponibles, una vez por club, sin ponderarlos por partidos futuros. Las habilidades resumen los cuatro mejores jugadores de campo en cada atributo y el portero del once del snapshot; se usan diferencias entre equipos. Son valoraciones del juego, no mediciones de rendimiento del partido.

\subsection{Transformaciones aprendidas dentro del entrenamiento}
Imputación por mediana, indicadores de ausencia, escalado, poda por correlación y MI/PCA se ajustan dentro del entrenamiento de cada partición, tanto interior como exterior. Las columnas totalmente vacías se conservan con valor neutro para mantener el esquema. La poda conserva la primera variable de un par con correlación absoluta mayor que 0,95. MI elige cinco o diez variables; PCA se evalúa con cinco componentes o 95\% de varianza. Las rejillas se fijaron antes de calcular los resultados revisados.

Se diferencia entre una tabla de diagnóstico y una decisión de selección: ninguna columna se elimina por un veredicto descriptivo sin que la transformación correspondiente se aplique y evalúe dentro del pipeline. Esta separación sigue las recomendaciones sobre ingeniería de variables de Zheng y Casari~\cite{zheng2018}. El predictor puro sin cuotas no utiliza precios para construir entradas, ponderaciones ni calibradores.

\section{Metodología y modelos}
\label{sec:modelos}
\subsection{Primero: modelos con cuotas}
Una cuota decimal $o_j$ se transforma en probabilidad mediante $p_j=(1/o_j)/\sum_k(1/o_k)$. Se calcula el promedio de probabilidades de las casas completas y fiables, no el inverso de la cuota promedio. Se comparan el consenso directo, Pinnacle fiable con respaldo, una LR sobre dos variables de referencia, una LR de consenso con margen y modelos con mercado más fundamentos. Así se distingue el efecto de elegir el precio de referencia del efecto de aprender a recalibrarlo.

\subsection{Después: un predictor sin cuotas}
La segunda rama recibe solamente historia deportiva, plantilla e individualidades. Se ensaya clasificación directa H/D/A y regresión de diferencia de goles. Para esta última, una función aprendida en una temporada anterior convierte la predicción de goles en tres probabilidades; después el regresor se reajusta con todo el pasado disponible. Se evalúan regresión lineal, RF, HGB, SVR y su promedio lineal/RF/HGB. La elección final se hace por log loss, no por RMSE aislado.

\subsection{Familias comunes, ensembles y ajuste}
Ambas ramas comparan LR, HGB, XGBoost, Random Forest, MLP y SVM. La MLP promedia tres semillas, sin parada temprana aleatoria. El SVM se limita a los 9.000 partidos más recientes tras ordenar por fecha y utiliza calibración multinomial en el 20\% posterior del entrenamiento, con separación estricta por fecha. Los hiperparámetros de clasificación se ajustan para clasificación, no se copian del problema de regresión.

Cada rama tiene promedio, ponderación convexa y stacking multinomial. Los pesos y el metamodelo se aprenden con predicciones de temporadas anteriores. Se generan predicciones de 2017/18 para inicializar el primer fold de evaluación de 2018/19; así no se sustituye ese fold por una estrategia distinta. La regularización y la comparación temporal siguen el marco de selección de modelos de James et al.~\cite{james2023}.

\subsection{Respuesta data-centric}
Se da más peso a los errores del favorito del consenso: factores 2, 3 y 5; igualdad de peso total entre aciertos y fallos; inverso de la probabilidad del resultado y su raíz. Se ensayan submuestreo y sobremuestreo 50/50 con tres semillas y recalibración posterior del esquema balanceado. Todos se construyen exclusivamente con resultados de entrenamiento. Cuando una variante deportiva usa estos pesos se denomina \emph{entrenamiento guiado por mercado}; no reemplaza al predictor sin cuotas puro.

\section{Diseño experimental}
\label{sec:diseno}
El entrenamiento y desarrollo comprenden 2014/15--2023/24. Se evalúan seis temporadas exteriores, 2018/19--2023/24, mediante ventana expansiva. Para cada una se elige la configuración utilizando las dos temporadas anteriores como validación interior; cada ajuste interior también utiliza únicamente su pasado. Las dos primeras temporadas modeladas proporcionan historia de entrenamiento antes de las ventanas de ajuste.

La comparación principal usa exactamente los mismos partidos y el log loss agregado por partido. Dentro de 0,0005 del mínimo se prefiere menor complejidad: LR, HGB, XGBoost, RF, MLP, SVM y ensembles; en regresión se mantiene un orden explícito de simplicidad. Se elige un candidato por rama. La selección entre familias a partir de esas seis temporadas conserva sesgo de desarrollo; no se presenta como una validación anidada de todo el proceso científico.

\textbf{Test revisado.} Los resultados del periodo 2024/25--2025/26 ya se conocían antes de esta revisión. Se congelaron las decisiones usando solo train/val y después se generaron predicciones corregidas para ambas ramas. No se afirma que eliminar o reentrenar el modelo convierta esos partidos en evidencia nueva: es una reevaluación sobre un test observado, no una prueba prospectiva independiente.

Los intervalos usan bootstrap por bloques de temporada, año ISO y semana ISO, con 5.000 remuestreos. Incluir el año evita agrupar dos semanas de años distintos durante temporadas extendidas. Se declaran cinco comparaciones principales por partición y se aplica Benjamini--Hochberg~\cite{benjamini1995}; los desgloses por liga son descriptivos. Los indicadores de estabilidad se interpretan como diagnósticos de cambio~\cite{siddiqi2017}, sin asegurar ausencia de drift porque una tendencia no resulte significativa.

\section{Resultados y análisis}
\label{sec:resultados}
\subsection{Enfoque con cuotas: aportación frente al precio directo}
"""
    body += rf"La selección elige {model_label(chosen['con']['candidate'].split('/')[-1])}, con log loss {f(vcon.log_loss)} frente a {f(vm.log_loss)} del consenso en validación. En el test revisado, los valores son {f(tcon.log_loss)} y {f(tm.log_loss)}. El Cuadro~\ref{{tab:con}} compara las familias; no se equipara una mejora de log loss con una mejora de ordenación. AUC y calibración se examinan de forma complementaria."
    rows = [[model_label(s), f(result.loc['con/'+s, 'log_loss']), f(result.loc['con/'+s, 'AUC_macro'], 3), f(result.loc['con/'+s, 'ECE_masa'])]
            for s in ['LR_REF', 'LR_consenso', 'LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM']]
    body += latex_table(['Modelo', 'LL', 'AUC', 'ECE'], rows, 'tab:con', 'Modelos con cuotas: predicciones exteriores de validación, con ajustes temporales interiores. LR completa incluye fundamentos; LR de referencia y de consenso solo mercado.')
    body += r"La Fig.~\ref{fig:modelos} muestra el rango de resultados en ambos enfoques. Se conservan todas las configuraciones revisadas en las evaluaciones posteriores; los ensembles no reutilizan predicciones de una versión anterior del SVM ni hiperparámetros descartados."
    body += latex_figure('revision_modelos', 'fig:modelos', 'Comparación de familias con cuotas (izquierda) y sin cuotas (derecha), sobre los mismos seis folds. Las escalas se ajustan por panel para hacer visibles las diferencias.', True)
    body += r"\subsection{Enfoque sin cuotas: construcción de un precio propio}"
    body += rf"La selección deportiva elige {model_label(chosen['sin']['candidate'].split('/')[-1])}. Obtiene log loss {f(vsin.log_loss)} en validación y {f(tsin.log_loss)} en el test revisado; sus accuracy 1X2 son {100*vsin.accuracy:.1f}\% y {100*tsin.accuracy:.1f}\%. Estos porcentajes usan todos los partidos, incluidos empates. El Cuadro~\ref{{tab:sin}} permite contrastar clasificación directa y regresión convertida en probabilidad."
    rows = [[model_label(s), f(result.loc['sin/'+s, 'log_loss']), f(result.loc['sin/'+s, 'AUC_macro'], 3), f(result.loc['sin/'+s, 'ECE_masa'])]
            for s in ['LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM', 'GD_LIN', 'GD_RF', 'GD_HGB', 'GD_SVR', 'GD_ALL']]
    body += latex_table(['Modelo', 'LL', 'AUC', 'ECE'], rows, 'tab:sin', 'Modelos sin cuotas. GD identifica regresión de diferencia de goles con calibrador temporal; ALL promedia regresión lineal, RF y HGB antes de calibrar.')
    body += r"La Fig.~\ref{fig:ablacion} responde P2: se compara historia sola, historia con plantilla e historia con plantilla y habilidades bajo la misma familia LR y selección de C. La diferencia observada pertenece a esa ablación; no se atribuye a todas las familias por extensión. La antigüedad de los ratings en 2024/25 puede limitar su valor."
    body += latex_figure('revision_ablacion', 'fig:ablacion', 'Ablación deportiva con LR. Todas las transformaciones se aprenden dentro del entrenamiento y los ratings preceden al partido.')
    body += r"\subsection{Comparación común de probabilidades}El Cuadro~\ref{tab:final} reúne referencias y modelos finales sobre la misma cohorte."
    rows = [[LABEL[k], f(m.loc[(v, k), 'log_loss']), f(m.loc[(t, k), 'log_loss']), f(m.loc[(t, k), 'AUC_macro'], 3), f(m.loc[(t, k), 'ECE_masa'])]
            for k in ['prior', 'consenso', 'Pinnacle_respaldo', 'modelo_con', 'modelo_sin']]
    body += latex_table(['Referencia/modelo', 'LL val.', 'LL test', 'AUC test', 'ECE test'], rows, 'tab:final', 'Comparación principal. Test revisado significa periodo previamente observado, reevaluado tras congelar las decisiones; ECE de igual masa.')
    body += r"Los contrastes pareados se detallan en el Cuadro~\ref{tab:contrastes}. Los intervalos miden incertidumbre del remuestreo condicionado a estos procedimientos; no corrigen por sí solos todo el sesgo de exploración."
    rows = []
    for r in c['primary'].query('split == "test_revisado"').itertuples():
        rows.append([LABEL[r.a]+' vs. '+LABEL[r.b], f(r.delta), f'[{f(r.lo)}, {f(r.hi)}]', f(r.p_BH, 3)])
    body += latex_table(['Contraste', r'$\Delta$ LL', r'IC 95\%', '$p$ BH'], rows, 'tab:contrastes', 'Diferencias pareadas en test revisado; negativo favorece al primer modelo. Familia de cinco contrastes predefinidos.')
    body += r"La Fig.~\ref{fig:cal} compara calibración por clase y la Fig.~\ref{fig:temporadas} la estabilidad temporal (P1). La variación entre temporadas no se interpreta como cambio causado por un único factor: pueden variar fuerza, cobertura y composición de ligas y casas."
    body += latex_figure('revision_calibracion', 'fig:cal', 'Calibración en validación: modelo con cuotas arriba y sin cuotas abajo, frente al consenso. Diez grupos de igual masa por clase.', True)
    body += latex_figure('revision_temporadas', 'fig:temporadas', 'Log loss por temporada para cada modelo y sus referencias. El fondo destaca el test revisado; las predicciones de validación son exteriores.', True)
    body += r"\subsection{¿Cuánto se aproxima el predictor deportivo al mercado?}"
    body += rf"En validación, la correlación con el consenso es {f(sim.corr_logit, 3)} para $\ln(p_H/p_A)$ y {f(sim.corr_pD, 3)} para la probabilidad de empate; coinciden en el favorito en el {100*sim.mismo_favorito:.1f}\% de los partidos. La fracción de ganancia sobre el prior es {100*sim.fraccion_ganancia_prior:.1f}\%, definida como $(LL_{{prior}}-LL_{{sin}})/(LL_{{prior}}-LL_{{mercado}})$. Esta razón depende del prior y de la métrica: no significa poseer ese porcentaje de toda la información de una casa. La Fig.~\ref{{fig:predictor}} muestra el parecido sin confundirlo con igualdad de calidad."
    body += latex_figure('revision_predictor', 'fig:predictor', 'Probabilidades deportivas frente al consenso en validación. La diagonal indica igualdad; la densidad muestra dónde se concentran las discrepancias.', True)
    body += r"Las probabilidades pueden transformarse en cuotas justas mediante $1/p$: estas no incluyen un margen comercial. El acierto del ganador se publica solo como indicador condicionado a partidos que realmente no empataron; la regla siempre elige entre H y A comparando sus probabilidades. No debe confundirse con accuracy 1X2 ni con una regla operativa que conozca de antemano cuáles partidos empatarán."
    body += r"\subsection{PCA, ensembles y reponderación}La Fig.~\ref{fig:reduccion} responde P3 con las mismas alternativas en ambas ramas. Todas significa que no se añade MI/PCA, manteniendo el preprocesamiento común. La Fig.~\ref{fig:ens} compara ensembles con una inicialización temporal homogénea."
    body += latex_figure('revision_reduccion', 'fig:reduccion', 'MI frente a PCA en LR para ambos enfoques: diferencia respecto a la LR sin reducción adicional. No se presupone que reducir variables deba mejorar.', True)
    body += latex_figure('revision_ensembles', 'fig:ens', 'Ensembles frente a LR completa, evaluados en los seis folds con predicciones previas para aprender pesos y stacking.', True)
    ens = pd.read_csv(c['out']/'ensemble_comparisons.csv').set_index('candidate')
    e = ens.loc['sin/mean']
    body += rf"Frente a la LR deportiva con historia y plantilla, el promedio sin cuotas reduce LL en {f(-e.delta)} (IC 95\% de la diferencia [{f(e.lo)}, {f(e.hi)}]; $p$ BH={f(e.p_BH, 3)}), con mejora en cinco de seis temporadas. El ponderado logra un valor algo menor, pero el promedio queda dentro de la tolerancia predefinida de 0,0005 y se prefiere por simplicidad. Con cuotas, el ponderado queda prácticamente igual a la LR de referencia y su intervalo incluye cero. Por tanto, el beneficio del ensemble depende de la rama; estos contrastes de desarrollo no sustituyen un test futuro."
    body += r"La Fig.~\ref{fig:pesos} muestra el experimento solicitado por el profesor. El objetivo sigue siendo calibración sobre la distribución real de partidos, no maximizar el acierto en el subconjunto donde el favorito pierde. Se reporta el efecto medio sin afirmar que todos los esquemas empeoren en cada temporada; la recalibración se evalúa en las cinco temporadas para las que existe una anterior con predicciones del mismo procedimiento."
    body += latex_figure('revision_pesos', 'fig:pesos', 'Pesos y remuestreo guiados por errores del mercado. La rama sin entradas de cuotas se identifica como guiada por mercado cuando usa esos pesos.', True)
    body += r"\subsection{Backtest con reglas fijas}La Fig.~\ref{fig:bt} responde P4. Se apuesta una unidad al lado con mayor $p_j o_j-1$ si supera 0,02, sin volver a elegir el umbral por ROI. Se comparan los dos modelos y las referencias directas con precios observados de Bet365 y Pinnacle; esta última excluye el periodo de fuente no fiable. Un retorno puntual positivo no basta para declarar rentabilidad cuando el intervalo incluye pérdidas."
    body += latex_figure('revision_backtest', 'fig:bt', 'ROI por unidad apostada e intervalos por bloques para los dos modelos. Las cantidades de apuestas dependen de la casa y su cobertura fiable.', True)
    body += r"""
\section{Discusión, limitaciones y ética}
\label{sec:discusion}
La comparación con cuotas mide el aporte adicional a un precio ya informativo; el predictor sin cuotas mide cuánto puede construirse con datos deportivos. Tienen valor científico aunque ninguno supere al mercado. Un modelo de menor log loss realiza una predicción probabilística mejor según esa regla; si el AUC cambia poco, eso sugiere que buena parte de la diferencia reside en calibración, sin convertirlo en una identidad demostrada.

Los resultados actuales reemplazan los números anteriores porque se corrigieron entradas y procedimientos, no porque se buscara reproducir una conclusión previa. Cada transformación llega a las predicciones evaluadas y se documenta la configuración por fold. La disponibilidad temporal elimina una fuga, pero puede deteriorar el rendimiento al utilizar plantillas antiguas: ese coste es preferible a presentar una estimación retrospectiva como disponible en su día.

Persisten limitaciones: no hay alineaciones reales ni lesiones; la historia cubre solo liga; las cuotas carecen de sincronización exacta; los ratings de algunas temporadas son antiguos; las rejillas son acotadas; y el test revisado ya había sido observado. Ningún intervalo convierte ese último periodo en validación prospectiva. Una próxima evaluación debería fijar primero el protocolo y esperar partidos nuevos. Tampoco se descarta drift por falta de significación ni se extrapola el resultado a otras ligas o deportes.

Las fuentes se utilizan para fines académicos, sin redistribuir sus CSV crudos. No se modelan datos sensibles de personas. La calidad desigual de cobertura puede perjudicar a equipos con menos historia, por lo que se publica la cobertura en lugar de ocultarla con imputaciones globales. El análisis de rentabilidad es experimental y no constituye recomendación de apuestas.

\section{Conclusiones}
\label{sec:conclusiones}
"""
    body += rf"Se entregan dos predictores con un protocolo común. Con cuotas se obtiene log loss {f(vcon.log_loss)} en validación y {f(tcon.log_loss)} en test revisado; sin cuotas, {f(vsin.log_loss)} y {f(tsin.log_loss)}. La referencia de consenso obtiene {f(vm.log_loss)} y {f(tm.log_loss)}. Estas comparaciones permiten valorar por separado el ajuste del precio y la capacidad deportiva propia."
    body += r"""

P1 se responde mediante resultados por temporada y diagnósticos de cobertura, sin afirmar ausencia universal de cambio. P2 se responde con una ablación explícita de historia, plantilla y habilidades. P3 se responde comparando MI y PCA sobre los mismos folds y transformaciones internas. P4 se responde con backtests de reglas fijas, intervalos y referencias de mercado: no debe deducirse rentabilidad solo de un promedio positivo. Los ensembles y la reponderación se juzgan por sus resultados corregidos en cada rama, sin exigir que superen a un método sencillo.

Como continuación, interesa incorporar snapshots más frecuentes, alineaciones y ausencias verificables, así como datos xG históricos. Los modelos de conteo de goles ofrecen otro contraste~\cite{dixon1997,karlis2003}. El siguiente test debe ser verdaderamente futuro respecto al protocolo congelado.
"""
    bibliography = old[old.index(r'\begin{thebibliography}'):]
    bibliography = bibliography.replace('2025.\n\bibitem{kagglefifa}', 'consultado en noviembre de 2026.\n\bibitem{kagglefifa}')
    # Close related float blocks, allowing the following methods discussion to
    # fill the page while a two-column figure waits for the next page top.
    preamble = preamble.replace(r'\usepackage[section]{placeins}', r'\usepackage{placeins}')
    for section in ('Resultados y análisis',):
        body = body.replace(r'\section{'+section+'}', r'\FloatBarrier'+'\n'+r'\section{'+section+'}')
    (ROOT/'docs/informe.tex').write_text(preamble+body+'\n'+r'\FloatBarrier'+'\n'+bibliography)


def presentation(c):
    # Visual direction: white canvas; teal = with market, blue = without market;
    # Segoe UI/system sans; direct comparative plots and large readable evidence.
    m = c['final']; val, test = 'validacion', 'test_revisado'
    chosen = c['frozen']['selected']; sim = c['similarity'].query('split == "validacion" and reference == "consenso"').iloc[0]
    def metric(split, name, col='log_loss'): return f(m.loc[(split, name), col])
    def img(name, alt): return f'<img src="img/{name}.png" alt="{html.escape(alt)}">'
    def table(split):
        rows = ''.join(f'<tr><td>{LABEL[k]}</td><td>{metric(split,k)}</td><td>{metric(split,k,"AUC_macro")}</td><td>{metric(split,k,"ECE_masa")}</td></tr>' for k in ('consenso','Pinnacle_respaldo','modelo_con','modelo_sin'))
        return '<table><thead><tr><th>Modelo / referencia</th><th>Log loss</th><th>AUC</th><th>ECE masa</th></tr></thead><tbody>'+rows+'</tbody></table>'
    slides = []
    def slide(title, body, note='', cls=''):
        n = len(slides)+1
        slides.append(f'<section class="slide {cls}" data-n="{n}"><h2>{title}</h2><div class="content">{body}</div><footer><span>{note}</span><span>{n} / 13</span></footer></section>')
    slide('Predecir fútbol con y sin cuotas', '<p class="lead">Dos preguntas, el mismo estándar de evaluación.</p><div class="two"><div class="con"><h3>Con cuotas</h3><p>¿Qué añade un modelo al precio del mercado?</p></div><div class="sin"><h3>Sin cuotas</h3><p>¿Qué probabilidades podemos construir con información deportiva?</p></div></div><p class="authors">José Luis Alva · Juan Carlos Ticlia · Elmer Villegas · Joseph Cose</p>', 'Machine Learning · UTEC · Sección 3')
    slide('Comparar probabilidades, no solo aciertos', '<ul><li>Salida: P(local), P(empate), P(visitante).</li><li><b>Log loss:</b> penaliza la confianza equivocada.</li><li><b>AUC:</b> mide ordenación; <b>ECE:</b> calibración.</li><li>El acierto del ganador excluye empates reales: no es accuracy 1X2.</li></ul>', 'P1 estabilidad · P2 plantilla · P3 MI/PCA · P4 rentabilidad')
    slide('Datos con disponibilidad histórica acreditada', img('revision_datos','Distribución y cobertura')+'<p>36.150 partidos originales; 36.119 comparables. Ratings anteriores a cada partido, con faltantes y antigüedad visibles.</p>', '9 ligas · 2014/15–2025/26 · calentamiento 2013/14')
    slide('Entrenar con pasado y documentar el test observado', '<div class="timeline"><p>2014/15–2017/18: historia y ajuste inicial</p><p>2018/19–2023/24: seis folds exteriores; ajuste interior con pasado</p><p class="test">2024/25–2025/26: reevaluación tras congelar decisiones</p></div><ul><li>Imputación, escalado, poda y selección dentro de cada entrenamiento.</li><li>El test ya fue observado: reentrenar no lo vuelve desconocido.</li></ul>', 'Mismos partidos y métricas para ambos enfoques')
    slide('Un experimento común para dos entradas distintas', '<div class="two"><div class="con"><h3>Mercado + fundamentos</h3><p>Consenso y Pinnacle fiable; comparación directa sin aprendizaje.</p></div><div class="sin"><h3>Solo fundamentos</h3><p>Elo, forma, plantilla, habilidades y contexto histórico.</p></div></div><p>LR · HGB · XGBoost · RF · MLP · SVM</p><p>Sin cuotas: también diferencia de goles convertida a probabilidades.</p>', 'Rejillas fijadas · calibración temporal · configuraciones revisadas propagadas')
    slide('Con cuotas: medir qué añade el aprendizaje', f'<p class="lead con-text">Elegido: {model_label(chosen["con"]["candidate"].split("/")[-1])}</p>'+table(val)+f'<p>En validación: {metric(val,"modelo_con")} frente a {metric(val,"consenso")} del consenso.</p>', 'Las referencias directas separan elección del precio y aprendizaje')
    slide('Con cuotas: pérdidas, ordenación y calibración', img('revision_calibracion_con','Calibración del modelo con cuotas')+'<p>Una mejora de log loss no implica necesariamente ordenar mejor: contrastamos también AUC y calibración.</p>', 'Curvas por clase · validación exterior · diez grupos de igual masa')
    slide('Sin cuotas: construir el predictor deportivo', img('revision_ablacion','Ablación sin cuotas')+'<p>Historia → plantilla → habilidades. Los ratings se cruzan por fecha y pueden proceder de una versión anterior.</p>', 'Comparación bajo LR; cada C se elige con temporadas anteriores')
    slide('Sin cuotas: un precio propio que podemos contrastar', f'<p class="lead sin-text">Elegido: {model_label(chosen["sin"]["candidate"].split("/")[-1])} · LL validación {metric(val,"modelo_sin")}</p>'+img('revision_predictor','Probabilidades propias frente al consenso')+f'<p>Correlación logit con consenso {sim.corr_logit:.3f}; favorito coincidente {100*sim.mismo_favorito:.1f} %. Cuotas justas = 1/P, sin margen.</p>', 'Parecido al mercado no equivale a igual calidad ni rentabilidad')
    slide('Comparación final con el mismo periodo', table(test)+'<p>La comparación utiliza los mismos 5.999 partidos. Los contrastes pareados y sus intervalos están en el informe.</p>', 'Test revisado ya observado · no es evidencia prospectiva independiente')
    slide('PCA, ensembles y respuesta al profesor', '<div class="two">'+img('revision_ensembles','Ensembles en ambos enfoques')+img('revision_reduccion','MI y PCA en ambos enfoques')+'</div><p>El ensemble mejora sin cuotas; con cuotas se prefiere LR. También probamos pesos y resampling 50/50; esas variantes son guiadas por mercado.</p>', 'Predicciones previas inicializan el primer ensemble; no se cambia de método en ese fold')
    slide('Rentabilidad y límites de los datos', img('revision_backtest','Backtest de ambos modelos')+'<p>τ=0,02 fijo; una unidad por apuesta. Pinnacle se excluye desde el 23/07/2025. Los precios no tienen sincronización exacta entre casas.</p>', 'Un ROI puntual positivo no demuestra rentabilidad · no es consejo de apuestas')
    slide('Dos contribuciones, una comparación honesta', f'<div class="two"><div class="con"><h3>Con cuotas</h3><p>Log loss revisado: <b>{metric(test,"modelo_con")}</b></p><p>Medimos el aporte respecto al precio directo.</p></div><div class="sin"><h3>Sin cuotas</h3><p>Log loss revisado: <b>{metric(test,"modelo_sin")}</b></p><p>Construimos probabilidades con información deportiva.</p></div></div><p>Limitaciones: ratings antiguos, alineaciones del juego, solo ligas y test observado. La siguiente prueba debe usar partidos futuros.</p><p class="repo">github.com/jcarlos-t/MLProject</p>', 'Preguntas · resultados reproducibles · ambos enfoques con igual rigor')
    style = '''
:root{--ink:#0f172a;--mut:#475569;--con:#0d9488;--sin:#2563eb}*{box-sizing:border-box}body{margin:0;background:#e2e8f0;color:var(--ink);font-family:"Segoe UI",Arial,sans-serif;overflow:hidden}#stage{width:1280px;height:720px;position:absolute;left:50%;top:50%;transform-origin:center}.slide{position:absolute;inset:0;background:white;padding:45px 64px 58px;display:none;flex-direction:column}.slide.active{display:flex}h2{font-size:38px;line-height:1.15;margin:0 0 24px;font-weight:700;max-width:1100px}h3{font-size:27px;margin:0 0 16px}.content{flex:1;display:flex;flex-direction:column;gap:15px;min-height:0;justify-content:center}.content>img{max-height:330px;max-width:1120px;object-fit:contain;align-self:center}p,li{font-size:23px;line-height:1.38;margin:0}ul{margin:0;padding-left:28px}li{margin:15px 0}.lead{font-size:30px}.authors{margin-top:35px;font-size:20px;color:var(--mut)}.two{display:flex;gap:34px;align-items:center}.two>div{flex:1;padding:25px 26px;background:#f8fafc;border-left:5px solid}.two>img{width:49%;max-height:295px;object-fit:contain}.con{border-color:var(--con)!important}.sin{border-color:var(--sin)!important}.con-text{color:var(--con)}.sin-text{color:var(--sin)}footer{position:absolute;bottom:20px;left:64px;right:64px;display:flex;justify-content:space-between;font-size:14px;color:var(--mut);gap:25px}footer span:first-child{max-width:1030px}table{border-collapse:collapse;width:100%;font-size:23px}th,td{text-align:right;padding:12px 15px;border-bottom:1px solid #cbd5e1}th:first-child,td:first-child{text-align:left}thead{background:#f1f5f9}tbody tr:nth-last-child(2){color:var(--con);font-weight:600}tbody tr:last-child{color:var(--sin);font-weight:600}.timeline p{padding:18px 22px;margin-bottom:12px;background:#f1f5f9}.timeline .test{background:#fef3c7}.repo{color:var(--sin);font-size:20px}button{position:fixed;right:20px;bottom:12px;z-index:10;padding:8px 14px;border:1px solid #94a3b8;background:white;color:var(--ink);border-radius:4px;cursor:pointer}button:focus-visible{outline:3px solid var(--sin)}
@media print{@page{size:1280px 720px;margin:0}html,body{width:1280px;height:auto;overflow:visible;background:white}#stage{position:static;transform:none!important;height:auto}.slide{position:relative;display:flex!important;width:1280px;height:720px;break-after:page;page-break-after:always}.slide:last-child{break-after:auto;page-break-after:auto}button{display:none}}
'''
    script = '''let i=Math.max(0,Math.min(12,(parseInt(location.hash.slice(1))||1)-1));const slides=[...document.querySelectorAll('.slide')],stage=document.querySelector('#stage');function fit(){const s=Math.min(innerWidth/1280,innerHeight/720);stage.style.transform=`translate(-50%,-50%) scale(${s})`}function show(){slides.forEach((s,k)=>s.classList.toggle('active',k===i));history.replaceState(null,'','#'+(i+1))}addEventListener('resize',fit);addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();i=Math.min(12,i+1)}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();i=Math.max(0,i-1)}else if(e.key==='Home')i=0;else if(e.key==='End')i=12;show()});document.querySelector('button').addEventListener('click',()=>print());fit();show();'''
    path = ROOT/'docs/presentacion'; (path/'img').mkdir(exist_ok=True)
    for p in (ROOT/'docs/figures').glob('revision_*.png'): shutil.copy2(p, path/'img'/p.name)
    (path/'presentacion.html').write_text('<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fútbol con y sin cuotas</title><style>'+style+'</style></head><body><main id="stage">'+''.join(slides)+'</main><button aria-label="Imprimir presentación">Imprimir / PDF</button><script>'+script+'</script></body></html>')


def render(out=None):
    out = out or current(); c = context(out)
    if not (out/'market_guided_comparisons.csv').exists(): development_comparisons(out)
    figures(c); paper(c); presentation(c)
    # Both deliverables draw from exactly this exported table.
    c['final'].reset_index().to_csv(ROOT/'docs/resultados_revision.csv', index=False)
    print('Figuras, informe y 13 diapositivas actualizados desde', out.name)


if __name__ == '__main__': render()
