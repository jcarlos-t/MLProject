#!/usr/bin/env python3
"""Generate the nine teaching notebooks around the shared, tested implementation."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT/'scripts/notebooks'
BOOT = """from pathlib import Path
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd()
while not (ROOT/'scripts/mlcore').is_dir():
    if ROOT == ROOT.parent: raise RuntimeError('No se encontro la raiz del proyecto')
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'scripts'))
from mlcore.experiment import initialize, aligned
from mlcore.data import read_data, FOLDS, TEST
OUT = initialize()
print('Ejecucion verificada:', OUT.name)
"""


def write(name, title, intro, steps):
    cells = [nbf.v4.new_markdown_cell('# '+title+'\n\n'+intro),
             nbf.v4.new_markdown_cell('## Inicialización\n\nUsamos la misma identidad de datos, código y protocolo en todos los cuadernos.'),
             nbf.v4.new_code_cell(BOOT),
             nbf.v4.new_markdown_cell('**Qué encontramos y cómo se usa.** La identidad evita mezclar artefactos de protocolos distintos. Los cálculos científicos viven en `scripts/mlcore`; estos cuadernos explican, ejecutan y muestran cada etapa.')]
    for heading, before, code, after in steps:
        cells += [nbf.v4.new_markdown_cell('## '+heading+'\n\n'+before), nbf.v4.new_code_cell(code),
                  nbf.v4.new_markdown_cell('**Qué encontramos / decisión.** '+after)]
    nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
              'language_info': {'name': 'python', 'version': '3.14.8'}})
    nbf.write(nb, NB/name)


def main():
    write('01_eda.ipynb', 'Datos y EDA: dos enfoques, una cohorte',
      'Primero estudiamos las fuentes sin ajustar modelos. Las cuotas sirven de entrada en una rama y de referencia externa en la otra. No se imputa ni selecciona usando el test.', [
      ('Integridad de las fuentes', 'Las 117 tablas de partidos y 12 snapshots deben coincidir con las huellas verificadas.',
       "from mlcore.data import verify_inputs, load_matches\nmanifest, dates = verify_inputs()\nraw = load_matches()\nprint(len(manifest), 'crudos verificados')\nprint('Partidos por temporada:')\ndisplay(raw.groupby('Season').size().rename('n').to_frame())",
       'La ejecución se detiene si falta o cambia un crudo. El calentamiento 2013/14 solo alimentará historia.'),
      ('Exploración de train/val', 'Las distribuciones de desarrollo se calculan sin las dos temporadas reservadas.',
       "tv = raw[raw.Season.ge(1415) & ~raw.Season.isin(TEST)]\ndisplay(tv.FTR.value_counts(normalize=True).rename('proporcion').to_frame())\nprint('Duplicados:', int(raw.duplicated(['Div','Season','Date','HomeTeam','AwayTeam']).sum()))\nfig, ax = plt.subplots(figsize=(6,3)); tv.FTR.value_counts().reindex(['H','D','A']).plot.bar(ax=ax); ax.set_title('Resultados en desarrollo'); plt.show()\ndisplay(tv[['HS','AS','HST','AST','HF','AF']].isna().mean().rename('faltantes').to_frame())",
       'Los faltantes de estadísticas se mantienen. El objetivo principal es evaluar probabilidades, sin tratar el desbalance como un error que obligue a reequilibrar.'),
      ('Fechas acreditadas', 'Las fechas antiguas se recuperaron de la misma versión y actualización SoFIFA; cada entrada conserva la fuente y la huella del CSV.',
       "display(pd.DataFrame(dates['snapshots'])[['year','available_at','evidence','source']])",
       'Una versión anual no es información pretemporada por definición. Su uso se permite solo después de la fecha acreditada.')])
    write('02_feature.ipynb', 'Variables disponibles antes del partido',
      'Reconstruimos forma, fuerza histórica, plantilla y mercado con estados temporales. Las dos ramas usarán los mismos partidos de comparación.', [
      ('Construcción cronológica', 'Todos los partidos de una fecha leen el estado previo antes de actualizarlo con resultados del día. La implementación está en `mlcore/data.py`.',
       "from mlcore.data import prepare\nif not (OUT/'dataset.csv').exists(): prepare(OUT)\ndf, sets = read_data(OUT)\nprint('Cohorte:', df.shape, '| train/val:', int((~df.Season.isin(TEST)).sum()), '| test:', int(df.Season.isin(TEST).sum()))\ndisplay(pd.DataFrame({'grupo':sets.keys(), 'variables':[len(v) for v in sets.values()]}))",
       'Los partidos sin mercado quedan fuera de la comparación pareada pero ya alimentaron el historial. No se hace poda global por correlación.'),
      ('Disponibilidad y calidad', 'Se exige snapshot estrictamente anterior. Pinnacle se deja de usar como precio fiable desde el 23/07/2025.',
       "assert ((df.H_snapshot_at < df.Date) | df.H_snapshot_at.isna()).all()\nassert ((df.A_snapshot_at < df.Date) | df.A_snapshot_at.isna()).all()\nassert not df.loc[df.Date.ge('2025-07-23'),'pinnacle_fiable'].any()\ndisplay(pd.read_csv(OUT/'coverage.csv'))\nassert not any(c.startswith(('MKT','REF','PIN')) for c in sets['sin_cuotas'])",
       'La cobertura inferior al 100% y los ratings antiguos se reportan. No se rellenan habilidades con una media de toda la temporada.'),
      ('Separación y diagnóstico de variables', 'La revisión descriptiva no elimina columnas silenciosamente; las decisiones aprendidas se ejecutarán dentro de los folds.',
       "tv = df[~df.Season.isin(TEST)]\nreview = tv[sets['sin_cuotas']+sets['mercado']].agg(['count','mean','std']).T\nreview['missing_fraction'] = tv[review.index].isna().mean()\nreview.to_csv(OUT/'feature_diagnostics_trainval.csv')\ndisplay(review.head(15))\nprint('Orden cronologico:', df.Date.is_monotonic_increasing)",
       'Se exportan train_val y test en el directorio de ejecución. La tabla es diagnóstica; la selección efectiva será trazable por el pipeline.')])
    write('03_preprocesamiento.ipynb', 'Preprocesamiento dentro del entrenamiento',
      'Imputación, escalado, poda y reducción tienen que aprenderse con la parte de entrenamiento de cada partición, incluidas las interiores.', [
      ('Medianas y esquema de columnas', 'Mostramos un ajuste de demostración limitado a las primeras dos temporadas. No es un ajuste global reutilizado en validación.',
       "from mlcore.models import preprocessing\ndf, sets = read_data(OUT)\ntr = df[df.Season.lt(1617)]\nfor branch in ['con_cuotas','sin_cuotas']:\n    pre = preprocessing(); X = pre.fit_transform(tr[sets[branch]].to_numpy(float), tr.y.to_numpy())\n    print(branch, '| entrada:', len(sets[branch]), '| salida tras poda:', X.shape[1], '| filas:', len(tr))\n    assert np.isfinite(X).all()",
       'Los faltantes se resuelven dentro de cada entrenamiento, conservando columnas completamente vacías e indicadores. No se consulta el test para decidir F1/F2.'),
      ('MI frente a PCA', 'Se registran las alternativas antes de evaluar: MI-5, MI-10, PCA-5 y PCA-95%, con la misma familia LR y búsqueda de C en ambas ramas.',
       "from mlcore.models import specifications, GRIDS\ndisplay(pd.DataFrame([{'enfoque':s.branch,'modelo':s.name,'familia':s.family,'reduccion':s.reduction,'n_variables':len(s.columns),'configuraciones':len(GRIDS[s.family])} for s in specifications(sets)]))",
       'Los resultados se generarán en 04 y se interpretarán en 05. No se elige la mejor representación usando la temporada que se predice.'),
      ('Pruebas contra fuga', 'Las pruebas alteran resultados y ratings futuros y exigen que las variables del pasado no cambien.',
       "import subprocess\nr = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)\nprint(r.stdout+r.stderr)\nr.check_returncode()",
       'Se comprueban también ECE constante, cuotas inválidas, separación del SVM y bloques ISO. La unión cartesiana del calentamiento desaparece del nuevo cálculo.')])
    write('04_modelo.ipynb', 'Modelos con y sin cuotas: validación temporal',
      'Cada rama compara seis familias de clasificación. Sin cuotas también se evalúa diferencia de goles convertida a probabilidades. Ningún resultado de test participa en esta etapa.', [
      ('Ajustes interiores y predicciones exteriores', 'Las configuraciones de cada candidato se eligen con las dos temporadas anteriores a cada fold. Cada split se calcula una vez y se identifica por huellas de datos, código y protocolo.',
       "from mlcore.experiment import nested_cv\nif not (OUT/'oof_base.csv').exists(): nested_cv(OUT,jobs=4)\ndf, sets = read_data(OUT)\nbase = pd.read_csv(OUT/'oof_base.csv')\nassert not base.Season.isin(TEST).any()\ndisplay(pd.read_csv(OUT/'model_comparison.csv'))",
       'Las seis temporadas exteriores son comunes. Las predicciones de 2017/18 solo inicializan los ensembles; no se incluyen en el resultado principal.'),
      ('Identidad y alineación', 'Cada predicción se enlaza por identificador de partido, no por la posición de una tabla antigua.',
       "for candidate,g in base.groupby('candidate'):\n    assert not g.match_id.duplicated().any()\n    check = g.merge(df[['match_id','y']],on='match_id',validate='one_to_one',suffixes=('_p','_d'))\n    assert check.y_p.eq(check.y_d).all()\nprint('Predicciones alineadas:', base.candidate.nunique(), 'candidatos')",
       'Estas predicciones revisadas son las que consumirán evaluación y ensembles; no hay un camino paralelo con un SVM anterior.')])
    write('05_hiperparametros.ipynb', 'Selección temporal y reducción de variables',
      'La búsqueda y sus resultados son parte del experimento común. Los parámetros no se copian entre regresión y clasificación.', [
      ('Configuraciones por fold', 'Cada entrada registra temporadas interiores, pérdidas de la rejilla y configuración elegida.',
       "params = json.loads((OUT/'parameters_by_fold.json').read_text())\ndisplay(pd.DataFrame(params).head(25))\nassert all(max(p['inner']) < p['season'] for p in params)\nprint('Configuraciones exteriores registradas:',len(params))",
       'El SVM calibra después de su entrenamiento base. La MLP usa tres semillas y los árboles no hacen parada temprana con un split aleatorio.'),
      ('MI/PCA para ambas ramas', 'Comparamos las alternativas predefinidas con todas las variables después del preprocesamiento común.',
       "results = pd.read_csv(OUT/'model_comparison.csv')\nfor branch in ['con','sin']:\n    names = [branch+'/LR'+x for x in ['', '_MI5','_MI10','_PCA5','_PCA95']]\n    t = results.set_index('candidate').loc[names].copy()\n    t['delta_vs_LR'] = t.log_loss-t.iloc[0].log_loss\n    display(t[['log_loss','delta_vs_LR','AUC_macro','ECE_masa']])",
       'La reducción es útil solo si sus probabilidades mejoran. No se fuerza una explicación causal basada únicamente en qué método gana.')])
    write('06_mercado.ipynb', 'Comparar con el mercado directo',
      'Primero se mide el valor añadido de usar cuotas. Después se compara la señal deportiva con una referencia que no utiliza aprendizaje.', [
      ('Referencias observadas', 'Consenso y Pinnacle con respaldo son distribuciones directas. Las casas incompletas o no fiables no se imputan.',
       "from mlcore.experiment import references\nfrom mlcore.metrics import metrics\ndf, sets = read_data(OUT)\nrows=[]\nfor s in FOLDS:\n    va=df[df.Season.eq(s)]; tr=df[df.Season.lt(s)]\n    for name,p in references(va,tr).items(): rows.append(dict(Season=s,referencia=name,**metrics(va.y,p)))\nreferences_table=pd.DataFrame(rows)\ndisplay(references_table.groupby('referencia')[['log_loss','AUC_macro','ECE_masa']].mean())",
       'Las medias por temporada son un diagnóstico; la tabla principal final agregará por partido. El prior se estima con entrenamiento.'),
      ('Diagnósticos pareados e importancia', 'Calculamos comparaciones descriptivas de validación y permutación por grupos en LR. No se usan para mirar el test.',
       "from mlcore.report import validation_diagnostics\nif not (OUT/'validation_diagnostics.csv').exists(): validation_diagnostics(OUT)\ndisplay(pd.read_csv(OUT/'validation_diagnostics.csv'))\ndisplay(pd.read_csv(OUT/'importance_groups.csv').groupby(['branch','grupo']).delta.agg(['mean','std']))",
       'Estos contrastes diagnósticos corrigen su propia familia; las cinco comparaciones principales finales se declararán aparte. Importancia por permutación no equivale a causalidad.')])
    write('07_evaluacion.ipynb', 'Evaluación probabilística comparable',
      'Evaluamos las mismas familias y las mismas temporadas. ECE de masa y ancho son métricas distintas; el acierto condicionado del ganador se mantiene separado de accuracy 1X2.', [
      ('Métricas por enfoque y clase', 'Todas las métricas comparten la implementación de `mlcore/metrics.py`.',
       "df,sets=read_data(OUT); base=pd.read_csv(OUT/'oof_base.csv'); va=df[df.Season.isin(FOLDS)]\nfrom mlcore.metrics import metrics\nrows=[]\nfor branch in ['con','sin']:\n    for family in ['LR','HGB','XGB','RF','MLP','SVM']:\n        name=branch+'/'+family; p=aligned(base,name,va)\n        rows.append(dict(candidate=name,**metrics(va.y,p)))\ndisplay(pd.DataFrame(rows))",
       'Los resultados describen probabilidades de las tres clases. Una predicción de empate puede estar bien calibrada aunque rara vez sea el argmax.'),
      ('Curvas ROC exteriores', 'Mostramos LR con y sin cuotas frente al consenso, sin elegir modelos con estas curvas.',
       "from sklearn.metrics import roc_curve, roc_auc_score\nfig,axes=plt.subplots(1,3,figsize=(14,3.5))\nfor j,ax in enumerate(axes):\n    for name,color in [('con/LR','#0d9488'),('sin/LR','#2563eb')]:\n        p=aligned(base,name,va)[:,j]; fpr,tpr,_=roc_curve(va.y.eq(j),p)\n        ax.plot(fpr,tpr,color=color,label=f'{name}: {roc_auc_score(va.y.eq(j),p):.3f}')\n    ax.plot([0,1],[0,1],':',color='gray'); ax.set_title(['Local','Empate','Visitante'][j]); ax.legend(fontsize=8)\nplt.tight_layout(); plt.show()",
       'Se examina discriminación por clase; no se deduce que probabilidades sean equivalentes porque los AUC se parezcan.')])
    write('08_pesos_ensembles.ipynb', 'Pesos, remuestreo y ensembles en ambas ramas',
      'Se completa la respuesta al profesor antes de congelar modelos y evaluar el test revisado.', [
      ('Predicciones previas para ensembles', 'Los pesos y el stacking usan temporadas anteriores; 2017/18 inicializa el primer fold. No hay un método de respaldo distinto en 2018/19.',
       "from mlcore.experiment import feedback\nif not (OUT/'oof_candidates.csv').exists(): feedback(OUT)\ndisplay(pd.read_csv(OUT/'ensemble_weights.csv'))\nresults=pd.read_csv(OUT/'model_comparison.csv')\ndisplay(results[results.candidate.str.contains('mean|weighted|stack')])",
       'Ambos ensembles se evalúan en las mismas seis temporadas y consumen configuraciones revisadas.'),
      ('Pesos y remuestreo solicitados', 'Se prueban LR y HGB de ambas ramas. El entrenamiento ponderado según el mercado se etiqueta explícitamente.',
       "display(pd.read_csv(OUT/'market_guided_results.csv'))\nprint('Recalibracion posterior:',pd.read_csv(OUT/'oof_recalibration.csv').candidate.unique().tolist())",
       'No se afirma que todas las variantes empeoren en cada temporada. El modelo sin cuotas puro no recibe estos pesos; son experimentos adicionales guiados por mercado.'),
      ('Congelar una selección por enfoque', 'Se elige menor log loss; dentro de 0,0005, se prefiere simplicidad. El test revisado aún no se genera en esta etapa.',
       "from mlcore.experiment import freeze\nfrozen=freeze(OUT)\nprint(json.dumps(frozen['selected'],indent=2,ensure_ascii=False))",
       'La selección queda vinculada a huellas de predicciones y protocolo. Cambiarla exige una ejecución nueva y no permite llamar prospectivo al test observado.')])
    write('09_modelos_finales.ipynb', 'Dos modelos finales y evaluación revisada',
      'Los modelos se entrenan desde cero con train/val. El test ya se observó antes de la auditoría: esta evaluación se etiqueta con esa limitación.', [
      ('Entrenamiento final y evaluación cerrada', 'Las decisiones se leen del archivo congelado. Si la misma ejecución ya fue evaluada, se leen sus resultados en lugar de volver a seleccionar.',
       "from mlcore.experiment import evaluate\nfinal=evaluate(OUT)\ndisplay(final)\ndisplay(pd.read_csv(OUT/'primary_comparisons.csv'))",
       'Ambos enfoques reciben evaluación sobre los mismos partidos. La identidad del test no se restaura borrando un modelo.'),
      ('Probabilidades, cuotas justas y backtest', 'La semejanza al mercado se calcula por referencia. La rentabilidad utiliza tau fijo y solo precios considerados fiables.',
       "display(pd.read_csv(OUT/'similarity.csv'))\ndisplay(pd.read_csv(OUT/'backtest.csv'))\nimport joblib\ndf,sets=read_data(OUT); te=df[df.Season.isin(TEST)]\nfor branch in ['con','sin']:\n    model=joblib.load(OUT/f'modelo_{branch}_cuotas.joblib'); p=model.predict_proba(te.iloc[:3])\n    print(branch,'probabilidades',p,'cuotas justas',1/p)\n    assert np.isfinite(p).all() and np.allclose(p.sum(1),1)",
       'Las cuotas justas no incluyen margen comercial. Para usar el modelo deportivo se proporcionan sus variables históricas, nunca el resultado del partido.'),
      ('Entregables desde la misma tabla', 'Las figuras, el informe y las 13 diapositivas se generan desde esta ejecución, no desde cifras copiadas a mano.',
       "from mlcore.report import render\nrender(OUT)\nprint('Fuentes de informe y presentacion regeneradas; compilar PDF con scripts/report/build.py')",
       'Las conclusiones de ambas ramas deben seguir los resultados corregidos. La siguiente evaluación independiente requiere partidos futuros.')])
    # Both originals were archived before replacing the documented order.
    for old in ('08_modelo_final.ipynb','09_pesos_ensembles.ipynb'):
        p=NB/old
        if p.exists(): p.unlink()


if __name__ == '__main__': main()
