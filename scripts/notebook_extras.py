"""Exploration cells appended to the teaching notebooks.

Every analysis here is descriptive and uses train/val only, except the last
notebook, which describes the already frozen test predictions. Nothing in this
file changes a model, a threshold or the frozen selection.
Each entry: (heading, text before the code, code, finding written after seeing the output).
"""

EXTRAS = {
'01_eda.ipynb': [
('Cobertura de estadísticas por liga y temporada',
 'Una media global de faltantes puede esconder que el problema se concentra en pocas ligas. Lo revisamos por liga y temporada con los tiros del local.',
 r"""missing = tv.assign(missing=tv.HS.isna()).pivot_table(index='Div', columns='Season', values='missing', aggfunc='mean')
fig, ax = plt.subplots(figsize=(10, 4)); im = ax.imshow(missing, aspect='auto', vmin=0, vmax=1)
ax.set_yticks(range(len(missing)), missing.index); ax.set_xticks(range(len(missing.columns)), missing.columns, rotation=45)
fig.colorbar(im, ax=ax, label='Fracción sin tiros'); plt.show()
display(missing.round(2))""",
 'Los faltantes de tiros no están repartidos al azar: Bélgica, Países Bajos, Portugal y Turquía no tienen estadísticas de partido en 2014/15–2016/17 y están casi completas después (a Turquía le falta un 8 % en 2022/23). Las otras cinco ligas están completas. Es un faltante estructural: se conserva con su indicador en vez de borrar esas filas.'),
('Frecuencia del resultado por temporada',
 'Si la proporción de victorias locales, empates y visitantes cambia mucho entre temporadas, un prior fijo sería una mala referencia.',
 r"""rates = pd.crosstab(tv.Season, tv.FTR, normalize='index').reindex(columns=['H', 'D', 'A'])
rates.plot(marker='o', figsize=(9, 3), ylim=(0, 0.6)); plt.ylabel('Frecuencia observada'); plt.show()
display(rates.round(3))""",
 'La victoria local oscila entre 40,4 % (2020/21, jugada mayormente sin público) y 47,6 % (2016/17), y el empate entre 23,1 % y 26,0 %. El prior de cada temporada se calcula solo con temporadas anteriores. Este cambio ayuda a entender por qué 2020/21 es la temporada con mayor log loss para todos los modelos.'),
('Goles y ventaja de local por liga',
 'Comparamos las ligas entre sí y contrastamos la distribución de goles con una Poisson, que es el supuesto habitual de los modelos de conteo de goles.',
 r"""from scipy.stats import poisson
liga = tv.groupby('Div').agg(partidos=('FTR', 'size'), goles_local=('FTHG', 'mean'), goles_visitante=('FTAG', 'mean'),
    local=('FTR', lambda s: s.eq('H').mean()), empate=('FTR', lambda s: s.eq('D').mean()), visitante=('FTR', lambda s: s.eq('A').mean()))
display(liga.round(3))
k = np.arange(0, 9); fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
for ax, col, name in [(axes[0], 'FTHG', 'local'), (axes[1], 'FTAG', 'visitante')]:
    ax.bar(k, tv[col].value_counts(normalize=True).reindex(k, fill_value=0), alpha=.6, label='observado')
    ax.plot(k, poisson.pmf(k, tv[col].mean()), 'o-', color='black', label=f'Poisson (media {tv[col].mean():.2f})')
    ax.set_title('Goles del '+name); ax.set_xlabel('goles'); ax.legend()
plt.tight_layout(); plt.show()
indep = float(sum(poisson.pmf(i, tv.FTHG.mean())*poisson.pmf(i, tv.FTAG.mean()) for i in range(15)))
print(f'Empates observados: {tv.FTR.eq("D").mean():.3f} | esperados con dos Poisson independientes: {indep:.3f}')""",
 'La ventaja de local aparece en las nueve ligas: entre 42,7 % (Italia) y 45,5 % (España) de victorias locales, y unos 0,3 goles más para el local. Países Bajos y Alemania son las ligas con más goles. La distribución de goles se parece a una Poisson, y con dos Poisson independientes de media global se esperaría un 24,9 % de empates frente al 24,8 % observado: a este nivel agregado no hay un exceso de empates que explicar.'),
('Disponibilidad de cuotas por casa y temporada',
 'Aquí solo contamos si la cuota existe, no qué resultado hubo; por eso se muestran todas las temporadas. Una casa cuenta si publica las tres cuotas del partido.',
 r"""books = {'Bet365': 'B365', 'Bet&Win': 'BW', 'Pinnacle': 'PS', 'William Hill': 'WH'}
cohort = raw[raw.Season.ge(1415)]
avail = pd.DataFrame({name: cohort[[p+c for c in 'HDA']].notna().all(axis=1).groupby(cohort.Season).mean() for name, p in books.items()})
display(avail.round(3))
fig, ax = plt.subplots(figsize=(9, 2.6)); im = ax.imshow(avail.T, aspect='auto', vmin=0, vmax=1)
ax.set_yticks(range(len(avail.columns)), avail.columns); ax.set_xticks(range(len(avail)), avail.index, rotation=45)
fig.colorbar(im, ax=ax, label='Fracción con cuotas completas'); plt.show()""",
 'Hasta 2021/22 las cuatro casas cubren más del 99 % de los partidos; después William Hill baja a 93 % y 96 %. En 2024/25 Bet&Win cae a 62 % y William Hill a 75 %. En 2025/26 William Hill desaparece y Pinnacle solo aparece en el 48 %, además de excluirse desde el 23/07/2025 por la advertencia de la fuente. El consenso de las dos últimas temporadas se apoya en menos casas que el del entrenamiento: es una limitación que se declara; no se imputan cuotas.'),
('Acuerdo entre casas',
 'Si las cuatro casas dicen casi lo mismo, tiene sentido resumirlas en un consenso. Convertimos cada cuota en probabilidad de victoria local y comparamos.',
 r"""imp = {}
for name, p in books.items():
    inv = 1/tv[[p+c for c in 'HDA']].apply(pd.to_numeric, errors='coerce')
    imp[name] = inv.div(inv.sum(axis=1, min_count=3), axis=0).iloc[:, 0]
imp = pd.DataFrame(imp); display(imp.corr().round(4))
diff = (imp.max(axis=1)-imp.min(axis=1)).dropna()
print(f'Diferencia entre la casa más alta y la más baja en P(local): mediana {diff.median():.3f}, percentil 95 {diff.quantile(.95):.3f}, máximo {diff.max():.3f}')""",
 'Las probabilidades de las cuatro casas tienen una correlación de al menos 0,997. La diferencia entre la casa más alta y la más baja es de 1,7 puntos en la mediana y de 3,8 en el percentil 95, con casos aislados de hasta 17 puntos. Resumirlas en un consenso pierde muy poca información.'),
('Valores extremos',
 'Aplicamos la regla del rango intercuartílico (IQR) a goles, tiros y cuotas, y miramos los casos más extremos para decidir si son errores.',
 r"""cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'B365H', 'B365D', 'B365A']
x = tv[cols].apply(pd.to_numeric, errors='coerce'); q1, q3 = x.quantile(.25), x.quantile(.75); iqr = q3-q1
atipico = (x.lt(q1-1.5*iqr) | x.gt(q3+1.5*iqr)).mean()
display(pd.DataFrame({'mediana': x.median(), 'máximo': x.max(), 'fracción atípica (IQR)': atipico}).round(3))
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
x[['FTHG', 'FTAG']].plot.box(ax=axes[0]); x[['HST', 'AST']].plot.box(ax=axes[1]); x[['B365H', 'B365D', 'B365A']].plot.box(ax=axes[2], logy=True)
for ax, t in zip(axes, ['Goles', 'Tiros a puerta', 'Cuotas Bet365 (escala log)']): ax.set_title(t)
plt.tight_layout(); plt.show()
display(tv.loc[x.B365A.nlargest(3).index, ['Date', 'Div', 'HomeTeam', 'AwayTeam', 'B365H', 'B365A', 'FTHG', 'FTAG']])
display(tv.loc[(x.FTHG+x.FTAG).nlargest(3).index, ['Date', 'Div', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']])""",
 'La regla IQR marca como atípico el 8 % de los partidos por goles del local (cuatro o más) y cerca del 9–10 % de las cuotas, que tienen una cola larga. Los casos más extremos son reales: cuotas de 41 para el visitante en partidos del Barcelona en casa y goleadas como el 0–13 del Ajax al VVV Venlo. No se eliminan ni se recortan.'),
],
'02_feature.ipynb': [
('Señal individual de cada variable',
 'Antes de modelar, medimos cuánto se relaciona cada variable deportiva con la diferencia de goles. Añadimos el precio de referencia como punto de comparación.',
 r"""dev = df[~df.Season.isin(TEST)]
rho = dev[sets['sin_cuotas']+['REF_logit']].corrwith(dev.GD, method='spearman').sort_values(key=abs, ascending=False)
display(rho.head(15).rename('Spearman con la diferencia de goles').to_frame().round(3))
top = rho.head(15)[::-1]; fig, ax = plt.subplots(figsize=(8, 4))
top.plot.barh(ax=ax, color=['#0d9488' if k == 'REF_logit' else '#2563eb' for k in top.index])
ax.set_xlabel('Correlación de Spearman con la diferencia de goles'); plt.tight_layout(); plt.show()
print('Variables deportivas con |rho| < 0,02:', int((rho.drop('REF_logit').abs() < .02).sum()), 'de', len(sets['sin_cuotas']))""",
 'El precio de referencia es la variable más relacionada con la diferencia de goles (0,47). Las mejores variables deportivas llegan a 0,41–0,42 (Elo, tiros a puerta y goles recientes, plantilla) y muchas tienen valores casi iguales, señal de que miden lo mismo. Solo 6 de las 77 variables deportivas no muestran casi ninguna relación.'),
('Variables clave según el resultado',
 'Miramos cuatro diferencias local menos visitante (Elo, puntos recientes, once titular e historial de puntos) según cómo terminó el partido.',
 r"""keys = ['ELO_DIF', 'EW_Pts10_DIF', 'XI_DIF', 'HPPG_DIF']
fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
for ax, k in zip(axes, keys):
    dev.boxplot(column=k, by='FTR', ax=ax, showfliers=False, grid=False); ax.set_title(k); ax.set_xlabel('resultado')
plt.suptitle(''); plt.tight_layout(); plt.show()
display(dev.groupby('FTR')[keys].median().reindex(['H', 'D', 'A']).round(3))""",
 'Las medianas se ordenan como se espera: cuando gana el local su Elo es 37 puntos mayor y su once 1,8 puntos mejor; cuando gana el visitante es 52 puntos menor y 2,5 peor. En los empates las diferencias quedan cerca de cero. Las cajas se solapan mucho: ninguna variable separa por sí sola los tres resultados.'),
('Redundancia entre variables',
 'Muchas variables miden lo mismo de formas parecidas. Contamos los pares casi idénticos, que es lo que después elimina la poda por correlación.',
 r"""corr = dev[sets['sin_cuotas']].corr().abs()
pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().sort_values(ascending=False)
print('Pares con |r| > 0,95:', int((pairs > .95).sum()), 'de', len(pairs), '| con |r| > 0,80:', int((pairs > .8).sum()))
display(pairs.head(8).rename('|r|').to_frame().round(3))
sel = ['ELO_DIF', 'HPPG_DIF', 'EW_Pts10_DIF', 'EW_GD10_DIF', 'XGD', 'SQ_DIF', 'XI_DIF', 'SQz_DIF', 'REF_logit']
m = dev[sel].corr(); fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(m, vmin=-1, vmax=1, cmap='RdBu_r')
ax.set_xticks(range(len(sel)), sel, rotation=90); ax.set_yticks(range(len(sel)), sel)
for i in range(len(sel)):
    for j in range(len(sel)): ax.text(j, i, f'{m.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)
fig.colorbar(im, ax=ax); plt.tight_layout(); plt.show()""",
 'Hay 11 pares de variables con correlación mayor a 0,95 y 143 con más de 0,80; por ejemplo, la plantilla y el once titular (0,98) o las dos medidas de diferencia de goles reciente (1,00). Esto justifica la poda por correlación y la regularización. El precio de referencia también se correlaciona fuerte con las variables deportivas, lo que anticipa que añadirlas al precio aportará poco.'),
('Estabilidad de las variables en el tiempo',
 'El índice de estabilidad poblacional (PSI) compara la distribución de una variable en dos periodos. Como regla práctica, menos de 0,10 es estable y más de 0,25 es un cambio fuerte.',
 r"""def psi(a, b, bins=10):
    a, b = a.dropna(), b.dropna()
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins+1)[1:-1]))
    if len(a) == 0 or len(b) == 0 or len(edges) == 0: return np.nan
    pa = np.clip(np.bincount(np.digitize(a, edges), minlength=len(edges)+1)/len(a), 1e-4, None)
    pb = np.clip(np.bincount(np.digitize(b, edges), minlength=len(edges)+1)/len(b), 1e-4, None)
    return float(((pb-pa)*np.log(pb/pa)).sum())
early, late = dev[dev.Season.lt(1819)], dev[dev.Season.ge(1819)]
stab = pd.Series({c: psi(early[c], late[c]) for c in sets['sin_cuotas']+sets['mercado']}).sort_values(ascending=False)
display(stab.head(10).rename('PSI: 2014/15–2017/18 frente a 2018/19–2023/24').to_frame().round(3))
print('PSI > 0,25:', int((stab > .25).sum()), '| entre 0,10 y 0,25:', int(stab.between(.1, .25).sum()), '| menor que 0,10:', int((stab < .1).sum()))""",
 'Ninguna variable supera 0,25 y solo dos (las faltas recientes del local y del visitante) quedan entre 0,10 y 0,25; las otras 77 son estables. Las variables se comportan de forma parecida al principio y al final del periodo de desarrollo, así que entrenar con todo el pasado es razonable.'),
('Antigüedad de las valoraciones',
 'Cada partido usa la última versión del videojuego publicada antes de su fecha. Medimos cuántos días tiene esa versión en cada temporada de desarrollo.',
 r"""display(dev.groupby('Season').H_snapshot_age.describe()[['count', 'mean', '50%', 'max']].round(0))
fig, ax = plt.subplots(figsize=(9, 3)); dev.boxplot(column='H_snapshot_age', by='Season', ax=ax, grid=False, showfliers=False)
ax.set_title('Antigüedad de la valoración usada (días)'); plt.suptitle(''); plt.show()""",
 'La valoración usada tiene en la mediana entre 129 y 166 días. Hay una cola larga: el 1,6 % de las valoraciones tiene más de un año, y unos pocos clubes de Países Bajos, Portugal, Turquía, Bélgica e Italia usan versiones de varios años porque no aparecen en las ediciones siguientes del videojuego. Se conservan con su antigüedad real en lugar de inventar un valor.'),
('Equipos recién ascendidos',
 'Los equipos que no jugaron la temporada anterior en la liga tienen poca historia. Vemos cuántos partidos afecta y si les faltan valoraciones.',
 r"""g = dev.assign(nuevo=dev.NEW_H.eq(1) | dev.NEW_A.eq(1), sin_plantilla=dev.SQ_H.isna() | dev.SQ_A.isna())
display(g.groupby('nuevo').agg(partidos=('y', 'size'), local=('y', lambda s: s.eq(0).mean()), empate=('y', lambda s: s.eq(1).mean()),
                               visitante=('y', lambda s: s.eq(2).mean()), sin_plantilla=('sin_plantilla', 'mean')).round(3))
g2 = dev.assign(local_nuevo=dev.NEW_H.eq(1), visitante_nuevo=dev.NEW_A.eq(1))
print('Victoria local cuando el recién ascendido es el local:', round(g2[g2.local_nuevo & ~g2.visitante_nuevo].y.eq(0).mean(), 3),
      '| cuando es el visitante:', round(g2[~g2.local_nuevo & g2.visitante_nuevo].y.eq(0).mean(), 3))""",
 'El 26 % de los partidos tiene al menos un equipo que no jugó la temporada anterior en esa liga. A esos partidos les faltan valoraciones con más frecuencia (4,5 % frente a 1,0 %). Cuando el recién ascendido es local, la victoria local baja a 31,5 %; cuando es visitante, sube a 56,6 %. La variable binaria de recién ascendido recoge esa diferencia.'),
],
'03_preprocesamiento.ipynb': [
('Qué hace el preprocesamiento en un entrenamiento real',
 'Ajustamos el preprocesamiento con todo lo anterior a 2023/24 y contamos cuántas columnas entran, cuántos indicadores de dato faltante se añaden y cuáles elimina la poda.',
 r"""dev = df[~df.Season.isin(TEST)]; tr = dev[dev.Season.lt(2324)]
def nombres(pre, cols):
    imp, pruner = pre.steps[0][1], pre.steps[-1][1]
    todos = list(cols)+['falta_'+cols[i] for i in imp.indicator_.features_]
    return todos, [todos[i] for i in pruner.indices_]
for branch in ['con_cuotas', 'sin_cuotas']:
    cols = sets[branch]; pre = preprocessing().fit(tr[cols].to_numpy(float), tr.y.to_numpy())
    todos, quedan = nombres(pre, cols); fuera = [n for n in todos if n not in quedan]
    print(f'{branch}: {len(cols)} variables + {len(todos)-len(cols)} indicadores de faltante → {len(quedan)} tras la poda')
    print('  eliminadas por correlación > 0,95:', ', '.join(fuera) if fuera else 'ninguna')""",
 'Con cuotas entran 84 variables y se añaden 58 indicadores de faltante; tras la poda quedan 81. Sin cuotas, de 77 + 58 quedan 76. Casi todos los indicadores desaparecen porque son copias unos de otros: cuando falta una estadística de una liga o la plantilla de un club, faltan todas a la vez. La poda también elimina una variable de cada par casi idéntico (por ejemplo el once titular frente a la plantilla) y, con cuotas, el consenso, que repite al precio de referencia.'),
('Sensibilidad a las decisiones de preprocesamiento',
 'Las decisiones (mediana, indicador de faltante, escalado y poda en 0,95) se fijaron antes de evaluar. Aquí comprobamos cuánto cambiaría el log loss de validación de la LR sin cuotas con otras opciones. Es un análisis de sensibilidad: no se usa para cambiar el modelo.',
 r"""from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from mlcore.models import CorrelationPruner
cols = sets['sin_cuotas']
def variante(imputer='median', indicator=True, scale=True, prune=.95):
    steps = [SimpleImputer(strategy=imputer, keep_empty_features=True, add_indicator=indicator)]
    if scale: steps.append(StandardScaler())
    if prune: steps.append(CorrelationPruner(prune))
    return make_pipeline(*steps, LogisticRegression(C=.05, max_iter=5000))
variantes = {'elegido: mediana + indicador + escala + poda 0,95': {}, 'sin indicador de faltante': dict(indicator=False),
             'media en vez de mediana': dict(imputer='mean'), 'sin escalar': dict(scale=False), 'sin poda': dict(prune=None), 'poda en 0,80': dict(prune=.8)}
y_val = dev.loc[dev.Season.isin(FOLDS), 'y']; rows = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for name, kw in variantes.items():
        p = np.vstack([variante(**kw).fit(dev.loc[dev.Season.lt(s), cols].to_numpy(float), dev.loc[dev.Season.lt(s), 'y'])
                       .predict_proba(dev.loc[dev.Season.eq(s), cols].to_numpy(float)) for s in FOLDS])
        rows.append(dict(variante=name, log_loss=log_loss(y_val, p, labels=[0, 1, 2])))
t = pd.DataFrame(rows); t['delta'] = t.log_loss-t.log_loss.iloc[0]; display(t.round(5))""",
 'Ninguna alternativa cambia el log loss en más de 0,0007. Quitar el indicador o podar en 0,80 daría una mejora de ese tamaño, parecida a la tolerancia de 0,0005 con la que tratamos dos modelos como equivalentes; usar la media o no escalar casi no cambia nada. Las decisiones no se modifican: se fijaron antes de evaluar, y este análisis muestra que el resultado no depende de ellas.'),
('MI y PCA dentro de un entrenamiento',
 'Vemos qué variables conserva la información mutua y cuánta varianza resumen los componentes principales, usando solo el entrenamiento anterior a 2023/24.',
 r"""from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
for ax, branch in zip(axes, ['con_cuotas', 'sin_cuotas']):
    cols = sets[branch]; pre = preprocessing().fit(tr[cols].to_numpy(float), tr.y.to_numpy()); Z = pre.transform(tr[cols].to_numpy(float))
    mi = pd.Series(mutual_info_classif(Z, tr.y, random_state=42, n_neighbors=3), index=nombres(pre, cols)[1]).sort_values(ascending=False)
    var = np.cumsum(PCA(svd_solver='full').fit(Z).explained_variance_ratio_); n95 = int(np.searchsorted(var, .95)+1)
    print(branch, '· 10 variables con mayor información mutua:', ', '.join(mi.head(10).index))
    print(f'  PCA sobre {Z.shape[1]} columnas: 5 componentes explican el {var[4]:.0%} de la varianza; el 95 % necesita {n95}')
    ax.plot(range(1, len(var)+1), var); ax.axhline(.95, ls=':', color='gray'); ax.axvline(5, ls=':', color='gray')
    ax.set_title(branch, fontsize=9); ax.set_xlabel('componentes')
axes[0].set_ylabel('varianza acumulada'); plt.tight_layout(); plt.show()""",
 'Con cuotas, la información mutua pone primero el precio de referencia y después el historial de puntos, el Elo y la plantilla; sin cuotas, esas mismas variables deportivas encabezan la lista. En PCA, cinco componentes explican solo cerca de la mitad de la varianza (52–53 %) y hacen falta entre 45 y 47 para llegar al 95 %: resumir en cinco componentes pierde mucha información, lo que ayuda a entender por qué PCA-5 es la peor reducción.'),
],
'04_modelo.ipynb': [
('Log loss por familia y temporada',
 'Un promedio de seis temporadas puede esconder que un modelo gana solo en una. Desglosamos el log loss de cada familia por temporada de validación.',
 r"""from sklearn.metrics import log_loss
va = df[df.Season.isin(FOLDS)]; familias = [b+'/'+f for b in ('con', 'sin') for f in ('LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM')]; rows = []
for name in familias:
    p = aligned(base, name, va)
    for s in FOLDS:
        m = va.Season.eq(s).to_numpy(); rows.append(dict(modelo=name, temporada=s, log_loss=log_loss(va.y.to_numpy()[m], p[m], labels=[0, 1, 2])))
tabla = pd.DataFrame(rows).pivot(index='modelo', columns='temporada', values='log_loss').loc[familias]; display(tabla.round(4))
fig, ax = plt.subplots(figsize=(9, 4)); im = ax.imshow(tabla, aspect='auto', cmap='viridis_r')
ax.set_yticks(range(len(tabla)), tabla.index); ax.set_xticks(range(len(tabla.columns)), tabla.columns); fig.colorbar(im, ax=ax, label='log loss'); plt.show()
puesto = tabla.groupby(tabla.index.str[:3]).rank(); display(puesto.mean(axis=1).round(2).rename('puesto medio dentro de su enfoque (1 = mejor)').to_frame())""",
 'El orden es estable: la SVM es la peor familia de ambos enfoques en las seis temporadas y 2020/21 es la más difícil para todas. Con cuotas, el bosque aleatorio, HGB y XGBoost se reparten los primeros puestos y la red neuronal queda quinta; sin cuotas, XGBoost y la red neuronal son las mejores en promedio. Entre las cinco primeras familias las diferencias son de milésimas.'),
('Error de entrenamiento frente a validación',
 'Para hablar de sobreajuste hay que comparar el error en los datos con que se entrenó y en datos nuevos. Lo medimos en dos temporadas (2021/22 y 2023/24). La LR de solo precio, que tiene dos variables, sirve de referencia: su diferencia refleja solo que una temporada es más fácil o difícil que el pasado.',
 r"""from mlcore.report import training_gap
brecha = training_gap(OUT); display(brecha.sort_values('brecha_extra').round(4))""",
 'La LR de solo precio tiene una diferencia de −0,0070: las dos temporadas evaluadas fueron algo más fáciles que el pasado, y ese valor sirve de cero. Respecto a ella, la SVM casi no tiene brecha (0,002 con cuotas y 0,000 sin cuotas), pero su error ya es alto en el entrenamiento: su problema no es el sobreajuste. El bosque aleatorio tiene la mayor brecha (0,044 y 0,036) y aun así valida bien; la red neuronal con cuotas muestra 0,024. El sobreajuste existe en los modelos flexibles, pero no explica por sí solo el orden de la validación.'),
('Clasificar el resultado o predecir goles',
 'Sin cuotas probamos dos estrategias: clasificar directamente H/D/A o predecir la diferencia de goles y convertirla en probabilidades. Las ordenamos por log loss de validación.',
 r"""%matplotlib inline
res = pd.read_csv(OUT/'model_comparison.csv').set_index('candidate').log_loss
sin = res[res.index.str.startswith('sin/')].sort_values()
tipo = ['regresión de goles' if '/GD_' in k else 'ensemble' if k.split('/')[1] in ('mean', 'weighted', 'stack') else 'clasificación' for k in sin.index]
display(pd.DataFrame({'log_loss': sin.round(4), 'tipo': tipo}))
color = {'regresión de goles': '#f59e0b', 'ensemble': '#0d9488', 'clasificación': '#2563eb'}
fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(sin.index[::-1], (sin-res['sin/LR'])[::-1], color=[color[t] for t in tipo[::-1]])
ax.axvline(0, color='k', lw=.8); ax.set_xlabel('Δ log loss frente a sin/LR (negativo es mejor)'); plt.tight_layout(); plt.show()""",
 'El promedio de regresores de diferencia de goles (0,9794) supera a la LR con todas las variables, a HGB y a la red neuronal y empata con el bosque aleatorio, pero queda detrás de la LR con historia y plantilla, de XGBoost y de los ensembles. Los regresores individuales quedan más atrás. Predecir goles es una alternativa válida, pero no mejor que clasificar directamente.'),
],
'05_hiperparametros.ipynb': [
('Curvas de la rejilla',
 'Para cada familia mostramos la pérdida interior de las tres configuraciones (de la más simple a la más compleja) en cada temporada, restando el mínimo. Así se ve si la rejilla tiene un óptimo claro o es casi plana.',
 r"""from mlcore.models import GRIDS
P = pd.DataFrame(params); P = P[P.season.isin(FOLDS)]; fam = ['LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM']
fig, axes = plt.subplots(2, 6, figsize=(17, 5), sharex=True)
for i, branch in enumerate(('con', 'sin')):
    for j, f in enumerate(fam):
        sc = np.array(P[P.candidate.eq(branch+'/'+f)].scores.to_list()); rel = sc-sc.min(1, keepdims=True)
        axes[i, j].plot(range(3), rel.T, color='gray', alpha=.5); axes[i, j].plot(range(3), rel.mean(0), 'o-', color='#0d9488' if branch == 'con' else '#2563eb')
        axes[i, j].set_title(branch+'/'+f, fontsize=9); axes[i, j].set_xticks(range(3))
    axes[i, 0].set_ylabel('pérdida interior − mínimo')
plt.tight_layout(); plt.show()
display(P[P.candidate.isin([b+'/'+f for b in ('con', 'sin') for f in fam])].pivot(index='candidate', columns='season', values='chosen'))
for f in fam: print(f, GRIDS[f])""",
 'La elección es bastante estable. HGB con cuotas elige siempre árboles de un nivel, la LR sin cuotas siempre la regularización más fuerte (C = 0,001) y XGBoost sin cuotas siempre profundidad 2. La red neuronal prefiere la penalización más alta (alpha = 10) en las primeras temporadas y cambia después. En los árboles potenciados nunca se elige la opción más profunda: la rejilla no se queda corta por el lado de la complejidad.'),
('Sensibilidad de la LR a la regularización',
 'La rejilla de la LR solo tiene tres valores de C. Probamos un rango mucho más amplio en validación para comprobar que esos tres valores no dejan fuera una opción claramente mejor. No se usa para cambiar la selección.',
 r"""from mlcore.models import specifications, Predictor
from sklearn.metrics import log_loss
df, sets = read_data(OUT); specs = {s.id: s for s in specifications(sets)}; va = df[df.Season.isin(FOLDS)]
Cs = [1e-4, 1e-3, 1e-2, .05, .1, 1., 10., 100.]; rows = []
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    for name in ('con/LR_REF', 'con/LR', 'sin/LR_plantilla', 'sin/LR'):
        for C in Cs:
            p = np.vstack([Predictor(specs[name], dict(C=C)).fit(df[df.Season.lt(s)]).predict_proba(df[df.Season.eq(s)]) for s in FOLDS])
            rows.append(dict(modelo=name, C=C, log_loss=log_loss(va.y, p, labels=[0, 1, 2])))
sens = pd.DataFrame(rows).pivot(index='C', columns='modelo', values='log_loss'); display(sens.round(5))
fig, ax = plt.subplots(figsize=(8, 3.2)); (sens-sens.min()).plot(ax=ax, logx=True, marker='o'); ax.set_ylabel('log loss − mínimo de cada modelo')
ax.axvspan(.001, 1, alpha=.08, color='gray'); plt.show()""",
 'La LR de solo precio es plana para C ≥ 0,01 (0,9652) y solo empeora con una regularización muy fuerte. Las LR sin cuotas tienen su mínimo en C = 0,001, un valor de la rejilla, y empeoran tanto con 0,0001 como con valores mayores: el óptimo queda dentro del rango explorado. La LR con cuotas y todas las variables tendría su mínimo en 0,01 (0,9693), que la rejilla no incluye, pero sigue lejos de la LR de solo precio. Ampliar la rejilla no cambiaría la selección.'),
],
'06_mercado.ipynb': [
('Margen de cada casa',
 'El margen es lo que la casa cobra: cuánto supera a 1 la suma de las probabilidades implícitas. Lo medimos por casa y temporada en desarrollo.',
 r"""%matplotlib inline
dev = df[~df.Season.isin(TEST)]; books = {'Bet365': 'B365', 'Bet&Win': 'BW', 'Pinnacle': 'PS', 'William Hill': 'WH'}
marg = pd.DataFrame({n: (1/dev[[p+c for c in 'HDA']]).sum(axis=1, min_count=3).sub(1).groupby(dev.Season).mean() for n, p in books.items()})
display((100*marg).round(2)); (100*marg).plot(marker='o', figsize=(8, 3)); plt.ylabel('Margen medio (%)'); plt.show()""",
 'Pinnacle cobra el margen más bajo (entre 2,3 % y 3,5 %), frente al 5–6 % de Bet365 y al 5,5–8 % de Bet&Win y William Hill. Por eso se usa como referencia mientras es fiable: un margen menor deforma menos la probabilidad.'),
('Sesgo del favorito y la sorpresa',
 'Agrupamos todas las probabilidades del consenso (local, empate y visitante) en diez grupos y comparamos lo que el mercado dice con lo que pasa. Si el mercado fuera perfecto, la diferencia sería cero en todos los grupos.',
 r"""%matplotlib inline
p = dev[['MKT_pH', 'MKT_pD', 'MKT_pA']].to_numpy().ravel(); o = np.eye(3)[dev.y].ravel()
t = pd.DataFrame({'p': p, 'o': o}); t['grupo'] = pd.qcut(t.p, 10, duplicates='drop')
t = t.groupby('grupo', observed=True).agg(n=('p', 'size'), implicita=('p', 'mean'), observada=('o', 'mean')); t['diferencia'] = t.observada-t.implicita
display(t.round(4))
fig, ax = plt.subplots(figsize=(7, 3)); ax.bar(range(len(t)), t.diferencia, color=['#dc2626' if v < 0 else '#0d9488' for v in t.diferencia])
ax.set_xticks(range(len(t)), [f'{v:.2f}' for v in t.implicita], rotation=45); ax.axhline(0, color='k', lw=.8)
ax.set_xlabel('probabilidad implícita media del grupo'); ax.set_ylabel('observada − implícita'); plt.tight_layout(); plt.show()""",
 'El mercado da demasiada probabilidad a las sorpresas y muy poca a los favoritos: en el grupo de probabilidades más bajas dice 10,8 % y ocurre 9,3 %; en el de más altas dice 69,4 % y ocurre 72,1 %. Es un error sistemático, relacionado con repartir el margen de forma proporcional, y es justo el tipo de error que una recalibración puede corregir.'),
('Qué corrige la LR de precio',
 'Comparamos, en las seis temporadas de validación, el ajuste que la LR aplica a la referencia con el error que la referencia tenía realmente en cada grupo de probabilidad.',
 r"""%matplotlib inline
va = df[df.Season.isin(FOLDS)]; base = pd.read_csv(OUT/'oof_base.csv'); q = aligned(base, 'con/LR_REF', va); r = va[['REF_pH', 'REF_pD', 'REF_pA']].to_numpy()
t = pd.DataFrame({'ref': r.ravel(), 'ajuste': (q-r).ravel(), 'obs': np.eye(3)[va.y].ravel()}); t['grupo'] = pd.qcut(t.ref, 10, duplicates='drop')
g = t.groupby('grupo', observed=True).agg(referencia=('ref', 'mean'), ajuste_modelo=('ajuste', 'mean'), observada=('obs', 'mean')); g['error_referencia'] = g.observada-g.referencia
display(g.round(4))
fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(g.referencia, g.error_referencia, 'o-', label='observada − referencia')
ax.plot(g.referencia, g.ajuste_modelo, 's-', label='ajuste que aplica la LR'); ax.axhline(0, color='k', lw=.8); ax.legend(); ax.set_xlabel('probabilidad de la referencia'); plt.tight_layout(); plt.show()
print('Ajuste absoluto medio por probabilidad:', round(float(np.abs(q-r).mean()), 4), '| máximo:', round(float(np.abs(q-r).max()), 4))""",
 'El ajuste de la LR va en la misma dirección que el error de la referencia: resta cerca de 0,8 puntos a los resultados poco probables y suma 1,8 a los favoritos claros, donde la referencia se quedaba 2,2 puntos corta. El cambio medio es de 0,9 puntos de probabilidad. La LR no aporta información nueva: corrige una parte del sesgo sistemático del precio.'),
('Desacuerdo entre casas',
 'Dividimos la validación en cuatro grupos según cuánto discrepan las casas entre sí y vemos en cuáles gana más la LR de precio.',
 r"""from mlcore.metrics import losses
g = va.assign(ll_consenso=losses(va.y, va[['MKT_pH', 'MKT_pD', 'MKT_pA']].to_numpy()), ll_modelo=losses(va.y, q),
              grupo=pd.qcut(va.MKT_STD, 4, labels=['bajo', 'medio-bajo', 'medio-alto', 'alto']))
t = g.groupby('grupo', observed=True).agg(partidos=('y', 'size'), desacuerdo=('MKT_STD', 'mean'), ll_consenso=('ll_consenso', 'mean'), ll_modelo=('ll_modelo', 'mean'))
t['modelo − consenso'] = t.ll_modelo-t.ll_consenso; display(t.round(4))""",
 'El desacuerdo entre casas es mayor en los partidos con un favorito claro, que son también los más fáciles de predecir (log loss 0,83 frente a 1,04 en el grupo de menor desacuerdo). La ventaja de la LR sobre el consenso crece con el desacuerdo, de −0,0002 a −0,0027, lo que coincide con que su corrección se concentra en favoritos y sorpresas.'),
],
'07_evaluacion.ipynb': [
('Calibración por clase',
 'Para cada clase agrupamos los partidos en diez grupos de probabilidad y comparamos la probabilidad media con la frecuencia real. Un modelo bien calibrado cae sobre la diagonal.',
 r"""from mlcore.metrics import ece
series = {'Consenso directo': va[['MKT_pH', 'MKT_pD', 'MKT_pA']].to_numpy(), 'Con cuotas (LR de precio)': aligned(base, 'con/LR_REF', va),
          'Sin cuotas (LR historia y plantilla)': aligned(base, 'sin/LR_plantilla', va)}
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8)); rows = []
for j, ax in enumerate(axes):
    for name, p in series.items():
        g = pd.DataFrame({'p': p[:, j], 'o': va.y.eq(j).to_numpy()}); g['bin'] = pd.qcut(g.p, 10, duplicates='drop'); t = g.groupby('bin', observed=True)[['p', 'o']].mean()
        ax.plot(t.p, t.o, 'o-', label=name, ms=3)
        rows.append(dict(serie=name, clase='HDA'[j], ECE_masa=ece(p[:, j], va.y.eq(j)), ECE_ancho=ece(p[:, j], va.y.eq(j), equal_mass=False)))
    ax.plot([0, 1], [0, 1], ':', color='gray'); ax.set_title(['Local', 'Empate', 'Visitante'][j]); ax.set_xlabel('probabilidad predicha'); ax.set_xlim(0, t.p.max()+.08); ax.set_ylim(0, t.p.max()+.08)
axes[0].set_ylabel('frecuencia observada'); axes[0].legend(fontsize=7); plt.tight_layout(); plt.show()
E = pd.DataFrame(rows); display(E.pivot(index='serie', columns='clase', values='ECE_masa').round(4)); display(E.pivot(index='serie', columns='clase', values='ECE_ancho').round(4))""",
 'La LR de precio tiene el menor error de calibración en las tres clases, con las dos formas de medirlo (grupos de igual tamaño o de igual ancho). La LR sin cuotas es la peor calibrada en victoria local y empate, y queda entre las otras dos en victoria visitante.'),
('Bootstrap por bloques frente a bootstrap simple',
 'Los partidos de una misma semana no son independientes. Comparamos el intervalo que se obtiene remuestreando semanas completas con el que saldría remuestreando partidos sueltos.',
 r"""from mlcore.metrics import compare, losses
d = losses(va.y, series['Con cuotas (LR de precio)'])-losses(va.y, series['Consenso directo'])
blk = compare(va, d, np.zeros(len(d))); rng = np.random.default_rng(2026); iid = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)])
lo, hi = np.quantile(iid, [.025, .975])
print(f'Δ log loss (LR de precio − consenso): {d.mean():.5f}')
print(f'IC 95 % por bloques semanales:        [{blk["lo"]:.5f}; {blk["hi"]:.5f}]  ancho {blk["hi"]-blk["lo"]:.5f}')
print(f'IC 95 % con partidos sueltos:         [{lo:.5f}; {hi:.5f}]  ancho {hi-lo:.5f}')
print(f'Razón de anchos (bloques / simple): {(blk["hi"]-blk["lo"])/(hi-lo):.2f}')""",
 'Los dos intervalos son prácticamente iguales (razón de anchos 1,00): en estos datos, que los partidos de una misma semana compartan condiciones casi no cambia la incertidumbre de la diferencia. Mantenemos el bootstrap por bloques porque es la opción prudente.'),
('Ventana creciente frente a ventana móvil',
 'Nuestro protocolo entrena con todo el pasado. Una alternativa es entrenar solo con las tres temporadas más recientes, por si el fútbol antiguo ya no sirve. Comparamos ambos esquemas con dos LR de regularización fija.',
 r"""from mlcore.models import specifications, Predictor
from sklearn.metrics import log_loss
specs = {s.id: s for s in specifications(sets)}; rows = []
for name in ('con/LR_REF', 'sin/LR_plantilla'):
    for s in FOLDS:
        past = sorted(x for x in df.Season.unique() if x < s); obj = df[df.Season.eq(s)]
        for esquema, seasons in [('creciente (todo el pasado)', past), ('móvil (3 temporadas)', past[-3:])]:
            p = Predictor(specs[name], dict(C=.05)).fit(df[df.Season.isin(seasons)]).predict_proba(obj)
            rows.append(dict(modelo=name, esquema=esquema, temporada=s, log_loss=log_loss(obj.y, p, labels=[0, 1, 2])))
t = pd.DataFrame(rows).pivot_table(index=['modelo', 'esquema'], columns='temporada', values='log_loss'); t['media'] = t.mean(axis=1); display(t.round(4))""",
 'Entrenar con todo el pasado es igual o mejor que usar solo tres temporadas: con cuotas la diferencia es mínima (0,9651 frente a 0,9655) y sin cuotas la ventana creciente gana por 0,0011, sobre todo en las dos últimas temporadas. Los datos antiguos siguen siendo útiles, lo que coincide con la estabilidad de las variables vista en el cuaderno 02.'),
('Log loss por liga',
 'Comprobamos si alguna liga es mucho más difícil de predecir y si la distancia entre el modelo sin cuotas y el mercado es parecida en todas.',
 r"""L = pd.DataFrame({'Div': va.Div.to_numpy(), 'Consenso': losses(va.y, series['Consenso directo']), 'Con cuotas': losses(va.y, series['Con cuotas (LR de precio)']),
                  'Sin cuotas': losses(va.y, series['Sin cuotas (LR historia y plantilla)'])}).groupby('Div').mean()
L['con − consenso'] = L['Con cuotas']-L['Consenso']; L['sin − consenso'] = L['Sin cuotas']-L['Consenso']; display(L.round(4))
L[['Consenso', 'Con cuotas', 'Sin cuotas']].plot.bar(figsize=(9, 3), ylim=(.85, 1.05)); plt.ylabel('log loss'); plt.show()""",
 'La dificultad cambia mucho entre ligas: de 0,92 en Países Bajos y Portugal a 1,00 en Francia y Turquía. La LR de precio mejora al consenso en ocho de las nueve ligas (Francia es la excepción). La distancia de la LR sin cuotas al mercado va de 0,004 en Turquía a 0,020 en Italia.'),
('Por qué el empate casi nunca es la clase más probable',
 'Miramos la distribución de la probabilidad de empate que da el modelo con cuotas y si, donde el modelo dice más empate, de verdad hay más empates.',
 r"""p = series['Con cuotas (LR de precio)']
print(f'P(empate) predicha: mínimo {p[:, 1].min():.3f}, mediana {np.median(p[:, 1]):.3f}, máximo {p[:, 1].max():.3f}')
print(f'Partidos donde el empate es la clase más probable: {(p.argmax(1) == 1).mean():.2%} | empates reales: {va.y.eq(1).mean():.2%}')
g = pd.DataFrame({'pD': p[:, 1], 'empate': va.y.eq(1).to_numpy()}); g['grupo'] = pd.qcut(g.pD, 5)
display(g.groupby('grupo', observed=True).agg(partidos=('pD', 'size'), pD_media=('pD', 'mean'), empates_reales=('empate', 'mean')).round(3))
fig, ax = plt.subplots(figsize=(7, 2.8)); ax.hist(p[:, 1], bins=40, alpha=.6, label='P(empate)'); ax.hist(p[:, [0, 2]].max(1), bins=40, alpha=.6, label='P(victoria más probable)')
ax.legend(); ax.set_xlabel('probabilidad'); plt.show()""",
 'La probabilidad de empate casi nunca pasa de 0,31 (mediana 0,265), así que el empate es la clase más probable en solo el 0,11 % de los partidos, aunque ocurre en el 24,9 %. Aun así está bien calibrada: donde el modelo dice 15 % hay 14 % de empates y donde dice 31 % hay 31 %. El modelo distingue partidos con más y con menos empate; lo que no puede hacer es convertir el empate en el resultado más probable.'),
],
'08_pesos_ensembles.ipynb': [
('Recalibración de la variante balanceada',
 'La variante que iguala el peso de aciertos y fallos del favorito se puede recalibrar con la temporada anterior. Comparamos antes y después en las cinco temporadas donde hay recalibración disponible. Estos pesos usan información del mercado.',
 r"""from mlcore.metrics import metrics
df, sets = read_data(OUT); frame = df[df.Season.isin(FOLDS[1:])]
recal = pd.read_csv(OUT/'oof_recalibration.csv'); weighted = pd.read_csv(OUT/'oof_market_guided.csv'); rows = []
for name in recal.candidate.unique():
    ref = name.removesuffix('_recal').rsplit('/', 1)[0]+'/none'
    for label, source, candidate in [('sin pesos', weighted, ref), ('balance', weighted, name.removesuffix('_recal')), ('balance recalibrado', recal, name)]:
        rows.append(dict(modelo=name.rsplit('/', 1)[0], variante=label, **metrics(frame.y, aligned(source, candidate, frame))))
display(pd.DataFrame(rows)[['modelo', 'variante', 'log_loss', 'ECE_masa']].round(4))""",
 'Recalibrar corrige la calibración de la variante balanceada (el ECE de la LR con cuotas baja de 0,023 a 0,009), pero su log loss sigue por encima del entrenamiento sin pesos en los cuatro casos. Los pesos no aportan nada que una recalibración pueda rescatar.'),
('Pesos del ensemble ponderado por temporada',
 'El ensemble ponderado aprende, con las temporadas anteriores, cuánto peso dar a cada modelo. Vemos si esos pesos son estables.',
 r"""w = pd.read_csv(OUT/'ensemble_weights.csv'); fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
for ax, branch in zip(axes, ('con', 'sin')):
    t = w[w.branch.eq(branch)].set_index('season').drop(columns='branch').dropna(axis=1, how='all'); t.columns = [c.split('/')[1] for c in t.columns]
    t.plot.bar(stacked=True, ax=ax, width=.8, colormap='tab10'); ax.set_title('Pesos del ensemble ponderado · '+branch); ax.legend(fontsize=7, ncol=2); ax.set_ylim(0, 1)
    print(branch, '· peso medio:', t.mean().round(2).to_dict())
plt.tight_layout(); plt.show()""",
 'Con cuotas, el ponderado pone en promedio el 93 % del peso en la LR de solo precio y casi todo el resto en la LR completa: es casi un único modelo. Sin cuotas el peso se reparte entre las dos LR reducidas, la red neuronal y los árboles. La SVM casi no recibe peso en ninguno de los dos enfoques.'),
('Por qué promediar ayuda sin cuotas y no con cuotas',
 'Promediar modelos sirve cuando son parecidos en calidad y se equivocan en partidos distintos. Medimos la correlación entre las pérdidas por partido de los miembros de cada ensemble.',
 r"""from mlcore.metrics import losses
from mlcore.experiment import ensemble_member_names
va = df[df.Season.isin(FOLDS)]; base = pd.read_csv(OUT/'oof_base.csv')
for branch in ('con', 'sin'):
    miembros = ensemble_member_names(branch); L = pd.DataFrame({k.split('/')[1]: losses(va.y, aligned(base, k, va)) for k in miembros}); c = L.corr()
    fuera = c.where(~np.eye(len(c), dtype=bool)).stack(); media = losses(va.y, np.mean([aligned(base, k, va) for k in miembros], axis=0)).mean()
    print(f'{branch}: correlación media entre pérdidas {fuera.mean():.3f} (mínima {fuera.min():.3f}) | mejor miembro {L.mean().idxmin()} = {L.mean().min():.4f} | peor {L.mean().idxmax()} = {L.mean().max():.4f} | promedio simple = {media:.4f}')
    display(L.mean().sort_values().round(4).rename('log loss de cada miembro').to_frame().T)""",
 'La correlación entre las pérdidas de los miembros es igual de alta en ambos enfoques (0,96), así que esa no es la diferencia. Lo que cambia es la calidad relativa: con cuotas hay un miembro claramente mejor (0,9652) y promediarlo con otros peores lo empeora (0,9673); sin cuotas siete miembros están entre 0,9787 y 0,9801 y su promedio (0,9776) mejora al mejor de ellos.'),
('Pesos a los partidos sorpresivos, temporada por temporada',
 'El resultado agregado dice que dar más peso a las sorpresas empeora. Comprobamos si eso pasa en todas las temporadas o solo en el total.',
 r"""from sklearn.metrics import log_loss
rows = []
for cand in sorted(weighted.candidate.unique()):
    branch, family, scheme = cand.split('/')
    if scheme == 'none': continue
    for s in FOLDS:
        f = va[va.Season.eq(s)]
        rows.append(dict(modelo=branch+'/'+family, esquema=scheme, temporada=s,
                         delta=log_loss(f.y, aligned(weighted, cand, f), labels=[0, 1, 2])-log_loss(f.y, aligned(weighted, f'{branch}/{family}/none', f), labels=[0, 1, 2])))
t = pd.DataFrame(rows); display(t.pivot_table(index=['modelo', 'esquema'], columns='temporada', values='delta').round(4))
print('Combinaciones modelo/esquema/temporada que mejoran al entrenamiento sin pesos:', int((t.delta < 0).sum()), 'de', len(t))
display(t[t.delta < 0].round(5))""",
 'Con cuotas, todos los esquemas empeoran en las seis temporadas. Sin cuotas hay 11 combinaciones de 192 con una mejora pequeña (como mucho 0,0025), siempre con los esquemas suaves (balance y remuestreo) y entre 2018/19 y 2020/21. Los pesos fijos 2, 3 y 5 y los inversos empeoran siempre. En el agregado ninguna variante mejora al entrenamiento sin pesos. Estas variantes usan el mercado para definir los pesos, así que no forman parte del modelo deportivo puro.'),
],
'09_modelos_finales.ipynb': [
('Resultados por temporada y por liga',
 'Desglosamos el log loss de los dos modelos finales y de las referencias para ver si el resultado global se repite en cada temporada y en cada liga.',
 r"""bs = pd.read_csv(OUT/'by_season.csv').pivot(index='Season', columns='candidate', values='log_loss')[['prior', 'consenso', 'Pinnacle_respaldo', 'modelo_con', 'modelo_sin']]
display(bs.round(4)); bl = pd.read_csv(OUT/'by_league.csv')
for split in bl.split.unique():
    t = bl[bl.split.eq(split)].pivot(index='Div', columns='candidate', values='log_loss')[['consenso', 'modelo_con', 'modelo_sin']]
    t['con − consenso'] = t.modelo_con-t.consenso; t['sin − consenso'] = t.modelo_sin-t.consenso
    print(split, '· ligas donde el modelo con cuotas mejora al consenso:', int((t['con − consenso'] < 0).sum()), 'de', len(t)); display(t.round(4))""",
 'El modelo con cuotas queda por debajo del consenso en siete de las ocho temporadas (la excepción es 2020/21) y el modelo sin cuotas queda por encima del mercado en todas. Por liga, el modelo con cuotas mejora al consenso en ocho de nueve en validación y en seis de nueve en test: la mejora global no se reparte por igual, y en Inglaterra, Alemania y Francia el test no la muestra.'),
('Calibración de los modelos finales en el test',
 'Repetimos el diagrama de calibración con las predicciones del test, solo para describirlas: no se ajusta nada con este resultado.',
 r"""%matplotlib inline
from mlcore.metrics import ece
tp = pd.read_csv(OUT/'test_predictions.csv'); te = df[df.Season.isin(TEST)]; fig, axes = plt.subplots(1, 3, figsize=(14, 3.6)); rows = []
for j, ax in enumerate(axes):
    for name, color in [('consenso', '#475569'), ('modelo_con', '#0d9488'), ('modelo_sin', '#2563eb')]:
        p = aligned(tp, name, te)[:, j]; g = pd.DataFrame({'p': p, 'o': te.y.eq(j).to_numpy()}); g['bin'] = pd.qcut(g.p, 10, duplicates='drop'); t = g.groupby('bin', observed=True)[['p', 'o']].mean()
        ax.plot(t.p, t.o, 'o-', color=color, ms=3, label=name); rows.append(dict(serie=name, clase='HDA'[j], ECE=ece(p, te.y.eq(j))))
    ax.plot([0, 1], [0, 1], ':', color='gray'); ax.set_title(['Local', 'Empate', 'Visitante'][j]); ax.set_xlim(0, t.p.max()+.08); ax.set_ylim(0, t.p.max()+.08)
axes[0].legend(fontsize=8); plt.tight_layout(); plt.show(); display(pd.DataFrame(rows).pivot(index='serie', columns='clase', values='ECE').round(4))""",
 'En el test el modelo con cuotas vuelve a estar mejor calibrado que el consenso en las tres clases (por ejemplo, 0,021 frente a 0,027 en victoria local). El modelo sin cuotas tiene aquí el menor ECE; eso no significa que prediga mejor, porque su log loss es mayor: está bien calibrado, pero distingue peor unos partidos de otros.'),
('Reglas de decisión para el empate',
 'El experimento adicional de empates cambia la regla con la que se elige la clase. Mostramos el modelo original y la regla elegida en cada enfoque.',
 r"""dr = pd.read_csv(ROOT/'docs/resultados_empates.csv')
display(dr[dr.family.isin(['base', 'boost', 'detector_threshold'])][['split', 'branch', 'family', 'log_loss', 'F1_macro', 'accuracy', 'recall_D', 'precision_D']].round(4))""",
 'Las reglas de decisión suben el F1 macro (de 0,40 a 0,48 con cuotas y de 0,39 a 0,47 sin cuotas) y llevan el recall del empate a 36–40 %, a cambio de 3 a 4 puntos de accuracy. La regla elegida con cuotas (multiplicar la probabilidad de empate) no cambia el log loss; la elegida sin cuotas (el detector con umbral) lo empeora ligeramente. Son reglas para elegir etiqueta, no mejores probabilidades.'),
],
}
