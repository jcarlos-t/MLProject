# ML_Project — Predicción de resultados de fútbol (1X2)

**Predicción probabilística del resultado de partidos de fútbol mediante Machine Learning**, usando únicamente información disponible **antes** del partido: forma de los equipos (últimos 5 partidos, desfazados), calidad de plantilla (ratings FIFA/FC de jugadores) y cuotas de apuestas pre-partido.

## Integrantes

Proyecto del curso de Machine Learning (UTEC), segunda etapa (modelado, evaluación y discusión):

- José Luis Alva Espinoza
- Juan Carlos Ticlia Maqui
- Elmer José Manuel Villegas Suárez
- Joseph Anderson Cose Rojas

**Informe:** `docs/informe.pdf` (formato IEEE, fuente en `docs/informe.tex`). **Presentación:** `docs/presentacion/presentacion.pdf` (13 diapositivas, 10 minutos; fuente HTML en `docs/presentacion/presentacion.html`, se navega con las flechas; alineada con las cifras del informe, incluido el test holdout). **Cuadernos:** `scripts/notebooks/`, que se leen y ejecutan en orden 01 a 08 (ver `scripts/notebooks/README.md`). **Modelo final:** `scripts/data/modelo_final.json` (se genera con `08_modelo_final.ipynb`).

---

## Contexto académico: qué dice la literatura

Este proyecto se apoya en dos trabajos de referencia que marcan la metodología, los modelos y la forma de evaluar, y los contrasta con el resultado propio (ver *Resultados*).

### 1. Tammouch, Elouafi & Essadik (2024) — *Betting on Machine Learning: Extracting Patterns from Football's Anarchic Odds*

- Predice el resultado (1X2) de la Premier League con estadísticas de partido y **selección de características**.
- Compara **Deep Neural Networks, Decision Trees y Random Forest**, con **Mutual Information (MI)** y **PCA** para reducir dimensionalidad.
- Resultados clave:
  - **DNN + MI es el mejor**: accuracy ~55.1% (baseline claro vs ~33% de azar).
  - Decision Trees ~41–46%; no se benefician mucho de MI.
  - **Random Forest falla (~26%)** en datasets pequeños y de alta dimensionalidad ("curse of dimensionality").
  - **MI > PCA**: PCA descarta features de baja varianza que sí tienen valor predictivo.
- Retos identificados que asumimos: equipos ascendidos/descendidos crean huecos de datos (imputación necesaria), y el sobreajuste por pocas muestras frente a muchas features.

### 2. Stübinger, Mangold & Knoll (2020) — *Machine Learning in Football Betting: Prediction of Match Results Based on Player Characteristics* (Applied Sciences, 10(1), 46)

- **No predice la clase 1X2, sino la diferencia de goles (regresión)** con **habilidades individuales de los jugadores** (FIFA) del once declarado de cada equipo.
- Ligas top europeas (5 grandes y segundas divisiones), con la regla de que los jugadores se resumen por habilidad (top-4 de campo y portero).
- **Estrategia de betting**: apuesta solo si la predicción de diferencia de goles es extrema (`|ŷ| > 2`), con cuotas de Bet365.
- Resultados reportados:
  - El **ensemble ALL** (promedio de RF + Boosting + SVM + regresión lineal) logra **+1.58% por apuesta**; RF (+0.43%) y Boosting (+0.72%) positivos.
  - Los bench marks ingenuos **pierden dinero**: BET (−4.53%), HOM (−4.60%), RAN (−8.17%).
- Conclusión clave del paper: **combinar modelos (ensembles) y ser selectivo/conservador supera a cualquier modelo individual y a las heurísticas de apuestas**.

### Qué tomamos de cada uno

| Lección | Fuente | Aplicación en el proyecto |
|---|---|---|
| Usar solo información previa (features lagged + cuotas de apertura + ratings) | Ambos | `02_feature.ipynb`: forma rolling sin el partido actual, `MKT_*`, plantilla |
| Selección de features con MI frente a PCA | Tammouch 2024 | `04_modelo.ipynb` Paso 9: 2 conjuntos × 3 modelos × 9 métodos |
| Habilidades individuales y regresión de la diferencia de goles | Stübinger 2020 | `SK_*` (33 habilidades del once, top-4 + portero) y modelos LIN/RF/HGB/ALL |
| Ensemble de modelos | Stübinger 2020 | `ALL` de clasificación y de regresión; se compara con LR |
| Backtest con umbral y cuotas reales | Stübinger 2020 | `06_mercado.ipynb` Paso 7 y `08_modelo_final.ipynb`: umbral `τ` fijo a priori (0.02), regla `|ŷ| > 2` |
| Split temporal | Ambos | Ventana expansiva de 6 folds (2018/19–2023/24), nunca aleatorio; 2024/25 y 2025/26 son un test holdout que se evalúa una sola vez |

---

## Qué vamos a hacer (metodología)

### Objetivo

Estimar la distribución de probabilidad `P(Home), P(Draw), P(Away)` para cada partido y,
dado que el modelo esté bien calibrado, compararla contra las cuotas del mercado para
detectar **apuestas con valor (value bets)**.

### Pipeline

1. **Datos** (ver `docs/Data_Dictionary.md`; guía de reproducción en `scripts/download/README.md`):
   - `scripts/download/fetch_data.py` descarga crudos -> `scripts/data/raw/` (117 CSVs de partidos, football-data.co.uk: 9 ligas × 13 temporadas; 2013/14 es solo calentamiento del historial y 2014/15–2025/26 son las temporadas modeladas) y `scripts/data/players/raw/` (12 snapshots de ratings de jugadores, Kaggle: FIFA 15 → FC 26). El snapshot FC25 viene de sofifa, capturado en julio de 2025: final de la temporada 2024/25 y sin fichajes de verano.
   - `scripts/download/process_data.py` **solo consolida** -> `matches.csv` (36,150 partidos, 9 ligas, 12 temporadas), `matches_warmup.csv` (2013/14), `players/raw/players_all.csv` y `matches_sq.csv` (base unificada). Resuelve el mapeo de clubes EA → football-data (110 colisiones corregidas; cobertura 100%).
   - `scripts/notebooks/01_eda.ipynb` (EDA): limpia la base unificada **sin imputar nada** (159 columnas de cuotas: 138 con >40% de nulos y 3 con 20–40% eliminadas, 18 conservadas tal como se observaron; `Time`, `Referee`, `HFKC/AFKC` y las columnas del descanso descartadas; la plantilla ya no tiene nulos) y exporta `matches_clean.csv` (36,150 × 74, con los nulos estructurales de la fuente) y genera las 8 figuras de `docs/figures/`.
2. **Feature engineering** (`scripts/notebooks/02_feature.ipynb`):
   - Forma: media de los últimos 5 partidos de cada equipo, **sin incluir el actual**, con calentamiento 2013/14 y ventana acolchada con la media de la liga anterior a la fecha. Las estadísticas de partido no se imputan (NaN estructural en 11.0% de los partidos).
   - Mercado: `MKT_pD`, `MKT_logit = ln(pH/pA)` y `MKT_margin`, del consenso de probabilidades implícitas de B365/BW/PS/WH (cuotas **observadas** de apertura). Reemplaza a `AvgH/D/A`, que solo existen desde 2019/20. Se excluyen 31 partidos sin ninguna cuota.
   - Plantilla: `HomeTeam_sq_top11_mean` y `DIF_sq_top11 = local − visitante`.
   - Habilidades: 33 variables `SK_*` (top-4 de campo y portero del once titular, diferencia local − visitante) y target de regresión `GD = FTHG − FTAG`.
   - **Variables nuevas** (Pasos 11–15): precio de apertura de **Pinnacle** con respaldo del consenso (`REF_logit`, `REF_pD`), dispersión entre casas y brecha Pinnacle–resto, **Elo**, medias exponenciales de puntos y de diferencia de goles, goles esperados tipo Dixon–Coles, tiros a puerta ponderados, tendencia al empate, contexto (descanso, congestión, semana de temporada, recién llegados) y plantilla normalizada por liga; comprobadas contra fuga (perturbación del Elo, recálculo a mano) y **revisadas** en una tabla única (`revision_variables.csv`: 25 de 93 variables redundantes, inestables o sin información).
   - Salidas: `dataset_final.csv` (36,119 × 40), `feature_sets.json` (F1 = 13 variables sin NaN para LR/SVM/MLP; F2 = 33, con NaN para árboles; clave `EXTRA` con las nuevas), `dataset_skills.csv`, `dataset_extra.csv` (27 variables nuevas), `backtest_odds.csv`, `mi_ranking.csv`, `revision_variables.csv`.
3. **Selección de features**: `mutual_info_classif` como ranking (primer 80% por fecha) y comparación MI vs PCA con validación temporal en `04_modelo.ipynb` (Paso 9).
4. **Modelos** (`scripts/notebooks/04_modelo.ipynb`, `06_mercado.ipynb`):
   - Referencias: prior y mercado crudo.
   - **LR**, **HGB**, **XGBoost**, **SVM (RBF)**, **MLP** y **ensemble** (promedio de probabilidades). La **LR sobre las tres variables de mercado es el *baseline*** (el modelo más simple con aprendizaje); el *prior* y el mercado crudo son *referencias sin aprendizaje*. Además, regresión de la diferencia de goles (LIN, RF, HGB y ALL) convertida a probabilidades 1X2.
   - Hiperparámetros elegidos en la ventana 2016/17–2017/18 (disjunta de la evaluación) con desempate por simplicidad.
5. **Evaluación** (probabilística y predictiva), validación de **ventana expansiva** con 6 folds (2018/19–2023/24) y un **test holdout** (2024/25 y 2025/26, 5,999 partidos) que no participa en ninguna decisión:
   - **Log loss** (primaria), **RPS**, accuracy, acierto del ganador sin empates, RMSE/MAE de GD.
   - **Calibración**: ECE y pendiente de calibración, con normalización proporcional y por potencia del margen.
   - Comparaciones pareadas con *bootstrap*, McNemar, Benjamini–Hochberg.
6. **Preprocesamiento, hiperparámetros y evaluación ampliada** (`03_preprocesamiento.ipynb`, `05_hiperparametros.ipynb`, `07_evaluacion.ipynb`) y **modelo final** (`08_modelo_final.ipynb`):
   - **Curvas ROC y AUC** (uno contra el resto, por resultado y macro), matrices de confusión y balanced accuracy.
   - **Calibración**: ECE de igual ancho y de igual masa, MCE, ECE *top-label*, pendiente e intercepto, diagramas de fiabilidad y recalibración posterior (*vector scaling*).
   - **Validación cruzada**: la temporal expansiva se compara con K-fold aleatorio y *leave-one-season-out*; selección de hiperparámetros con **validación anidada temporal**; rejillas extendidas para detectar óptimos en el borde.
   - **Inferencia**: *bootstrap* por bloques (temporada–semana) y Benjamini–Hochberg sobre todas las comparaciones.
   - **Justificación empírica del preprocesamiento**: representación de cuotas, escalado, faltantes, asimetría, WOE/IV, memoria de la forma (EWMA, Elo) y **PSI** entre temporadas.
   - **Modelo final** (`08_modelo_final.ipynb`): regresión logística de 2 variables (`REF_logit`, `REF_pD`) sobre la cuota de apertura de Pinnacle (respaldo: consenso), sin `MKT_margin`, con `C` por validación anidada; se **entrena solo con 2014/15–2023/24**, se guarda en `modelo_final.json` / `.joblib` y se usa con `predecir_1x2`; su evaluación en el test holdout (Paso 8) escribe `modelo_final_test.csv`.
7. **Pruebas de eficiencia del mercado y backtest** (`06_mercado.ipynb`):
   - Modelo de residuo (`GD − ŷ_mercado`), *encompassing* apertura vs cierre de Pinnacle, CLV frente a la línea de cierre y backtest de apuestas de una unidad con cuotas de apertura de Bet365 y Pinnacle (umbral `τ` elegido en los 4 primeros folds, reportado en los 2 últimos de validación y, sin volver a elegirlo, aplicado una vez al test holdout).

### Resultados principales

| Pregunta | Resultado |
|---|---|
| Mejor modelo (log loss, 6 folds de validación) | **Modelo final (LR de 2 variables sobre Pinnacle): 0.9653**; *baseline* (LR de 3 variables sobre cuotas) 0.9657; mercado crudo 0.9664; prior 1.0720. Frente al mercado −0.0011 [−0.0018, −0.0004], p ajustado 0.004, mismo AUC (0.684) |
| **Test holdout (2024/25–2025/26, 5,999 partidos)** | **Modelo final: 0.9651**; mercado crudo 0.9667; Pinnacle cruda 0.9660; *baseline* (LR de 3 variables) 0.9652; prior 1.0750. Frente al consenso de 4 casas −0.0016 [−0.0027, −0.0006], p ajustado 0.004, 2/2 temporadas (la referencia más débil); frente a Pinnacle cruda −0.0009 (p ajustado 0.045, en el límite; 0.063 con otra semilla del bootstrap) y frente al *baseline* −0.0001 (sin diferencia). Casi toda la ventaja está en los 4,440 partidos con Pinnacle observada (−0.0020 vs consenso); con el respaldo de consenso (1,559 partidos, 2025/26) es −0.0004 y no se distingue de 0. Sin sobreajuste a la selección (validación 0.9653). Por liga solo Turquía es significativa (en validación, Portugal) |
| ¿Algo supera al mercado? | No: plantilla, forma, estadísticas, 33 habilidades, HGB, MLP, XGBoost, SVM y ensemble empatan o empeoran (+0.001 a +0.028) |
| Con cuotas vs sin cuotas (ganador, sin empates) | 71.6% vs 70.4% (el benchmark sin cuotas usa plantilla z + Elo + EWMA + habilidades); cuando discrepan acierta el modelo con cuotas el 56.1% (McNemar p < 0.0001); residuo R² = −0.0006 |
| Plantilla frente a forma | Sin cuotas la plantilla mejora el log loss en 0.039; con cuotas, ≈ 0 incluso al inicio de la temporada |
| MI vs PCA | MI gana con cuotas (6/6 folds); PCA ≥ MI sin cuotas; reducir casi no mejora a usar todas las variables (la mayor ganancia: MLP con MI-5 y cuotas, 0.0024) |
| Discriminación (AUC) | Mercado ≈ LR (AUC macro 0.684): local 0.73, visitante 0.74, empate 0.59; el SVM tiene `P(empate)` invertida (AUC 0.44) con `probability=True` |
| Robustez de la evaluación | K-fold aleatorio y leave-one-season-out (solo train/val) dan 0.0001–0.0014 menos log loss (más optimistas); IC por bloques ×1.01 respecto al de partidos; 9 de 11 contrastes significativos tras BH (no lo son la LR ni el MLP frente al mercado crudo); validación anidada reproduce el log loss (±0.0001) |
| Preprocesamiento | Convertir cuotas a probabilidad importa (+0.0057 las crudas); escalado, faltantes, transformaciones y WOE no mejoran; `MKT_margin` es no estacionaria (PSI 0.6–1.5) |
| Features de fuerza de equipo | Sin cuotas, Elo + EWMA de goles + plantilla z mejoran el benchmark 0.0068 (≈ 0.9884 → 0.9817; ya aplicado en el benchmark de `06_mercado.ipynb`) y los tiros a puerta ponderados otros 0.0027 (árboles, *p* ajustado 0.017); con cuotas ninguna variable nueva aporta (17 comparaciones, Benjamini–Hochberg) |
| Mercado | El cierre subsume a la apertura (log loss Pinnacle 0.9658 → 0.9620); el modelo sin cuotas no anticipa el movimiento (r = −0.030, el cierre corrige sus discrepancias; CLV −0.18 pp frente a −0.01 pp del lado contrario) |
| Calibración | Favoritos +2 a +4 pp, longshots −1.5 pp con normalización proporcional; casi desaparece con normalización por potencia |
| Backtest | Sin cuotas: ROI −7.7% a −12.5% (promete +15% a +31%); LR con B365 y τ = 0: +9.7% [+0.4%, +21.3%] (1 de 40 configuraciones sin corrección múltiple; con Pinnacle −6.4%), no se sostiene; regla `|ŷ| > 2`: 706 apuestas, −2.9% [−6.3%, +0.1%] (no reproduce el +1.58% del paper). Modelo final en el test (τ = 0.02 fijo): Pinnacle −2.6% [−17.9%, +11.4%], B365 +11.4% [−4.1%, +29.0%]: sin retorno significativo |

Conclusión: en 2014–2026 el mercado es eficiente respecto a los fundamentos disponibles; el detalle y las limitaciones están en `docs/informe.tex`.

### Cómo leer el proyecto (tres preguntas)

Todo el repositorio responde a una sola pregunta (¿se puede predecir 1X2 mejor que las cuotas?) desde tres ángulos. Cada cuaderno sirve a uno:

| Pregunta | Respuesta | Cuadernos |
|---|---|---|
| **A. ¿Qué predicen los fundamentos?** (plantilla, forma, Elo, habilidades, sin cuotas) | Algo (70.4% de acierto del ganador frente a 71.6% con cuotas), pero menos que el precio | 01–03, 04 (bloques), 06 (Paso 1) |
| **B. ¿Qué modelo es mejor sobre las cuotas?** | Ninguno supera a una regresión logística; el modelo final es una LR de 2 variables sobre Pinnacle que **recalibra** el precio. Frente a Pinnacle cruda y al *baseline* no hay ventaja clara | 04, 05, 07, 08 |
| **C. ¿Hay dinero?** | No se demuestra | 06 (Pasos 5–7), 08 (Paso 6 y 8) |

### Salvedades que conviene recordar

- **Sesgo de selección.** Variables, reglas de aceptación y modelo final se eligieron mirando las mismas 6 temporadas de validación; solo el test holdout (2024/25–2025/26) está libre de él. La corrección de Benjamini–Hochberg es por familia, no global.
- **El test mezcla dos regímenes de precio.** En 2025/26 Pinnacle cubre 48% de los partidos y William Hill 0%; el resto usa el consenso. Se reporta separado (`08_modelo_final.ipynb`, Paso 8).
- **Fuga leve que no llega al test** (el MLP ya no usa `early stopping`: su número de iteraciones se elige en la ventana de ajuste): el corte del 80% de `02_feature` (ranking MI y revisión de variables) incluye partidos de validación; las habilidades faltantes se rellenan con la media de liga y temporada.
- **Snapshots de jugadores heterogéneos:** FC25 (julio de 2025, fin de temporada) y FC26 (21 sep. 2025, un mes dentro de la temporada) frente a los demás (lanzamiento). Solo afectan a las dos temporadas del test, y ningún resultado reportado las usa.
- **Umbral del backtest.** `τ` = 0.02 se fija a priori (antes se elegía por ROI máximo de validación, que es inestable).
- **Reproducibilidad.** Los datos crudos no se versionan; `scripts/download/manifest.py` guarda y comprueba sus huellas SHA-256 (`data_manifest.json`) para saber si una descarga nueva difiere de la usada en el informe.

### Cómo se trataron los problemas típicos

- **Equipos ascendidos / sin historial**: calentamiento con 2013/14 y ventana acolchada con la media de la liga anterior a la fecha (solo 0.4% de los registros de 2014/15 no tiene ningún partido previo). El mapeo de clubes está auditado (100% de cobertura).
- **Curse of dimensionality**: pocas variables bien elegidas, regularización y una comparación explícita MI vs PCA; con 36 mil partidos y 43–46 variables no aparece.
- **Desbalanceo leve de clases** (H=44%, D=25%, A=31%): no se rebalancea (el objetivo es probabilidad calibrada), el desbalance es el prior.
- **Ausencias estructurales**: tiros, córners, faltas y tarjetas no existen en 4 ligas antes de 2017/18; no se imputan y los modelos lineales usan F1 (sin NaN) y los de árboles F2.
- **Temporada actual (2526) incompleta**: `fetch_data.py` no vuelve a descargar archivos existentes; para refrescarla, borrar `raw/*_2526.csv` y re-ejecutar. En 2025/26 Pinnacle tiene solo ~48% de cobertura de cierre.

---

## Estructura del repositorio

```
ML_Project/
├── README.md                  # este documento
├── requirements.txt           # dependencias pinneadas (Python)
├── docs/
│   ├── informe.tex            # paper del proyecto (IEEEtran) y informe.pdf compilado
│   ├── Data_Dictionary.md     # diccionario de datos (todas las columnas)
│   ├── presentacion/          # presentacion.html (fuente), presentacion.pdf (entregable) e img/ con las figuras usadas
│   └── figures/               # figuras del informe (eda_*, prep_*, res_*, hp_*, mercado_*, final_*; las p_* son recortes de un solo panel para el informe)
├── scripts/
│   ├── download/
│   │   ├── README.md          # guía de reproducibilidad de los datos
│   │   ├── fetch_data.py      # descarga de datos crudos
│   │   └── process_data.py    # solo consolidación de crudos
│   ├── split_holdout.py       # divide dataset_final.csv en train_val.csv y test.csv (data/split/); test = 2024/25 y 2025/26
│   ├── report/
│   │   ├── extraer_figuras.py   # extrae de los cuadernos ejecutados las figuras completas de docs/figures
│   │   └── recortar_figuras.py  # recorta los paneles clave de esas figuras para el informe (docs/figures/p_*.png)
│   ├── notebooks/
│   │   ├── 01_eda.ipynb          # EDA de partidos -> exporta matches_clean.csv (figuras del informe)
│   │   ├── 02_feature.ipynb      # feature engineering sobre matches_clean -> dataset_final.csv
│   │   ├── 03_preprocesamiento.ipynb  # justificación empírica de cada preprocesamiento, Elo/EWMA, WOE/IV, PSI
│   │   ├── 04_modelo.ipynb       # comparación de bloques/modelos con CV temporal; selección de hiperparámetros base (Paso 6a)
│   │   ├── 05_hiperparametros.ipynb   # validación anidada, rejillas extendidas, SVM corregido, ensemble con SVR
│   │   ├── 06_mercado.ipynb      # ¿quién gana? cuotas vs sin cuotas, calibración del mercado, pruebas de eficiencia y backtest
│   │   ├── 07_evaluacion.ipynb        # CV (temporal vs K-fold vs LOSO), ROC/AUC, ECE/MCE, recalibración, bootstrap por bloques + BH
│   │   └── 08_modelo_final.ipynb      # modelo final: selección, evaluación completa, backtest, guardado y uso
│   └── data/                  # generado por los scripts (gitignored)
│       ├── matches.csv        # 36,150 partidos (consolidado crudo; 36,119 modelables tras excluir 31 sin cuotas)
│       ├── matches_sq.csv     # base unificada: 225 cols, 36 de squad
│       ├── matches_clean.csv  # dataset limpio, sin nulos (lo genera 01_eda.ipynb)
│       ├── dataset_final.csv  # 33 features + target + metadatos (lo genera 02_feature.ipynb)
│       ├── split/             # train_val.csv (30,120 partidos, 2014/15–2023/24) y test.csv (5,999, 2024/25–2025/26); lo genera scripts/split_holdout.py
│       ├── backtest_odds.csv  # cuotas B365 (apertura) y Pinnacle (cierre) alineadas fila a fila
│       ├── mi_ranking.csv     # ranking de mutual information (train)
│       ├── feature_sets.json  # F1 (sin NaN), F2 (completo), SK_habilidades y targets
│       ├── dataset_extra.csv  # 27 variables nuevas (Elo, EWMA, REF, empate, contexto...) alineadas con dataset_final (lo genera 02_feature.ipynb)
│       ├── revision_variables.csv  # tabla de revisión de las 93 variables (MI, IV, PSI, veredicto)
│       ├── dataset_skills.csv # 33 habilidades del once titular (local - visitante) + FTHG, FTAG, GD
│       ├── hiperparametros.json  # hiperparámetros base elegidos (04_modelo.ipynb, Paso 6a)
│       ├── hiperparametros_modelos.json  # SVM, XGBoost, MLP (04_modelo.ipynb, Paso 6)
│       ├── hiperparametros_extendidos.json  # rejillas extendidas (05_hiperparametros.ipynb, Paso 2)
│       ├── modelo_final.json / modelo_final.joblib / modelo_final_oof.csv / modelo_final_test.csv  # modelo final (entrenado solo con train/val), sus predicciones fuera de muestra en validación y en el test (08_modelo_final.ipynb)
│       ├── modelos_oof.csv / ablation_results.csv / skills_*.csv / seleccion_caracteristicas.csv  # salidas de 04_modelo.ipynb
│       ├── backtest_resultados.csv / backtest_precio_blando.csv / mercado_pruebas.csv  # salidas de 06_mercado.ipynb
│       ├── predicciones_oof.csv  # predicciones fuera de muestra de los 6 folds (06_mercado.ipynb, Paso 2)
│       ├── matches_warmup.csv # 2013/14: solo historial de forma, no se modela
│       ├── raw/               # 117 CSVs de football-data.co.uk
│       └── players/           # snapshots Kaggle, normalized, squad_quality, club_map
```

---

## Reproducibilidad

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# 1) Datos (guía detallada en scripts/download/README.md)
.venv/bin/python scripts/download/fetch_data.py    # descarga data/raw y data/players/raw
.venv/bin/python scripts/download/process_data.py  # consolida matches.csv, players_all.csv y matches_sq.csv

# 2) Partición train/val/test (test holdout = 2024/25 y 2025/26), después de 02_feature.ipynb:
#    .venv/bin/python scripts/split_holdout.py   # escribe data/split/train_val.csv y test.csv

# 3) Notebooks (desde scripts/notebooks/, en este orden):
#    1) 01_eda.ipynb     -> limpia matches_sq.csv y exporta matches_clean.csv (+ figuras)
#    2) 02_feature.ipynb -> feature engineering: exporta dataset_final.csv, dataset_skills.csv, dataset_extra.csv, revision_variables.csv
#    3) 03_preprocesamiento.ipynb -> justificación del preprocesamiento, nuevas variables, PSI
#    4) 04_modelo.ipynb  -> bloques, modelos (LR/HGB/XGBoost/SVM/MLP), hiperparámetros base (genera hiperparametros.json y modelos_oof.csv), MI vs PCA
#    5) 05_hiperparametros.ipynb  -> validación anidada, rejillas extendidas (genera hiperparametros_extendidos.json)
#    6) 06_mercado.ipynb -> predicciones fuera de muestra, pruebas de eficiencia del mercado y backtest
#    7) 07_evaluacion.ipynb       -> CV, ROC/AUC, calibración, bootstrap por bloques (usa modelos_oof.csv)
#    8) 08_modelo_final.ipynb     -> modelo final (solo train/val), evaluación, backtest, test holdout (Paso 8) y guardado (genera modelo_final.json)

# 4) Figuras e informe: .venv/bin/python scripts/report/extraer_figuras.py && .venv/bin/python scripts/report/recortar_figuras.py; cd docs && tectonic informe.tex
```

> **Nota**: usar siempre `.venv/bin/python`. El mapeo de clubes (EA → football-data) se hace en `scripts/download/process_data.py` (paso 5: overrides + `rapidfuzz` como respaldo). Cobertura actual de squad quality: 100% de los slots equipo-partido.
