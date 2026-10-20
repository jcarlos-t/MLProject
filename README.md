# ML_Project — Predicción de resultados de fútbol (1X2)

**Predicción probabilística del resultado de partidos de fútbol mediante Machine Learning**, usando únicamente información disponible **antes** del partido: forma de los equipos (últimos 5 partidos, desfazados), calidad de plantilla (ratings FIFA/FC de jugadores) y cuotas de apuestas pre-partido.

## Integrantes

Proyecto del curso de Machine Learning (UTEC), segunda etapa (modelado, evaluación y discusión):

- José Luis Alva Espinoza
- Juan Carlos Ticlia Maqui
- Elmer José Manuel Villegaz Suárez
- Joseph Anderson Cose Rojas

**Informe:** `docs/informe.pdf` (formato IEEE, fuente en `docs/informe.tex`). **Presentación:** `docs/presentacion/presentacion.pdf` (fuente HTML en `docs/presentacion/presentacion.html`, se navega con las flechas). **Cuadernos:** `scripts/notebooks/`, que se leen y ejecutan en orden 01 a 08 (ver `scripts/notebooks/README.md`). **Modelo final:** `scripts/data/modelo_final.json` (se genera con `08_modelo_final.ipynb`).

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
| Backtest con umbral y cuotas reales | Stübinger 2020 | `06_mercado.ipynb` Paso 7: umbral elegido en validación, regla `|ŷ| > 2` |
| Split temporal | Ambos | Ventana expansiva de 8 folds (2018/19–2025/26), nunca aleatorio |

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
   - **LR**, **HGB**, **XGBoost**, **SVM (RBF)**, **MLP** y **ensemble** (promedio de probabilidades), más regresión de la diferencia de goles (LIN, RF, HGB y ALL) convertida a probabilidades 1X2.
   - Hiperparámetros elegidos en la ventana 2016/17–2017/18 (disjunta de la evaluación) con desempate por simplicidad.
5. **Evaluación** (probabilística y predictiva), validación de **ventana expansiva** con 8 folds (2018/19–2025/26):
   - **Log loss** (primaria), **RPS**, accuracy, acierto del ganador sin empates, RMSE/MAE de GD.
   - **Calibración**: ECE y pendiente de calibración, con normalización proporcional y por potencia del margen.
   - Comparaciones pareadas con *bootstrap*, McNemar, Benjamini–Hochberg.
6. **Preprocesamiento, hiperparámetros y evaluación ampliada** (`03_preprocesamiento.ipynb`, `05_hiperparametros.ipynb`, `07_evaluacion.ipynb`) y **modelo final** (`08_modelo_final.ipynb`):
   - **Curvas ROC y AUC** (uno contra el resto, por resultado y macro), matrices de confusión y balanced accuracy.
   - **Calibración**: ECE de igual ancho y de igual masa, MCE, ECE *top-label*, pendiente e intercepto, diagramas de fiabilidad y recalibración posterior (*vector scaling*).
   - **Validación cruzada**: la temporal expansiva se compara con K-fold aleatorio y *leave-one-season-out*; selección de hiperparámetros con **validación anidada temporal**; rejillas extendidas para detectar óptimos en el borde.
   - **Inferencia**: *bootstrap* por bloques (temporada–semana) y Benjamini–Hochberg sobre todas las comparaciones.
   - **Justificación empírica del preprocesamiento**: representación de cuotas, escalado, faltantes, asimetría, WOE/IV, memoria de la forma (EWMA, Elo) y **PSI** entre temporadas.
   - **Modelo final** (`08_modelo_final.ipynb`): regresión logística de 2 variables (`REF_logit`, `REF_pD`) sobre la cuota de apertura de Pinnacle (respaldo: consenso), sin `MKT_margin`, con `C` por validación anidada; se guarda en `modelo_final.json` / `.joblib` y se usa con `predecir_1x2`.
7. **Pruebas de eficiencia del mercado y backtest** (`06_mercado.ipynb`):
   - Modelo de residuo (`GD − ŷ_mercado`), *encompassing* apertura vs cierre de Pinnacle, CLV frente a la línea de cierre y backtest de apuestas de una unidad con cuotas de apertura de Bet365 y Pinnacle (umbral `τ` elegido en los 4 primeros folds y reportado en los 4 últimos).

### Resultados principales

| Pregunta | Resultado |
|---|---|
| Mejor modelo (log loss, 8 folds) | **Modelo final (LR de 2 variables sobre Pinnacle): 0.9652**; LR anterior de 3 variables 0.9656; mercado crudo 0.9665; prior 1.0727. Frente al mercado −0.0013 [−0.0018, −0.0007], p ajustado 0.001, mismo AUC (0.684) |
| ¿Algo supera al mercado? | No: plantilla, forma, estadísticas, 33 habilidades, HGB, MLP, XGBoost, SVM y ensemble empatan o empeoran (+0.002 a +0.027) |
| Con cuotas vs sin cuotas (ganador, sin empates) | 71.8% vs 70.0%; cuando discrepan acierta el modelo con cuotas el 57.5% (McNemar p < 0.0001); residuo R² = −0.0007 |
| Plantilla frente a forma | Sin cuotas la plantilla mejora el log loss en 0.041; con cuotas, ≈ 0 incluso al inicio de la temporada |
| MI vs PCA | MI gana con cuotas (8/8 folds); PCA ≥ MI sin cuotas; ninguna supera a usar todas las variables |
| Discriminación (AUC) | Mercado ≈ LR (AUC macro 0.684): local 0.73, visitante 0.74, empate 0.59; el SVM tiene `P(empate)` invertida (AUC 0.44) con `probability=True` |
| Robustez de la evaluación | K-fold aleatorio y leave-one-season-out dan 0.0002–0.0011 menos log loss (más optimistas); IC por bloques ×1.02 respecto al de partidos; 11/11 contrastes significativos tras BH; validación anidada reproduce el log loss (±0.0001) |
| Preprocesamiento | Convertir cuotas a probabilidad importa (+0.0054 las crudas); escalado, faltantes, transformaciones y WOE no mejoran; `MKT_margin` es no estacionaria (PSI 0.6–1.5) |
| Features de fuerza de equipo | Sin cuotas, Elo + EWMA de goles + plantilla z mejoran el benchmark 0.0054 (0.9865 → 0.9811) y los tiros a puerta ponderados otros 0.0031 (árboles, *p* ajustado 0.017); con cuotas ninguna variable nueva aporta (17 comparaciones, Benjamini–Hochberg) |
| Mercado | El cierre subsume a la apertura (log loss Pinnacle 0.9640 → 0.9603); el modelo sin cuotas no anticipa el movimiento (r = −0.002; CLV −0.12 pp) |
| Calibración | Favoritos +2 a +4 pp, longshots −1.5 pp con normalización proporcional; casi desaparece con normalización por potencia |
| Backtest | Sin cuotas: ROI −8.5% a −14.8% (promete +20%); LR con B365: +5.1% [−1.8%, +11.8%], no significativo; regla `|ŷ| > 2`: 824 apuestas, −0.8% [−3.4%, +1.8%] |

Conclusión: en 2014–2026 el mercado es eficiente respecto a los fundamentos disponibles; el detalle y las limitaciones están en `docs/informe.tex`.

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
│   └── figures/               # figuras del informe (eda_*, prep_*, res_*, hp_*, mercado_*, final_*; las p_* son recortes de un solo panel para el informe)
├── scripts/
│   ├── download/
│   │   ├── README.md          # guía de reproducibilidad de los datos
│   │   ├── fetch_data.py      # descarga de datos crudos
│   │   └── process_data.py    # solo consolidación de crudos
│   ├── report/
│   │   └── recortar_figuras.py  # recorta los paneles clave de las figuras de los cuadernos para el informe (docs/figures/p_*.png)
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
│       ├── matches.csv        # 36,150 partidos (consolidado crudo)
│       ├── matches_sq.csv     # base unificada: 225 cols, 36 de squad
│       ├── matches_clean.csv  # dataset limpio, sin nulos (lo genera 01_eda.ipynb)
│       ├── dataset_final.csv  # 33 features + target + metadatos (lo genera 02_feature.ipynb)
│       ├── backtest_odds.csv  # cuotas B365 (apertura) y Pinnacle (cierre) alineadas fila a fila
│       ├── mi_ranking.csv     # ranking de mutual information (train)
│       ├── feature_sets.json  # F1 (sin NaN), F2 (completo), SK_habilidades y targets
│       ├── dataset_extra.csv  # 27 variables nuevas (Elo, EWMA, REF, empate, contexto...) alineadas con dataset_final (lo genera 02_feature.ipynb)
│       ├── revision_variables.csv  # tabla de revisión de las 93 variables (MI, IV, PSI, veredicto)
│       ├── dataset_skills.csv # 33 habilidades del once titular (local - visitante) + FTHG, FTAG, GD
│       ├── hiperparametros.json  # hiperparámetros base elegidos (04_modelo.ipynb, Paso 6a)
│       ├── hiperparametros_modelos.json  # SVM, XGBoost, MLP (04_modelo.ipynb, Paso 6)
│       ├── hiperparametros_extendidos.json  # rejillas extendidas (05_hiperparametros.ipynb, Paso 2)
│       ├── modelo_final.json / modelo_final.joblib / modelo_final_oof.csv  # modelo final y sus predicciones fuera de muestra (08_modelo_final.ipynb)
│       ├── modelos_oof.csv / ablation_results.csv / skills_*.csv / seleccion_caracteristicas.csv  # salidas de 04_modelo.ipynb
│       ├── backtest_resultados.csv / backtest_precio_blando.csv / mercado_pruebas.csv  # salidas de 06_mercado.ipynb
│       ├── predicciones_oof.csv  # predicciones fuera de muestra de los 8 folds (06_mercado.ipynb, Paso 2)
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

# 2) Notebooks (desde scripts/notebooks/, en este orden):
#    1) 01_eda.ipynb     -> limpia matches_sq.csv y exporta matches_clean.csv (+ figuras)
#    2) 02_feature.ipynb -> feature engineering: exporta dataset_final.csv, dataset_skills.csv, dataset_extra.csv, revision_variables.csv
#    3) 03_preprocesamiento.ipynb -> justificación del preprocesamiento, nuevas variables, PSI
#    4) 04_modelo.ipynb  -> bloques, modelos (LR/HGB/XGBoost/SVM/MLP), hiperparámetros base (genera hiperparametros.json y modelos_oof.csv), MI vs PCA
#    5) 05_hiperparametros.ipynb  -> validación anidada, rejillas extendidas (genera hiperparametros_extendidos.json)
#    6) 06_mercado.ipynb -> predicciones fuera de muestra, pruebas de eficiencia del mercado y backtest
#    7) 07_evaluacion.ipynb       -> CV, ROC/AUC, calibración, bootstrap por bloques (usa modelos_oof.csv)
#    8) 08_modelo_final.ipynb     -> modelo final, evaluación, backtest y guardado (genera modelo_final.json)

# 3) Informe: cd docs && tectonic informe.tex
```

> **Nota**: usar siempre `.venv/bin/python`. El mapeo de clubes (EA → football-data) se hace en `scripts/download/process_data.py` (paso 5: overrides + `rapidfuzz` como respaldo). Cobertura actual de squad quality: 100% de los slots equipo-partido.
