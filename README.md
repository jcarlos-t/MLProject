# ML_Project — Predicción de resultados de fútbol (1X2)

**Predicción probabilística del resultado de partidos de fútbol mediante Machine Learning**, usando únicamente información disponible **antes** del partido: forma de los equipos (últimos 5 partidos, desfazados), calidad de plantilla (ratings FIFA/FC de jugadores) y cuotas de apuestas pre-partido.

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
| Usar solo información previa (features lagged + cuotas de apertura + ratings) | Ambos | `feature.ipynb`: forma rolling sin el partido actual, `MKT_*`, plantilla |
| Selección de features con MI frente a PCA | Tammouch 2024 | `modelo.ipynb` Paso 9: 2 conjuntos × 3 modelos × 9 métodos |
| Habilidades individuales y regresión de la diferencia de goles | Stübinger 2020 | `SK_*` (33 habilidades del once, top-4 + portero) y modelos LIN/RF/HGB/ALL |
| Ensemble de modelos | Stübinger 2020 | `ALL` de clasificación y de regresión; se compara con LR |
| Backtest con umbral y cuotas reales | Stübinger 2020 | `mercado.ipynb` Paso 7: umbral elegido en validación, regla `|ŷ| > 2` |
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
   - `scripts/notebooks/eda.ipynb` (EDA): limpia la base unificada (159 columnas de cuotas: 138 con >40% de nulos eliminadas, 3 banderas, 18 imputadas con KNN; `Time` y `Referee` descartadas por ser estructurales; squad imputado con la media liga-temporada), exporta `matches_clean.csv` (36,150 × 80, sin nulos) y genera las 8 figuras de `docs/figures/`.
2. **Feature engineering** (`scripts/notebooks/feature.ipynb`):
   - Forma: media de los últimos 5 partidos de cada equipo, **sin incluir el actual**, con calentamiento 2013/14 y ventana acolchada con la media de la liga anterior a la fecha. Las estadísticas de partido no se imputan (NaN estructural en 11.0% de los partidos).
   - Mercado: `MKT_pD`, `MKT_logit = ln(pH/pA)` y `MKT_margin`, del consenso de probabilidades implícitas de B365/BW/PS/WH (cuotas **observadas** de apertura). Reemplaza a `AvgH/D/A`, que solo existen desde 2019/20. Se excluyen 31 partidos sin ninguna cuota.
   - Plantilla: `HomeTeam_sq_top11_mean` y `DIF_sq_top11 = local − visitante`.
   - Habilidades: 33 variables `SK_*` (top-4 de campo y portero del once titular, diferencia local − visitante) y target de regresión `GD = FTHG − FTAG`.
   - Salidas: `dataset_final.csv` (36,119 × 40), `feature_sets.json` (F1 = 13 variables sin NaN para LR/SVM/MLP; F2 = 33, con NaN para árboles), `dataset_skills.csv`, `backtest_odds.csv`, `mi_ranking.csv`.
3. **Selección de features**: `mutual_info_classif` como ranking (primer 80% por fecha) y comparación MI vs PCA con validación temporal en `modelo.ipynb` (Paso 9).
4. **Modelos** (`scripts/notebooks/modelo.ipynb`, `mercado.ipynb`):
   - Referencias: prior y mercado crudo.
   - **LR**, **HGB**, **XGBoost**, **SVM (RBF)**, **MLP** y **ensemble** (promedio de probabilidades), más regresión de la diferencia de goles (LIN, RF, HGB y ALL) convertida a probabilidades 1X2.
   - Hiperparámetros elegidos en la ventana 2016/17–2017/18 (disjunta de la evaluación) con desempate por simplicidad.
5. **Evaluación** (probabilística y predictiva), validación de **ventana expansiva** con 8 folds (2018/19–2025/26):
   - **Log loss** (primaria), **RPS**, accuracy, acierto del ganador sin empates, RMSE/MAE de GD.
   - **Calibración**: ECE y pendiente de calibración, con normalización proporcional y por potencia del margen.
   - Comparaciones pareadas con *bootstrap*, McNemar, Benjamini–Hochberg.
6. **Pruebas de eficiencia del mercado y backtest** (`mercado.ipynb`):
   - Modelo de residuo (`GD − ŷ_mercado`), *encompassing* apertura vs cierre de Pinnacle, CLV frente a la línea de cierre y backtest de apuestas de una unidad con cuotas de apertura de Bet365 y Pinnacle (umbral `τ` elegido en los 4 primeros folds y reportado en los 4 últimos).

### Resultados principales

| Pregunta | Resultado |
|---|---|
| Mejor modelo (log loss, 8 folds) | **LR sobre 3 variables de cuotas: 0.9656**; mercado crudo 0.9665; prior 1.0726 |
| ¿Algo supera al mercado? | No: plantilla, forma, estadísticas, 33 habilidades, HGB, MLP, XGBoost, SVM y ensemble empatan o empeoran (+0.002 a +0.027) |
| Con cuotas vs sin cuotas (ganador, sin empates) | 71.8% vs 70.0%; cuando discrepan acierta el modelo con cuotas el 57.5% (McNemar p < 0.0001); residuo R² = −0.0007 |
| Plantilla frente a forma | Sin cuotas la plantilla mejora el log loss en 0.041; con cuotas, ≈ 0 incluso al inicio de la temporada |
| MI vs PCA | MI gana con cuotas (8/8 folds); PCA ≥ MI sin cuotas; ninguna supera a usar todas las variables |
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
│   └── figures/               # figuras del EDA usadas por el informe
├── scripts/
│   ├── download/
│   │   ├── README.md          # guía de reproducibilidad de los datos
│   │   ├── fetch_data.py      # descarga de datos crudos
│   │   └── process_data.py    # solo consolidación de crudos
│   ├── notebooks/
│   │   ├── eda.ipynb          # EDA de partidos -> exporta matches_clean.csv (figuras del informe)
│   │   ├── feature.ipynb      # feature engineering sobre matches_clean -> dataset_final.csv
│   │   ├── modelo.ipynb       # comparación de bloques/modelos con CV temporal (mercado, plantilla, forma, habilidades; clasificación y regresión)
│   │   └── mercado.ipynb      # ¿quién gana? cuotas vs sin cuotas, desacuerdo, calibración del mercado, hiperparámetros
│   └── data/                  # generado por los scripts (gitignored)
│       ├── matches.csv        # 36,150 partidos (consolidado crudo)
│       ├── matches_sq.csv     # base unificada: 225 cols, 36 de squad
│       ├── matches_clean.csv  # dataset limpio, sin nulos (lo genera eda.ipynb)
│       ├── dataset_final.csv  # 33 features + target + metadatos (lo genera feature.ipynb)
│       ├── backtest_odds.csv  # cuotas B365 (apertura) y Pinnacle (cierre) alineadas fila a fila
│       ├── mi_ranking.csv     # ranking de mutual information (train)
│       ├── feature_sets.json  # F1 (sin NaN), F2 (completo), SK_habilidades y targets
│       ├── dataset_skills.csv # 33 habilidades del once titular (local - visitante) + FTHG, FTAG, GD
│       ├── hiperparametros.json  # hiperparámetros elegidos (mercado.ipynb, Paso 1)
│       ├── hiperparametros_modelos.json  # SVM, XGBoost, MLP (modelo.ipynb, Paso 6)
│       ├── modelos_oof.csv / ablation_results.csv / skills_*.csv / seleccion_caracteristicas.csv  # salidas de modelo.ipynb
│       ├── backtest_resultados.csv / backtest_precio_blando.csv / mercado_pruebas.csv  # salidas de mercado.ipynb
│       ├── predicciones_oof.csv  # predicciones fuera de muestra de los 8 folds (mercado.ipynb, Paso 2)
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
#    1) eda.ipynb     -> limpia matches_sq.csv y exporta matches_clean.csv (+ figuras)
#    2) feature.ipynb -> feature engineering y exporta dataset_final.csv, dataset_skills.csv
#    3) mercado.ipynb -> hiperparámetros, predicciones fuera de muestra, mercado y backtest
#    4) modelo.ipynb  -> bloques, modelos (LR/HGB/XGBoost/SVM/MLP), MI vs PCA, preguntas de investigación

# 3) Informe: cd docs && tectonic informe.tex
```

> **Nota**: usar siempre `.venv/bin/python`. El mapeo de clubes (EA → football-data) se hace en `scripts/download/process_data.py` (paso 5: overrides + `rapidfuzz` como respaldo). Cobertura actual de squad quality: 100% de los slots equipo-partido.
