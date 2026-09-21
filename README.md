# ML_Project — Predicción de resultados de fútbol (1X2)

**Predicción probabilística del resultado de partidos de fútbol mediante Machine Learning**, usando únicamente información disponible **antes** del partido: forma de los equipos (últimos 5 partidos, desfazados), calidad de plantilla (ratings FIFA/FC de jugadores) y cuotas de apuestas pre-partido.

---

## Contexto académico: qué dice la literatura

Este proyecto se apoya en dos trabajos de referencia que marcan la metodología, los modelos y la forma de evaluar.

### 1. Tammouch, Elouafi & Essadik (2024) — *Betting on Machine Learning: Extracting Patterns from Football's Anarchic Odds*

- Predice el resultado (1X2) de la Premier League con estadísticas de partido y **selección de características**.
- Compara **Deep Neural Networks, Decision Trees y Random Forest**, con **Mutual Information (MI)** y **PCA** para reducir dimensionalidad.
- Resultados clave:
  - **DNN + MI es el mejor**: accuracy ~55.1% (baseline claro vs ~33% de azar).
  - Decision Trees ~41–46%; suprir no se benefician mucho de MI.
  - **Random Forest falla (~26%)** en datasets pequeños y de alta dimensionalidad ("curse of dimensionality").
  - **MI > PCA**: PCA descarta features de baja varianza que sí tienen valor predictivo.
- Retos identificados que asumimos: equipos ascendidos/descendidos crean huecos de datos (imputación necesaria), y el sobreajuste por pocas muestras frente a muchas features.

### 2. Knoll & Stübinger (2020) — *Machine-Learning-Based Statistical Arbitrage Football Betting* (Applied Sciences)

- **No predice la clase 1X2, sino la diferencia de goles (regresión)** con características de partido + **habilidades individuales de los jugadores** (40 features por jugador).
- 47,856 partidos de las 5 grandes ligas y sus segundas divisiones (2006–2018). Cada temporada se divide en **periodo de formación** (primeros 5 matchdays) y **periodo de trading** (resto), evitando look-ahead bias.
- **Estrategia de betting (statistical arbitrage)**: apuesta solo si la predicción de diferencia de goles es extrema (`|ŷ| > 2`), usando cuotas Bet365.
- Resultados:
  - El **ensemble ALL** (promedio de RF + Boosting + SVM + regresión lineal) logra **+1.58% por apuesta**; RF (+0.43%) y Boosting (+0.72%) positivos.
  - Los bench marks ingenuos **pierden dinero**: BET (−4.53%), HOM (−4.60%), RAN (−8.17%).
  - Las estrategias **aversas al riesgo** (apuestas de cuota baja, = resultados probables) son las más rentables a largo plazo.
- Conclusión clave: **combinar modelos (ensembles) y ser selectivo/conservador supera a cualquier modelo individual y a las heurísticas de apuestas**.

### Qué tomamos de cada uno

| Lección | Fuente | Aplicación en el proyecto |
|---|---|---|
| Usar solo información previa (features lagged + cuotas pre-partido + ratings pre-temporada) | Ambos | Features en `matches_sq.csv` |
| Selección de features con MI (evitar PCA que pierde info) | Tammouch 2024 | `mutual_info_classif` antes de entrenar |
| Modelos robustos a datos pequeños: evitar overfit | Tammouch 2024 | RF con regularización, XGBoost, CV temporal |
| Player/squad features mejoran la predicción | Knoll & Stübinger 2020 | Features `*_sq_*` de calidad de plantilla |
| Ensemble/blending de modelos > modelo individual | Knoll & Stübinger 2020 | Voting/average de probabilidades |
| Backtest de apuestas con umbral de confianza y cuotas reales | Knoll & Stübinger 2020 | Regla de "value bet" + ROI por apuesta |
| División temporemporal formación/trading para evitar look-ahead | Knoll & Stübinger 2020 | Split temporal por fecha, nunca aleatorio |

---

## Qué vamos a hacer (metodología)

### Objetivo

Estimar la distribución de probabilidad `P(Home), P(Draw), P(Away)` para cada partido y,
dado que el modelo esté bien calibrado, compararla contra las cuotas del mercado para
detectar **apuestas con valor (value bets)**.

### Pipeline

1. **Datos** (ver `data/DATA_DICTIONARY.md`):
   - `scripts/download/fetch_data.py` descarga crudos; `process_data.py` **solo consolida** → `matches.csv` (30,182 partidos, 9 ligas, 10 temporadas) y `players/raw/players_all.csv`.
   - `scripts/eda_y_analitycs/eda.ipynb` (EDA + analytics de partidos): decide calidad y exporta `matches_clean.csv` (183 cols tipadas).
   - `scripts/eda_y_analitycs/eda_jugadores.ipynb` (EDA + analytics de jugadores): decide limpieza + features de jugadores y exporta `matches_sq.csv` (36 columnas de squad quality: 17 features + bandera `_sq_mapped` por lado; las versiones z-score se calculan en el modelado).
   - Diccionario completo de columnas en `DATA_DICTIONARY.md`.
2. **Feature engineering**:
   - Rolling / lagged por equipo: media de los últimos 5 partidos de goles (a favor/en contra), tiros, tiros a puerta, corners, faltas, tarjetas, puntos, fuerza defensiva (clean sheets). Esto respeta el requisito de "solo información previa".
   - Squad quality directa (pre-temporada): `*_sq_overall_mean`, `top11_mean`, `*_top_mean` por línea + versión relativa `_z` (decisión en `eda_jugadores.ipynb`: **usar las `_z`**).
   - Home advantage: features separadas para local y visitante.
3. **Selección de features**: `mutual_info_classif` sobre el total de features; comparar con entrenar sin selección.
4. **Modelos**:
   - Baseline: predecir la clase mayoritaria (o las cuotas del mercado).
   - **Logistic Regression** (lineal, interpretable), **Random Forest**, **XGBoost**, y opcionalmente **MLP/DNN** para replicar el hallazgo de Tammouch sobre redes.
   - **Ensemble**: promedio de probabilidades (blending) de los mejores modelos.
5. **Evaluación** (probabilística y predictiva):
   - Accuracy, F1 macro.
   - **Log Loss y Brier Score** (el objetivo es buena calibración, no solo acierto).
   - **Calibración**: reliability diagram + Brier decomposition.
   - Validación **temporal** (train en temporadas pasadas, test en la siguiente), respetando orden cronológico.
6. **Backtest de apuestas (statistical arbitrage)**:
   - Convertir probabilidades del modelo vs cuotas **de mercado** (apertura `AvgH/D/A` y cierre `AvgCH/D/A`/Pinnacle `PSCH/D/A`, ya descontando el overround).
   - Reglas de apuesta con umbral de "valor": apostar solo si `prob_modelo × cuota > 1 + threshold`.
   - Métricas: ROI/ROI por apuesta, nº de apuestas, racha, comparación con bench marks HOM/BET/RAN.
   - Si nuestro modelo es buen predictor, debe reproducir el resultado del paper de Knoll & Stübinger: **retornos positivos y significativos** frente a estrategias ingenuas.

### Cómo se tratarán los problemas típicos

- **Equipos ascendidos / sin historial** (problema de Tammouch): imputación con el rendimiento medio de equipos de mitad de tabla de la temporada previa; los slots sin squad quality quedan marcados con `_sq_mapped=0` y se imputan.
- **Curse of dimensionality** (RF falló en el paper): pocas features bien elegidas (MI), CV, regularización y pipelines simples primero.
- **Desbalanceo leve de clases** (H=44%, D=25%, A=31%): no se rebalancea (el objetivo es probabilidad calibrada), se usa el desbalance como prior.
- **Temporada actual (2526) incompleta**: `matches.csv` incluye todos los partidos hasta hoy; para el backtest se puede optar por quedarnos hasta 2425.

---

## Estructura del repositorio

```
ML_Project/
├── README.md                  # este documento
├── DATA_DICTIONARY.md         # diccionario de datos (todas las columnas)
├── data.md                    # documentación general de datos y reproducción
├── data/
│   ├── matches.csv            # 30,182 partidos, 186 columnas (consolidado crudo)
│   ├── matches_clean.csv      # base tipada de partidos (183 cols, lo genera eda.ipynb)
│   ├── matches_sq.csv      # squad quality consolidado: 222 cols (219 útiles), 36 col. de squad
│   ├── raw/                   # 90 CSVs de football-data.co.uk
│   └── players/               # snapshots Kaggle, players_all, normalized, squad_quality, club_map
├── scripts/
│   ├── download/
│   │   ├── fetch_data.py      # descarga de datos crudos
│   │   └── process_data.py    # solo consolidación de crudos
│   └── eda_y_analitycs/       # notebooks unificados (EDA + analytics):
│                              #   eda.ipynb (partidos), eda_jugadores.ipynb (jugadores)
└── notebooks/                 # (pendiente: modelado final, backtest)
```

---

## Reproducibilidad

```bash
python -m venv .venv
.venv/bin/pip install pandas rapidfuzz kagglehub

.venv/bin/python scripts/download/fetch_data.py    # descarga data/raw y data/players/raw
.venv/bin/python scripts/download/process_data.py  # consolida matches.csv + players_all.csv

# Notebooks (en orden, desde scripts/eda_y_analitycs/):
#   1) eda_jugadores.ipynb    -> genera matches_sq.csv (features de jugadores + join + analytics)
#   2) eda.ipynb              -> genera matches_clean.csv (base tipada) + analytics de partidos
```

> **Nota**: usar siempre `.venv/bin/python`. Sin `rapidfuzz` en el entorno, el mapeo de clubes (en `eda_jugadores.ipynb`) pierde ~26 mapeos fuzzy (cobertura de 97.1% → 96.0%).

Shap
IV

ventanas por meses: std, mean, media, min, max, tendencia (pendientes)