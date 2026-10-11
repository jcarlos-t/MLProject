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

1. **Datos** (ver `docs/Data_Dictionary.md`; guía de reproducción en `scripts/download/README.md`):
   - `scripts/download/fetch_data.py` descarga crudos -> `scripts/data/raw/` (117 CSVs de partidos, football-data.co.uk: 9 ligas × 13 temporadas; 2013/14 es solo calentamiento del historial y 2014/15–2025/26 son las temporadas modeladas) y `scripts/data/players/raw/` (12 snapshots de ratings de jugadores, Kaggle: FIFA 15 → FC 26); el snapshot FC25 viene de sofifa, capturado en julio de 2025: final de la temporada 2024/25 y sin fichajes de verano).
   - `scripts/download/process_data.py` **solo consolida** -> `matches.csv` (36,150 partidos, 9 ligas, 12 temporadas), `players/raw/players_all.csv` y `matches_sq.csv` (base unificada).
   - `scripts/notebooks/eda.ipynb` (EDA): limpia la base unificada (columnas de cuotas con >40% de nulos eliminadas, imputación KNN de cuotas y de estadísticas de partido, squad imputado con la media liga-temporada; `Time` descartada por ser estructural desde 2019/20) y exporta `matches_clean.csv` (sin nulos).
   - `scripts/notebooks/feature.ipynb` (feature engineering): parte del dataset limpio, construye features de historial (rolling últimos 5), mercado (consenso de probabilidades implícitas B365/BW/PS/WH, `MKT_*`), squad (`DIF_sq_top11`) y un ranking con `mutual_info_classif` (`mi_ranking.csv`), y exporta `dataset_final.csv` (33 features + target + metadatos), `feature_sets.json` (F1 = 13 features sin NaN para LR/SVM/MLP; F2 = las 33, con NaN en tiros/córners/faltas/tarjetas donde la fuente no los publica) y `backtest_odds.csv` (cuotas B365 de apertura y Pinnacle de cierre, observadas).
   - Diccionario completo de columnas en `docs/Data_Dictionary.md`.
2. **Feature engineering**:
   - Rolling / lagged por equipo: media de los últimos 5 partidos de goles (a favor/en contra), tiros, tiros a puerta, corners, faltas, tarjetas, puntos, fuerza defensiva (clean sheets). Esto respeta el requisito de "solo información previa".
   - Squad quality directa (pre-temporada): se conserva `*_sq_top11_mean` (media del once titular) por lado + banderas `*_sq_mapped`, y se crea la ventaja neta `DIF_sq_top11 = local − visitante` (decisión en `feature.ipynb`).
   - Mercado: `MKT_pH/pD/pA` = promedio entre casas de las probabilidades implícitas normalizadas de las cuotas de **apertura**, y `MKT_margin` (overround medio). Reemplaza a `AvgH/D/A`, que solo existen desde 2019/20.
   - Home advantage: features separadas para local y visitante.
3. **Selección de features**: `mutual_info_classif` (solo en train) como ranking; `dataset_final.csv` conserva todas las features y el top-k se decide en la validación temporal del modelado.
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
- **Temporada actual (2526) incompleta**: `fetch_data.py` no vuelve a descargar archivos existentes; para refrescarla, borrar `raw/*_2526.csv` y re-ejecutar. En 2025/26 Pinnacle (`PSH*`) tiene solo ~48% de cobertura.

---

## Estructura del repositorio

```
ML_Project/
├── README.md                  # este documento
├── requirements.txt           # dependencias pinneadas (Python)
├── docs/
│   ├── informe.tex            # paper del proyecto (IEEEtran)
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
#    1) eda.ipynb     -> limpia matches_sq.csv y exporta matches_clean.csv
#    2) feature.ipynb -> feature engineering sobre matches_clean.csv y exporta dataset_final.csv
```

> **Nota**: usar siempre `.venv/bin/python`. El mapeo de clubes (EA → football-data) se hace en `scripts/download/process_data.py` (paso 5: overrides + `rapidfuzz` como respaldo). Cobertura actual de squad quality: 100% de los slots equipo-partido.
