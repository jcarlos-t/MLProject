# Reproducibilidad de los datos

Este directorio contiene los scripts que **descargan y consolidan** los datos crudos.
No hacen EDA ni modelado (eso vive en `../notebooks/`).

## Qué hace cada script

| Script | Entrada | Salida | Descripción |
|---|---|---|---|
| `fetch_data.py` | Internet (football-data.co.uk, Kaggle) | `data/raw/<league>_<season>.csv` (117 CSVs de partidos, incl. 2013/14 de calentamiento) y `data/players/raw/*.csv` (12 snapshots de ratings de jugadores) | Descarga de datos crudos. |
| `process_data.py` | `data/raw/` y `data/players/raw/` | `data/matches.csv`, `data/players/raw/players_all.csv`, `data/players/normalized/players_*.csv`, `data/players/squad_quality_players.csv`, `data/players/club_map.csv`, `data/matches_sq.csv` | Consolidación y join por (Season, Div, Team). No hace EDA ni decide calidad. |

> Todas las rutas relativas apuntan a `scripts/data/` (las salidas se escriben ahí, junto a los datos ya descargados).

## Requisitos

- Python 3.10+
- Conexión a internet (primera descarga)
- Usa el entorno virtual del proyecto:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Orden de ejecución

```bash
# 1) Descarga crudos (football-data.co.uk + snapshots de Kaggle)
.venv/bin/python scripts/download/fetch_data.py

# 2) Consolida partidos, jugadores y squad quality
.venv/bin/python scripts/download/process_data.py
```

(Si los archivos ya existen en `scripts/data/`, la descarga los salta/sobrescribe según
los mensajes del script; es seguro re-ejecutar.)

## Notas

- **Flujo aguas arriba del modelado**: `../notebooks/01_eda.ipynb` limpia `matches_sq.csv` y exporta
  `matches_clean.csv` (dataset limpio); `../notebooks/02_feature.ipynb` parte de ese archivo limpio y exporta
  `dataset_final.csv` (33 features + target), `dataset_extra.csv` (25 variables nuevas) y `backtest_odds.csv`.
- **Kaggle**: los 10 snapshots provienen de datasets públicos (no requieren credenciales para
  descargar vía `kagglehub`). Están definidos en `KAGGLE_DATASETS` dentro de `fetch_data.py`;
  algunos usan el filtro `fifa_version` para fijar el snapshot de esa temporada.
- **Snapshots**: hay exactamente un snapshot por temporada (FIFA 15 → FC 26, temporadas 2014/15–2025/26) para evitar mezclar versiones de
  ratings de EA/FC en el cruce de `process_data.py`.
- El mapeo de clubes (EA → football-data) se hace en `process_data.py` (paso 5: overrides + `rapidfuzz`).
  Cobertura de squad quality: 100%.
## Huellas de los datos

`manifest.py` calcula el SHA-256 de cada CSV crudo y lo guarda en `data_manifest.json` (versionado). `python scripts/download/manifest.py --check` compara los datos locales con ese manifiesto: si una descarga nueva difiere (por ejemplo, football-data.co.uk actualizó 2025/26), lo avisa.
