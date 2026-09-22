# Reproducibilidad de los datos

Este directorio contiene los scripts que **descargan y consolidan** los datos crudos.
No hacen EDA ni modelado (eso vive en `../eda/`).

## Qué hace cada script

| Script | Entrada | Salida | Descripción |
|---|---|---|---|
| `fetch_data.py` | Internet (football-data.co.uk, Kaggle) | `data/raw/<league>_<season>.csv` (90 CSVs de partidos) y `data/players/raw/*.csv` (10 snapshots de ratings de jugadores) | Descarga de datos crudos. |
| `process_data.py` | `data/raw/` y `data/players/raw/` | `data/matches.csv`, `data/players/raw/players_all.csv`, `data/players/normalized/players_*.csv`, `data/players/squad_quality_players.csv`, `data/players/club_map.csv`, `data/matches_sq.csv` | Consolidación y join por (Season, Div, Team). No hace EDA ni decide calidad. |

> Todas las rutas relativas apuntan a `scripts/data/` (las salidas se escriben ahí, junto a los datos ya descargados).

## Requisitos

- Python 3.10+
- Conexión a internet (primera descarga)
- Leverage el entorno virtual del proyecto:

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

- **Flujo aguas arriba del modelado**: `../eda/eda.ipynb` limpia `matches_sq.csv` y exporta
  `matches_clean.csv` (dataset limpio); `../eda/feature.ipynb` parte de ese archivo limpio y exporta
  `dataset_final.csv` (40 features top-MI + target).
- **Kaggle**: los 10 snapshots provienen de datasets públicos (no requieren credenciales para
  descargar vía `kagglehub`). Están definidos en `KAGGLE_DATASETS` dentro de `fetch_data.py`;
  algunos usan el filtro `fifa_version` para fijar el snapshot de esa temporada.
- **Snapshots**: hay exactamente un snapshot por temporada para evitar mezclar versiones de
  ratings de EA/FC en el cruce de `process_data.py`.
- El mapeo fuzzy de clubes (EA → football-data) se hace en `../eda/feature.ipynb` y requiere
  `rapidfuzz`; sin él la cobertura cae de ~97.1% a ~96.0%.