# Data Documentation

## 1. Data Origin

### Match Data (football-data.co.uk)

- **Source**: https://football-data.co.uk/data.php
- **Files**: `data/raw/<DIV>_<SEASON>.csv` (90 CSVs)
- **Cobertura**: 10 temporadas (2016/17 a 2025/26), 9 ligas europeas
- **Ligas**: E0 (Premier League), D1 (Bundesliga), SP1 (La Liga), I1 (Serie A), F1 (Ligue 1), N1 (Eredivisie), P1 (Primeira Liga), T1 (Super Lig), B1 (Pro League)
- **Acceso**: gratuito, sin API key, descarga directa por HTTP
- **Actualización**: semanal (domingos y miércoles)

### Player Ratings (Kaggle)

- **Source**: https://www.kaggle.com/datasets/ (múltiples datasets públicos)
- **Files**: `data/players/raw/*.csv` (10 snapshots)
- **Datasets**:
  - `luisfucros/fifa-players` (2017-2022): ratings FIFA 17-22
  - `stefanoleone992/ea-sports-fc-24-complete-player-dataset` (2023-2024): ratings FC 23-24. Unico archivo `male_players.csv` con `fifa_version` 15-24; el script lo filtra a `fifa_version==23` y `fifa_version==24` respectivamente
  - `akshyakumarkc/fifa-25-player-ratings` (2025): ratings FIFA 25
  - `rovnez/fc-26-fifa-26-player-data` (2026): ratings FC 26
- **Acceso**: público, descarga via `kagglehub`, **sin API key** (los IDs de dataset usados son los publicos correctos; el dataset de 2023-2024 es `stefanoleone992/...`, no `stefanoleone/...` que requiere autenticacion)

---

## 2. General Characteristics

### matches.csv (output consolidated)

| Columna | Descripcion |
|---------|-------------|
| `Div` | Codigo de liga (E0, D1, SP1, I1, F1, N1, P1, T1, B1) |
| `Season` | Temporada (ej: 1617 = 2016/17) |
| `Date`, `HomeTeam`, `AwayTeam` | Identificacion del partido |
| `FTHG`, `FTAG` | Goles local/visitante |
| `FTR` | Resultado: H (local), D (empate), A (visitante) = **target** |
| `HS/AS`, `HST/AST`, `HC/AC` | Tiros, tiros a puerta, corners |
| `HF/AF`, `HY/AY`, `HR/AR` | Faltas, tarjetas amarillas/rojas |
| Cuotas (`B365H/D/A`, `AvgH/D/A`, etc.) | Cuotas de apuestas (apertura y cierre) |

- **Total**: 30,182 partidos, ~4.4 MB
- **Distribucion target**: H=44.3%, D=24.8%, A=30.9%

### matches_sq.csv (output enriched)

- `matches.csv` + 18 columnas de squad quality (9 por equipo local, 9 por visitante)
- **Features por equipo**: `n_players`, `overall_mean`, `overall_max`, `top11_mean`, `top15_mean`, `gk_top_mean`, `def_top_mean`, `mid_top_mean`, `att_top_mean`
- **Total**: 61 columnas, ~97.1% cobertura de team-slots (los slots sin jugadores mapeados quedan `NA` con `_sq_mapped=0`)

### squad_quality_players.csv

- Agregados por (season, club): media overall, top-11, top-15, medias por linea (GK, DEF, MID, ATT)

### club_map.csv

- Mapeo club EA Sports -> club canonico de football-data (DIV:Equipo)

---

## 3. Scripts

### `scripts/fetch_data.py` — Descarga de datos

**Que hace**: Descarga automaticamente toda la data cruda necesaria.

1. **football-data.co.uk**: descarga 90 CSVs (10 temporadas x 9 ligas) via HTTP con `urllib`
2. **Kaggle**: descarga 10 snapshots de ratings de jugadores via `kagglehub`

**Uso**:
```bash
.venv/bin/python scripts/fetch_data.py
```

**Requisitos**: `pip install kagglehub pandas` (en el venv)

**Output**: `data/raw/` y `data/players/raw/`

---

### `scripts/process_data.py` — Pipeline de procesamiento

**Que hace**: Ejecuta el pipeline completo de transformacion y enriquecimiento de datos en 5 pasos:

1. **Consolidacion** (`data/matches.csv`): Lee los 90 CSVs de `data/raw/`, agrega columna `Season`, y concatena en un unico archivo
2. **Normalizacion** (`data/players/normalized/`): Estandariza esquemas de los 10 snapshots de Kaggle a columnas comunes: `season, club, league, overall, pos`
3. **Squad Quality** (`data/players/squad_quality_players.csv`): Agrega por (season, club) metricas de calidad de plantilla: media overall, top-11 mean, top-15 mean, medias por linea (GK/DEF/MID/ATT)
4. **Mapeo de clubes** (`data/players/club_map.csv`): Mapea nombres de clubes EA Sports a nombres canonicos de football-data via overrides manuales, matching exacto, subconjunto de tokens, y fuzzy matching (rapidfuzz)
5. **Merge** (`data/matches_sq.csv`): Une `matches.csv` con `squad_quality_players.csv` via `club_map.csv`, agregando 18 features de plantilla (9 local + 9 visitante)

**Uso**:
```bash
.venv/bin/python scripts/process_data.py
```

**Requisitos**: `pip install pandas rapidfuzz` (en el venv)

**Output**: `data/matches.csv`, `data/matches_sq.csv`, `data/players/squad_quality_players.csv`, `data/players/club_map.csv`

---

## 4. Tratamiento Aplicado

| Paso | Archivo generado | Descripcion |
|------|-----------------|-------------|
| Descarga | `data/raw/*.csv` | Descarga sin transformacion |
| Descarga | `data/players/raw/*.csv` | Descarga + filtro por `fifa_version` (2023/2024) |
| Consolidacion | `data/matches.csv` | Union de CSVs + columna Season |
| Normalizacion | `data/players/normalized/*.csv` | Estandarizacion de esquemas |
| Squad Quality | `data/players/squad_quality_players.csv` | Agregacion estadistica por club |
| Club Mapping | `data/players/club_map.csv` | Normalizacion de nombres EA -> football-data |
| Merge | `data/matches_sq.csv` | Enriquecimiento de partidos con features de plantilla |

### Environment (venv)

Los scripts dependen de `pandas`, `rapidfuzz` y `kagglehub`. Se recomienda usar el entorno virtual `ML_Project/.venv`:

```bash
# Crear el venv (una sola vez)
python -m venv .venv

# Instalar dependencias (una sola vez)
.venv/bin/pip install pandas rapidfuzz kagglehub

# Ejecutar scripts con el Python del venv
.venv/bin/python scripts/fetch_data.py
.venv/bin/python scripts/process_data.py
```

**Importante**: `rapidfuzz` (fuzzy matching de clubes en el paso 4) **solo está disponible en el venv**. Si se ejecuta con el `python` global del sistema sin `rapidfuzz` instalado, el pipeline baja de 2,279 mapeos (con 26 fuzzy) a 2,253 (sin fuzzy), reduciendo la cobertura de `matches_sq.csv` de ~97.1% a ~96.0%. Usa siempre `.venv/bin/python`.

### Notas importantes

- Las estadisticas de partido (tiros, corners) son **post-partido**: deben usarse desfazadas (lagged) como features historicas
- Los nombres de equipo estan en ingles: normalizar antes de unir con otras fuentes
- Las cuotas de apuestas son pre-partido y pueden usarse directamente
- Para reproducir: ejecutar primero `fetch_data.py`, luego `process_data.py`
