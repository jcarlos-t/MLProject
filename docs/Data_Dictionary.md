# Diccionario de Datos — ML_Project (fútbol 1X2)

Predicción probabilística del resultado de partidos de fútbol (local / empate / visitante).
Fuente: **football-data.co.uk** (partidos y cuotas) + **Kaggle** (ratings de jugadores FIFA/FC).

> **Versión:** base `matches_sq.csv` de **2014/15–2025/26 (36,150 partidos, 225 columnas)**. Los conteos de `Únicos`/`Nulos` de las tablas se calcularon sobre esa base. Para el modelado, `scripts/notebooks/eda.ipynb` la reduce a 80 columnas sin nulos (`matches_clean.csv`) y `feature.ipynb` construye `dataset_final.csv` (36,119 filas; ver §3). La temporada 2013/14 (`matches_warmup.csv`, 2,918 partidos) solo alimenta el historial de forma y no forma parte de esta tabla.

---

## 0. Conceptos clave del sistema

| Término | Qué es | Ejemplo en datos |
|---------|--------|------------------|
| **Liga (`Div`)** | Campeonato nacional. Los códigos son de football-data: cada uno identifica una liga europea. | `E0` = Premier League, `D1` = Bundesliga, `SP1` = La Liga, `I1` = Serie A, `F1` = Ligue 1, `N1` = Eredivisie, `P1` = Primeira Liga, `T1` = Süper Lig, `B1` = Pro League |
| **Temporada (`Season`)** | Ciclo anual de un campeonato, codificado con el año de inicio y el de fin. | `1415` = temporada 2014/15 … `2526` = 2025/26 |
| **Resultado 1X2 (`FTR`)** | **Target** del modelo: el partido se salda con victoria del local, empate o victoria del visitante. | `H`, `D`, `A` |
| **Equipo local / visitante** | `HomeTeam` juega en su estadio; `AwayTeam` de localía. El local tiene ventaja estadística (~44% de victorias). | `HomeTeam=Man United`, `AwayTeam=Man City` |
| **Cuota decimal** | Multiplicador que paga la casa si aciertas; >1 y menor = más probable. | Bet365 `B365H = 1.16` → local muy favorito |
| **Código de casa** | Prefijo que identifica la casa/referencia en el nombre de la columna (`B365`=Bet365, `PS`=Pinnacle, `Max`=máx. de mercado…). Ver §1.4.1. | `PSCH` → Pinnacle (PS), cierre (C), local (H) |
| **Cuota de apertura vs cierre** | Apertura = inicios de la semana; **cierre** = publicada al cerrar el mercado (considerada la mejor estimación de la 'probabilidad real'). | `B365H` (apertura) vs `B365CH` (cierre) |
| **Overround** | Margen de la casa: `1/H + 1/D + 1/A` siempre >1. Para comparar con el modelo hay que **normalizar** las cuotas (dividir por su suma). | `1/2.4 + 1/3.3 + 1/3.25 ≈ 1.05` |
| **Snapshot de jugadores** | Captura anual de ratings **EA Sports (FIFA/FC)** usada como 'valoración real' de cada jugador esa temporada. | Snapshots `2015.csv` … `2026.csv` |
| **Squad quality** | **Calidad de plantilla** agregada por (temporada, club) a partir de los ratings de los jugadores (medias, máximo y top-3 por línea). Es pre-temporada → usable directa. | `overall_mean`, `top11_mean`, `gk_top3_mean` |
| **Mapeo de clubes** | Traducción del nombre del club EA → nombre canónico de football-data. | `Man United` (football-data) = `Manchester United` (EA) |
| **Team-slot / `_sq_mapped`** | Cada partido tiene 2 'huecos' de equipo (local y visitante). `_sq_mapped=1` si ese hueco se mapeó a un club con squad; `0` si quedó `NA` (ascendidos, filtros de temporada…). | `HomeTeam_sq_mapped = 1` |
| **Estadísticas post-partido** | `HS`, `AS`, `HC`… se conocen **después** del partido. Para modelar solo con información previa hay que desfazarlas (rolling de los últimos N partidos por equipo). | `HS` (tiros del local) se usa media de los últimos 5 |

---

## 1. `scripts/data/matches_sq.csv` — Dataset principal (base unificada)

Generado por `scripts/download/process_data.py`: **join** de `matches.csv` con la
**squad quality** de local y visitante vía `Season|Div|Team` → `club_map.csv`
(un snapshot por temporada, sin mezclar versiones ni lookahead).

- **Una fila = 1 partido**. Cobertura: **9 ligas × 12 temporadas (2014/15–2025/26)** = **36,150 partidos**.
- **36,150 filas × 225 columnas** = 189 del partido + 36 de plantilla (17×2 lados + 2 flags).
- Las columnas vienen tipadas (int/float/str). Algunas columnas de partido se registran como
  `float` pero son conteos enteros (p. ej. `HS`); las cuotas son `float` continuas.
- Las features derivadas de plantilla (ventaja neta `DIF_sq_top11`) **no vienen aquí**: se
  construyen en el feature engineering (`scripts/notebooks/feature.ipynb`).

### 1.1 Identificación del partido

> Cada fila se identifica de forma única por `Div` + `Season` + `Date` + `HomeTeam` + `AwayTeam`.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `Div` | str | `E0` | 9 | 0 | **Liga** (código football-data). 9 ligas europeas. |
| `Season` | int | `1617` | 12 | 0 | **Temporada** como código de 4 dígitos: `1415` = 2014/15 … `2526` = 2025/26. |
| `Date` | str | `13/08/16` | 2634 | 0 | Fecha del partido (dd/mm/aa). |
| `Time` | str | `20:00` | 47 | 14920 | Hora de inicio (solo desde 2019/20; nulo estructural en 41.3% de las filas). |
| `HomeTeam` | str | `Burnley` | 296 | 0 | Equipo **local** (nombre en inglés). 296 equipos distintos. |
| `AwayTeam` | str | `Swansea` | 296 | 0 | Equipo **visitante**. |
| `Referee` | str | `J Moss` | 53 | 31590 | Árbitro (solo la Premier League lo publica; 87.4% de nulos). |


### 1.2 Resultado (target)

> `FTR` es el target del modelo. `HTHG/HTAG/HTR` son del medio tiempo (referencia, no target).

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `FTHG` | int | `0` | 11 | 0 | Goles del **local** al tiempo completo. |
| `FTAG` | int | `1` | 11 | 0 | Goles del **visitante** al tiempo completo. |
| `FTR` | str | `A` | 3 | 0 | **TARGET 1X2**: resultado a tiempo completo (H/D/A). |
| `HTHG` | float | `0` | 7 | 39 | Goles del local al **medio tiempo**. |
| `HTAG` | float | `0` | 8 | 39 | Goles del visitante al medio tiempo. |
| `HTR` | str | `D` | 3 | 39 | Resultado 1X2 al medio tiempo. |


### 1.3 Estadísticas de partido (post-partido)

> Información **after the match**: para el modelo usarlas **desfazadas** (media rolling de los últimos N partidos por equipo), nunca el valor del propio partido.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `HS` | float | `10` | 44 | 3513 | Tiros del local. |
| `AS` | float | `17` | 39 | 3513 | Tiros del visitante. |
| `HST` | float | `3` | 21 | 3513 | Tiros a puerta del local. |
| `AST` | float | `9` | 20 | 3513 | Tiros a puerta del visitante. |
| `HF` | float | `10` | 34 | 3993 | Faltas cometidas por el local. |
| `AF` | float | `14` | 34 | 3993 | Faltas cometidas por el visitante. |
| `HC` | float | `7` | 24 | 3513 | Córners del local. |
| `AC` | float | `4` | 20 | 3513 | Córners del visitante. |
| `HY` | float | `3` | 11 | 3513 | Tarjetas amarillas del local. |
| `AY` | float | `2` | 11 | 3512 | Tarjetas amarillas del visitante. |
| `HR` | float | `0` | 4 | 3512 | Tarjetas rojas del local. |
| `AR` | float | `0` | 4 | 3513 | Tarjetas rojas del visitante. |
| `HFKC` | float | `13` | 26 | 35672 | Free kicks concedidos por el local (98.7% de nulos; se descarta). |
| `AFKC` | float | `6` | 29 | 35672 | Free kicks concedidos por el visitante. |


### 1.4 Cuotas de apertura — 1X2

> Cuotas decimales pre-partido. `AvgH/D/A` y `MaxH/D/A` = referencia agregada de todo el mercado; `Bb*` = agregado Betbrain.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `B365H` | float | `2.4` | 172 | 58 | Cuota **Bet365** (apertura). |
| `B365D` | float | `3.3` | 71 | 58 | Idem Bet365. |
| `B365A` | float | `3.25` | 170 | 58 | Idem Bet365. |
| `BWH` | float | `2.45` | 177 | 1263 | Cuota **Bet&Win** (apertura). |
| `BWD` | float | `3.1` | 81 | 1263 | Idem Bet&Win. |
| `BWA` | float | `2.95` | 187 | 1263 | Idem Bet&Win. |
| `IWH` | float | `2.5` | 204 | 7683 | Cuota **Interwetten** (apertura). |
| `IWD` | float | `3.3` | 119 | 7683 | Idem Interwetten. |
| `IWA` | float | `2.65` | 215 | 7683 | Idem Interwetten. |
| `LBH` | float | `2.45` | 130 | 22047 | Cuota **Ladbrokes** (apertura). |
| `LBD` | float | `3.25` | 51 | 22047 | Idem Ladbrokes. |
| `LBA` | float | `3.1` | 131 | 22047 | Idem Ladbrokes. |
| `PSH` | float | `2.47` | 1264 | 1703 | Cuota **Pinnacle** (apertura); mercado muy eficiente. |
| `PSD` | float | `3.32` | 985 | 1703 | Idem Pinnacle. |
| `PSA` | float | `3.19` | 2079 | 1703 | Idem Pinnacle. |
| `WHH` | float | `2.5` | 124 | 4152 | Cuota **William Hill** (apertura). |
| `WHD` | float | `3.2` | 52 | 4152 | Idem WH. |
| `WHA` | float | `2.9` | 123 | 4152 | Idem WH. |
| `VCH` | float | `2.5` | 128 | 6090 | Cuota **VC Bet** (apertura). |
| `VCD` | float | `3.2` | 60 | 6090 | Idem VC Bet. |
| `VCA` | float | `3.25` | 129 | 6090 | Idem VC Bet. |
| `Bb1X2` | float | `55` | 45 | 21230 | Nº de casas usadas en el agregado Betbrain. |
| `BbMxH` | float | `2.55` | 746 | 21230 | Cuota **máxima** Betbrain 1X2 (apertura). |
| `BbAvH` | float | `2.43` | 1019 | 21230 | Cuota **promedio** Betbrain 1X2 (apertura). |
| `BbMxD` | float | `3.35` | 701 | 21230 | Máximo Betbrain 1X2. |
| `BbAvD` | float | `3.21` | 748 | 21230 | Promedio Betbrain 1X2. |
| `BbMxA` | float | `3.3` | 1122 | 21230 | Máximo Betbrain 1X2. |
| `BbAvA` | float | `3.1` | 1710 | 21230 | Promedio Betbrain 1X2. |
| `MaxH` | float | `1.16` | 689 | 14950 | **Máximo de mercado** (apertura) para local. |
| `MaxD` | float | `10` | 557 | 14950 | Máximo de mercado (empate). |
| `MaxA` | float | `23` | 989 | 14950 | Máximo de mercado (visitante). |
| `AvgH` | float | `1.14` | 1045 | 14950 | **Promedio de mercado** (apertura) para local. |
| `AvgD` | float | `8.75` | 785 | 14950 | Promedio de mercado (empate). |
| `AvgA` | float | `19.83` | 1662 | 14950 | Promedio de mercado (visitante). |

### 1.4.1 Cómo leer los nombres de las cuotas

Toda columna de cuota sigue el patrón `CODIGO + mercado + RESULTADO` (y una `C` extra = **cierre**):

- **Resultado (sufijo)**: `H` = Local (Home), `D` = Empate (Draw), `A` = Visitante (Away).
- **`C` antes del resultado** = cuota de **cierre** (publicada al cerrar el mercado). Sin `C` = **apertura**.
  Ej.: `PSH` = Pinnacle apertura local; `PSCH` = Pinnacle **cierre** local.
- **`>2.5` / `<2.5`** = mercado Over/Under 2.5 goles.
- **`AHH`/`AHA`** = Asian Handicap lado local/visitante; **`AHh`/`AHCh`/`BbAHh`** = **línea** del hándicap.
- **`Bb`** = agregado **Betbrain** (no es una casa): `Bb1X2`/`BbOU`/`BbAH` = nº de casas del agregado;
  `BbMx*`/`BbAv*` = cuota máxima/promedio del agregado.
- **`Max`** / **`Avg`** = cuota **máxima**/**promedio** de todo el mercado.

| Código | Casa / referencia | Columnas (apertura → cierre) |
|---|---|---|
| `B365` | **Bet365** | `B365H/D/A`, `B365>2.5`, `B365AHH/AHA` → `B365CH/D/A`, `B365C>2.5`, `B365CAHH/AHA` |
| `BW` | **Bet&Win** | `BWH/D/A` → `BWCH/D/A` |
| `IW` | **Interwetten** | `IWH/D/A` → `IWCH/D/A` |
| `LB` | **Ladbrokes** | `LBH/D/A` → `LBCH/D/A` |
| `PS` | **Pinnacle** (muy eficiente) | `PSH/D/A`, `P>2.5`, `PAHH/AHA` → `PSCH/D/A`, `PC>2.5`, `PCAHH/AHA` |
| `WH` | **William Hill** | `WHH/D/A` → `WHCH/D/A` |
| `VC` | **VC Bet** | `VCH/D/A` → `VCCH/D/A` |
| `BF` | **Betfair Exchange** | `BFH/D/A` → `BFCH/D/A` |
| `1XB` | **Betfair Match Odds** | `1XBH/D/A` → `1XBCH/D/A` |
| `BFE` | Betfair Exchange (datapoint) | `BFEH/D/A`, `BFE>2.5`, `BFEAHH/AHA` → `BFECH/D/A`, `BFEC>2.5`, `BFECAHH/AHA` |
| `BFD` | Betfair datapoint (solo 2425+) | `BFDH/D/A` → `BFDCH/D/A` |
| `BMGM` | **BetMGM** (solo 2425+) | `BMGMH/D/A` → `BMGMCH/D/A` |
| `BV` | **BetVictor** (solo 2425+) | `BVH/D/A` → `BVCH/D/A` |
| `CL` | **Colossus** (solo 2425+) | `CLH/D/A` → `CLCH/D/A` |
| `Max` | **Máximo de mercado** | `MaxH/D/A`, `Max>2.5`, `MaxAHH/AHA` → `MaxCH/D/A`, `MaxC>2.5`, `MaxCAHH/AHA` |
| `Avg` | **Promedio de mercado** | `AvgH/D/A`, `Avg>2.5`, `AvgAHH/AHA` → `AvgCH/D/A`, `AvgC>2.5`, `AvgCAHH/AHA` |

### 1.5 Cuotas de apertura — mercados secundarios

> Over/Under 2.5 y Asian Handicap de apertura. `AHh`/`BbAHh`/`AHCh` son la **línea** (hándicap), el resto son cuotas a cada lado.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `B365>2.5` | float | `1.4` | 101 | 14975 | Over/Under 2.5 — cuota Bet365 a **más** de 2.5 goles. |
| `B365<2.5` | float | `3` | 94 | 14975 | Idem Bet365 a menos. |
| `P>2.5` | float | `1.4` | 225 | 16612 | Pinnacle Over/Under 2.5 (>2.5). |
| `P<2.5` | float | `3.11` | 341 | 16612 | Pinnacle Over/Under 2.5 (<2.5). |
| `Max>2.5` | float | `1.45` | 210 | 14951 | Máximo de mercado Over 2.5. |
| `Max<2.5` | float | `3.11` | 301 | 14951 | Máximo de mercado Under 2.5. |
| `Avg>2.5` | float | `1.41` | 213 | 14951 | Promedio de mercado Over 2.5. |
| `Avg<2.5` | float | `2.92` | 352 | 14951 | Promedio de mercado Under 2.5. |
| `BbOU` | float | `40` | 38 | 21230 | Nº de casas usadas en el agregado Over/Under. |
| `BbMx>2.5` | float | `2.4` | 207 | 21230 | Máximo Betbrain Over 2.5. |
| `BbAv>2.5` | float | `2.3` | 183 | 21230 | Promedio Betbrain Over 2.5. |
| `BbMx<2.5` | float | `1.68` | 260 | 21231 | Máximo Betbrain Under 2.5. |
| `BbAv<2.5` | float | `1.61` | 336 | 21231 | Promedio Betbrain Under 2.5. |
| `AHh` | float | `-2.25` | 28 | 14958 | **Línea** del Asian Handicap de apertura (ej. `-2.25` = local con -2.25 goles). |
| `B365AHH` | float | `1.96` | 57 | 15017 | Asian Handicap Bet365 — lado local. |
| `B365AHA` | float | `1.94` | 59 | 15017 | Asian Handicap Bet365 — lado visitante. |
| `PAHH` | float | `1.97` | 62 | 16528 | Asian Handicap Pinnacle — local. |
| `PAHA` | float | `1.95` | 58 | 16528 | Asian Handicap Pinnacle — visitante. |
| `MaxAHH` | float | `1.97` | 65 | 14952 | Asian máximo de mercado — local. |
| `MaxAHA` | float | `2` | 68 | 14952 | Asian máximo de mercado — visitante. |
| `AvgAHH` | float | `1.94` | 59 | 14951 | Asian promedio de mercado — local. |
| `AvgAHA` | float | `1.94` | 98 | 14951 | Asian promedio de mercado — visitante. |
| `BbAH` | float | `32` | 33 | 21231 | Nº de casas del agregado Asian. |
| `BbAHh` | float | `-0.25` | 27 | 21230 | Línea Asian del agregado Betbrain. |
| `BbMxAHH` | float | `2.13` | 134 | 21230 | Máximo Betbrain Asian — local. |
| `BbAvAHH` | float | `2.06` | 134 | 21230 | Promedio Betbrain Asian — local. |
| `BbMxAHA` | float | `1.86` | 130 | 21230 | Máximo Betbrain Asian — visitante. |
| `BbAvAHA` | float | `1.81` | 128 | 21230 | Promedio Betbrain Asian — visitante. |


### 1.6 Betfair (Exchange) — apertura

> Exchange (apuestas entre usuarios). Disponible aprox. desde 2020/21. `1XB*` = Betfair Match Odds; `BFE*` = datapoint.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `BFH` | float | `1.6` | 99 | 33136 | **Betfair Exchange** 1X2 (apertura). |
| `BFD` | float | `4.33` | 44 | 33136 | Idem Betfair. |
| `BFA` | float | `5` | 101 | 33136 | Idem Betfair. |
| `1XBH` | float | `1.68` | 521 | 33213 | **Betfair Match Odds** 1X2 (apertura). |
| `1XBD` | float | `4.32` | 407 | 33213 | Idem. |
| `1XBA` | float | `5.03` | 636 | 33213 | Idem. |
| `BFEH` | float | `1.66` | 228 | 30347 | Betfair Exchange 1X2 (datapoint apertura). |
| `BFED` | float | `4.5` | 136 | 30347 | Idem. |
| `BFEA` | float | `5.6` | 240 | 30347 | Idem. |
| `BFE>2.5` | float | `1.59` | 154 | 30575 | Betfair Exchange Over 2.5 (apertura). |
| `BFE<2.5` | float | `2.64` | 174 | 30575 | Betfair Exchange Under 2.5. |
| `BFEAHH` | float | `2.1` | 81 | 31376 | Betfair Exchange Asian — local (apertura). |
| `BFEAHA` | float | `1.88` | 77 | 31376 | Betfair Exchange Asian — visitante. |


### 1.7 Cuotas de cierre — 1X2

> Publicadas al **cerrar** el mercado; se consideran la mejor estimación de la 'probabilidad real'. `AvgC*` disponible desde 2019/20; Pinnacle/Bet365 desde antes.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `PSCH` | float | `2.79` | 1285 | 1603 | **Pinnacle cierre** 1X2 — mejor estimación de 'probabilidad real' del mercado. |
| `PSCD` | float | `3.16` | 1008 | 1603 | Idem Pinnacle cierre. |
| `PSCA` | float | `2.89` | 2086 | 1603 | Idem Pinnacle cierre. |
| `B365CH` | str | `1.14` | 155 | 14950 | **Bet365 cierre** 1X2. |
| `B365CD` | float | `9.5` | 59 | 14950 | Idem. |
| `B365CA` | float | `21` | 154 | 14950 | Idem. |
| `BWCH` | float | `1.14` | 171 | 16177 | Bet&Win cierre. |
| `BWCD` | float | `9` | 76 | 16177 | Idem. |
| `BWCA` | float | `20` | 177 | 16177 | Idem. |
| `IWCH` | float | `1.15` | 186 | 22577 | Interwetten cierre. |
| `IWCD` | float | `8` | 113 | 22577 | Idem. |
| `IWCA` | float | `18` | 198 | 22577 | Idem. |
| `WHCH` | float | `1.11` | 124 | 19049 | William Hill cierre. |
| `WHCD` | float | `9.5` | 53 | 19049 | Idem. |
| `WHCA` | float | `21` | 124 | 19049 | Idem. |
| `VCCH` | float | `1.14` | 118 | 20968 | VC Bet cierre. |
| `VCCD` | float | `9.5` | 57 | 20968 | Idem. |
| `VCCA` | float | `23` | 122 | 20968 | Idem. |
| `MaxCH` | float | `1.16` | 890 | 14949 | **Máximo de mercado cierre** 1X2. |
| `MaxCD` | float | `10.5` | 705 | 14949 | Idem empate. |
| `MaxCA` | float | `23` | 1175 | 14949 | Idem visitante. |
| `AvgCH` | float | `1.14` | 1070 | 14949 | **Promedio de mercado cierre** 1X2. |
| `AvgCD` | float | `9.52` | 827 | 14949 | Idem empate. |
| `AvgCA` | float | `19.18` | 1682 | 14949 | Idem visitante. |


### 1.8 Cuotas de cierre — mercados secundarios

> Over/Under y Asian Handicap de cierre.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `B365C>2.5` | float | `1.3` | 106 | 14951 | Bet365 **cierre** Over 2.5. |
| `B365C<2.5` | float | `3.5` | 104 | 14951 | Bet365 cierre Under 2.5. |
| `PC>2.5` | float | `1.34` | 251 | 16579 | Pinnacle cierre Over 2.5. |
| `PC<2.5` | float | `3.44` | 377 | 16579 | Pinnacle cierre Under 2.5. |
| `MaxC>2.5` | float | `1.36` | 241 | 14949 | Máximo mercado cierre Over 2.5. |
| `MaxC<2.5` | float | `3.76` | 351 | 14949 | Máximo mercado cierre Under 2.5. |
| `AvgC>2.5` | float | `1.32` | 239 | 14949 | Promedio mercado cierre Over 2.5. |
| `AvgC<2.5` | float | `3.43` | 391 | 14949 | Promedio mercado cierre Under 2.5. |
| `AHCh` | float | `-2.25` | 30 | 14950 | **Línea Asian cierre**. |
| `B365CAHH` | float | `1.91` | 62 | 14954 | Asian cierre Bet365 — local. |
| `B365CAHA` | float | `1.99` | 61 | 14954 | Asian cierre Bet365 — visitante. |
| `PCAHH` | float | `1.94` | 89 | 16509 | Asian cierre Pinnacle — local. |
| `PCAHA` | float | `1.98` | 88 | 16509 | Asian cierre Pinnacle — visitante. |
| `MaxCAHH` | float | `1.99` | 82 | 14950 | Asian cierre máximo mercado — local. |
| `MaxCAHA` | float | `2.07` | 86 | 14950 | Idem visitante. |
| `AvgCAHH` | float | `1.9` | 64 | 14949 | Asian cierre promedio mercado — local. |
| `AvgCAHA` | float | `1.99` | 68 | 14949 | Idem visitante. |


### 1.9 Betfair (Exchange) — cierre

> Exchange de cierre: `BFCH/D/A`, `1XBCH/D/A`, `BFECH/D/A` (datapoint) y sus Over/Under/Asian.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `BFCH` | float | `1.62` | 101 | 33132 | **Betfair Exchange cierre** 1X2. |
| `BFCD` | float | `4` | 46 | 33132 | Idem. |
| `BFCA` | float | `5` | 102 | 33132 | Idem. |
| `1XBCH` | float | `1.66` | 575 | 33132 | Betfair Match Odds cierre. |
| `1XBCD` | float | `4.15` | 423 | 33132 | Idem. |
| `1XBCA` | float | `5.33` | 737 | 33132 | Idem. |
| `BFECH` | float | `1.72` | 229 | 30342 | Betfair Exchange (datapoint cierre) 1X2. |
| `BFECD` | float | `4.2` | 95 | 30342 | Idem. |
| `BFECA` | float | `5.4` | 236 | 30342 | Idem. |
| `BFEC>2.5` | float | `1.68` | 156 | 30343 | Betfair Exchange cierre Over 2.5. |
| `BFEC<2.5` | float | `2.46` | 164 | 30343 | Idem Under. |
| `BFECAHH` | float | `1.9` | 62 | 30346 | Betfair Exchange cierre Asian — local. |
| `BFECAHA` | float | `2.08` | 62 | 30346 | Idem visitante. |


### 1.10 Cuotas de casas recientes (2024/25–2025/26)

> Betfair datapoint, BetMGM, BetVictor, Colossus y Ladbrokes-cierre: solo temporadas 2425–2526 (muchos nulos antes).

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `BFDH` | float | `1.3` | 76 | 33175 | Betfair datapoint (apertura). |
| `BFDD` | float | `6` | 31 | 33175 | Idem. |
| `BFDA` | float | `9.5` | 77 | 33174 | Idem. |
| `BMGMH` | float | `1.29` | 195 | 33186 | **BetMGM** (apertura). |
| `BMGMD` | float | `6.5` | 67 | 33186 | Idem. |
| `BMGMA` | float | `9` | 192 | 33186 | Idem. |
| `BVH` | float | `1.3` | 103 | 33261 | **BetVictor** (apertura). |
| `BVD` | float | `6` | 43 | 33261 | Idem. |
| `BVA` | float | `8.5` | 102 | 33261 | Idem. |
| `CLH` | float | `1.33` | 91 | 33994 | **Colossus** (apertura). |
| `CLD` | float | `5.75` | 38 | 33994 | Idem. |
| `CLA` | float | `8` | 88 | 33994 | Idem. |
| `BFDCH` | float | `1.3` | 78 | 33219 | Betfair datapoint **cierre**. |
| `BFDCD` | float | `6` | 32 | 33219 | Idem. |
| `BFDCA` | float | `9.5` | 81 | 33219 | Idem. |
| `BMGMCH` | float | `1.3` | 195 | 33170 | BetMGM cierre. |
| `BMGMCD` | float | `6.25` | 69 | 33170 | Idem. |
| `BMGMCA` | float | `9` | 193 | 33170 | Idem. |
| `BVCH` | float | `1.29` | 102 | 33229 | BetVictor cierre. |
| `BVCD` | float | `6` | 46 | 33229 | Idem. |
| `BVCA` | float | `9` | 103 | 33229 | Idem. |
| `CLCH` | float | `1.3` | 91 | 34064 | Colossus cierre. |
| `CLCD` | float | `6` | 41 | 34064 | Idem. |
| `CLCA` | float | `8` | 89 | 34064 | Idem. |
| `LBCH` | float | `1.3` | 90 | 33928 | Ladbrokes cierre. |
| `LBCD` | float | `5.75` | 41 | 33928 | Idem. |
| `LBCA` | float | `8` | 89 | 33928 | Idem. |


### 1.11 Columnas basura

> 100% vacías (artefactos de la concatenación de crudos). Descartar antes de modelar.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `Unnamed: 105` | float | `NA` | 0 | 36150 | **Basura**: 100% vacía (artefacto de concatenación). Descartar. |
| `Unnamed: 119` | float | `NA` | 0 | 36150 | Basura: 100% vacía. Descartar. |
| `Unnamed: 120` | float | `NA` | 0 | 36150 | Basura: 100% vacía. Descartar. |


### 1.12 Squad quality — local y visitante

18 columnas para **local** (`HomeTeam_sq_*`) y las mismas 18 para **visitante** (`AwayTeam_sq_*`), miradas desde el lado del equipo en cuestión. Features **crudas** de plantilla (un snapshot por temporada, pre-partido → usables directas).

| Campo (base `HomeTeam_sq_*` / `AwayTeam_sq_*`) | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `HomeTeam_sq_n_players` | float | `33` | 22 | 0 | Squad **HomeTeam**: nº de jugadores del club en el snapshot de la temporada. |
| `HomeTeam_sq_overall_mean` | float | `66.1515` | 1459 | 0 | Squad **HomeTeam**: media del rating overall de toda la plantilla. |
| `HomeTeam_sq_overall_max` | float | `79` | 28 | 0 | Squad **HomeTeam**: rating overall máximo de la plantilla. |
| `HomeTeam_sq_top11_mean` | float | `74.1818` | 248 | 0 | Squad **HomeTeam**: media de los 11 ratings más altos (el once titular). |
| `HomeTeam_sq_top15_mean` | float | `73.4` | 320 | 0 | Squad **HomeTeam**: media de los 15 ratings más altos. |
| `HomeTeam_sq_gk_mean` | float | `63` | 193 | 0 | Squad **HomeTeam**: media de ratings de los porteros (GK). |
| `HomeTeam_sq_gk_top_mean` | float | `76` | 33 | 0 | Squad **HomeTeam**: mejor rating entre los porteros (GK) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_gk_top3_mean` | float | `70` | 112 | 0 | Squad **HomeTeam**: media de los 3 mejores ratings entre los porteros (GK) (base titular de la línea). |
| `HomeTeam_sq_def_mean` | float | `67.2727` | 694 | 0 | Squad **HomeTeam**: media de ratings de los defensas (DEF). |
| `HomeTeam_sq_def_top_mean` | float | `75` | 27 | 0 | Squad **HomeTeam**: mejor rating entre los defensas (DEF) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_def_top3_mean` | float | `74.6667` | 75 | 0 | Squad **HomeTeam**: media de los 3 mejores ratings entre los defensas (DEF) (base titular de la línea). |
| `HomeTeam_sq_mid_mean` | float | `65.4167` | 891 | 0 | Squad **HomeTeam**: media de ratings de los centrocampistas (MID). |
| `HomeTeam_sq_mid_top_mean` | float | `79` | 27 | 0 | Squad **HomeTeam**: mejor rating entre los centrocampistas (MID) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_mid_top3_mean` | float | `74.3333` | 78 | 0 | Squad **HomeTeam**: media de los 3 mejores ratings entre los centrocampistas (MID) (base titular de la línea). |
| `HomeTeam_sq_att_mean` | float | `68.6` | 479 | 0 | Squad **HomeTeam**: media de ratings de los delanteros (ATT). |
| `HomeTeam_sq_att_top_mean` | float | `75` | 33 | 0 | Squad **HomeTeam**: mejor rating entre los delanteros (ATT) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_att_top3_mean` | float | `73` | 110 | 0 | Squad **HomeTeam**: media de los 3 mejores ratings entre los delanteros (ATT) (base titular de la línea). |
| `HomeTeam_sq_mapped` | int | `1` | 1 | 0 | Flag: `1` si el slot HomeTeam tiene squad mapeada; `0` si quedó en `NA`. Imputar si 0. |

---

## 2. Dominios (variables categóricas)

| Variable | Tipo | Categorías (con frecuencia sobre 36,150) | Nulos |
|---|---|---|---|
| `Div` | str | `E0=4560; SP1=4560; I1=4560; F1=4237; T1=4006; D1=3672; P1=3672; N1=3598; B1=3285` | 0 |
| `Season` | int (categórica) | 12 temporadas: `1415`–`2526` (una por año, ej. `1415`, `2122`, `2425`) | 0 |
| `FTR` | str | `H=16044; A=11093; D=9013` → H=gana local, D=empate, A=gana visitante (44.4% / 30.7% / 24.9%) | 0 |
| `HTR` | str | `D=14679; H=12355; A=9077` (H/D/A del primer tiempo) | 39 |
| `HomeTeam` / `AwayTeam` | str | 296 equipos distintos (no se listan; ej.: Arsenal, Crystal Palace, Man City…) | 0 |
| `Time` | str | 47 horarios distintos (ej.: 20:00, 14:00…) | 14,920 |
| `Referee` | str | 53 árbitros (ej.: A Taylor, M Oliver, C Pawson…) | 31,590 |
| `Home/AwayTeam_sq_mapped` | int | `1` = squad mapeada (100% de los slots; tras corregir 110 colisiones de mapeo) | 0 |

> **Nota**: el target `FTR` está levemente desbalanceado (H≈44%, D≈25%, A≈31%): no se rebalancea
> (el objetivo es probabilidad **calibrada**), se usa el desbalance como prior.

---

## 3. Archivos derivados de los notebooks

| Archivo | Lo genera | Filas × cols | Contenido |
|---|---|---|---|
| `matches_warmup.csv` | `process_data.py` | 2,918 | Temporada 2013/14; solo alimenta el historial de forma, no se modela |
| `matches_clean.csv` | `eda.ipynb` | 36,150 × 80 | Base limpia sin nulos (159 cuotas: 138 eliminadas, 3 banderas `IW*_Ausente`, 18 imputadas con KNN; `Time`, `Referee`, `HFKC/AFKC` descartadas) |
| `dataset_final.csv` | `feature.ipynb` | 36,119 × 40 | 33 *features* (F2) + `FTR` + metadatos; excluye 31 partidos sin ninguna cuota |
| `dataset_skills.csv` | `feature.ipynb` | 36,119 × 41 | 33 habilidades `SK_*` (local − visitante) + `FTHG`, `FTAG`, `GD` |
| `feature_sets.json` | `feature.ipynb` | — | `F1_sin_nan` (13), `F2_completo` (33), `SK_habilidades` (33), `targets` |
| `backtest_odds.csv` | `feature.ipynb` | 36,119 | Cuotas de apertura (B365, Pinnacle) y cierre (Pinnacle), observadas |
| `mi_ranking.csv`, `mi_skills.csv` | `feature.ipynb` | — | Ranking de información mutua (primer 80% por fecha) |
| `hiperparametros.json`, `hiperparametros_modelos.json` | `mercado.ipynb`, `modelo.ipynb` | — | Hiperparámetros elegidos en la ventana 2016/17–2017/18 |
| `predicciones_oof.csv` | `mercado.ipynb` | 24,183 × 35 | Predicciones fuera de muestra de los 8 *folds* |
| `modelos_oof.csv`, `ablation_results.csv`, `skills_clasificacion.csv`, `skills_regresion.csv`, `seleccion_caracteristicas.csv` | `modelo.ipynb` | — | Resultados de bloques, modelos, habilidades y MI vs PCA |
| `mercado_pruebas.csv`, `backtest_resultados.csv`, `backtest_precio_blando.csv` | `mercado.ipynb` | — | Pruebas de eficiencia del mercado y del *backtest* |

### 3.1 Features del modelo (`dataset_final.csv`)

| Bloque | Variables | F1 | F2 |
|---|---|---|---|
| Mercado | `MKT_logit = ln(pH/pA)`, `MKT_pD`, `MKT_margin` (overround) | sí | sí |
| Plantilla | `HomeTeam_sq_top11_mean`, `DIF_sq_top11` (local − visitante) | sí | sí |
| Forma (goles, puntos, portería a cero) | `H_/A_` `GF_5`, `GA_5`, `Pts_5`, `CS_5` | sí | sí |
| Forma (tiros, tiros a puerta, córners, faltas, tarjetas) | `H_/A_` `SGF_5`, `SGA_5`, `SOTF_5`, `SOTA_5`, `CF_5`, `CA_5`, `FF_5`, `FA_5`, `CardF_5`, `CardA_5` | no | sí (con NaN estructural) |

La forma es la media de los últimos 5 partidos del equipo, **sin incluir el actual**, con calentamiento 2013/14 y relleno con la media de la liga anterior a la fecha. Las estadísticas de partido no se imputan: quedan en NaN donde la fuente no las publica (11.0% de los partidos). `MKT_*` usa solo cuotas de apertura **observadas** de B365, BW, PS y WH (nanmean entre casas; las cuotas de cierre no son *features*).

---
