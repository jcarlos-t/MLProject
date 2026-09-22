# Diccionario de Datos — ML_Project (fútbol 1X2)

Predicción probabilística del resultado de partidos de fútbol (local / empate / visitante).
Fuente: **football-data.co.uk** (partidos y cuotas) + **Kaggle** (ratings de jugadores FIFA/FC).

---

## 0. Conceptos clave del sistema

| Término | Qué es | Ejemplo en datos |
|---------|--------|------------------|
| **Liga (`Div`)** | Campeonato nacional. Los códigos son de football-data: cada uno identifica una liga europea. | `E0` = Premier League, `D1` = Bundesliga, `SP1` = La Liga, `I1` = Serie A, `F1` = Ligue 1, `N1` = Eredivisie, `P1` = Primeira Liga, `T1` = Süper Lig, `B1` = Pro League |
| **Temporada (`Season`)** | Ciclo anual de un campeonato, codificado con el año de inicio y el de fin. | `1617` = temporada 2016/17 … `2526` = 2025/26 |
| **Resultado 1X2 (`FTR`)** | **Target** del modelo: el partido se salda con victoria del local, empate o victoria del visitante. | `H`, `D`, `A` |
| **Equipo local / visitante** | `HomeTeam` juega en su estadio; `AwayTeam` de localía. El local tiene ventaja estadística (~44% de victorias). | `HomeTeam=Man United`, `AwayTeam=Man City` |
| **Cuota decimal** | Multiplicador que paga la casa si aciertas; >1 y menor = más probable. | Bet365 `B365H = 1.16` → local muy favorito |
| **Código de casa** | Prefijo que identifica la casa/referencia en el nombre de la columna (`B365`=Bet365, `PS`=Pinnacle, `Max`=máx. de mercado…). Ver §1.4.1. | `PSCH` → Pinnacle (PS), cierre (C), local (H) |
| **Cuota de apertura vs cierre** | Apertura = inicios de la semana; **cierre** = publicada al cerrar el mercado (considerada la mejor estimación de la 'probabilidad real'). | `B365H` (apertura) vs `B365CH` (cierre) |
| **Overround** | Margen de la casa: `1/H + 1/D + 1/A` siempre >1. Para comparar con el modelo hay que **normalizar** las cuotas (dividir por su suma). | `1/2.4 + 1/3.3 + 1/3.25 ≈ 1.05` |
| **Snapshot de jugadores** | Captura anual de ratings **EA Sports (FIFA/FC)** usada como 'valoración real' de cada jugador esa temporada. | Snapshots `2017.csv` … `2026.csv` |
| **Squad quality** | **Calidad de plantilla** agregada por (temporada, club) a partir de los ratings de los jugadores (medias, máximo y top-3 por línea). Es pre-temporada → usable directa. | `overall_mean`, `top11_mean`, `gk_top3_mean` |
| **Mapeo de clubes** | Traducción del nombre del club EA → nombre canónico de football-data. | `Man United` (football-data) = `Manchester United` (EA) |
| **Team-slot / `_sq_mapped`** | Cada partido tiene 2 'huecos' de equipo (local y visitante). `_sq_mapped=1` si ese hueco se mapeó a un club con squad; `0` si quedó `NA` (ascendidos, filtros de temporada…). | `HomeTeam_sq_mapped = 1` |
| **Estadísticas post-partido** | `HS`, `AS`, `HC`… se conocen **después** del partido. Para modelar solo con información previa hay que desfazarlas (rolling de los últimos N partidos por equipo). | `HS` (tiros del local) se usa media de los últimos 5 |

---

## 1. `scripts/data/matches_sq.csv` — Dataset principal (base unificada)

Generado por `scripts/download/process_data.py`: **join** de `matches.csv` con la
**squad quality** de local y visitante vía `Season|Div|Team` → `club_map.csv`
(un snapshot por temporada, sin mezclar versiones ni lookahead).

- **Una fila = 1 partido**. Cobertura: **9 ligas × 10 temporadas (2016/17–2025/26)**, y = **30,182 partidos**.
- **30,182 filas × 222 columnas** = 186 del partido + 36 de plantilla (17×2 lados + 2 flags).
- Las columnas vienen tipadas (int/float/str). Algunas columnas de partido se registran como
  `float` pero son conteos enteros (p. ej. `HS`); las cuotas son `float` continuas.
- Las features derivadas de plantilla (ventaja neta `DIF_sq_top11`) **no vienen aquí**: se
  construyen en el feature engineering (`scripts/eda/feature.ipynb`).

### 1.1 Identificación del partido

> Cada fila se identifica de forma única por `Div` + `Season` + `Date` + `HomeTeam` + `AwayTeam`.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `Div` | str | `E0` | 9 | 0 | **Liga** (código football-data). 9 ligas europeas. |
| `Season` | int | `1617` | 10 | 0 | **Temporada** como código de 4 dígitos: `1617` = 2016/17 … `2526` = 2025/26. |
| `Date` | str | `13/08/16` | 2152 | 0 | Fecha del partido (dd/mm/aa). |
| `Time` | str | `20:00` | 47 | 8952 | Hora de inicio (solo temporadas recientes). |
| `HomeTeam` | str | `Burnley` | 280 | 0 | Equipo **local** (nombre en inglés). 280 equipos distintos. |
| `AwayTeam` | str | `Swansea` | 280 | 0 | Equipo **visitante**. |
| `Referee` | str | `J Moss` | 50 | 26382 | Árbitro (casi solo 2016/17–2018/19 y 2025/26). |


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
| `HS` | float | `10` | 43 | 1197 | Tiros del local. |
| `AS` | float | `17` | 39 | 1197 | Tiros del visitante. |
| `HST` | float | `3` | 21 | 1197 | Tiros a puerta del local. |
| `AST` | float | `9` | 20 | 1197 | Tiros a puerta del visitante. |
| `HF` | float | `10` | 34 | 1677 | Faltas cometidas por el local. |
| `AF` | float | `14` | 34 | 1677 | Faltas cometidas por el visitante. |
| `HC` | float | `7` | 24 | 1197 | Córners del local. |
| `AC` | float | `4` | 20 | 1197 | Córners del visitante. |
| `HY` | float | `3` | 11 | 1196 | Tarjetas amarillas del local. |
| `AY` | float | `2` | 11 | 1196 | Tarjetas amarillas del visitante. |
| `HR` | float | `0` | 4 | 1196 | Tarjetas rojas del local. |
| `AR` | float | `0` | 4 | 1197 | Tarjetas rojas del visitante. |
| `HFKC` | float | `13` | 26 | 29704 | Free kicks concedidos por el local (solo temporadas recientes). |
| `AFKC` | float | `6` | 29 | 29704 | Free kicks concedidos por el visitante. |


### 1.4 Cuotas de apertura — 1X2

> Cuotas decimales pre-partido. `AvgH/D/A` y `MaxH/D/A` = referencia agregada de todo el mercado; `Bb*` = agregado Betbrain.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `B365H` | float | `2.4` | 172 | 56 | Cuota **Bet365** (apertura). |
| `B365D` | float | `3.3` | 70 | 56 | Idem Bet365. |
| `B365A` | float | `3.25` | 170 | 56 | Idem Bet365. |
| `BWH` | float | `2.45` | 169 | 1260 | Cuota **Bet&Win** (apertura). |
| `BWD` | float | `3.1` | 74 | 1260 | Idem Bet&Win. |
| `BWA` | float | `2.95` | 175 | 1260 | Idem Bet&Win. |
| `IWH` | float | `2.5` | 200 | 7647 | Cuota **Interwetten** (apertura). |
| `IWD` | float | `3.3` | 118 | 7647 | Idem Interwetten. |
| `IWA` | float | `2.65` | 212 | 7647 | Idem Interwetten. |
| `LBH` | float | `2.45` | 120 | 22030 | Cuota **Ladbrokes** (apertura). |
| `LBD` | float | `3.25` | 48 | 22030 | Idem Ladbrokes. |
| `LBA` | float | `3.1` | 120 | 22030 | Idem Ladbrokes. |
| `PSH` | float | `2.47` | 1189 | 1684 | Cuota **Pinnacle** (apertura); mercado muy eficiente. |
| `PSD` | float | `3.32` | 944 | 1684 | Idem Pinnacle. |
| `PSA` | float | `3.19` | 1980 | 1684 | Idem Pinnacle. |
| `WHH` | float | `2.5` | 124 | 4143 | Cuota **William Hill** (apertura). |
| `WHD` | float | `3.2` | 50 | 4143 | Idem WH. |
| `WHA` | float | `2.9` | 122 | 4143 | Idem WH. |
| `VCH` | float | `2.5` | 121 | 6087 | Cuota **VC Bet** (apertura). |
| `VCD` | float | `3.2` | 56 | 6087 | Idem VC Bet. |
| `VCA` | float | `3.25` | 122 | 6087 | Idem VC Bet. |
| `Bb1X2` | float | `55` | 45 | 21230 | Nº de casas usadas en el agregado Betbrain. |
| `BbMxH` | float | `2.55` | 647 | 21230 | Cuota **máxima** Betbrain 1X2 (apertura). |
| `BbAvH` | float | `2.43` | 891 | 21230 | Cuota **promedio** Betbrain 1X2 (apertura). |
| `BbMxD` | float | `3.35` | 603 | 21230 | Máximo Betbrain 1X2. |
| `BbAvD` | float | `3.21` | 676 | 21230 | Promedio Betbrain 1X2. |
| `BbMxA` | float | `3.3` | 970 | 21230 | Máximo Betbrain 1X2. |
| `BbAvA` | float | `3.1` | 1460 | 21230 | Promedio Betbrain 1X2. |
| `MaxH` | float | `1.16` | 689 | 8982 | **Máximo de mercado** (apertura) para local. |
| `MaxD` | float | `10` | 557 | 8982 | Máximo de mercado (empate). |
| `MaxA` | float | `23` | 989 | 8982 | Máximo de mercado (visitante). |
| `AvgH` | float | `1.14` | 1045 | 8982 | **Promedio de mercado** (apertura) para local. |
| `AvgD` | float | `8.75` | 785 | 8982 | Promedio de mercado (empate). |
| `AvgA` | float | `19.83` | 1662 | 8982 | Promedio de mercado (visitante). |

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
| `B365>2.5` | float | `1.4` | 101 | 9007 | Over/Under 2.5 — cuota Bet365 a **más** de 2.5 goles. |
| `B365<2.5` | float | `3` | 94 | 9007 | Idem Bet365 a menos. |
| `P>2.5` | float | `1.4` | 225 | 10644 | Pinnacle Over/Under 2.5 (>2.5). |
| `P<2.5` | float | `3.11` | 341 | 10644 | Pinnacle Over/Under 2.5 (<2.5). |
| `Max>2.5` | float | `1.45` | 210 | 8983 | Máximo de mercado Over 2.5. |
| `Max<2.5` | float | `3.11` | 301 | 8983 | Máximo de mercado Under 2.5. |
| `Avg>2.5` | float | `1.41` | 213 | 8983 | Promedio de mercado Over 2.5. |
| `Avg<2.5` | float | `2.92` | 352 | 8983 | Promedio de mercado Under 2.5. |
| `BbOU` | float | `40` | 38 | 21230 | Nº de casas usadas en el agregado Over/Under. |
| `BbMx>2.5` | float | `2.4` | 199 | 21230 | Máximo Betbrain Over 2.5. |
| `BbAv>2.5` | float | `2.3` | 180 | 21230 | Promedio Betbrain Over 2.5. |
| `BbMx<2.5` | float | `1.68` | 250 | 21230 | Máximo Betbrain Under 2.5. |
| `BbAv<2.5` | float | `1.61` | 317 | 21230 | Promedio Betbrain Under 2.5. |
| `AHh` | float | `-2.25` | 28 | 8990 | **Línea** del Asian Handicap de apertura (ej. `-2.25` = local con -2.25 goles). |
| `B365AHH` | float | `1.96` | 57 | 9049 | Asian Handicap Bet365 — lado local. |
| `B365AHA` | float | `1.94` | 59 | 9049 | Asian Handicap Bet365 — lado visitante. |
| `PAHH` | float | `1.97` | 62 | 10560 | Asian Handicap Pinnacle — local. |
| `PAHA` | float | `1.95` | 58 | 10560 | Asian Handicap Pinnacle — visitante. |
| `MaxAHH` | float | `1.97` | 65 | 8984 | Asian máximo de mercado — local. |
| `MaxAHA` | float | `2` | 68 | 8984 | Asian máximo de mercado — visitante. |
| `AvgAHH` | float | `1.94` | 59 | 8983 | Asian promedio de mercado — local. |
| `AvgAHA` | float | `1.94` | 98 | 8983 | Asian promedio de mercado — visitante. |
| `BbAH` | float | `32` | 33 | 21230 | Nº de casas del agregado Asian. |
| `BbAHh` | float | `-0.25` | 26 | 21230 | Línea Asian del agregado Betbrain. |
| `BbMxAHH` | float | `2.13` | 134 | 21230 | Máximo Betbrain Asian — local. |
| `BbAvAHH` | float | `2.06` | 134 | 21230 | Promedio Betbrain Asian — local. |
| `BbMxAHA` | float | `1.86` | 128 | 21230 | Máximo Betbrain Asian — visitante. |
| `BbAvAHA` | float | `1.81` | 126 | 21230 | Promedio Betbrain Asian — visitante. |


### 1.6 Betfair (Exchange) — apertura

> Exchange (apuestas entre usuarios). Disponible aprox. desde 2020/21. `1XB*` = Betfair Match Odds; `BFE*` = datapoint.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `BFH` | float | `1.6` | 99 | 27168 | **Betfair Exchange** 1X2 (apertura). |
| `BFD` | float | `4.33` | 44 | 27168 | Idem Betfair. |
| `BFA` | float | `5` | 101 | 27168 | Idem Betfair. |
| `1XBH` | float | `1.68` | 521 | 27245 | **Betfair Match Odds** 1X2 (apertura). |
| `1XBD` | float | `4.32` | 407 | 27245 | Idem. |
| `1XBA` | float | `5.03` | 636 | 27245 | Idem. |
| `BFEH` | float | `1.66` | 228 | 24379 | Betfair Exchange 1X2 (datapoint apertura). |
| `BFED` | float | `4.5` | 136 | 24379 | Idem. |
| `BFEA` | float | `5.6` | 240 | 24379 | Idem. |
| `BFE>2.5` | float | `1.59` | 154 | 24607 | Betfair Exchange Over 2.5 (apertura). |
| `BFE<2.5` | float | `2.64` | 174 | 24607 | Betfair Exchange Under 2.5. |
| `BFEAHH` | float | `2.1` | 81 | 25408 | Betfair Exchange Asian — local (apertura). |
| `BFEAHA` | float | `1.88` | 77 | 25408 | Betfair Exchange Asian — visitante. |


### 1.7 Cuotas de cierre — 1X2

> Publicadas al **cerrar** el mercado; se consideran la mejor estimación de la 'probabilidad real'. `AvgC*` disponible desde 2019/20; Pinnacle/Bet365 desde antes.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `PSCH` | float | `2.79` | 1219 | 1600 | **Pinnacle cierre** 1X2 — mejor estimación de 'probabilidad real' del mercado. |
| `PSCD` | float | `3.16` | 958 | 1600 | Idem Pinnacle cierre. |
| `PSCA` | float | `2.89` | 1973 | 1600 | Idem Pinnacle cierre. |
| `B365CH` | str | `1.14` | 155 | 8982 | **Bet365 cierre** 1X2. |
| `B365CD` | float | `9.5` | 59 | 8982 | Idem. |
| `B365CA` | float | `21` | 154 | 8982 | Idem. |
| `BWCH` | float | `1.14` | 171 | 10209 | Bet&Win cierre. |
| `BWCD` | float | `9` | 76 | 10209 | Idem. |
| `BWCA` | float | `20` | 177 | 10209 | Idem. |
| `IWCH` | float | `1.15` | 186 | 16609 | Interwetten cierre. |
| `IWCD` | float | `8` | 113 | 16609 | Idem. |
| `IWCA` | float | `18` | 198 | 16609 | Idem. |
| `WHCH` | float | `1.11` | 124 | 13081 | William Hill cierre. |
| `WHCD` | float | `9.5` | 53 | 13081 | Idem. |
| `WHCA` | float | `21` | 124 | 13081 | Idem. |
| `VCCH` | float | `1.14` | 118 | 15000 | VC Bet cierre. |
| `VCCD` | float | `9.5` | 57 | 15000 | Idem. |
| `VCCA` | float | `23` | 122 | 15000 | Idem. |
| `MaxCH` | float | `1.16` | 890 | 8981 | **Máximo de mercado cierre** 1X2. |
| `MaxCD` | float | `10.5` | 705 | 8981 | Idem empate. |
| `MaxCA` | float | `23` | 1175 | 8981 | Idem visitante. |
| `AvgCH` | float | `1.14` | 1070 | 8981 | **Promedio de mercado cierre** 1X2. |
| `AvgCD` | float | `9.52` | 827 | 8981 | Idem empate. |
| `AvgCA` | float | `19.18` | 1682 | 8981 | Idem visitante. |


### 1.8 Cuotas de cierre — mercados secundarios

> Over/Under y Asian Handicap de cierre.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `B365C>2.5` | float | `1.3` | 106 | 8983 | Bet365 **cierre** Over 2.5. |
| `B365C<2.5` | float | `3.5` | 104 | 8983 | Bet365 cierre Under 2.5. |
| `PC>2.5` | float | `1.34` | 251 | 10611 | Pinnacle cierre Over 2.5. |
| `PC<2.5` | float | `3.44` | 377 | 10611 | Pinnacle cierre Under 2.5. |
| `MaxC>2.5` | float | `1.36` | 241 | 8981 | Máximo mercado cierre Over 2.5. |
| `MaxC<2.5` | float | `3.76` | 351 | 8981 | Máximo mercado cierre Under 2.5. |
| `AvgC>2.5` | float | `1.32` | 239 | 8981 | Promedio mercado cierre Over 2.5. |
| `AvgC<2.5` | float | `3.43` | 391 | 8981 | Promedio mercado cierre Under 2.5. |
| `AHCh` | float | `-2.25` | 30 | 8982 | **Línea Asian cierre**. |
| `B365CAHH` | float | `1.91` | 62 | 8986 | Asian cierre Bet365 — local. |
| `B365CAHA` | float | `1.99` | 61 | 8986 | Asian cierre Bet365 — visitante. |
| `PCAHH` | float | `1.94` | 89 | 10541 | Asian cierre Pinnacle — local. |
| `PCAHA` | float | `1.98` | 88 | 10541 | Asian cierre Pinnacle — visitante. |
| `MaxCAHH` | float | `1.99` | 82 | 8982 | Asian cierre máximo mercado — local. |
| `MaxCAHA` | float | `2.07` | 86 | 8982 | Idem visitante. |
| `AvgCAHH` | float | `1.9` | 64 | 8981 | Asian cierre promedio mercado — local. |
| `AvgCAHA` | float | `1.99` | 68 | 8981 | Idem visitante. |


### 1.9 Betfair (Exchange) — cierre

> Exchange de cierre: `BFCH/D/A`, `1XBCH/D/A`, `BFECH/D/A` (datapoint) y sus Over/Under/Asian.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `BFCH` | float | `1.62` | 101 | 27164 | **Betfair Exchange cierre** 1X2. |
| `BFCD` | float | `4` | 46 | 27164 | Idem. |
| `BFCA` | float | `5` | 102 | 27164 | Idem. |
| `1XBCH` | float | `1.66` | 575 | 27164 | Betfair Match Odds cierre. |
| `1XBCD` | float | `4.15` | 423 | 27164 | Idem. |
| `1XBCA` | float | `5.33` | 737 | 27164 | Idem. |
| `BFECH` | float | `1.72` | 229 | 24374 | Betfair Exchange (datapoint cierre) 1X2. |
| `BFECD` | float | `4.2` | 95 | 24374 | Idem. |
| `BFECA` | float | `5.4` | 236 | 24374 | Idem. |
| `BFEC>2.5` | float | `1.68` | 156 | 24375 | Betfair Exchange cierre Over 2.5. |
| `BFEC<2.5` | float | `2.46` | 164 | 24375 | Idem Under. |
| `BFECAHH` | float | `1.9` | 62 | 24378 | Betfair Exchange cierre Asian — local. |
| `BFECAHA` | float | `2.08` | 62 | 24378 | Idem visitante. |


### 1.10 Cuotas de casas recientes (2024/25–2025/26)

> Betfair datapoint, BetMGM, BetVictor, Colossus y Ladbrokes-cierre: solo temporadas 2425–2526 (muchos nulos antes).

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `BFDH` | float | `1.3` | 76 | 27207 | Betfair datapoint (apertura). |
| `BFDD` | float | `6` | 31 | 27207 | Idem. |
| `BFDA` | float | `9.5` | 77 | 27206 | Idem. |
| `BMGMH` | float | `1.29` | 195 | 27218 | **BetMGM** (apertura). |
| `BMGMD` | float | `6.5` | 67 | 27218 | Idem. |
| `BMGMA` | float | `9` | 192 | 27218 | Idem. |
| `BVH` | float | `1.3` | 103 | 27293 | **BetVictor** (apertura). |
| `BVD` | float | `6` | 43 | 27293 | Idem. |
| `BVA` | float | `8.5` | 102 | 27293 | Idem. |
| `CLH` | float | `1.33` | 91 | 28026 | **Colossus** (apertura). |
| `CLD` | float | `5.75` | 38 | 28026 | Idem. |
| `CLA` | float | `8` | 88 | 28026 | Idem. |
| `BFDCH` | float | `1.3` | 78 | 27251 | Betfair datapoint **cierre**. |
| `BFDCD` | float | `6` | 32 | 27251 | Idem. |
| `BFDCA` | float | `9.5` | 81 | 27251 | Idem. |
| `BMGMCH` | float | `1.3` | 195 | 27202 | BetMGM cierre. |
| `BMGMCD` | float | `6.25` | 69 | 27202 | Idem. |
| `BMGMCA` | float | `9` | 193 | 27202 | Idem. |
| `BVCH` | float | `1.29` | 102 | 27261 | BetVictor cierre. |
| `BVCD` | float | `6` | 46 | 27261 | Idem. |
| `BVCA` | float | `9` | 103 | 27261 | Idem. |
| `CLCH` | float | `1.3` | 91 | 28096 | Colossus cierre. |
| `CLCD` | float | `6` | 41 | 28096 | Idem. |
| `CLCA` | float | `8` | 89 | 28096 | Idem. |
| `LBCH` | float | `1.3` | 90 | 27960 | Ladbrokes cierre. |
| `LBCD` | float | `5.75` | 41 | 27960 | Idem. |
| `LBCA` | float | `8` | 89 | 27960 | Idem. |


### 1.11 Columnas basura

> 100% vacías (artefactos de la concatenación de crudos). Descartar antes de modelar.

| Campo | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `Unnamed: 105` | float | `NA` | 0 | 30182 | **Basura**: 100% vacía (artefacto de concatenación). Descartar. |
| `Unnamed: 119` | float | `NA` | 0 | 30182 | Basura: 100% vacía. Descartar. |
| `Unnamed: 120` | float | `NA` | 0 | 30182 | Basura: 100% vacía. Descartar. |


### 1.12 Squad quality — local y visitante

18 columnas para **local** (`HomeTeam_sq_*`) y las mismas 18 para **visitante** (`AwayTeam_sq_*`), miradas desde el lado del equipo en cuestión. Features **crudas** de plantilla (un snapshot por temporada, pre-partido → usables directas).

| Campo (base `HomeTeam_sq_*` / `AwayTeam_sq_*`) | Tipo | Ejemplo real | Únicos | Nulos | Descripción |
|---|---|---|---|---|---|
| `HomeTeam_sq_n_players` | float | `33` | 41 | 873 | Squad **HomeTeam**: nº de jugadores del club en el snapshot de la temporada. |
| `HomeTeam_sq_overall_mean` | float | `66.1515` | 1242 | 873 | Squad **HomeTeam**: media del rating overall de toda la plantilla. |
| `HomeTeam_sq_overall_max` | float | `79` | 32 | 873 | Squad **HomeTeam**: rating overall máximo de la plantilla. |
| `HomeTeam_sq_top11_mean` | float | `74.1818` | 244 | 873 | Squad **HomeTeam**: media de los 11 ratings más altos (el once titular). |
| `HomeTeam_sq_top15_mean` | float | `73.4` | 316 | 873 | Squad **HomeTeam**: media de los 15 ratings más altos. |
| `HomeTeam_sq_gk_mean` | float | `63` | 193 | 873 | Squad **HomeTeam**: media de ratings de los porteros (GK). |
| `HomeTeam_sq_gk_top_mean` | float | `76` | 33 | 873 | Squad **HomeTeam**: mejor rating entre los porteros (GK) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_gk_top3_mean` | float | `70` | 104 | 873 | Squad **HomeTeam**: media de los 3 mejores ratings entre los porteros (GK) (base titular de la línea). |
| `HomeTeam_sq_def_mean` | float | `67.2727` | 645 | 873 | Squad **HomeTeam**: media de ratings de los defensas (DEF). |
| `HomeTeam_sq_def_top_mean` | float | `75` | 30 | 873 | Squad **HomeTeam**: mejor rating entre los defensas (DEF) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_def_top3_mean` | float | `74.6667` | 78 | 873 | Squad **HomeTeam**: media de los 3 mejores ratings entre los defensas (DEF) (base titular de la línea). |
| `HomeTeam_sq_mid_mean` | float | `65.4167` | 807 | 873 | Squad **HomeTeam**: media de ratings de los centrocampistas (MID). |
| `HomeTeam_sq_mid_top_mean` | float | `79` | 29 | 873 | Squad **HomeTeam**: mejor rating entre los centrocampistas (MID) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_mid_top3_mean` | float | `74.3333` | 82 | 873 | Squad **HomeTeam**: media de los 3 mejores ratings entre los centrocampistas (MID) (base titular de la línea). |
| `HomeTeam_sq_att_mean` | float | `68.6` | 455 | 891 | Squad **HomeTeam**: media de ratings de los delanteros (ATT). |
| `HomeTeam_sq_att_top_mean` | float | `75` | 34 | 891 | Squad **HomeTeam**: mejor rating entre los delanteros (ATT) (el **máximo**, pese al nombre `top`). |
| `HomeTeam_sq_att_top3_mean` | float | `73` | 107 | 891 | Squad **HomeTeam**: media de los 3 mejores ratings entre los delanteros (ATT) (base titular de la línea). |
| `HomeTeam_sq_mapped` | int | `1` | 2 | 0 | Flag: `1` si el slot HomeTeam tiene squad mapeada; `0` si quedó en `NA`. Imputar si 0. |

---

## 2. Dominios (variables categóricas)

| Variable | Tipo | Categorías (con frecuencia sobre 30,182) | Nulos |
|---|---|---|---|
| `Div` | str | `E0=3800; SP1=3800; I1=3800; F1=3477; T1=3394; D1=3060; P1=3060; N1=2986; B1=2805` | 0 |
| `Season` | int (categórica) | 10 temporadas: `1617`–`2526` (una por año, ej. `1617`, `2122`, `2425`) | 0 |
| `FTR` | str | `H=13371; A=9334; D=7477` → H=gana local, D=empate, A=gana visitante | 0 |
| `HTR` | str | `D=12140; H=10314; A=7689` (H/D/A del primer tiempo) | 39 |
| `HomeTeam` / `AwayTeam` | str | 280 equipos distintos (no se listan; ej.: Crystal Palace, Everton, Man City…) | 0 |
| `Time` | str | 47 horarios distintos (ej.: 20:00, 14:00…) | 8,952 |
| `Referee` | str | 50 árbitros (ej.: A Taylor, M Oliver, C Pawson…) | 26,382 |
| `Home/AwayTeam_sq_mapped` | int | `1` = squad mapeada, `0` = queda `NA` | 0 |

> **Nota**: el target `FTR` está levemente desbalanceado (H≈44%, D≈25%, A≈31%): no se rebalancea
> (el objetivo es probabilidad **calibrada**), se usa el desbalance como prior.

---