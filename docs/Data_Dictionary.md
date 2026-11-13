# Diccionario de datos — revisión temporal

La base científica vigente está en el directorio señalado por `scripts/data/revision_actual.txt`. Los nombres de archivos antiguos se conservan solo en el archivo histórico. Cada fila modelable corresponde a un partido; no confundir información cruda disponible después del encuentro con sus variables históricas previas.

## Identidad y particiones

| Campo | Significado | Uso |
|---|---|---|
| `match_id` | Huella estable de liga, temporada, fecha y equipos | Enlace de datos y predicciones |
| `row_id` | Posición cronológica en esta ejecución | Diagnóstico; no usar para unir versiones diferentes |
| `Date` | Fecha del partido | Orden temporal, sin hora de captura de cuotas |
| `Div` | E0, D1, SP1, I1, F1, N1, P1, T1 o B1 | Liga |
| `Season` | Año inicial/final, p. ej. 2324 | Folds y particiones, nunca predictor numérico |
| `HomeTeam`, `AwayTeam` | Identidades canónicas de equipos | Construcción histórica |
| `y` | H=0, D=1, A=2 | Target de clasificación |
| `GD` | Goles locales menos visitantes | Target de regresión |
| `FTR`, `FTHG`, `FTAG`, estadísticas del encuentro | Resultado y observaciones postpartido | Solo targets o actualización de historia posterior; excluidos de todas las listas predictoras |

Los 36.150 partidos originales generan 36.119 comparables; 31 sin ninguna cuota válida alimentan historia pero no la comparación pareada. Train/val: 30.120; test: 5.999. El calentamiento 2013/14 no se modela.

## Historia deportiva: 38 variables

Los prefijos H/A indican el lado del equipo. Todos los partidos de una fecha leen la historia anterior y después actualizan los estados; nunca ven resultados del mismo día.

- `H_*_5`, `A_*_5`: goles a favor/en contra, puntos, portería a cero, tiros, tiros a puerta, córners, faltas y tarjetas en los cinco partidos previos. Una ventana incompleta usa solo un prior de liga anterior; estadísticas sin historia conservan NaN.
- `H_HPPG`, `A_HPPG`, `HPPG_DIF`: puntos históricos suavizados con un prior de liga anterior a la fecha.
- `ELO_DIF`: diferencia de Elo previo por liga, K=20, ventaja local 60 y regresión estacional de 25% hacia 1500.
- `EW_Pts5_DIF`, `EW_Pts10_DIF`, `EW_GD5_DIF`, `EW_GD10_DIF`, `EW_SOTD5_DIF`, `EW_SOTD10_DIF`: diferencias de medias exponenciales con semivida de 5/10 partidos.
- `DRAW_PROP`, `TOTG_PROP`, `XGD`: tendencia histórica al empate, goles totales y contraste simple de ataque/defensa históricos. XGD no es un modelo Dixon–Coles estimado.
- `REST_H/A`, `CONG_H/A`: días de descanso con tope 14 y partidos previos de liga en 14 días. No ven copas ni torneos continentales.
- `NEW_H/A`: ausencia del equipo en la historia observada de la liga en la temporada anterior.
- `SEASON_WEEK`: semana desde el 1 de julio del año de inicio; origen de calendario conocido, sin usar el primer partido futuro de una temporada.

## Plantilla: 6 variables y 33 habilidades

- `SQ_H`, `SQ_DIF`: media de los once mejores ratings del plantel disponible y diferencia entre equipos.
- `XI_H`, `XI_DIF`: media de los titulares asignados por el juego y diferencia. No son alineaciones de cada partido.
- `SQz_H`, `SQz_DIF`: normalización respecto a clubes con snapshots ya disponibles, una vez por club, no ponderada por partidos futuros.
- `SK_*`: 33 diferencias de habilidades; las de campo promedian el top-4 del once del juego, las de portero su portero. Las ausencias se conservan hasta el imputador de entrenamiento.
- `H_snapshot_at`, `A_snapshot_at`: fecha verificable del snapshot utilizado. Debe ser estrictamente anterior al partido; campo de trazabilidad, no predictor.
- `H_snapshot_age`, `A_snapshot_age`: antigüedad en días, para diagnóstico de cobertura, no utilizadas como predictores.

Se utiliza la versión verificable más reciente disponible por club. Los snapshots antiguos se acreditaron por código de actualización SoFIFA y concordancia de ratings con el CSV original de FC24 que conserva fechas. El manifiesto temporal identifica fuente y huellas. En ausencia de una versión actual se usa una anterior disponible o NaN; no se impone cobertura del 100%.

## Mercado

| Campo | Definición |
|---|---|
| `MKT_pH/D/A` | Promedio entre probabilidades normalizadas de casas completas y fiables |
| `MKT_logit` | ln(pH/pA) del consenso |
| `MKT_margin` | Overround medio observado; su estabilidad debe evaluarse |
| `REF_pH/D/A` | Pinnacle cuando es completa y fiable; consenso en otro caso |
| `REF_logit` | ln(pH/pA) de la referencia |
| `MKT_STD` | Dispersión de logits entre casas disponibles |
| `PIN_GAP` | Logit Pinnacle menos el resto; neutro si Pinnacle no es fiable/disponible |
| `pinnacle_fiable` | Tres cuotas válidas y fecha anterior a 23/07/2025 |
| `n_casas` | Número de ternas completas válidas en el consenso |

Cada terna decimal debe ser finita y mayor que uno. Una casa incompleta no se imputa. La normalización proporcional es `(1/o)/sum(1/o)`. Los datos son capturas pre-cierre, no necesariamente el primer precio ni capturas sincronizadas.

## Listas predictoras y resultados

`feature_sets.json` declara historia (38), plantilla (6), habilidades (33), rama sin cuotas (77), referencia de mercado (2), consenso (3) y rama completa con cuotas (84). Las variantes de mercado y ablaciones usan subconjuntos explícitos. Metadatos, targets y cuotas de cierre no entran en ninguna lista.

El pipeline aprende imputación, indicadores, escalado, poda y MI/PCA con el entrenamiento de cada split. `specifications.json`, `parameters_by_fold.json` y los metadatos de cada modelo describen qué se ajustó. Los identificadores de candidato son internos; el informe utiliza nombres legibles.

Las predicciones tienen `match_id`, temporada, target y tres probabilidades en orden H/D/A. Las tablas de métricas identifican partición, modelo y definición del ECE. `acierto_ganador_condicional` elige H frente a A solo entre partidos que realmente no empataron; `accuracy` considera las tres clases en todos los partidos.

`frozen.json` vincula selección a las huellas del protocolo y OOF. `evaluation_complete.json` identifica las predicciones del test. El modelo sin cuotas puro no utiliza pesos ni variables del mercado; las variantes ponderadas según el favorito son experimentos guiados por mercado.
