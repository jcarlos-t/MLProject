---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Calibri', Arial, sans-serif;
    background: #FFFFFF;
    color: #2D2D2D;
    position: relative;
    padding: 0;
  }
  section.dark {
    background: #1F4E79;
    color: #FFFFFF;
  }
  section table { position: absolute; }
---

<!-- _class: dark -->
<!-- _paginate: false -->
<img src="images/icons/motivation_white.png" style="position:absolute; left:86.00%; top:8.89%; width:7.00%; height:12.44%; " />
<div style="position:absolute; left:7.00%; top:16.00%; width:86.00%; height:37.33%; color:#FFFFFF; font-weight:bold; font-size:29px; line-height:1.3;">¿Se pueden estimar probabilidades 1X2 bien calibradas para un mercado de apuestas de fútbol altamente eficiente?</div>
<div style="position:absolute; left:7.00%; top:56.00%; width:86.00%; height:7.11%; color:#A0BBDD; font-size:16px;">Etapa 1 (P1) — Problema, Dataset y Análisis Exploratorio de Datos</div>
<div style="position:absolute; left:7.00%; top:64.89%; width:20.00%; height:0.62%; background:#2E75B6;"></div>
<div style="position:absolute; left:7.00%; top:67.56%; width:86.00%; height:16.00%; color:#CADCFC; font-size:14px; line-height:1.5;">José Luis Alva Espinoza &nbsp;·&nbsp; Juan Carlos Ticlia Maqui &nbsp;·&nbsp; Elmer José Villegas Suárez &nbsp;·&nbsp; Joseph Anderson Cose Rojas<br/>Departamento de Computación — Universidad de Ingeniería y Tecnología (UTEC), Lima, Perú</div>
<div style="position:absolute; left:7.00%; top:87.11%; width:86.00%; height:6.22%; color:#A0BBDD; font-size:13px;">Septiembre 2026</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Un mercado de apuestas eficiente solo se puede superar con probabilidades calibradas, no con más aciertos</div>
<img src="images/icons/motivation_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:24.89%; width:53.00%; height:58.67%; color:#2D2D2D; font-size:18px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:16px;">Fútbol: pocos goles, azar alto, fuerte efecto de contexto.</li><li style="margin-bottom:16px;">Techo empírico: ~55% de acierto en la clase ganadora.</li><li style="margin-bottom:16px;">Mercado de apuestas altamente eficiente — las cuotas ya condensan la información pública.</li><li style="margin-bottom:16px;">ML solo aporta valor con probabilidades bien calibradas.</li></ul></div>
<div style="position:absolute; left:60.00%; top:24.89%; width:33.00%; height:44.44%; border:0.75px solid #CCCCCC; overflow:hidden;"><img src="images/photos/motivation_photo.jpg" style="width:100%; height:100%; object-fit:cover; display:block;" /></div>
<div style="position:absolute; left:60.00%; top:70.04%; width:33.00%; height:4.44%; color:#777777; font-size:10px; font-style:italic;">Estadio de fútbol — Wikimedia Commons (CC0)</div>
<div style="position:absolute; left:5.00%; top:92.44%; width:90.00%; height:5.33%; color:#777777; font-size:12px; ">Tammouch, Elouafi &amp; Essadik (2024); Knoll &amp; Stübinger (2020)</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">El objetivo: predecir 1X2 con probabilidades calibradas usando solo datos pre-partido</div>
<img src="images/icons/question_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:11.00%; top:24.00%; width:78.00%; height:22.22%; background:#EBF3FA; border:1.2px solid #2E75B6; border-radius:8px; display:flex; align-items:center; justify-content:center; text-align:center; color:#1F4E79; font-size:16px; padding:0 20px;">Dado un partido con local H y visitante A, estimar P(FTR = c | x) para c ∈ {H, D, A},<br/>donde x agrupa forma reciente, calidad de plantilla, localía y cuotas pre-partido.</div>
<div style="position:absolute; left:5.00%; top:50.67%; width:90.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Hipótesis de trabajo</div>
<div style="position:absolute; left:5.00%; top:57.78%; width:90.00%; height:33.78%; color:#2D2D2D; font-size:16px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">Las cuotas de mercado dominan sobre las features estadísticas individuales.</li><li style="margin-bottom:8px;">La calidad de plantilla aporta señal complementaria, sobre todo al inicio de temporada.</li><li style="margin-bottom:8px;">Un modelo mal calibrado puede acertar más pero perder dinero en el backtest.</li></ul></div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">La tarea es clasificación multiclase probabilística; log loss y Brier score priman sobre accuracy</div>
<img src="images/icons/metrics_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:23.11%; width:43.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Tarea de ML</div>
<div style="position:absolute; left:5.00%; top:30.22%; width:43.00%; height:30.22%; color:#2D2D2D; font-size:16px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">Multiclase (3 clases), salida probabilística.</li><li style="margin-bottom:8px;">Métrica primaria: log loss.</li><li style="margin-bottom:8px;">Complementarias: accuracy, F1, Brier, reliability diagrams.</li></ul></div>
<div style="position:absolute; left:53.00%; top:23.11%; width:42.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Cuota → probabilidad</div>
<div style="position:absolute; left:53.00%; top:30.22%; width:42.00%; height:26.67%; color:#2D2D2D; font-size:16px; line-height:1.6;">p_c = (1/o_c) / Σ_j (1/o_j)<br/><br/>Value bet si p̂ · o &gt; 1.</div>
<div style="position:absolute; left:53.00%; top:58.67%; width:42.00%; height:12.44%; color:#777777; font-size:12px; font-style:italic;">El edge puede aparecer en local, empate o visitante — no solo en el favorito.</div>
<div style="position:absolute; left:5.00%; top:63.11%; width:36.00%; height:25.78%; border:0.75px solid #CCCCCC; overflow:hidden;"><img src="images/photos/metrics_photo.jpg" style="width:100%; height:100%; object-fit:cover; display:block;" /></div>
<div style="position:absolute; left:5.00%; top:89.60%; width:36.00%; height:4.44%; color:#777777; font-size:10px; font-style:italic;">Dardos — kallerna, Wikimedia Commons (CC BY-SA 3.0)</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Construimos un dataset propio de 30,182 partidos combinando cuotas de mercado y calidad de plantilla</div>
<img src="images/icons/dataset_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:23.11%; width:43.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Fuentes</div>
<div style="position:absolute; left:5.00%; top:30.22%; width:43.00%; height:26.67%; color:#2D2D2D; font-size:16px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;"><b>football-data.co.uk: </b>90 CSVs, 9 ligas, 10 temporadas.</li><li style="margin-bottom:8px;"><b>Kaggle (FIFA/FC): </b>10 snapshots de ratings, proxy de plantilla.</li></ul></div>
<div style="position:absolute; left:5.00%; top:57.78%; width:43.00%; height:31.11%; border:0.75px solid #CCCCCC; overflow:hidden;"><img src="images/photos/dataset_photo.jpg" style="width:100%; height:100%; object-fit:cover; display:block;" /></div>
<div style="position:absolute; left:5.00%; top:89.60%; width:43.00%; height:4.44%; color:#777777; font-size:10px; font-style:italic;">Centro de datos — Wikimedia Commons (CC0)</div>
<div style="position:absolute; left:53.00%; top:23.11%; width:42.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Dataset consolidado</div>
<table style="position:absolute; left:53.00%; top:30.22%; width:42.00%; height:46.22%; border-collapse:collapse; font-size:13px;"><tr><td style="padding:5px 8px; border:0.5px solid #CCCCCC;">Partidos</td><td style="padding:5px 8px; border:0.5px solid #CCCCCC; text-align:right;">30,182</td></tr><tr><td style="padding:5px 8px; border:0.5px solid #CCCCCC;">Columnas útiles</td><td style="padding:5px 8px; border:0.5px solid #CCCCCC; text-align:right;">219 (de 222)</td></tr><tr><td style="padding:5px 8px; border:0.5px solid #CCCCCC;">Ligas / Temporadas</td><td style="padding:5px 8px; border:0.5px solid #CCCCCC; text-align:right;">9 / 10</td></tr><tr><td style="padding:5px 8px; border:0.5px solid #CCCCCC;">Equipos</td><td style="padding:5px 8px; border:0.5px solid #CCCCCC; text-align:right;">280</td></tr><tr><td style="padding:5px 8px; border:0.5px solid #CCCCCC;">Cobertura squad quality</td><td style="padding:5px 8px; border:0.5px solid #CCCCCC; text-align:right;">97.1%</td></tr><tr><td style="padding:5px 8px; border:0.5px solid #CCCCCC;">Nulos tras limpieza</td><td style="padding:5px 8px; border:0.5px solid #CCCCCC; text-align:right;">0</td></tr></table>
<div style="position:absolute; left:53.00%; top:79.11%; width:42.00%; height:10.67%; color:#777777; font-size:12px; ">Clave compuesta por partido: (Div, Season, Date, HomeTeam, AwayTeam) — sin duplicados.</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">FTR está levemente desbalanceada (44% local) y se usará como prior, no se rebalanceará</div>
<img src="images/icons/target_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<img src="images/charts/target_distribution.png" style="position:absolute; left:5.00%; top:23.11%; width:55.00%; height:64.00%; object-fit:contain;" />
<div style="position:absolute; left:62.00%; top:23.11%; width:33.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Lectura</div>
<div style="position:absolute; left:62.00%; top:30.22%; width:33.00%; height:51.56%; color:#2D2D2D; font-size:15px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:10px;">30,182 partidos totales.</li><li style="margin-bottom:10px;">Ventaja de localía: 1.56 vs. 1.25 goles promedio (local vs. visitante).</li><li style="margin-bottom:10px;">Desbalance estructural y leve → se usa como prior informativo.</li></ul></div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Las cuotas medias de mercado son la señal predictiva más fuerte antes del partido</div>
<img src="images/icons/odds_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<img src="images/eda/eda_cuotas_vs_ftr.png" style="position:absolute; left:5.00%; top:23.11%; width:55.00%; height:64.00%; object-fit:contain;" />
<div style="position:absolute; left:5.00%; top:83.20%; width:55.00%; height:6.22%; background:#FFF2CC; border:1px solid #E6C800; border-radius:6px; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:13px; color:#7A5200; text-align:center;">AvgH baja → más victorias locales</div>
<div style="position:absolute; left:62.00%; top:23.11%; width:33.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Qué observar</div>
<div style="position:absolute; left:62.00%; top:30.22%; width:33.00%; height:53.33%; color:#2D2D2D; font-size:14px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">AvgH/AvgD/AvgA equivalen casi al cierre de Pinnacle (acierto del favorito: 53.9% vs. 54.5%).</li><li style="margin-bottom:8px;">Información mutua con FTR ≈ 0.11 en clases H y A.</li><li style="margin-bottom:8px;">Se usan como baseline del mercado (menos ruido al promediar casas).</li></ul></div>
<div style="position:absolute; left:5.00%; top:92.44%; width:90.00%; height:5.33%; color:#777777; font-size:12px; ">Fig.: cuotas medias 1X2 según resultado real FTR (EDA propio).</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">La calidad de plantilla aporta señal complementaria, con menor magnitud que las cuotas</div>
<img src="images/icons/squad_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<img src="images/eda/eda_squad_vs_ftr.png" style="position:absolute; left:5.00%; top:23.11%; width:55.00%; height:64.00%; object-fit:contain;" />
<div style="position:absolute; left:62.00%; top:23.11%; width:33.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Qué observar</div>
<div style="position:absolute; left:62.00%; top:30.22%; width:33.00%; height:53.33%; color:#2D2D2D; font-size:15px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:10px;">Mayor diferencia de top11 (local − visitante) → mayor proporción de victorias locales.</li><li style="margin-bottom:10px;">Magnitud menor que el efecto de las cuotas.</li><li style="margin-bottom:10px;">Más útil en jornadas tempranas, cuando la forma reciente es escasa.</li></ul></div>
<div style="position:absolute; left:5.00%; top:92.44%; width:90.00%; height:5.33%; color:#777777; font-size:12px; ">Fig.: diferencia de calidad del once titular según FTR (EDA propio).</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Las cuotas del mismo mercado son casi colineales (r &gt; 0.9): hay redundancia, no señal nueva</div>
<img src="images/icons/correlation_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<img src="images/eda/eda_corr_detalle.png" style="position:absolute; left:5.00%; top:23.11%; width:55.00%; height:64.00%; object-fit:contain;" />
<div style="position:absolute; left:62.00%; top:23.11%; width:33.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Qué observar</div>
<div style="position:absolute; left:62.00%; top:30.22%; width:33.00%; height:53.33%; color:#2D2D2D; font-size:14px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">Multicolinealidad confirmada entre casas de apuestas del mismo mercado.</li><li style="margin-bottom:8px;">Justifica imputación KNN y quedarse con una única referencia (Avg) + banderas de ausencia.</li><li style="margin-bottom:8px;">Goles y tiros correlacionan r ≈ 0.6–0.7; squad correlaciona poco con cuotas → señal complementaria.</li></ul></div>
<div style="position:absolute; left:5.00%; top:92.44%; width:90.00%; height:5.33%; color:#777777; font-size:12px; ">Fig.: detalle de la matriz de correlación de Pearson (EDA propio).</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Una limpieza explícita de 156 columnas de cuotas deja el dataset sin valores nulos</div>
<img src="images/icons/cleaning_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<img src="images/charts/missing_values.png" style="position:absolute; left:5.00%; top:23.11%; width:55.00%; height:62.22%; object-fit:contain;" />
<div style="position:absolute; left:62.00%; top:23.11%; width:33.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Otras variables</div>
<div style="position:absolute; left:62.00%; top:30.22%; width:33.00%; height:53.33%; color:#2D2D2D; font-size:14px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">Unnamed 105/119/120 (100%) → artefactos, eliminadas.</li><li style="margin-bottom:8px;">Referee (87.4%) eliminada; Time (29.7%) → Time_hora.</li><li style="margin-bottom:8px;">HTR (39 nulos) reconstruida desde HTHG/HTAG.</li><li style="margin-bottom:8px;">Sin duplicados: clave compuesta única por partido.</li></ul></div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Los outliers en goles, tiros y cuotas son partidos reales y se conservan</div>
<img src="images/icons/outliers_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<img src="images/eda/eda_boxplot_repr.png" style="position:absolute; left:5.00%; top:23.11%; width:55.00%; height:64.00%; object-fit:contain;" />
<div style="position:absolute; left:62.00%; top:23.11%; width:33.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Regla IQR</div>
<div style="position:absolute; left:62.00%; top:30.22%; width:33.00%; height:53.33%; color:#2D2D2D; font-size:14px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">Atípico si está fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR].</li><li style="margin-bottom:8px;">Goles hasta 10–13, tiros hasta 46 → marcadores reales, no errores.</li><li style="margin-bottom:8px;">Cuotas AvgA hasta 41.22 → partidos muy desequilibrados, informativos.</li><li style="margin-bottom:8px;">Decisión: no se eliminan.</li></ul></div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">El mayor riesgo metodológico es la fuga de información temporal (data leakage)</div>
<img src="images/icons/leakage_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:24.89%; width:56.00%; height:58.67%; color:#2D2D2D; font-size:18px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:16px;"><b>Post-partido: </b>goles, tiros, tarjetas → deben pasar a medias rolling de los últimos N partidos.</li><li style="margin-bottom:16px;"><b>Look-ahead: </b>toda agregación usa solo el pasado → split train/test cronológico, nunca aleatorio.</li><li style="margin-bottom:16px;"><b>Plantilla por temporada: </b>un snapshot de ratings no predice una temporada anterior.</li></ul></div>
<div style="position:absolute; left:63.00%; top:24.89%; width:32.00%; height:45.33%; border:0.75px solid #CCCCCC; overflow:hidden;"><img src="images/photos/leakage_photo.jpg" style="width:100%; height:100%; object-fit:cover; display:block;" /></div>
<div style="position:absolute; left:63.00%; top:70.93%; width:32.00%; height:4.44%; color:#777777; font-size:10px; font-style:italic;">Relojes de arena — Aaaatu, Wikimedia Commons (CC BY-SA 4.0)</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">La cobertura por liga y la exclusión de casas de apuestas también pueden introducir sesgo</div>
<img src="images/icons/bias_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:23.11%; width:43.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Sesgo de cobertura</div>
<div style="position:absolute; left:5.00%; top:30.22%; width:43.00%; height:35.56%; color:#2D2D2D; font-size:16px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">E0: 3,800 partidos; B1: 2,805.</li><li style="margin-bottom:8px;">Ligas con menos historia, sub-representadas.</li><li style="margin-bottom:8px;">Mitigación: validación por liga y temporada.</li></ul></div>
<div style="position:absolute; left:53.00%; top:23.11%; width:42.00%; height:6.22%; color:#2E75B6; font-weight:bold; font-size:20px;">Sesgo de cuotas</div>
<div style="position:absolute; left:53.00%; top:30.22%; width:42.00%; height:35.56%; color:#2D2D2D; font-size:16px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">Excluir casas con nulos puede sesgar si la ausencia no es aleatoria.</li><li style="margin-bottom:8px;">Mitigación: bandera binaria _Ausente.</li><li style="margin-bottom:8px;">Privacidad: sin PII; jugadores seudonimizados.</li></ul></div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Los próximos pasos convierten el EDA en un pipeline temporalmente correcto hacia P2</div>
<img src="images/icons/nextsteps_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:21.33%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:24.89%; width:56.00%; height:62.22%; color:#2D2D2D; font-size:16px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:10px;"><b>1. Preprocesamiento: </b>features rolling (5 partidos), split cronológico.</li><li style="margin-bottom:10px;"><b>2. Features: </b>squad quality + cuotas de apertura, z-scores.</li><li style="margin-bottom:10px;"><b>3. Selección: </b>mutual_info_classif vs. sin selección.</li><li style="margin-bottom:10px;"><b>4. Modelos: </b>logística, Random Forest, XGBoost, MLP, ensemble.</li><li style="margin-bottom:10px;"><b>5. Evaluación: </b>log loss, Brier, calibración, validación por temporada.</li><li style="margin-bottom:10px;"><b>6. Backtest: </b>value bets vs. cuotas de cierre, ROI.</li></ul></div>
<div style="position:absolute; left:63.00%; top:24.89%; width:32.00%; height:56.89%; border:0.75px solid #CCCCCC; overflow:hidden;"><img src="images/photos/nextsteps_photo.jpg" style="width:100%; height:100%; object-fit:cover; display:block;" /></div>
<div style="position:absolute; left:63.00%; top:82.49%; width:32.00%; height:4.44%; color:#777777; font-size:10px; font-style:italic;">Camino de montaña — Laluttam, Wikimedia Commons (CC BY-SA 4.0)</div>

---

<!-- _class: dark -->
<div style="position:absolute; left:5.00%; top:5.33%; width:90.00%; height:8.00%; color:#A0BBDD; font-size:20px;">Conclusiones</div>
<img src="images/icons/conclusions_white.png" style="position:absolute; left:89.50%; top:4.27%; width:4.50%; height:8.00%; " />
<div style="position:absolute; left:5.00%; top:13.87%; width:90.00%; height:0.62%; background:#2E75B6;"></div>
<div style="position:absolute; left:5.00%; top:17.78%; width:90.00%; height:62.22%; color:#FFFFFF; font-size:18px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:18px;"><b>1. Dataset listo para modelar: </b>30,182 partidos, 219 columnas útiles, 0 nulos.</li><li style="margin-bottom:18px;"><b>2. Señal dominante: </b>cuotas de mercado; squad quality complementa, no redunda.</li><li style="margin-bottom:18px;"><b>3. Riesgo crítico: </b>fuga temporal → exige rolling features y split cronológico.</li><li style="margin-bottom:18px;"><b>4. Próximo hito (P2): </b>selección de features, comparación de modelos, backtest.</li></ul></div>
<div style="position:absolute; left:5.00%; top:87.11%; width:90.00%; height:7.11%; color:#A0BBDD; font-size:14px;">github.com/jcarlos-t/MLProject</div>

---

<div style="position:absolute; left:5.00%; top:4.44%; width:83.50%; height:8.89%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">Referencias</div>
<img src="images/icons/references_accent.png" style="position:absolute; left:90.00%; top:5.33%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:14.22%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:20.44%; width:90.00%; height:74.67%; color:#2D2D2D; font-size:14px;"><p style="margin:0 0 10px 0;">Y. Tammouch, A. Elouafi, I. Essadik, "Betting on Machine Learning: Extracting Patterns from Football's Anarchic Odds," 2024.</p><p style="margin:0 0 10px 0;">J. Knoll, J. Stübinger, "Machine-Learning-Based Statistical Arbitrage Football Betting," Applied Sciences, vol. 10, no. 19, 2020.</p><p style="margin:0 0 10px 0;">M. J. Dixon, S. G. Coles, "Modelling Association Football Scores and Inefficiencies in the Football Betting Market," Applied Statistics, vol. 46, no. 2, 1997.</p><p style="margin:0 0 10px 0;">D. Karlis, I. Ntzoufras, "Analysis of sports data using bivariate Poisson models," The Statistician, vol. 52, no. 3, 2003.</p><p style="margin:0 0 10px 0;">football-data.co.uk, "Football Data Historical Results and Betting Odds," 2025.</p><p style="margin:0 0 10px 0;">Kaggle, "FIFA / EA Sports FC Player Ratings Datasets," 2025.</p></div>

---

<div style="position:absolute; left:5.00%; top:2.67%; width:90.00%; height:6.22%; color:#777777; font-size:14px; font-style:italic;">Apéndice A — Metodología del backtest</div>
<div style="position:absolute; left:5.00%; top:10.67%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">El backtest usa un split de 3 etapas y Kelly fraccional para no sobreajustar el umbral</div>
<img src="images/icons/backtest_accent.png" style="position:absolute; left:90.00%; top:11.56%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:27.56%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<div style="position:absolute; left:5.00%; top:40.89%; width:56.00%; height:51.56%; color:#2D2D2D; font-size:16px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:12px;"><b>Regla de valor: </b>apostar si p_modelo · cuota &gt; 1 + τ (cuotas de cierre).</li><li style="margin-bottom:12px;"><b>Split en 3 etapas: </b>entrenar → elegir τ en validación → fijar τ y medir ROI en test.</li><li style="margin-bottom:12px;"><b>Tamaño de apuesta: </b>Kelly fraccional.</li><li style="margin-bottom:12px;"><b>Benchmark: </b>debe superar a local-siempre, favorito-siempre y azar.</li></ul></div>
<div style="position:absolute; left:63.00%; top:40.89%; width:32.00%; height:42.67%; border:0.75px solid #CCCCCC; overflow:hidden;"><img src="images/photos/backtest_photo.jpg" style="width:100%; height:100%; object-fit:cover; display:block;" /></div>
<div style="position:absolute; left:63.00%; top:84.27%; width:32.00%; height:4.44%; color:#777777; font-size:10px; font-style:italic;">Monedas — Wikimedia Commons (CC0)</div>

---

<div style="position:absolute; left:5.00%; top:2.67%; width:90.00%; height:6.22%; color:#777777; font-size:14px; font-style:italic;">Apéndice B — Análisis multivariado adicional</div>
<div style="position:absolute; left:5.00%; top:10.67%; width:83.50%; height:16.00%; color:#1F4E79; font-weight:bold; font-size:26px; line-height:1.25;">La PCA de cuotas no muestra clusters separables: el problema tiene margen estrecho entre clases</div>
<img src="images/icons/pca_accent.png" style="position:absolute; left:90.00%; top:11.56%; width:4.20%; height:7.47%; " />
<div style="position:absolute; left:5.00%; top:27.56%; width:90.00%; height:0.53%; background:#CCCCCC;"></div>
<img src="images/eda/eda_pca.png" style="position:absolute; left:5.00%; top:39.11%; width:55.00%; height:53.33%; object-fit:contain;" />
<div style="position:absolute; left:62.00%; top:39.11%; width:33.00%; height:53.33%; color:#2D2D2D; font-size:14px; line-height:1.4;"><ul style="margin:0; padding-left:20px;"><li style="margin-bottom:8px;">2 componentes retienen casi toda la varianza del mercado 1X2 (cuotas casi colineales).</li><li style="margin-bottom:8px;">PC1 extremo (cuota local muy baja) se concentra en victorias locales.</li><li style="margin-bottom:8px;">Nube central: partidos equilibrados y empates.</li></ul></div>
