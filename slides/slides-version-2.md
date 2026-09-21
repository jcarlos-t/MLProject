<!-- Slide number: 1 -->

![slide 1 image 1](images/slide1_img1.png)

¿Se pueden estimar probabilidades 1X2 bien calibradas para un mercado de apuestas de fútbol altamente eficiente?

Etapa 1 (P1) — Problema, Dataset y Análisis Exploratorio de Datos

José Luis Alva Espinoza  ·  Juan Carlos Ticlia Maqui  ·  Elmer José Villegas Suárez  ·  Joseph Anderson Cose Rojas
Departamento de Computación — Universidad de Ingeniería y Tecnología (UTEC), Lima, Perú

Septiembre 2026



<!-- Slide number: 2 -->

Un mercado de apuestas eficiente solo se puede superar con probabilidades calibradas, no con más aciertos

![slide 2 image 1](images/slide2_img1.png)

Fútbol: pocos goles, azar alto, fuerte efecto de contexto.
Techo empírico: ~55% de acierto en la clase ganadora.
Mercado de apuestas altamente eficiente — las cuotas ya condensan la información pública.
ML solo aporta valor con probabilidades bien calibradas.

![slide 2 image 2](images/slide2_img2.jpg)

Estadio de fútbol — Wikimedia Commons (CC0)

Tammouch, Elouafi & Essadik (2024); Knoll & Stübinger (2020)



<!-- Slide number: 3 -->

El objetivo: predecir 1X2 con probabilidades calibradas usando solo datos pre-partido

![slide 3 image 1](images/slide3_img1.png)

Dado un partido con local H y visitante A, estimar P(FTR = c | x) para c ∈ {H, D, A},
donde x agrupa forma reciente, calidad de plantilla, localía y cuotas pre-partido.

Hipótesis de trabajo

Las cuotas de mercado dominan sobre las features estadísticas individuales.
La calidad de plantilla aporta señal complementaria, sobre todo al inicio de temporada.
Un modelo mal calibrado puede acertar más pero perder dinero en el backtest.



<!-- Slide number: 4 -->

La tarea es clasificación multiclase probabilística; log loss y Brier score priman sobre accuracy

![slide 4 image 1](images/slide4_img1.png)

Tarea de ML

Cuota → probabilidad

Multiclase (3 clases), salida probabilística.
Métrica primaria: log loss.
Complementarias: accuracy, F1, Brier, reliability diagrams.

p_c = (1/o_c) / Σ_j (1/o_j)

Value bet si p̂ · o > 1.

El edge puede aparecer en local, empate o visitante — no solo en el favorito.

![slide 4 image 2](images/slide4_img2.jpg)

Dardos — kallerna, Wikimedia Commons (CC BY-SA 3.0)



<!-- Slide number: 5 -->

Construimos un dataset propio de 30,182 partidos combinando cuotas de mercado y calidad de plantilla

![slide 5 image 1](images/slide5_img1.png)

Fuentes

Dataset consolidado

football-data.co.uk: 90 CSVs, 9 ligas, 10 temporadas.
Kaggle (FIFA/FC): 10 snapshots de ratings, proxy de plantilla.

Partidos | 30,182

Columnas útiles | 219 (de 222)

Ligas / Temporadas | 9 / 10

Equipos | 280

Cobertura squad quality | 97.1%

Nulos tras limpieza | 0

![slide 5 image 2](images/slide5_img2.jpg)

Clave compuesta por partido: (Div, Season, Date, HomeTeam, AwayTeam) — sin duplicados.

Centro de datos — Wikimedia Commons (CC0)



<!-- Slide number: 6 -->

FTR está levemente desbalanceada (44% local) y se usará como prior, no se rebalanceará

![slide 6 image 1](images/slide6_img1.png)

[Gráfico: COLUMN_CLUSTERED (51)]

- H (local): 44.3

- A (visitante): 30.9

- D (empate): 24.8

Lectura

30,182 partidos totales.
Ventaja de localía: 1.56 vs. 1.25 goles promedio (local vs. visitante).
Desbalance estructural y leve → se usa como prior informativo.



<!-- Slide number: 7 -->

Las cuotas medias de mercado son la señal predictiva más fuerte antes del partido

![slide 7 image 1](images/slide7_img1.png)

![slide 7 image 2](images/slide7_img2.png)

Qué observar

AvgH/AvgD/AvgA equivalen casi al cierre de Pinnacle (acierto del favorito: 53.9% vs. 54.5%).
Información mutua con FTR ≈ 0.11 en clases H y A.
Se usan como baseline del mercado (menos ruido al promediar casas).

AvgH baja → más victorias locales

Fig.: cuotas medias 1X2 según resultado real FTR (EDA propio).



<!-- Slide number: 8 -->

La calidad de plantilla aporta señal complementaria, con menor magnitud que las cuotas

![slide 8 image 1](images/slide8_img1.png)

![slide 8 image 2](images/slide8_img2.png)

Qué observar

Mayor diferencia de top11 (local − visitante) → mayor proporción de victorias locales.
Magnitud menor que el efecto de las cuotas.
Más útil en jornadas tempranas, cuando la forma reciente es escasa.

Fig.: diferencia de calidad del once titular según FTR (EDA propio).



<!-- Slide number: 9 -->

Las cuotas del mismo mercado son casi colineales (r > 0.9): hay redundancia, no señal nueva

![slide 9 image 1](images/slide9_img1.png)

![slide 9 image 2](images/slide9_img2.png)

Qué observar

Multicolinealidad confirmada entre casas de apuestas del mismo mercado.
Justifica imputación KNN y quedarse con una única referencia (Avg) + banderas de ausencia.
Goles y tiros correlacionan r ≈ 0.6–0.7; squad correlaciona poco con cuotas → señal complementaria.

Fig.: detalle de la matriz de correlación de Pearson (EDA propio).



<!-- Slide number: 10 -->

Una limpieza explícita de 156 columnas de cuotas deja el dataset sin valores nulos

![slide 10 image 1](images/slide10_img1.png)

[Gráfico: BAR_CLUSTERED (57)]

- Eliminadas (>40%): 83.0

- Bandera Ausente (20-40%): 58.0

- Imputadas KNN (0-20%): 15.0

Otras variables

Unnamed 105/119/120 (100%) → artefactos, eliminadas.
Referee (87.4%) eliminada; Time (29.7%) → Time_hora.
HTR (39 nulos) reconstruida desde HTHG/HTAG.
Sin duplicados: clave compuesta única por partido.



<!-- Slide number: 11 -->

Los outliers en goles, tiros y cuotas son partidos reales y se conservan

![slide 11 image 1](images/slide11_img1.png)

![slide 11 image 2](images/slide11_img2.png)

Regla IQR

Atípico si está fuera de [Q1 − 1.5·IQR, Q3 + 1.5·IQR].
Goles hasta 10–13, tiros hasta 46 → marcadores reales, no errores.
Cuotas AvgA hasta 41.22 → partidos muy desequilibrados, informativos.
Decisión: no se eliminan.



<!-- Slide number: 12 -->

El mayor riesgo metodológico es la fuga de información temporal (data leakage)

![slide 12 image 1](images/slide12_img1.png)

Post-partido: goles, tiros, tarjetas → deben pasar a medias rolling de los últimos N partidos.
Look-ahead: toda agregación usa solo el pasado → split train/test cronológico, nunca aleatorio.
Plantilla por temporada: un snapshot de ratings no predice una temporada anterior.

![slide 12 image 2](images/slide12_img2.jpg)

Relojes de arena — Aaaatu, Wikimedia Commons (CC BY-SA 4.0)



<!-- Slide number: 13 -->

La cobertura por liga y la exclusión de casas de apuestas también pueden introducir sesgo

![slide 13 image 1](images/slide13_img1.png)

Sesgo de cobertura

Sesgo de cuotas

E0: 3,800 partidos; B1: 2,805.
Ligas con menos historia, sub-representadas.
Mitigación: validación por liga y temporada.

Excluir casas con nulos puede sesgar si la ausencia no es aleatoria.
Mitigación: bandera binaria _Ausente.
Privacidad: sin PII; jugadores seudonimizados.



<!-- Slide number: 14 -->

Los próximos pasos convierten el EDA en un pipeline temporalmente correcto hacia P2

![slide 14 image 1](images/slide14_img1.png)

1. Preprocesamiento: features rolling (5 partidos), split cronológico.
2. Features: squad quality + cuotas de apertura, z-scores.
3. Selección: mutual_info_classif vs. sin selección.
4. Modelos: logística, Random Forest, XGBoost, MLP, ensemble.
5. Evaluación: log loss, Brier, calibración, validación por temporada.
6. Backtest: value bets vs. cuotas de cierre, ROI.

![slide 14 image 2](images/slide14_img2.jpg)

Camino de montaña — Laluttam, Wikimedia Commons (CC BY-SA 4.0)



<!-- Slide number: 15 -->

![slide 15 image 1](images/slide15_img1.png)

Conclusiones

1. Dataset listo para modelar: 30,182 partidos, 219 columnas útiles, 0 nulos.
2. Señal dominante: cuotas de mercado; squad quality complementa, no redunda.
3. Riesgo crítico: fuga temporal → exige rolling features y split cronológico.
4. Próximo hito (P2): selección de features, comparación de modelos, backtest.

github.com/jcarlos-t/MLProject



<!-- Slide number: 16 -->

Referencias

![slide 16 image 1](images/slide16_img1.png)

Y. Tammouch, A. Elouafi, I. Essadik, "Betting on Machine Learning: Extracting Patterns from Football's Anarchic Odds," 2024.

J. Knoll, J. Stübinger, "Machine-Learning-Based Statistical Arbitrage Football Betting," Applied Sciences, vol. 10, no. 19, 2020.

M. J. Dixon, S. G. Coles, "Modelling Association Football Scores and Inefficiencies in the Football Betting Market," Applied Statistics, vol. 46, no. 2, 1997.

D. Karlis, I. Ntzoufras, "Analysis of sports data using bivariate Poisson models," The Statistician, vol. 52, no. 3, 2003.

football-data.co.uk, "Football Data Historical Results and Betting Odds," 2025.

Kaggle, "FIFA / EA Sports FC Player Ratings Datasets," 2025.



<!-- Slide number: 17 -->

Apéndice A — Metodología del backtest

El backtest usa un split de 3 etapas y Kelly fraccional para no sobreajustar el umbral

![slide 17 image 1](images/slide17_img1.png)

Regla de valor: apostar si p_modelo · cuota > 1 + τ (cuotas de cierre).
Split en 3 etapas: entrenar → elegir τ en validación → fijar τ y medir ROI en test.
Tamaño de apuesta: Kelly fraccional.
Benchmark: debe superar a local-siempre, favorito-siempre y azar.

![slide 17 image 2](images/slide17_img2.jpg)

Monedas — Wikimedia Commons (CC0)



<!-- Slide number: 18 -->

Apéndice B — Análisis multivariado adicional

La PCA de cuotas no muestra clusters separables: el problema tiene margen estrecho entre clases

![slide 18 image 1](images/slide18_img1.png)

![slide 18 image 2](images/slide18_img2.png)

2 componentes retienen casi toda la varianza del mercado 1X2 (cuotas casi colineales).
PC1 extremo (cuota local muy baja) se concentra en victorias locales.
Nube central: partidos equilibrados y empates.


