# Lectura y ejecución de los nueve cuadernos

La implementación científica común vive en `scripts/mlcore`. Cada celda de código está precedida por su propósito y seguida de interpretación. Se ejecutan en orden; 08 completa experimentos y congela selección, y 09 exporta **dos** modelos y evalúa el test.

| # | Cuaderno | Propósito |
|---|---|---|
| 01 | `01_eda.ipynb` | Integridad de fuentes, fechas de snapshots, resultados y goles por liga, cobertura de cuotas, acuerdo entre casas y valores extremos |
| 02 | `02_feature.ipynb` | Historia cronológica, cruces de ratings y precios fiables; señal, redundancia y estabilidad de las variables |
| 03 | `03_preprocesamiento.ipynb` | Transformaciones internas, pruebas contra fuga y sensibilidad a las decisiones de preprocesamiento |
| 04 | `04_modelo.ipynb` | Clasificación de ambas ramas y regresión deportiva, CV temporal, resultados por temporada y error de entrenamiento frente a validación |
| 05 | `05_hiperparametros.ipynb` | Configuraciones elegidas por fold, MI/PCA, curvas de la rejilla y sensibilidad a la regularización |
| 06 | `06_mercado.ipynb` | Referencias directas, contrastes de desarrollo e importancia; márgenes, sesgo del favorito y qué corrige la LR de precio |
| 07 | `07_evaluacion.ipynb` | Métricas comunes, curvas ROC y calibración por clase, bootstrap por bloques, esquemas de validación y resultados por liga |
| 08 | `08_pesos_ensembles.ipynb` | Pesos, resampling y recalibración, ensembles (pesos y por qué ayudan) y selección congelada |
| 09 | `09_modelos_finales.ipynb` | Exportación, test de ambos enfoques por temporada y liga, calibración, reglas de empate y entregables |

Las salidas se enlazan por una identidad de ejecución que incorpora entradas, código y protocolo. No se reutilizan modelos antiguos por una caché anónima.

Las métricas nuevas se consultan en `docs/resultados_revision.csv` y en el directorio que indica `scripts/data/revision_actual.txt`. Las cifras anteriores se conservan en el archivo histórico local y no deben citarse como resultados vigentes.
