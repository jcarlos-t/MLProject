# Lectura y ejecución de los nueve cuadernos

La implementación científica común vive en `scripts/mlcore`. Cada celda de código está precedida por su propósito y seguida de interpretación. Se ejecutan en orden; 08 completa experimentos y congela selección, y 09 exporta **dos** modelos y evalúa el test.

| # | Cuaderno | Propósito |
|---|---|---|
| 01 | `01_eda.ipynb` | Integridad de fuentes, exploración de desarrollo y fechas de snapshots |
| 02 | `02_feature.ipynb` | Historia cronológica, cruces de ratings y precios fiables |
| 03 | `03_preprocesamiento.ipynb` | Transformaciones internas y pruebas contra fuga |
| 04 | `04_modelo.ipynb` | Clasificación de ambas ramas y regresión deportiva, CV temporal |
| 05 | `05_hiperparametros.ipynb` | Configuraciones elegidas por fold, MI/PCA |
| 06 | `06_mercado.ipynb` | Referencias directas, contrastes de desarrollo e importancia |
| 07 | `07_evaluacion.ipynb` | Métricas comunes y curvas por clase |
| 08 | `08_pesos_ensembles.ipynb` | Pesos, resampling, ensembles y selección congelada |
| 09 | `09_modelos_finales.ipynb` | Exportación, test de ambos enfoques y entregables |

Las salidas se enlazan por una identidad de ejecución que incorpora entradas, código y protocolo. No se reutilizan modelos antiguos por una caché anónima.

Las métricas nuevas se consultan en `docs/resultados_revision.csv` y en el directorio que indica `scripts/data/revision_actual.txt`. Las cifras anteriores se conservan en el archivo histórico local y no deben citarse como resultados vigentes.
