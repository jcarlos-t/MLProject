# Fútbol 1X2 con y sin cuotas

Proyecto de Machine Learning de UTEC (Sección 3). Estima P(local), P(empate) y P(visitante) en nueve ligas europeas, con dos contribuciones de igual importancia:

1. **Con cuotas:** medir cuánto añade el aprendizaje frente a probabilidades implícitas directas del mercado.
2. **Sin cuotas:** construir un predictor deportivo con historia, plantilla y habilidades, y medir su distancia al mercado.

Equipo: José Luis Alva Espinoza, Juan Carlos Ticlia Maqui, Elmer José Manuel Villegas Suárez y Joseph Anderson Cose Rojas.

## Protocolo corregido

- 36.150 partidos originales; 36.119 comparables. Los 31 sin mercado participan en el historial previo, pero no en la comparación pareada.
- Calentamiento: 2013/14. Train/val: 2014/15–2023/24, 30.120 partidos. Seis folds exteriores: 2018/19–2023/24.
- Ajuste temporal interior con las dos temporadas anteriores a cada fold, incluyendo imputación, escalado, poda y MI/PCA dentro del entrenamiento.
- Test 2024/25–2025/26: 5.999 partidos, **previamente observado y reevaluado tras la auditoría**. Borrar o reentrenar un modelo no vuelve desconocido ese periodo.
- La selección se congela antes de generar las predicciones revisadas del test. Un resultado posterior no modifica esa selección.
- Ratings: doce fechas verificadas en `scripts/download/snapshot_availability.json`. Cruce hacia atrás por club, estrictamente antes de la fecha del partido. Si falta el snapshot, se usa una versión previa verificable o se conserva el faltante.
- Cuotas **pre-cierre observadas**, no necesariamente apertura real ni capturas simultáneas. Pinnacle no se considera fiable desde el 23/07/2025; se usa el consenso de otras casas y se restringe su backtest al periodo fiable.
- El modelo deportivo puro no recibe cuotas como variables, pesos ni entradas de calibración. Las variantes reponderadas por errores del favorito se identifican como **guiadas por mercado**.

## Reproducción

Usar Python 3.10+ y el entorno del proyecto:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/download/fetch_data.py
.venv/bin/python scripts/download/manifest.py --check
.venv/bin/python scripts/download/process_data.py
.venv/bin/python scripts/run_revision.py --jobs 4
.venv/bin/python scripts/split_holdout.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/report/build.py
```

El paso de descarga se omite si ya están los 129 crudos. Si una fuente cambió, la comprobación falla: no actualizar el manifiesto para ocultar diferencias. Los PDF requieren `tectonic`, `chromium` y `pdfinfo`; no son dependencias de pip.

La ejecución puede separarse con `--stage prepare`, `cv`, `feedback`, `freeze`, `evaluate`. Las rejillas predefinidas tienen 81 ajustes por temporada, ocho temporadas de cálculo compartido y 29 candidatos entre ambas ramas. La MLP usa tres semillas; SVM/SVR se limitan a 9.000 filas ordenadas. Las rejillas son acotadas: no representan una búsqueda exhaustiva.

Cada ejecución vive en `scripts/data/runs/<huella>/`; la huella incluye crudos, registro temporal, tabla de identidad de clubes, código científico y protocolo. Las predicciones intermedias se reutilizan únicamente dentro de esa identidad. El modelo final siempre se ajusta desde cero con train/val antes de la evaluación de esa ejecución. `scripts/data/revision_actual.txt` señala la vigente.

Los cuadernos 01–09 pueden ejecutarse en orden: si ya están las salidas válidas, las leen. Para ejecutarlos:

```bash
.venv/bin/python scripts/execute_notebooks.py
```

## Qué se compara

Ambas ramas evalúan LR, HGB, XGBoost, Random Forest, MLP y SVM con calibración temporal; MI-5/10 y PCA-5/95% en LR; promedio, pesos convexos y stacking. Sin cuotas también se evalúan regresión lineal, RF, HGB, SVR y promedio de diferencia de goles con calibrador temporal. Las ablaciones separan historia, plantilla y habilidades.

La métrica primaria es log loss agregado por partido. Se publica también RPS, Brier, AUC macro, accuracy 1X2, ECE de igual masa/ancho y recall de empate. El **acierto del ganador está condicionado a partidos que realmente no empataron**: no equivale a acertar el 1X2 en ese porcentaje de todos los encuentros.

Cada rama elige su candidato por menor log loss. Dentro de 0,0005, se prefiere simplicidad según un orden predefinido. Los cinco contrastes finales por partición usan bootstrap por bloques y Benjamini–Hochberg; los diagnósticos de desarrollo y desgloses no son pruebas globales de todas las hipótesis exploradas.

Los esquemas data-centric mantienen factores 2/3/5, igualdad de peso acierto/fallo, inverso y raíz de la probabilidad del resultado, submuestreo y sobremuestreo 50/50 con tres semillas. Se incluye recalibración posterior. No se presupone que deban ganar ni perder en todas las temporadas.

## Uso de ambos modelos

Resultados de la ejecución revisada `01afe756130bafc6` (menor log loss es mejor):

| Modelo / referencia | Validación temporal | Test revisado, previamente observado |
|---|---:|---:|
| Consenso directo | 0,966435 | 0,966804 |
| Con cuotas: LR de referencia | **0,965199** | **0,965034** |
| Sin cuotas: promedio de ocho predictores | 0,977599 | 0,979290 |

El ensemble sin cuotas mejora frente a la LR con historia y plantilla: diferencia de log loss −0,001061, IC 95% [−0,002022; −0,000101], p ajustado 0,036 en validación. Con cuotas, el ensemble ponderado no mejora claramente la LR de referencia. El modelo sin cuotas alcanza el 89,4% de la reducción de pérdida del consenso respecto al prior en validación; esta razón no mide un porcentaje de toda la información del mercado. Ninguna rama demuestra rentabilidad de apuestas con este protocolo.

Las salidas vigentes incluyen `modelo_con_cuotas.joblib` y `modelo_sin_cuotas.joblib`, sus metadatos JSON, variables exactas y predicciones enlazadas por identificador. La API `predict_proba(frame)` valida el esquema. El ejemplo siguiente reconstruye las variables de un partido usando solo resultados anteriores:

```bash
.venv/bin/python scripts/predict.py --branch sin --date 2026-11-15 --league E0 --home Arsenal --away Chelsea
.venv/bin/python scripts/predict.py --branch con --date 2026-11-15 --league E0 --home Arsenal --away Chelsea --quotes '{"B365":[2.1,3.5,3.6],"BW":[2.05,3.4,3.7]}'
```

El ejemplo no incorpora partidos nuevos automáticamente: la historia disponible termina donde terminan los crudos congelados. Para producción hacen falta historia actualizada y un protocolo futuro independiente. La salida incluye probabilidades y **cuotas justas** 1/P, sin margen comercial; no es recomendación de apuestas.

## Entregables y trazabilidad

- `docs/informe.tex` / `informe.pdf`: informe IEEE, máximo 12 páginas, con resultados de ambos enfoques.
- `docs/presentacion/presentacion.html` / `presentacion.pdf`: 13 diapositivas, 10 minutos; flechas para navegar e impresión PDF.
- `docs/resultados_revision.csv`: única tabla numérica utilizada por ambos entregables.
- `docs/AUDITORIA_PROYECTO.md`: hallazgos iniciales y estado de resolución.
- `scripts/mlcore/`: implementación compartida y probada; `scripts/notebooks/`: exposición ejecutada de cada etapa.

Los resultados anteriores, cuadernos y documentos se preservan localmente en `scripts/data/archive/pre_revision_20261112/`. No se mezclan con resultados revisados. Las figuras antiguas que ya no cita el informe son antecedentes; las vigentes llevan prefijo `revision_` y se generan directamente desde los resultados, sin índices frágiles de celdas ni recortes pendientes.

Los CSV crudos no se versionan ni redistribuyen. Se conservan las fuentes y huellas para uso académico. La auditoría corrige fugas detectadas, pero no elimina los límites de cobertura, ratings antiguos, alineaciones del juego, partidos solo de liga, sincronización de cuotas y test observado.
