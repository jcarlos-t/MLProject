# Auditoría del proyecto — 12 de noviembre de 2026

> La evidencia de esta auditoría describe la versión inicial. El estado de la implementación posterior figura al final; las cifras históricas no deben confundirse con resultados revisados.

Revisión del código de los nueve cuadernos, scripts de descarga, consolidación, partición y figuras; datos locales, modelos y predicciones exportados; README, diccionario, informe y presentación HTML. Los números de celda de este documento comienzan en cero, como los índices de los scripts de extracción de figuras.

Se hicieron comprobaciones independientes sobre los datos y pruebas aisladas de funciones. No se volvieron a ejecutar todas las rejillas de entrenamiento ni se eligieron modelos nuevos usando el test. Se verificaron las predicciones del test ya existentes. Esta auditoría no modifica código, datos, modelos ni entregables: únicamente añade este documento.

**Qué hace el proyecto**

Estima probabilidades H/D/A para fútbol y pregunta si los fundamentos aportan información adicional a las cuotas. Consolida resultados y ratings FIFA/FC, limpia, construye forma histórica y variables de mercado/plantilla, compara modelos con seis temporadas de validación y termina con una LR sobre `REF_logit` y `REF_pD`.

Flujo: `fetch_data.py` → `process_data.py` → 01 EDA → 02 variables → 03 preprocesamiento → 04 modelos → 05 revisión de hiperparámetros → 06 mercado/backtest → 07 evaluación → 08 modelo final/test → 09 pesos/ensembles. `split_holdout.py` se ejecuta por separado después de 02. Las figuras y los textos son entregables derivados.

Hay 36.150 partidos originales y 36.119 modelables: 30.120 de train/val y 5.999 de test. El modelo exportado usa C=0,01 y se entrenó hasta 2023/24. La conclusión central es una recalibración pequeña del precio, sin rentabilidad demostrada.

**1. Alta — Ratings posteriores al partido se usan como si fueran previos**

Ubicación: `scripts/download/process_data.py`, `PLAYER_SPECS`, normalización y `join_matches_squad`; `02_feature.ipynb`, celda 32, habilidades. El cruce solo utiliza temporada y equipo. No se compara la fecha del partido con la fecha de disponibilidad del snapshot; la normalización elimina esa información temporal.

Comprobación con los CSV originales:

| Snapshot | Fecha registrada en el CSV | Partidos anteriores de su temporada |
|---|---|---:|
| FIFA 23 | `update_as_of = 2022-09-26` | 610 |
| FC 24 | `update_as_of = 2023-09-22` | 404 |
| FC 25 | `version = 2025-07-17` | 3.018 |
| FC 26 | `fifa_update_date = 2025-09-19` | 358 |

Los dos primeros afectan a la validación, no solamente al test. La documentación reconoce el problema de FC25/26, pero trata los snapshots anteriores como si fueran pretemporada. Es una fuga temporal: se usa una versión posterior de ratings, clubes y posiciones para partidos anteriores. No hace falta que incluya el resultado explícitamente para violar la disponibilidad histórica.

Alcance: comparaciones con plantilla/habilidades y el benchmark sin cuotas. La LR final de dos variables no consume ratings. No se ha cuantificado cuánto cambian sus competidores al reparar esta fuga.

Acción: conservar `available_at`/fecha de actualización y hacer un cruce temporal hacia atrás. Para partidos anteriores a la disponibilidad de un snapshot, usar una versión anterior realmente disponible.

**2. Alta — La calibración temporal del SVM corregido no es temporal**

Ubicación: `05_hiperparametros.ipynb`, celda 8, `SVMCal.fit`; también los recortes de filas de 03/04/05.

`process_data.py` concatena por temporada y después por liga. El dataset final conserva ese orden; tiene **98 retrocesos de fecha**. Sin ordenar, se toma `ix[-9000:]` como historia reciente y después `X[:k]`/`X[k:]` como base/calibración.

Con las 9.000 filas reales del primer fold:

- El entrenamiento base llega al **20/05/2018**.
- La calibración comienza el **28/07/2017**.
- **1.184 partidos del entrenamiento base** son posteriores al inicio de la calibración.
- En el último fold son **1.257**. La calibración también queda sesgada hacia las ligas situadas al final del archivo.

El fold exterior sigue excluido del entrenamiento: no es una filtración directa de su target. Sí invalida la descripción de calibración con el 20% más reciente y el propósito de evitar mirar el futuro dentro del ajuste.

Acción: ordenar todos los índices por fecha antes del cap y separar la calibración por una fecha explícita.

**3. Alta en el experimento afectado — El prior de HPPG utiliza resultados del holdout**

Ubicación: `03_preprocesamiento.ipynb`, celda 20: `prior = lg[col].mean()` para puntos; `NEW = NEW.fillna(NEW.median())`.

`lg` contiene calentamiento y todas las temporadas, incluidas 2024/25 y 2025/26. Su media de puntos se calcula desde `FTR`, de modo que resultados del test influyen en las variables históricas `HPPG_DIF` que se comparan en validación. La diferencia local−visitante no cancela el prior cuando tienen distintas cantidades de historial.

Comprobación: el prior completo es **1,3755887171** y sin el test **1,3763040914**. Cambia HPPG en **19.482 partidos de train/val** de la base original; el cambio máximo, contando toda la base, es aproximadamente **0,000700**. El tamaño observado es pequeño, pero la dependencia del target futuro existe. Además, las medianas de las alternativas se ajustan globalmente; hay 138 registros equipo-partido modelables sin rolling alternativo inicial.

Alcance: experimentos de memoria/forma de 03. HPPG no entra directamente en la LR final ni en el bloque de fundamentos que usa 06. No equivale a demostrar que las predicciones finales del test estén contaminadas.

Acción: prior histórico por fecha y un imputador ajustado dentro de cada fold.

**4. Media — Normalización de plantilla calculada con toda la temporada predicha**

Ubicación: `02_feature.ipynb`, celda 41, construcción de `SQz_*` y `XIz_*`.

Las medias/desviaciones por liga-temporada utilizan todos los slots equipo-partido de esa temporada, incluidos partidos futuros. Los clubes además quedan ponderados por su número de apariciones, no una vez por plantilla. Añadir partidos futuros puede cambiar la variable de un partido antiguo. En temporadas incompletas o formatos desiguales el efecto es especialmente difícil de justificar.

Esto es uso de distribución futura, distinto de leer targets futuros. Una normalización sobre el conjunto de plantillas conocido antes de empezar podría ser válida, pero no es lo que implementa el cálculo sobre todos los partidos. Afecta al benchmark sin cuotas y a candidatos con plantilla; no a las dos variables finales de mercado.

Acción: definir el universo disponible en la fecha de predicción y calcular una referencia histórica o de plantillas disponibles, sin ponderarla por partidos futuros.

**5. Alta para las comparaciones — Las mejoras de 05 no se propagan**

Ubicación: `05_hiperparametros.ipynb`, celdas 8/11; lectores en 06/07/09.

05 selecciona rejillas extendidas, corrige la calibración del SVM, genera `PF` y escribe `hiperparametros_extendidos.json`. Sin embargo:

- Ningún cuaderno posterior lee ese JSON.
- `PF` no se exporta como predicciones reutilizables.
- 07 evalúa `modelos_oof.csv` de 04, que conserva el SVM original.
- 09 aprende los ensembles sobre esas mismas predicciones originales, no sobre las versiones mejoradas.
- 06 sigue leyendo `hiperparametros.json` de 04.

Así, una corrección existe y se comenta, pero no pasa a la comparación unificada. La conclusión de que el ensemble no supera a la LR describe el ensemble de las versiones originales. No prueba qué ocurriría con los modelos corregidos e integrados; no se debe anticipar que cambiarían o no el ganador.

Acción: versionar y exportar predicciones de validación de las configuraciones revisadas y consumirlas explícitamente. Mantener el test reservado.

**6. Media — La revisión de variables es informativa, no una selección aplicada**

Ubicación: `02_feature.ipynb`, celdas 53/56.

`revision_variables.csv` contiene 93 variables: **68 conservar**, **12 sin información**, **9 redundantes** y **4 inestables**. No tiene lectores posteriores. Los conjuntos exportados tampoco se filtran por ese veredicto: ocho variables marcadas para descartar siguen en F2 y otras diecisiete en EXTRA. Incluso `REF_pD`, usada por el modelo final, figura como redundante.

No implica que haya que eliminar automáticamente `REF_pD`: la redundancia por correlación no sustituye la utilidad conjunta. El problema es presentar un veredicto como decisión de selección si después no gobierna nada. `mi_ranking.csv` también queda como salida descriptiva; la selección MI efectiva de 04 sí se recalcula dentro de los folds.

La revisión añade otra inconsistencia: calcula PSI para 2425/2526 y correlaciones con toda la base. Por tanto, no es una revisión ciega al holdout. No encontré una lectura automática que traslade esos veredictos al modelo final.

Acción: llamarla diagnóstico exploratorio o definir una política de selección verificable. Separar los diagnósticos de test de cualquier decisión de desarrollo.

**7. Alta para la interpretación — Se llama “apertura” a una captura previa al cierre**

Ubicación: nombres y explicaciones de mercado en README, informe, presentación y cuadernos 02/06/08.

La [documentación de Football-Data](https://football-data.co.uk/data.php) distingue una primera captura realizada **después** de abrir el mercado y el cierre. Sus [notas](https://football-data.co.uk/notes.txt) describen capturas los viernes y martes. Las columnas sin C no garantizan el primer precio publicado por la casa.

No es fuga postpartido por sí misma. Sí cambia la pregunta experimental: se compara contra una captura pre-cierre, no necesariamente contra apertura real. El dataset tampoco permite acreditar sincronización exacta entre casas. Además, las variables históricas se calculan antes del partido, no necesariamente antes del instante en que se capturó la cuota.

Acción: usar “cuotas pre-cierre observadas”, precisar el horizonte de predicción y evitar inferir una estrategia ejecutable con precios simultáneos sin timestamps.

**8. Alta — Pinnacle en 2025/26 tiene un problema de calidad adicional a los NaN**

Ubicación: `02_feature.ipynb`, celda 41, `ps_ok`; `08_modelo_final.ipynb`, celda 5, `psok`.

La [fuente oficial](https://football-data.co.uk/data.php) advierte que desde el **23/07/2025** la API pública de Pinnacle es poco fiable y sus cuotas pueden estar desactualizadas respecto a otras casas, tanto pre-cierre como cierre.

El proyecto comprueba si existe el número, pero lo acepta como precio válido sin una política de calidad por fecha. Separar “Pinnacle observada” de “consenso” aborda cobertura, no vigencia. Esto afecta a entradas del modelo final y al backtest con Pinnacle en la segunda temporada del test. No permite afirmar que todas esas cuotas sean incorrectas ni cuantificar aquí el efecto.

Acción: declarar la limitación y definir un tratamiento por régimen de fuente con datos de desarrollo o una evaluación futura. No elegir retrospectivamente el tratamiento que más mejore este test.

**9. Media — El primer fold del ensemble ponderado/stacking usa otro método**

Ubicación: `09_pesos_ensembles.ipynb`, celda 11, rama `prev.sum() == 0`.

En 2018/19 no se aprenden pesos ni stacking: ambos usan el promedio sin SVM. Sin embargo, sus métricas se agregan como si los seis folds fueran del mismo procedimiento.

Reconstruí el promedio ponderado con la función del cuaderno y las predicciones guardadas:

- Diferencia media ponderado−LR en todos los partidos: **+0,0003175**.
- Diferencia del método de respaldo en el primer fold: **+0,0015185**.
- Diferencia en los cinco folds donde sí se aprenden pesos: **+0,0000817**.
- El primer fold aporta aproximadamente **78,5%** de la desventaja total.

Esto no demuestra superioridad del ponderado, pero sí limita la interpretación del +0,0003. Es una política con arranque de respaldo, no seis evaluaciones homogéneas de pesos aprendidos.

Acción: generar predicciones de una temporada anterior para inicializar el metamodelo, o reportar el arranque por separado y comparar métodos en exactamente los mismos folds.

**10. Media — Algunos experimentos reutilizan hiperparámetros de otro problema o del futuro de su fold**

Ubicación: `09_pesos_ensembles.ipynb`, celdas 2/11; `08_modelo_final.ipynb`, celda 14.

El Random Forest clasificador recibe `RF_reg`: hiperparámetros elegidos por RMSE de regresión sobre fundamentos, no por log loss de clasificación sobre F2. Es un benchmark con parámetros prestados, no un RF ajustado para esa tarea.

Además, 09 usa `MF['C'] = 0.01` en todos los folds del modelo final. Ese C definitivo se eligió en 08 usando 2022/23–2023/24, temporadas futuras para los primeros folds de 09. Las predicciones finales OOF de 08 sí usan C anidado. Hay dos definiciones de validación del “modelo final”, aunque sus log loss redondeados coincidan.

Acción: ajustar cada tarea en su ventana autorizada y reutilizar los C elegidos por fold cuando se afirma validación anidada.

**11. Media — Las afirmaciones estadísticas exceden lo comprobado**

Ubicación: `docs/informe.tex`, discusión del test y conclusiones; presentación, diapositiva de test.

“No hay sobreajuste a la selección” se deduce de 0,9653 en validación frente a 0,9651 en test. Dos pérdidas parecidas de periodos distintos no prueban ausencia de sesgo de selección: la dificultad de los periodos y los precios de referencia también cambian. La evidencia defendible es que la mejora respecto al consenso se mantiene en el holdout utilizado.

Las múltiples familias de exploración no quedan corregidas globalmente y la validación anidada de C no anida todo el proceso de elección de variables/modelo. Esto está parcialmente reconocido por el propio proyecto, pero convive con formulaciones absolutas. Tampoco no encontrar una mejora significativa prueba que el mercado sea eficiente frente a todos los modelos posibles: la conclusión debe limitarse a los datos, procedimientos y potencia de estos experimentos.

Acción: conservar los resultados y moderar esas inferencias; no convertir una ausencia de evidencia en una demostración de ausencia.

**12. Media — El join de calentamiento crea 8.514.724 filas inútiles**

Ubicación: `03_preprocesamiento.ipynb`, celda 20, `HA = H.join(A, how='inner')`.

Todos los partidos de calentamiento tienen `rid=-1`. Hay 2.918 entradas con ese índice en H y otras 2.918 en A. El join genera su producto cartesiano y solo después ejecuta `HA[HA.index >= 0]`.

Prueba aislada: 100 entradas de calentamiento por lado y dos partidos útiles producen 10.002 filas, de las que sobreviven dos. Con los tamaños reales se crean **8.514.724 combinaciones de calentamiento**, además de las filas útiles. El resultado que sobrevive puede ser correcto, pero el coste de memoria y tiempo es innecesario y puede tumbar la ejecución.

Acción: filtrar `rid>=0` antes de la unión y exigir `validate='one_to_one'`.

**13. Media — Errores y desajustes concretos de documentación**

| Afirmación | Evidencia actual |
|---|---|
| Ninguna variante de pesos/remuestreo mejora en una temporada | LR sin cuotas + peso 50/50 mejora en **2/6**; su sobremuestreo, en **1/6**. Siguen siendo peores en promedio. Está en los CSV y salidas de 09. |
| Correlaciones 0,94 / 0,85 “con Pinnacle” | 09 calcula esas cifras contra el **consenso**. Recalculadas contra REF/Pinnacle son **0,941159 / 0,835616**; el segundo redondea a 0,84, no 0,85. |
| Una sola definición de acierto del ganador | 06 compara el signo de GD; 09 usa argmax de las tres probabilidades tras excluir empates reales. El benchmark sin cuotas da aproximadamente **70,4%** frente a **70,13%**, respectivamente. |
| Dataset limpio sin nulos / 80 columnas | `matches_clean.csv` tiene **74 columnas y 87.721 celdas nulas**, conservadas deliberadamente. Hay frases obsoletas en README y diccionario. |
| Tau elegido en los cuatro primeros folds | Un párrafo del README lo dice, mientras el código actual fija **0,02**. No existe actualmente esa selección por ROI. |
| FC26 fechado siempre el 21/09/2025 | El nombre del archivo alude a esa fecha, pero `fifa_update_date` es **19/09/2025**. Conviene distinguir captura y actualización. |

Estas discrepancias no invierten por sí mismas el resultado central, pero impiden que un lector sepa qué cifra corresponde a qué procedimiento.

**14. Media/baja — Reproducción incompleta y continuación tras fallos**

Ubicación: `scripts/download/fetch_data.py`, bloque principal de `process_data.py`, scripts de figuras y guías.

La descarga captura excepciones y termina imprimiendo `Done` sin fallo de proceso. La consolidación salta archivos ausentes; el bloque principal ejecuta todos los pasos aunque alguno anterior devuelva False y termina sin `sys.exit(1)`. En una ejecución parcial se pueden mezclar derivados viejos y nuevos. El manifiesto ayuda a detectarlo, pero no se comprueba automáticamente antes del procesamiento.

`extraer_figuras.py` omite `prep_revision` y `prep_empate`, requeridas por `recortar_figuras.py`. Actualmente existen y coinciden con las salidas de 02; no hay evidencia de figuras obsoletas hoy. El problema es que no se regeneran con la cadena documentada de extracción y recorte.

08 requiere los archivos de `split_holdout.py`: ejecutar exclusivamente los nueve cuadernos no basta para una reconstrucción limpia. El README principal menciona el script manual, pero la afirmación de que cada cuaderno depende solo de los anteriores es incompleta. La receta detallada también omite 09.

Acción: detener la cadena ante fallos, comprobar el manifiesto como paso explícito y definir una receta completa con dependencias y figuras.

**15. Baja — Robustez de inferencia y métricas auxiliares**

Ubicación: `08_modelo_final.ipynb`, celdas 2/26; `06_mercado.ipynb`, celda 17.

- `predecir_1x2` no valida tres cuotas finitas y mayores que uno. Una prueba con `(0,3,4)` devuelve tres NaN. El respaldo tampoco filtra casas incompletas como sí hace la construcción del dataset; una casa con NaN puede invalidarlo todo.
- La función construye siempre dos variables Pinnacle, pero la selección admite otros candidatos e incluso tres variables. Si gana otro candidato con datos nuevos, la función exportada no corresponde al modelo elegido o falla por dimensión. Con el candidato actual funciona.
- ECE de igual masa falla para una probabilidad constante: `qcut(..., duplicates='drop')` deja bins NaN. El prior del test aparece con ECE NaN; debería tratarse como un único tramo. Las pendientes de calibración sobre una constante tampoco son interpretables.
- En una figura se equipara ROI justo con `-overround`. Bajo normalización proporcional, el valor exacto es `1 / sum(1/odds) - 1`, no `1 - sum(1/odds)`. Es una aproximación no etiquetada.

**16. Media — La poda efectiva de variables tampoco está dentro de cada fold**

Ubicación: `02_feature.ipynb`, celdas 20/26/29.

La poda por correlación se ajusta una vez con el primer 80% cronológico del dataset: el corte real es **05/02/2024** e incluye **16.962 partidos de las temporadas de validación**. A diferencia del ranking MI descriptivo que después no se lee, esta poda sí modifica las columnas de `dataset_final.csv` y los conjuntos F1/F2 que usan los modelos. Además, F1 se define comprobando ausencia de NaN en toda la base, incluido el test.

La correlación no utiliza el target, por lo que es fuga de distribución y no fuga directa del resultado. La selección MI/PCA de 04 sí se ajusta dentro del entrenamiento, pero eso no deshace la poda global anterior. Las variables de mercado finales tienen prioridad explícita en esa poda; no se ha demostrado aquí una alteración de sus probabilidades finales por este mecanismo.

Acción: fijar un esquema de variables con un periodo previo a la evaluación o ejecutar la poda y el tratamiento de faltantes dentro de cada fold. Explicitar las excepciones en la afirmación de que todo el preprocesamiento queda dentro del entrenamiento.

**Comprobaciones favorables y alcance del dictamen**

- `manifest.py --check`: **129 archivos**, cero faltantes, nuevos o diferentes.
- Sin duplicados en las claves revisadas; sin equipos que jueguen dos veces el mismo día en la base modelable.
- `dataset_final`, habilidades, extras y cuotas están alineados por las claves revisadas.
- Las cuatro tablas de predicciones mantienen `idx`/target correctos y sin índices duplicados.
- En los seis folds principales, la última fecha del entrenamiento precede a la primera de validación, incluso en 2020. El problema de orden afecta a los recortes por posición y a la calibración interna, no a esos límites exteriores.
- El modelo final exportado reproduce las probabilidades guardadas del test con diferencia máxima **1,11×10⁻¹⁶**.
- Log loss recalculado desde predicciones: **0,9653211092** en validación y **0,9651253927** en test, coherentes con 0,9653/0,9651.
- Las quince referencias de extracción de figuras tienen índices válidos. Las imágenes completas revisadas coinciden visualmente píxel a píxel con sus salidas actuales y las copias de presentación verificadas coinciden con las figuras originales.
- El código extraído de los nueve cuadernos y todos los scripts parsea correctamente; no hay errores almacenados en las salidas de los cuadernos.

No encontré evidencia de que la LR final de dos variables haya sido entrenada directamente con targets del holdout ni con estadísticas del propio partido. Sí hay fugas y problemas de evaluación en otros experimentos, además de una limitación de calidad de Pinnacle que alcanza al test final. Las cifras del modelo final son reproducibles con los artefactos actuales; las comparaciones y varias formulaciones necesitan las correcciones descritas.

Prioridad sugerida: reparar disponibilidad temporal de ratings y calibración del SVM; eliminar el prior/imputación global de 03; integrar las salidas mejoradas de 05; resolver el protocolo de calidad y denominación de cuotas; después reconciliar revisión de variables, ensembles, documentación y reproducción. Estas reparaciones deben desarrollarse en train/val, sin convertir el holdout ya observado en una nueva fuente de selección.


**Resolución de la auditoría — implementación posterior**

Las fuentes originales y resultados anteriores se preservan en `scripts/data/archive/pre_revision_20261112/`. El flujo vigente se implementa en `scripts/mlcore`; la ejecución se identifica por código, datos y protocolo. Se conserva el estado de test previamente observado, conforme a la decisión explícita del usuario.

| Hallazgo | Tratamiento en la revisión |
|---|---|
| 1. Ratings posteriores | Doce fechas verificadas por versión/metadatos, ligadas a huellas; cruce por club estrictamente anterior y respaldo de versiones anteriores. |
| 2. Calibración del SVM | Orden por fecha e identificador; cap después de ordenar; separación de calibración por fecha, sin días compartidos. |
| 3. HPPG e imputación global | Prior de liga anterior a la fecha; imputadores aprendidos dentro de cada entrenamiento interior/exterior. |
| 4. Normalización de plantilla | Universo de snapshots ya disponibles, una vez por club; sin conteos de partidos futuros. |
| 5. Mejoras sin propagación | Registro único de candidatos y predicciones corregidas; evaluación y ensembles consumen esas salidas. |
| 6. Revisión sin efecto | Tabla descriptiva separada de la selección; poda, MI y PCA efectivamente ejecutadas dentro de pipelines. |
| 7. Apertura frente a captura | Terminología corregida a pre-cierre observado. La falta de sincronización exacta permanece como limitación explícita. |
| 8. Pinnacle desactualizada | Excluida de referencia, consenso y backtest desde 23/07/2025; respaldo de otras casas. |
| 9. Arranque de ensembles | OOF de 2017/18 para entrenar pesos/stacking antes de 2018/19; ocho miembros por rama, incluyendo vistas LR parsimoniosas. |
| 10. Parámetros prestados/futuros | Rejillas específicas por tarea, selección interior por fold; ningún C definitivo se aplica retroactivamente como si fuera anidado. |
| 11. Inferencias excesivas | Eliminadas las pruebas pretendidas de ausencia de sobreajuste/drift o eficiencia universal; se distinguen desarrollo y test observado. |
| 12. Join cartesiano | Historia calculada por estados cronológicos; no se genera el join de millones de filas descartadas. |
| 13. Inconsistencias de cifras | Entregables desde tabla compartida; referencias separadas por nombre y una definición de acierto condicionado del ganador. |
| 14. Reproducción parcial | Fuentes verificadas antes de procesar, error fatal ante fallos, particiones de ejecución y figuras sin índices frágiles de celdas. |
| 15. Inferencia y métricas | Validación de cuotas, esquema explícito del modelo, API deportiva desde historia, ECE constante correcto y fórmula exacta de ROI en el cálculo. |
| 16. Poda global | Ajuste dentro de cada entrenamiento, incluido el interior; listas predictoras explícitas sin decidir disponibilidad usando el test. |

Las pruebas de regresión cubren perturbación de resultados y snapshots futuros, invariancia al añadir partidos futuros, fechas de ratings, orden temporal del SVM, entrenamiento de imputadores, alineación de pesos tras ordenar, serialización, ECE constante, cuotas inválidas, año ISO y reconstrucción de una predicción sin consultar resultados del día. Las garantías se limitan al código y las fuentes verificadas: no sustituyen una evaluación prospectiva con partidos nuevos.

**Verificación final de la implementación**

Ejecución vigente: `01afe756130bafc6`. Los artefactos previos están archivados y no alimentan esta ejecución. Su identidad incluye fuentes, registro temporal, mapa de clubes, código científico y protocolo; se verificaron las huellas de la configuración congelada y de las predicciones finales.

| Resultado | Validación temporal | Test revisado |
|---|---:|---:|
| Log loss del consenso directo | 0,966435 | 0,966804 |
| Log loss con cuotas: LR de referencia | 0,965199 | 0,965034 |
| Log loss sin cuotas: promedio de ocho predictores | 0,977599 | 0,979290 |

La revisión modifica la conclusión sobre ensembles: sin cuotas, el promedio mejora frente a LR con historia y plantilla (Δ LL −0,001061; IC 95% [−0,002022; −0,000101]; p BH 0,036), en cinco de seis temporadas. El ponderado tiene una pérdida algo menor, pero el promedio se elige por la tolerancia de simplicidad fijada de antemano. Con cuotas, el ponderado no muestra una mejora clara frente a LR de referencia. Son contrastes de desarrollo, sujetos al sesgo de selección entre las familias exploradas.

Comprobaciones completadas:

- 129 fuentes originales y registro de disponibilidad verificados, sin alterar los crudos.
- 12 pruebas de integridad temporal y numérica aprobadas.
- Nueve cuadernos ejecutados: todas las celdas de código tienen contador y ninguna salida contiene errores.
- Ambos modelos recargados desde disco reproducen las 5.999 probabilidades del test guardadas, con diferencia máxima de 3,33×10⁻¹⁶; las métricas recalculadas coinciden hasta 10⁻¹².
- Predictor sin cuotas sin variables de mercado; las variantes con pesos de mercado conservan su etiqueta diferenciada.
- Interfaz de inferencia ejecutada para ambas ramas: probabilidades válidas y cuotas justas, reconstruidas desde historia anterior a la fecha solicitada.
- Informe compilado de 8 páginas y presentación de 13 diapositivas, revisados visualmente; citas, bibliografía, referencias, figuras y orden de primera mención validados.
- README, diccionario, informe y presentación actualizados; cifras de los dos entregables generadas desde una tabla compartida.

El test sigue siendo previamente observado y no se ha recuperado su independencia borrando cachés. Tampoco se demuestra rentabilidad: los intervalos del modelo con cuotas incluyen pérdidas y el modelo sin cuotas presenta retorno negativo con las reglas evaluadas. Permanecen las limitaciones de datos señaladas en la auditoría, ahora tratadas explícitamente en el protocolo y los entregables.
