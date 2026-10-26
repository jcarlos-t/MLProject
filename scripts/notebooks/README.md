# Guía de lectura de los cuadernos

Se leen y se ejecutan **en este orden** (cada uno usa archivos que generan los anteriores). En todos, cada celda de código va precedida por una celda de texto (qué vamos a hacer y por qué) y seguida por otra (qué encontramos y, en los cuadernos 05 a 08, qué significa para nuestro caso).

| # | Cuaderno | Pregunta que responde | Resultado clave |
|---|---|---|---|
| 01 | `01_eda.ipynb` | ¿Cómo son los datos y qué problemas de calidad tienen? | Limpieza única y sin imputar: 36,150 partidos con NaN solo donde la fuente no publica |
| 02 | `02_feature.ipynb` | ¿Qué variables construimos, sin mirar el futuro, y cuáles valen la pena? | Bloques base (F1/F2), 27 variables nuevas (Pinnacle, Elo, EWMA, goles esperados, empate, contexto, once titular) y tabla de revisión |
| 03 | `03_preprocesamiento.ipynb` | ¿Cómo elegimos cada preprocesamiento? ¿Hay mejores variables? | La cuota→probabilidad importa; el resto casi no; Elo y tiros a puerta ayudan solo sin cuotas |
| 04 | `04_modelo.ipynb` | ¿Qué bloques de variables y qué modelos aportan? (incluye la selección de hiperparámetros base) | La LR sobre cuotas es la mejor; nada aporta sobre el precio |
| 05 | `05_hiperparametros.ipynb` | ¿Sesgó la selección de hiperparámetros? ¿Había óptimos en el borde? | No: la validación anidada reproduce el resultado; el óptimo es casi lineal |
| 06 | `06_mercado.ipynb` | ¿Quién gana, cuotas o fundamentos? ¿Dónde falla el mercado? ¿Hay dinero? | El mercado domina; no hay información que el cierre no tenga; sin retorno demostrable |
| 07 | `07_evaluacion.ipynb` | ¿Se sostiene el resultado con AUC, calibración, otros esquemas de CV y bootstrap por bloques? | Sí; el SVM tenía la probabilidad de empate invertida |
| 08 | `08_modelo_final.ipynb` | ¿Cuál es el modelo final, cómo se usa y qué rinde en el test holdout? | LR de 2 variables sobre Pinnacle; log loss 0.9653 en validación y **0.9651 en el test holdout** (consenso 0.9667, Pinnacle cruda 0.9660, *baseline* 0.9652): recalibra el precio, sin ventaja clara frente a Pinnacle ni al *baseline* |

**Terminología.** *Referencias sin aprendizaje*: el prior y el mercado crudo. ***Baseline***: la regresión logística sobre las tres variables de mercado ("LR (cuotas)"), el modelo más simple con aprendizaje. **Modelo final**: regresión logística de dos variables sobre la apertura de Pinnacle (en `08_modelo_final.ipynb` aparece como "MODELO FINAL" y el baseline como "Baseline (LR 3 var.)").

**Esquema train/val/test.** Las temporadas **2024/25 y 2025/26** (5,999 partidos) son un **test holdout**: los cuadernos 03–08 las excluyen de entrenamiento, validación y elección de hiperparámetros, variables y umbrales (`TEST_SEASONS`, `TEST_MASK`), y solo se evalúan una vez en el Paso 8 de `08_modelo_final.ipynb`. La validación son 6 folds temporales (2018/19–2023/24) y el modelo final se entrena con 2014/15–2023/24. Los cuadernos 01 y 02 solo construyen datos y variables.

**Por qué este orden.** Sigue las fases habituales de un proyecto de datos (comprensión de los datos → preparación → modelado → evaluación → modelo final): 01 explora y limpia; 02 y 03 preparan las variables y justifican el preprocesamiento; 04 y 05 comparan modelos y afinan hiperparámetros; 06 prueba el modelo contra el mercado y hace el *backtest*; 07 valida con más métricas; 08 entrena y exporta el modelo elegido. Cada cuaderno solo lee archivos de cuadernos anteriores, así que se pueden ejecutar de principio a fin sin errores.

Los cuadernos 01, 04 y 06 conservan en parte la forma "Hallazgo / Decisión" en la celda posterior; los demás usan "Qué encontramos / Qué significa para nuestro caso / Decisión".

**Tres preguntas.** Los cuadernos responden a A (qué predicen los fundamentos: 01–03, 06), B (qué modelo es mejor sobre las cuotas: 04, 05, 07, 08) y C (si hay dinero: 06 y 08). El umbral `τ` del *backtest* se fija a priori (0.02). Salvedades en el README principal.
