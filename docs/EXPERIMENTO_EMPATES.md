# Experimento adicional: detectar empates

Se prueban las hipótesis de diferencia de goles prevista y similitud de plantillas en las dos ramas. Es una extensión exploratoria. Los modelos principales y sus probabilidades se conservan.

**Protocolo.** Seis temporadas exteriores (2018/19–2023/24), con modelos ajustados solo con temporadas anteriores. Los parámetros de Poisson se eligen por log loss 1X2 y los del detector por log loss binario en hasta dos temporadas OOF anteriores. Las reglas de decisión se eligen por F1 macro usando esas temporadas anteriores. 2017/18 inicializa el primer fold; sus modelos usan valores iniciales fijados en el protocolo. Imputación y escalado se ajustan dentro de cada entrenamiento. La familia final se elige por F1 macro de validación, con tolerancia 0,002 a favor de la más sencilla; este uso de la validación para seleccionar familia conserva sesgo de desarrollo. El test no interviene en los ajustes, umbrales ni selección.

**Qué significa cada alternativa.** Poisson predice por separado los goles esperados de local y visitante y calcula P(H), P(D), P(A) mediante la diferencia de dos Poisson independientes. Su hipótesis de independencia es una limitación. Una diferencia **esperada** de goles ≤1 no equivale a empate: se prueba como regla, sin usar jamás los goles reales como entrada. La regla de plantilla utiliza la diferencia absoluta de rating agregado y conserva la decisión original si falta el rating. El detector aprende empate/no empate con diferencias absolutas y firmadas de fuerza, similitud de habilidades, goles e historial previos; la rama con cuotas añade probabilidades del mercado. Las valoraciones siguen siendo snapshots del juego.

**Probabilidades frente a decisiones.** Aumentar el peso del empate o aplicar un umbral puede mejorar F1 sin cambiar log loss: en esas variantes se conservan las probabilidades y se cambia solamente la clase elegida. El detector reemplaza P(D) y distribuye 1−P(D) entre H/A conservando la razón del predictor original. Se reportan F1 macro, F1 del empate, precisión y recall del empate, accuracy, frecuencia de predicción y log loss por separado.

**Señal de similitud de plantillas.** En validación, la frecuencia real de empate es 28.6% entre 6,224 partidos con diferencia de rating ≤2 y 22.9% entre 11,828 con diferencia >2; se excluyen los ratings faltantes de este diagnóstico. Es una asociación descriptiva, no causal. La similitud sí contiene señal, pero el empate sigue siendo minoría incluso entre equipos de calidad similar. No basta para decidir empate en todos esos partidos.

**Reglas congeladas para el test.** La rama con cuotas multiplica por 1.5 el valor de P(D) únicamente al elegir la etiqueta. La rama sin cuotas usa el detector y elige empate cuando P(D)≥0.275; si no, elige H/A según sus probabilidades relativas. Estos valores se eligieron con 2022/23–2023/24. Las reglas deportivas de comparación usan diferencia de rating ≤2 y bandas de goles esperados de 0.3 (con cuotas) y 0.4 (sin cuotas).

## Rama con cuotas

Familia elegida en validación: **Mayor peso al empate (decisión)**.

### validacion

| Método | F1 macro ↑ | F1 empate ↑ | Precisión empate | Recall empate | Accuracy ↑ | Predice empate | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original: mayor probabilidad | 0.4021 | 0.0040 | 0.4500 | 0.0020 | 53.87% | 0.11% | 0.965199 |
| Mayor peso al empate (decisión) | 0.4845 | 0.3297 | 0.3011 | 0.3642 | 50.93% | 30.06% | 0.965199 |
| Plantillas similares → empate | 0.4680 | 0.3311 | 0.2855 | 0.3939 | 47.97% | 34.28% | 0.965199 |
| Goles Poisson: mayor probabilidad | 0.3993 | 0.0000 | 0.0000 | 0.0000 | 53.70% | 0.00% | 0.971158 |
| Goles: diferencia esperada ≤ 1 | 0.3733 | 0.4182 | 0.2791 | 0.8340 | 39.70% | 74.27% | 0.971158 |
| Goles: banda elegida con pasado | 0.4778 | 0.3310 | 0.2959 | 0.3755 | 50.36% | 31.54% | 0.971158 |
| Detector de empate: probabilidades | 0.4103 | 0.0294 | 0.4059 | 0.0153 | 53.92% | 0.93% | 0.965622 |
| Detector de empate: umbral F1 | 0.4854 | 0.3564 | 0.3074 | 0.4240 | 50.18% | 34.27% | 0.965622 |

### test_revisado

| Método | F1 macro ↑ | F1 empate ↑ | Precisión empate | Recall empate | Accuracy ↑ | Predice empate | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original: mayor probabilidad | 0.4046 | 0.0039 | 0.5000 | 0.0019 | 53.86% | 0.10% | 0.965034 |
| Mayor peso al empate (decisión) | 0.4822 | 0.3274 | 0.2996 | 0.3610 | 50.48% | 30.94% | 0.965034 |
| Plantillas similares → empate | 0.4774 | 0.3516 | 0.3069 | 0.4117 | 48.77% | 34.44% | 0.965034 |
| Goles Poisson: mayor probabilidad | 0.3971 | 0.0000 | 0.0000 | 0.0000 | 53.14% | 0.00% | 0.972260 |
| Goles: diferencia esperada ≤ 1 | 0.3671 | 0.4228 | 0.2834 | 0.8325 | 39.39% | 75.41% | 0.972260 |
| Goles: banda elegida con pasado | 0.4800 | 0.3258 | 0.3004 | 0.3558 | 50.69% | 30.41% | 0.972260 |
| Detector de empate: probabilidades | 0.4131 | 0.0287 | 0.3710 | 0.0149 | 53.94% | 1.03% | 0.965845 |
| Detector de empate: umbral F1 | 0.4831 | 0.3564 | 0.3044 | 0.4299 | 49.56% | 36.26% | 0.965845 |

validacion: frente al original, Δ F1 macro **+0.0824**, IC 95% [+0.0759, +0.0887]; Δ accuracy **-2.94 puntos porcentuales**, IC [-3.61, -2.29].

test_revisado: frente al original, Δ F1 macro **+0.0776**, IC 95% [+0.0675, +0.0877]; Δ accuracy **-3.38 puntos porcentuales**, IC [-4.43, -2.34].

## Rama sin cuotas

Familia elegida en validación: **Detector de empate: umbral F1**.

### validacion

| Método | F1 macro ↑ | F1 empate ↑ | Precisión empate | Recall empate | Accuracy ↑ | Predice empate | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original: mayor probabilidad | 0.3928 | 0.0000 | 0.0000 | 0.0000 | 53.14% | 0.00% | 0.977599 |
| Mayor peso al empate (decisión) | 0.4679 | 0.2972 | 0.2872 | 0.3078 | 50.11% | 26.63% | 0.977599 |
| Plantillas similares → empate | 0.4660 | 0.3308 | 0.2855 | 0.3932 | 47.75% | 34.23% | 0.977599 |
| Goles Poisson: mayor probabilidad | 0.3945 | 0.0000 | 0.0000 | 0.0000 | 53.20% | 0.00% | 0.979196 |
| Goles: diferencia esperada ≤ 1 | 0.3777 | 0.4147 | 0.2789 | 0.8081 | 40.01% | 72.01% | 0.979196 |
| Goles: banda elegida con pasado | 0.4664 | 0.3179 | 0.2899 | 0.3521 | 49.62% | 30.19% | 0.979196 |
| Detector de empate: probabilidades | 0.3933 | 0.0013 | 0.7500 | 0.0007 | 53.15% | 0.02% | 0.978462 |
| Detector de empate: umbral F1 | 0.4744 | 0.3413 | 0.2955 | 0.4039 | 48.81% | 33.96% | 0.978462 |

### test_revisado

| Método | F1 macro ↑ | F1 empate ↑ | Precisión empate | Recall empate | Accuracy ↑ | Predice empate | Log loss ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original: mayor probabilidad | 0.3891 | 0.0000 | 0.0000 | 0.0000 | 52.36% | 0.00% | 0.979290 |
| Mayor peso al empate (decisión) | 0.4643 | 0.3597 | 0.2898 | 0.4740 | 47.46% | 41.99% | 0.979290 |
| Plantillas similares → empate | 0.4729 | 0.3514 | 0.3068 | 0.4110 | 48.31% | 34.39% | 0.979290 |
| Goles Poisson: mayor probabilidad | 0.3904 | 0.0000 | 0.0000 | 0.0000 | 52.48% | 0.00% | 0.981238 |
| Goles: diferencia esperada ≤ 1 | 0.3689 | 0.4146 | 0.2799 | 0.7987 | 39.29% | 73.25% | 0.981238 |
| Goles: banda elegida con pasado | 0.4656 | 0.3473 | 0.2929 | 0.4266 | 48.52% | 37.39% | 0.981238 |
| Detector de empate: probabilidades | 0.3891 | 0.0000 | 0.0000 | 0.0000 | 52.36% | 0.00% | 0.980380 |
| Detector de empate: umbral F1 | 0.4720 | 0.3382 | 0.2951 | 0.3961 | 48.46% | 34.46% | 0.980380 |

validacion: frente al original, Δ F1 macro **+0.0816**, IC 95% [+0.0756, +0.0878]; Δ accuracy **-4.33 puntos porcentuales**, IC [-5.04, -3.61].

test_revisado: frente al original, Δ F1 macro **+0.0829**, IC 95% [+0.0730, +0.0926]; Δ accuracy **-3.90 puntos porcentuales**, IC [-5.06, -2.75].

## Alcance de la comparación

Los intervalos usan 5.000 bootstrap por bloques de temporada/año ISO/semana. Son intervalos exploratorios condicionados al procedimiento; no corrigen la selección de la familia ni toda la multiplicidad de la exploración. No se mezclan las predicciones de decisión con el cálculo de log loss ni se sustituyen probabilidades por etiquetas 0/1.

Ejecución: `b58cd51d530c4d54`. Resultados, configuraciones por fold, selección congelada, predicciones y modelos guardados en `scripts/data/draw_runs/b58cd51d530c4d54`. El predictor exportado ofrece `predict_proba(frame)` y `predict(frame)`; las dos salidas pueden diferir de argmax cuando el objetivo de decisión es F1.

Reproducción: `.venv/bin/python scripts/experiment_draws.py`.

![Comparación de F1 y accuracy en ambas ramas](figures/empates_revision.png)
