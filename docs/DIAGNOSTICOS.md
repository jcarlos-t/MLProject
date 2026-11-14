# Diagnósticos de resultados

Ejecución: `39622169225decc0`.

## Consenso contra Bet365

Se recalcula la regla original (EV > 2 %, una unidad). Los siguientes análisis son descriptivos; no cambian precios, umbral ni selección.

- test: 181 apuestas, beneficio -84.15, ROI -46.49 %. Las diez mayores ganancias suman 62.75 unidades.
  Cuotas de esas ganancias: 5.50–12.00. Sin esas diez apuestas el ROI es -85.91 %.
  Cuota mediana apostada: 9.00; aciertos: 8.3 % de las apuestas.
  Bet365 ofrece el mejor precio publicado en el 67 % de estas apuestas y lo supera en 0.

- validacion: 809 apuestas, beneficio 205.23, ROI 25.37 %. Las diez mayores ganancias suman 178.00 unidades.
  Cuotas de esas ganancias: 14.00–26.00. Sin esas diez apuestas el ROI es 3.41 %.
  Cuota mediana apostada: 7.50; aciertos: 16.7 % de las apuestas.
  Bet365 ofrece el mejor precio publicado en el 73 % de estas apuestas y lo supera en 5.

Las cinco mayores ganancias de validación corresponden a partidos reales con un resultado inesperado:

- 2019-10-05 · Bayern Munich – Hoffenheim · resultado A · cuota 26.00 (mejor precio publicado 30.00)
- 2019-12-06 · Ajax – Willem II · resultado A · cuota 23.00 (mejor precio publicado 24.00)
- 2018-10-20 · Real Madrid – Levante · resultado A · cuota 19.00 (mejor precio publicado 20.95)
- 2019-01-29 · Newcastle – Man City · resultado H · cuota 19.00 (mejor precio publicado 19.00)
- 2019-09-25 · Paris SG – Reims · resultado A · cuota 19.00 (mejor precio publicado 23.00)

La regla apuesta casi siempre a resultados poco probables en los que Bet365 paga más que las demás casas. La ganancia de validación depende de unas pocas sorpresas y desaparece casi por completo al quitarlas; en test la misma regla pierde. Las cuotas apostadas no superan, salvo en casos contados, el mejor precio publicado, así que no hay indicios de cuotas erróneas: es variabilidad de apuestas de cuota alta. Los crudos no permiten acreditar que esos precios estuvieran disponibles a la vez, por lo que tampoco es una rentabilidad replicable.

## Recalibración de la variante balanceada

Comparación en las mismas cinco temporadas: 2019/20–2023/24. No comparar directamente con métricas de seis temporadas.

```text
            candidate       variant  log_loss  ECE_masa
 con/LR/balance_recal       balance  0.969014  0.022972
 con/LR/balance_recal balance_recal  0.967469  0.008830
con/HGB/balance_recal       balance  0.973792  0.029489
con/HGB/balance_recal balance_recal  0.970077  0.011652
 sin/LR/balance_recal       balance  0.981496  0.015728
 sin/LR/balance_recal balance_recal  0.981767  0.011405
sin/HGB/balance_recal       balance  0.981877  0.020236
sin/HGB/balance_recal balance_recal  0.981091  0.009908
```

## Disponibilidad y calibración

Las valoraciones usadas en 2024/25 tienen 504 días de antigüedad media. No hay una versión con fecha acreditada entre septiembre de 2023 y julio de 2025: la fuente de FC24 solo incluye la versión de lanzamiento, y los archivos de FC25 de esa época no traen fecha de publicación ni la alineación titular con la que se calcula el once. Por eso se conserva la versión anterior verificable y se declara la antigüedad.

El calibrador de diferencia de goles aprende con predicciones de una temporada posterior al entrenamiento base. Luego se reajusta el regresor con todo el pasado. No hay etiquetas futuras en ese ajuste, pero puede cambiar la distribución de la entrada al calibrador; debe evaluarse en datos futuros.
