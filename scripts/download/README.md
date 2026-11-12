# Fuentes, integridad y disponibilidad

`fetch_data.py` descarga 117 CSV de partidos (9 ligas × 13 temporadas, incluyendo calentamiento) y 12 snapshots de jugadores. No refresca silenciosamente archivos existentes. Una descarga fallida produce un estado de error y detiene la reproducción.

`manifest.py --check` exige las 129 huellas originales antes de procesar. `process_data.py` consolida, normaliza y mapea identidades; se detiene al fallar un paso. Su cruce de plantilla busca la última versión estrictamente anterior al partido, no la versión de toda la temporada.

El dataset científico se reconstruye desde esos crudos mediante `scripts/run_revision.py --stage prepare`. Conserva faltantes estructurales, cruza las habilidades por fecha, calcula historia sin resultados presentes/futuros y prepara las particiones. No utilizar derivados del archivo histórico para entrenar una versión corregida.

`snapshot_availability.json` conserva fecha, huella y evidencia de cada snapshot. Para FIFA 15–22 se contrastó el identificador de actualización incluido en `player_url` y todos los ratings overall compartidos con la misma versión del CSV original de FC24 (versión 4), que sí publica `update_as_of`. Los nombres de club pueden variar entre fuentes, por lo que se utiliza el identificador de versión y jugador. Las fechas explícitas de FC25 y FC26 son 17/07/2025 y 19/09/2025, respectivamente; el nombre del archivo FC26 alude a su captura del 21/09.

Las cuotas son capturas pre-cierre, no apertura real garantizada. La advertencia de Football-Data sobre Pinnacle desde 23/07/2025 se aplica a la selección de precios y al backtest. Consultar https://football-data.co.uk/data.php y https://football-data.co.uk/notes.txt.

No se publican datos crudos. Para reproducir una nueva descarga que difiera del manifiesto, primero se debe documentar la nueva versión de fuente y su evaluación; no sobrescribir las huellas solo para hacer pasar la comprobación.
