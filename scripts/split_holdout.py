"""Divide dataset_final.csv en train/val y test holdout por temporada.

Train/val : 2014/15-2023/24 (los folds de validacion son 2018/19-2023/24; cada uno entrena con lo anterior).
Test      : 2024/25 y 2025/26 (no se usa para entrenar, validar ni elegir nada; solo en 08_modelo_final.ipynb, Paso 8).

Salida: scripts/data/split/train_val.csv, test.csv y split_info.json. Uso: python scripts/split_holdout.py"""
import json
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent / 'data'
OUT = DATA / 'split'
TEST_SEASONS = [2425, 2526]

df = pd.read_csv(DATA / 'dataset_final.csv')
es_test = df['Season'].isin(TEST_SEASONS)
OUT.mkdir(exist_ok=True)
df[~es_test].to_csv(OUT / 'train_val.csv', index=False)
df[es_test].to_csv(OUT / 'test.csv', index=False)
info = dict(origen='dataset_final.csv', temporadas_test=TEST_SEASONS, filas_total=int(len(df)), filas_train_val=int((~es_test).sum()), filas_test=int(es_test.sum()),
            temporadas_train_val=[int(s) for s in sorted(df.loc[~es_test, 'Season'].unique())],
            partidos_por_temporada={int(s): int(n) for s, n in df['Season'].value_counts().sort_index().items()})
json.dump(info, open(OUT / 'split_info.json', 'w'), indent=1)
assert not set(df.loc[~es_test, 'Season']) & set(TEST_SEASONS) and df.loc[~es_test, 'Season'].max() < min(TEST_SEASONS)
print(f"train_val: {info['filas_train_val']} filas | test: {info['filas_test']} filas | guardado en {OUT}")
