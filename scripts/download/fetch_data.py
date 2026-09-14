#!/usr/bin/env python3
"""Download match data from football-data.co.uk and player ratings from Kaggle.

Usage:
    python scripts/fetch_data.py

Outputs:
    data/raw/<DIV>_<SEASON>.csv   (90 match CSVs from football-data.co.uk)
    data/players/raw/<file>.csv   (10 player rating snapshots from Kaggle)

Requirements:
    pip install kagglehub pandas
"""
import os
import sys
import urllib.request
import ssl
import shutil

try:
    import pandas as pd
except ImportError:
    pd = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PLAYERS_RAW = os.path.join(BASE_DIR, "data", "players", "raw")

SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
LEAGUES = ["E0", "D1", "SP1", "I1", "F1", "N1", "P1", "T1", "B1"]

# (dataset, source_file, local_name, fifa_version_filter)
# fifa_version_filter: entero para filtrar la columna `fifa_version` del CSV, o None
KAGGLE_DATASETS = [
    ("luisfucros/fifa-players", "players_17.csv", "2017.csv", None),
    ("luisfucros/fifa-players", "players_18.csv", "2018.csv", None),
    ("luisfucros/fifa-players", "players_19.csv", "2019.csv", None),
    ("luisfucros/fifa-players", "players_20.csv", "2020.csv", None),
    ("luisfucros/fifa-players", "players_21.csv", "2021.csv", None),
    ("luisfucros/fifa-players", "players_22.csv", "2022.csv", None),
    ("stefanoleone992/ea-sports-fc-24-complete-player-dataset", "male_players.csv", "2023.csv", 23),
    ("stefanoleone992/ea-sports-fc-24-complete-player-dataset", "male_players.csv", "2024.csv", 24),
    ("akshyakumarkc/fifa-25-player-ratings", "players_info.csv", "2025.csv", None),
    ("rovnez/fc-26-fifa-26-player-data", "FC26_20250921.csv", "2026.csv", None),
]

HEADERS = {"User-Agent": "Mozilla/5.0"}


def download_match_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    ctx = ssl.create_default_context()
    total = len(SEASONS) * len(LEAGUES)
    count = 0
    for season in SEASONS:
        for league in LEAGUES:
            count += 1
            url = f"https://football-data.co.uk/mmz4281/{season}/{league}.csv"
            out = os.path.join(RAW_DIR, f"{league}_{season}.csv")
            if os.path.exists(out):
                print(f"[{count}/{total}] skip {out} (exists)")
                continue
            print(f"[{count}/{total}] {url} -> {out}")
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, context=ctx) as resp:
                    data = resp.read()
                with open(out, "wb") as f:
                    f.write(data)
                print(f"  OK ({len(data)} bytes)")
            except Exception as e:
                print(f"  FAILED: {e}")


def download_player_data():
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        print("Skipping Kaggle player data download.")
        return

    os.makedirs(PLAYERS_RAW, exist_ok=True)
    downloaded = {}

    for dataset, filename, local_name, fifa_filter in KAGGLE_DATASETS:
        out = os.path.join(PLAYERS_RAW, local_name)
        if os.path.exists(out):
            print(f"skip {out} (exists)")
            continue

        cache_key = (dataset, filename)
        if cache_key not in downloaded:
            print(f"downloading {dataset}...")
            try:
                path = kagglehub.dataset_download(dataset)
                downloaded[cache_key] = path
            except Exception as e:
                err = str(e)
                if "403" in err or "permission" in err.lower():
                    print(f"  FAILED: {dataset} requiere autenticacion de Kaggle.")
                    print(f"  Solucion: ejecuta 'kaggle datasets download -d {dataset}' con tu API key (~/.kaggle/kaggle.json)")
                else:
                    print(f"  FAILED to download {dataset}: {e}")
                continue
        else:
            path = downloaded[cache_key]

        src = os.path.join(path, filename)
        if not os.path.exists(src):
            print(f"  WARNING: {filename} not found in {path}")
            continue

        if fifa_filter is not None:
            if pd is None:
                print(f"  WARNING: pandas requerido para filtrar {filename}; copiando sin filtrar")
                shutil.copy2(src, out)
            else:
                df = pd.read_csv(src)
                df = df[df["fifa_version"].astype(int) == fifa_filter]
                df.to_csv(out, index=False)
                print(f"  filtered fifa_version=={fifa_filter} -> {out} ({len(df):,} rows)")
            continue

        shutil.copy2(src, out)
        print(f"  copied to {out}")


if __name__ == "__main__":
    print("=== Downloading match data from football-data.co.uk ===")
    download_match_data()
    print()
    print("=== Downloading player ratings from Kaggle ===")
    download_player_data()
    print()
    print("Done.")
