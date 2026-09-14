#!/usr/bin/env python3
"""Procesamiento de datos sin EDA: consolidación + join por fecha y equipo.

Este script SOLO cruza/consolida tablas para construir la base unificada.
NO hace EDA ni analytics (no hay limpieza de calidad, ni z-scores relativos,
ni mutual information, ni modelos, ni ploteos / decisiones de análisis).
Eso queda en los notebooks de script/eda.

Pasos:
    1. Consolida CSVs de partidos        -> data/matches.csv
    2. Consolida snapshots de jugadores  -> data/players/raw/players_all.csv
    3. Normaliza snapshots a esquema estándar -> data/players/normalized/players_XXXX.csv
    4. Agrega squad quality por equipo   -> data/players/squad_quality_players.csv
    5. Mapea club EA -> equipo football-data    -> data/players/club_map.csv
    6. Join matches x (Season/Div/Team)  -> data/matches_sq.csv  (base unificada)

Usage:
    python scripts/download/process_data.py
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA = BASE_DIR / "data"
RAW = DATA / "raw"
PLAYERS = DATA / "players"
PLAYERS_RAW = PLAYERS / "raw"
NORM_DIR = PLAYERS / "normalized"

SEASONS = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
LEAGUES = ["E0", "D1", "SP1", "I1", "F1", "N1", "P1", "T1", "B1"]

# Snapshot (año) -> (fichero, columnas del esquema heterogéneo)
PLAYER_SPECS = {
    2017: ("2017.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2018: ("2018.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2019: ("2019.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2020: ("2020.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2021: ("2021.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2022: ("2022.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2023: ("2023.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2024: ("2024.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
    2025: ("2025.csv", dict(club="club", league="league", overall="ovr", pos="position")),
    2026: ("2026.csv", dict(club="club_name", league="league_name", overall="overall", pos="player_positions")),
}

LINE_GROUPS = ("gk", "def", "mid", "att")
SQUAD_FEAT = (["n_players", "overall_mean", "overall_max", "top11_mean", "top15_mean"]
              + [g for grp in LINE_GROUPS
                 for g in (f"{grp}_mean", f"{grp}_top_mean", f"{grp}_top3_mean")])

SEASON_SMAP = {j: f"{j - 2001:02d}{j - 2000:02d}" for j in range(2017, 2027)}


# ---------------------------------------------------------------------------
# Step 1: Consolidate match CSVs (solo join)
# ---------------------------------------------------------------------------
def consolidate_matches():
    print("=== Step 1: Consolidating match CSVs ===")
    rows = []
    for season in SEASONS:
        for league in LEAGUES:
            f = RAW / f"{league}_{season}.csv"
            if not f.exists():
                print(f"  skip {f.name} (not found)")
                continue
            try:
                df = pd.read_csv(f, dtype=str)
            except Exception as e:
                print(f"  skip {f.name}: {e}")
                continue
            if df.empty:
                continue
            df["Season"] = season
            df["Div"] = league
            rows.append(df)

    if not rows:
        print("  ERROR: no CSVs found. Run fetch_data.py first.")
        return False

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA / "matches.csv", index=False)
    print(f"  matches.csv: {len(out):,} rows")
    return True


# ---------------------------------------------------------------------------
# Step 2: Consolidate raw player snapshots (solo join, union de columnas)
# ---------------------------------------------------------------------------
def consolidate_players():
    print("=== Step 2: Consolidating raw player snapshots ===")
    frames = []
    for p in sorted(PLAYERS_RAW.glob("*.csv")):
        if p.name == "players_all.csv":
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
            continue
        df["season"] = int(p.stem)
        df["_src"] = p.name
        frames.append(df)

    if not frames:
        print("  ERROR: no snapshot CSVs found. Run fetch_data.py first.")
        return False

    out = pd.concat(frames, ignore_index=True, sort=False)
    PLAYERS_RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(PLAYERS_RAW / "players_all.csv", index=False)
    print(f"  players_all.csv: {len(out):,} rows | {len(out.columns)} columnas (union de esquemas)")
    return True


# ---------------------------------------------------------------------------
# Step 3: Normalizar snapshots a esquema estandar (season, club, league, overall, pos)
# ---------------------------------------------------------------------------
def normalize_players():
    print("=== Step 3: Normalizing player snapshots ===")
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for season, (fname, cmap) in PLAYER_SPECS.items():
        src = PLAYERS_RAW / fname
        if not src.exists():
            print(f"  skip {fname} (not found)")
            continue
        df = pd.read_csv(src, dtype=str)
        out = pd.DataFrame({"season": [season] * len(df)})
        for col in ("club", "league", "overall", "pos"):
            out[col] = df[cmap[col]].astype(str).str.strip()
        out["overall"] = pd.to_numeric(out["overall"].replace({"": None, "nan": None}), errors="coerce")
        out["overall"] = out["overall"].astype("Int64")
        out = out.dropna(subset=["overall"])
        out["pos"] = out["pos"].str.split(",").str[0].str.strip()
        out = out[out["club"] != ""].dropna(subset=["club"])
        out.to_csv(NORM_DIR / f"players_{season}.csv", index=False)
        frames.append(out)
        print(f"  {season}: {len(out):,}")

    if not frames:
        print("  ERROR: no snapshot CSVs found. Run fetch_data.py first.")
        return False
    players = pd.concat(frames, ignore_index=True)
    print(f"  normalized total: {len(players):,}")
    return True


# ---------------------------------------------------------------------------
# Step 4: Agregacion de squad quality por (season, club)
# ---------------------------------------------------------------------------
def squad_quality():
    print("=== Step 4: Aggregating squad quality per (season, club) ===")
    files = sorted(NORM_DIR.glob("players_*.csv"))
    if not files:
        print("  ERROR: no normalized snapshots. Run step 3 (normalize_players).")
        return False
    players = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    POS_GROUPS = {
        "GK": "gk", "CB": "def", "LB": "def", "LWB": "def", "RB": "def", "RWB": "def",
        "CDM": "mid", "CM": "mid", "CAM": "mid", "LM": "mid", "RM": "mid",
        "LW": "att", "RW": "att", "ST": "att", "CF": "att", "RF": "att", "LF": "att",
    }

    def agg_grp(g):
        ov = g["overall"].astype(int)
        top11 = ov.nlargest(11)
        top15 = ov.nlargest(15)
        line = {
            "league_raw": g["league"].mode().iloc[0] if g["league"].nunique() else "",
            "n_players": len(g),
            "overall_mean": ov.mean(),
            "overall_max": ov.max(),
            "top11_mean": top11.mean(),
            "top15_mean": top15.mean(),
        }
        for grp in LINE_GROUPS:
            s = ov[g["pos_group"] == grp]
            line[f"{grp}_n"] = len(s)
            line[f"{grp}_mean"] = s.mean() if len(s) else None
            line[f"{grp}_top_mean"] = s.max() if len(s) else None
            line[f"{grp}_top3_mean"] = s.nlargest(3).mean() if len(s) else None
        return pd.Series(line)

    allp = players.assign(pos_group=players["pos"].str.upper().map(POS_GROUPS).fillna("other"))
    sq = allp.groupby(["season", "club"]).apply(agg_grp, include_groups=False).reset_index()
    sq.to_csv(PLAYERS / "squad_quality_players.csv", index=False)
    print(f"  squad_quality_players.csv: {sq.shape[0]:,} (season, club)")
    return True


# ---------------------------------------------------------------------------
# Step 5: Mapeo de clubes EA -> (Div:Team) de football-data
# ---------------------------------------------------------------------------
def map_clubs():
    print("=== Step 5: Mapping EA clubs -> football-data teams ===")
    m_path = DATA / "matches.csv"
    sq_path = PLAYERS / "squad_quality_players.csv"
    if not m_path.exists() or not sq_path.exists():
        print("  ERROR: need matches.csv and squad_quality_players.csv (steps 1 and 4).")
        return False

    matches = pd.read_csv(m_path, dtype={"Div": str, "Season": str}, low_memory=False)
    sq = pd.read_csv(sq_path)

    TURK_FOLD = str.maketrans({"İ": "i", "ı": "i", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
                               "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C",
                               "ß": "ss"})

    def norm(s):
        s = str(s).replace("'", "").replace("\u2019", "")
        s = s.translate(TURK_FOLD)
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        s = s.lower()
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()

    def token_subset_match(ea_norm, canon_list, allow_prefix=True):
        et = ea_norm.split()
        best = None
        for ct, ref in canon_list:
            cn = ct.split()
            if not cn or len(cn) > len(et):
                continue
            ok = all(any(e == tok or (allow_prefix and e.startswith(tok)) for e in et) for tok in cn)
            if ok and (best is None or len(ct) > len(best[0])):
                best = (ct, ref)
        return best

    OVERRIDES = {
        "athletic club": "SP1:Ath Bilbao", "athletic club de bilbao": "SP1:Ath Bilbao",
        "athletic bilbao": "SP1:Ath Bilbao", "athletico madrid": "SP1:Ath Madrid",
        "atletico de madrid": "SP1:Ath Madrid", "atletico madrid": "SP1:Ath Madrid",
        "club atletico de madrid": "SP1:Ath Madrid", "espanyol": "SP1:Espanol",
        "real betis": "SP1:Betis", "deportivo alaves": "SP1:Alaves",
        "real sociedad": "SP1:Sociedad", "real valladolid cf": "SP1:Valladolid",
        "girona fc": "SP1:Girona", "granada cf": "SP1:Granada", "leganes": "SP1:Leganes",
        "ud las palmas": "SP1:Las Palmas", "ud almeria": "SP1:Almeria",
        "elche cf": "SP1:Elche", "sd eibar": "SP1:Eibar",
        "deportivo de la coruna": "SP1:La Coruna", "rc celta de vigo": "SP1:Celta",
        "real sp gijon": "SP1:Sp Gijon", "rcd mallorca": "SP1:Mallorca",
        "rayo vallecano": "SP1:Vallecano", "barcelona": "SP1:Barcelona",
        "fc barcelona": "SP1:Barcelona", "real madrid": "SP1:Real Madrid",
        "sevilla fc": "SP1:Sevilla", "valencia cf": "SP1:Valencia",
        "cf villarreal": "SP1:Villarreal", "osasuna": "SP1:Osasuna",
        "betis": "SP1:Betis", "celta": "SP1:Celta",
        "manchester city": "E0:Man City", "manchester united": "E0:Man United",
        "tottenham hotspur": "E0:Tottenham", "west ham united": "E0:West Ham",
        "leeds united": "E0:Leeds", "newcastle united": "E0:Newcastle",
        "wolverhampton wanderers": "E0:Wolves", "brighton & hove albion": "E0:Brighton",
        "nottingham forest": "E0:Nott'm Forest", "norwich city": "E0:Norwich",
        "southampton": "E0:Southampton", "afc bournemouth": "E0:Bournemouth",
        "crystal palace": "E0:Crystal Palace", "leicester city": "E0:Leicester",
        "aston villa": "E0:Aston Villa", "fulham": "E0:Fulham",
        "sheffield united": "E0:Sheffield United", "stoke city": "E0:Stoke",
        "middlesbrough": "E0:Middlesbrough", "watford": "E0:Watford",
        "west bromwich albion": "E0:West Brom", "luton town": "E0:Luton",
        "burnley": "E0:Burnley", "brentford": "E0:Brentford",
        "arsenal": "E0:Arsenal", "chelsea": "E0:Chelsea", "everton": "E0:Everton",
        "liverpool": "E0:Liverpool", "wolves": "E0:Wolves", "leeds": "E0:Leeds",
        "bayern munchen": "D1:Bayern Munich", "fc bayern munchen": "D1:Bayern Munich",
        "borussia dortmund": "D1:Dortmund", "rb leipzig": "D1:RB Leipzig",
        "bayer 04 leverkusen": "D1:Leverkusen", "bayer leverkusen": "D1:Leverkusen",
        "eintracht frankfurt": "D1:Ein Frankfurt", "fc koln": "D1:FC Koln",
        "borussia monchengladbach": "D1:M'gladbach", "vfl wolfsburg": "D1:Wolfsburg",
        "vfb stuttgart": "D1:Stuttgart", "fc schalke 04": "D1:Schalke 04",
        "werder bremen": "D1:Werder Bremen", "hertha bsc": "D1:Hertha",
        "holstein kiel": "D1:Holstein Kiel", "1. fsv mainz 05": "D1:Mainz",
        "tsg hoffenheim": "D1:Hoffenheim", "fc st. pauli": "D1:St Pauli",
        "frankfurt": "D1:Ein Frankfurt", "dortmund": "D1:Dortmund",
        "bayern munich": "D1:Bayern Munich", "stuttgart": "D1:Stuttgart",
        "wolfsburg": "D1:Wolfsburg", "freiburg": "D1:Freiburg", "mainz": "D1:Mainz",
        "leverkusen": "D1:Leverkusen", "hoffenheim": "D1:Hoffenheim",
        "ac milan": "I1:Milan", "as roma": "I1:Roma", "ssc napoli": "I1:Napoli",
        "inter": "I1:Inter", "internazionale": "I1:Inter", "juventus": "I1:Juventus",
        "ss lazio": "I1:Lazio", "acf fiorentina": "I1:Fiorentina", "atalanta": "I1:Atalanta",
        "torino": "I1:Torino", "genoa cfc": "I1:Genoa", "cagliari calcio": "I1:Cagliari",
        "bologna fc 1909": "I1:Bologna", "sassuolo": "I1:Sassuolo",
        "hellas verona": "I1:Verona", "venezia fc": "I1:Venezia",
        "milan": "I1:Milan", "roma": "I1:Roma", "lazio": "I1:Lazio",
        "fiorentina": "I1:Fiorentina", "napoli": "I1:Napoli",
        "paris saint-germain": "F1:Paris SG", "psg": "F1:Paris SG",
        "as monaco": "F1:Monaco", "olympique lyonnais": "F1:Lyon",
        "olympique de marseille": "F1:Marseille", "losc lille": "F1:Lille",
        "rc lens": "F1:Lens", "stade rennais": "F1:Rennes", "fc nantes": "F1:Nantes",
        "as saint-etienne": "F1:St Etienne", "ogc nice": "F1:Nice",
        "toulouse fc": "F1:Toulouse", "stade brestois 29": "F1:Brest",
        "lyon": "F1:Lyon", "marseille": "F1:Marseille", "lille": "F1:Lille",
        "lens": "F1:Lens", "rennes": "F1:Rennes", "nantes": "F1:Nantes",
        "montpellier": "F1:Montpellier", "nice": "F1:Nice", "toulouse": "F1:Toulouse",
        "brest": "F1:Brest", "reims": "F1:Reims",
        "ajax": "N1:Ajax", "psv eindhoven": "N1:PSV Eindhoven", "feyenoord": "N1:Feyenoord",
        "az alkmaar": "N1:AZ Alkmaar", "fc twente": "N1:Twente",
        "sc heerenveen": "N1:Heerenveen", "vitesse": "N1:Vitesse",
        "utrecht": "N1:Utrecht", "groningen": "N1:Groningen", "twente": "N1:Twente",
        "benfica": "P1:Benfica", "sporting cp": "P1:Sp Lisbon", "fc porto": "P1:Porto",
        "porto": "P1:Porto", "sc braga": "P1:Sp Braga", "braga": "P1:Sp Braga",
        "galatasaray": "T1:Galatasaray", "fenerbahce": "T1:Fenerbahce",
        "besiktas": "T1:Besiktas", "trabzonspor": "T1:Trabzonspor",
        "istanbul basaksehir": "T1:Buyuksehyr", "alanyaspor": "T1:Alanyaspor",
        "antalyaspor": "T1:Antalyaspor", "konyaspor": "T1:Konyaspor",
        "sivasspor": "T1:Sivasspor", "kayserispor": "T1:Kayserispor",
        "anderlecht": "B1:Anderlecht", "club brugge": "B1:Club Brugge",
        "krc genk": "B1:Genk", "genk": "B1:Genk", "gent": "B1:Gent",
        "standard liege": "B1:Standard", "charleroi": "B1:Charleroi",
        "antwerp": "B1:Antwerp", "mechelen": "B1:Mechelen", "kortrijk": "B1:Kortrijk",
    }

    NON_TARGET = {
        "salzburg", "fc red bull salzburg", "austria klagenfurt", "austria lustenau",
        "austria wien", "rapid wien", "sturm graz", "dynamo kyiv", "shakhtar donetsk",
        "al hilal", "al ittihad", "al nassr", "al ahli sfc",
    }
    LEAGUE_DIV_EXACT = {
        "english premier league": "E0", "premier league": "E0",
        "german 1. bundesliga": "D1", "bundesliga": "D1",
        "spain primera division": "SP1", "la liga": "SP1", "laliga ea sports": "SP1",
        "italian serie a": "I1", "french ligue 1": "F1", "ligue 1": "F1",
        "holland eredivisie": "N1", "eredivisie": "N1",
        "portuguese liga zon sagres": "P1", "liga portugal": "P1", "primeira liga": "P1",
        "turkish super lig": "T1", "super lig": "T1", "trendyol super lig": "T1",
        "belgian jupiler pro league": "B1", "jupiler pro league": "B1", "1a pro league": "B1",
    }
    OVERRIDES_NORM = {norm(k): v for k, v in OVERRIDES.items()}
    LEAGUE_DIV_NORM = {norm(k): v for k, v in LEAGUE_DIV_EXACT.items()}
    AMBIG_LABELS = {norm("Serie A"), norm("Serie A TIM")}

    def _div_of(league_raw):
        l = norm(league_raw)
        if l in LEAGUE_DIV_NORM:
            d = LEAGUE_DIV_NORM[l]
            return None if (d == "I1" and l in AMBIG_LABELS) else d
        return None

    sqm = sq.copy()
    sqm["div"] = sqm["league_raw"].map(_div_of)
    print(f"  clubs con div de liga objetivo: {sqm['div'].notna().sum()} / {len(sqm)}")

    mcl = matches[["Div", "Season", "HomeTeam", "AwayTeam"]].copy()
    canonical_teams = sorted(set(mcl["Div"].astype(str) + ":" + mcl["HomeTeam"]) |
                             set(mcl["Div"].astype(str) + ":" + mcl["AwayTeam"]))
    canon_ref = [(norm(t.split(":", 1)[1]), t) for t in canonical_teams]
    per_div = {}
    for c in canonical_teams:
        div, team = c.split(":", 1)
        per_div.setdefault(div, []).append((norm(team), team))
    global_exact = {norm(t.split(":", 1)[1]): t for t in canonical_teams}

    rows, review = [], []
    for (div, season, club), _ in sqm.groupby(["div", "season", "club"], dropna=False):
        key = norm(club)
        if key in NON_TARGET:
            continue
        if key in OVERRIDES_NORM:
            rows.append((season, club, OVERRIDES_NORM[key], 100)); continue
        if key in global_exact:
            rows.append((season, club, global_exact[key], 100)); continue
        if pd.isna(div):
            hit = token_subset_match(key, canon_ref, allow_prefix=False)
            if hit:
                rows.append((season, club, hit[1], 100))
            else:
                review.append((season, club, "", "", 0))
            continue
        pool = per_div.get(div, [])
        if not pool:
            review.append((season, club, div, "", 0)); continue
        hit = token_subset_match(key, pool)
        ref = f"{div}:{hit[1]}" if hit else None
        score = 100 if hit else 0
        if ref is None:
            try:
                from rapidfuzz import fuzz, process as rprocess
            except ImportError:
                fuzz = rprocess = None
            if rprocess is not None:
                cand = rprocess.extractOne(key, [p[0] for p in pool], scorer=fuzz.WRatio, score_cutoff=85)
                if cand is not None:
                    best, score, idx = cand
                    ref = f"{div}:{pool[idx][1]}"
        if ref is None:
            review.append((season, club, div, "", 0)); continue
        rows.append((season, club, ref, score))

    cm = pd.DataFrame(rows, columns=["season", "club", "div_team", "score"])
    cm[["div", "team"]] = cm["div_team"].str.split(":", expand=True)
    cm.to_csv(PLAYERS / "club_map.csv", index=False)
    low = cm[cm["score"] < 100]
    print(f"  club_map.csv: {len(cm):,} mapeos | exact/override: {len(cm) - len(low):,} | fuzzy: {len(low):,} | unmatched: {len(review):,}")
    return True


# ---------------------------------------------------------------------------
# Step 6: Join de matches con squad quality por (Fecha/Temporada + Equipo)
# ---------------------------------------------------------------------------
def join_matches_squad():
    print("=== Step 6: Join matches x squad (fecha/temporada + equipo) ===")
    m_path = DATA / "matches.csv"
    sq_path = PLAYERS / "squad_quality_players.csv"
    cm_path = PLAYERS / "club_map.csv"
    for p, name in [(m_path, "matches.csv"), (sq_path, "squad_quality_players.csv"), (cm_path, "club_map.csv")]:
        if not p.exists():
            print(f"  ERROR: {name} no encontrado. Ejecuta pasos previos.")
            return False

    m = pd.read_csv(m_path, dtype={"Div": str, "Season": str}, low_memory=False)
    sq = pd.read_csv(sq_path)
    cm = pd.read_csv(cm_path)

    def season_code(s):
        return s.astype(int).map(SEASON_SMAP).astype(str)

    sqj = sq.copy()
    sqj["season_code"] = season_code(sqj["season"])
    sqj["skc"] = sqj["season_code"] + "|" + sqj["club"].astype(str)

    cmj = cm.copy()
    cmj["season_code"] = season_code(cmj["season"])
    cmj["skc"] = cmj["season_code"] + "|" + cmj["club"].astype(str)
    cmj["skm"] = cmj["season_code"] + "|" + cmj["div"] + "|" + cmj["team"]

    # team-slot (Season|Div|Team) -> valores de squad
    joined = sqj.merge(cmj[["skc", "skm"]], on="skc", how="inner")
    sq_by_skm = {}
    for r in joined.itertuples(index=False):
        sq_by_skm[r.skm] = [getattr(r, f) for f in SQUAD_FEAT]

    for side in ("HomeTeam", "AwayTeam"):
        key = m["Season"] + "|" + m["Div"] + "|" + m[side]
        vals = key.map(sq_by_skm)
        for f in SQUAD_FEAT:
            m[f"{side}_sq_{f}"] = pd.NA
        m[f"{side}_sq_mapped"] = vals.notna().astype(int)
        hit = m.index[vals.notna()]
        if len(hit):
            arr = np.array(vals.loc[hit].tolist())
            for j, f in enumerate(SQUAD_FEAT):
                m.loc[hit, f"{side}_sq_{f}"] = arr[:, j]

    m.to_csv(DATA / "matches_sq.csv", index=False)
    hits = (m["HomeTeam_sq_mapped"] == 1).sum() + (m["AwayTeam_sq_mapped"] == 1).sum()
    total = len(m) * 2
    print(f"  matches_sq.csv: {len(m):,} filas | {len(m.columns)} columnas | cobertura: {hits / total:.4%}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    done = [
        consolidate_matches(),
        consolidate_players(),
        normalize_players(),
        squad_quality(),
        map_clubs(),
        join_matches_squad(),
    ]
    if all(done):
        print()
        print("=== Listo: base unificada en data/matches_sq.csv (sin EDA, solo joins). ===")
    else:
        print()
        print("=== ERRORES: revisa el paso fallido (data cruda en scripts/data, ejecuta fetch_data.py primero). ===")