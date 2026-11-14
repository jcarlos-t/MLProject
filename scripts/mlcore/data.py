"""Inputs verified by hash, point-in-time features, and immutable run identity."""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'scripts/data'
TEST = (2425, 2526)
FOLDS = (1819, 1920, 2021, 2122, 2223, 2324)
PIN_BAD_FROM = pd.Timestamp('2025-07-23')
KEY = ['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam']
STATS = ['GF', 'GA', 'Pts', 'CS', 'Shots', 'SOT', 'Corners', 'Fouls', 'Cards']
SK_OLD = {
    'crossing': 'attacking_crossing', 'finishing': 'attacking_finishing',
    'heading_accuracy': 'attacking_heading_accuracy', 'short_passing': 'attacking_short_passing',
    'volleys': 'attacking_volleys', 'dribbling': 'skill_dribbling', 'curve': 'skill_curve',
    'fk_accuracy': 'skill_fk_accuracy', 'long_passing': 'skill_long_passing',
    'ball_control': 'skill_ball_control', 'acceleration': 'movement_acceleration',
    'sprint_speed': 'movement_sprint_speed', 'agility': 'movement_agility',
    'reactions': 'movement_reactions', 'balance': 'movement_balance',
    'shot_power': 'power_shot_power', 'jumping': 'power_jumping', 'stamina': 'power_stamina',
    'strength': 'power_strength', 'long_shots': 'power_long_shots',
    'aggression': 'mentality_aggression', 'interceptions': 'mentality_interceptions',
    'positioning': 'mentality_positioning', 'vision': 'mentality_vision',
    'penalties': 'mentality_penalties', 'defensive_awareness': 'defending_marking_awareness',
    'standing_tackle': 'defending_standing_tackle', 'sliding_tackle': 'defending_sliding_tackle',
    'gk_diving': 'goalkeeping_diving', 'gk_handling': 'goalkeeping_handling',
    'gk_kicking': 'goalkeeping_kicking', 'gk_positioning': 'goalkeeping_positioning',
    'gk_reflexes': 'goalkeeping_reflexes',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_inputs():
    expected = json.loads((ROOT / 'scripts/download/data_manifest.json').read_text())['archivos']
    actual = {str(p.relative_to(DATA)): sha(p)
              for folder in ('raw', 'players/raw') for p in (DATA / folder).glob('*.csv')
              if p.name != 'players_all.csv'}
    if expected != actual:
        raise RuntimeError('Crudos distintos del manifiesto. Revisar antes de generar resultados.')
    availability = json.loads((ROOT / 'scripts/download/snapshot_availability.json').read_text())
    for entry in availability['snapshots']:
        if sha(DATA / f"players/raw/{entry['year']}.csv") != entry['raw_sha256']:
            raise RuntimeError('El snapshot no coincide con la evidencia de su fecha.')
    return expected, availability


def load_matches():
    """Rebuild from all 117 inputs; a missing file is fatal, not a partial dataset."""
    frames = []
    for p in sorted((DATA / 'raw').glob('*.csv')):
        div, season = p.stem.rsplit('_', 1)
        d = pd.read_csv(p, low_memory=False)
        d = d[d['FTR'].isin(['H', 'D', 'A'])].copy()
        d['Div'], d['Season'] = div, int(season)
        d['Date'] = pd.to_datetime(d['Date'], dayfirst=True, format='mixed', errors='raise')
        frames.append(d)
    d = pd.concat(frames, ignore_index=True, sort=False)
    if d.duplicated(KEY).any():
        raise ValueError('Partidos duplicados')
    d = d.sort_values(KEY[2:3] + ['Div', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)
    d['match_id'] = [hashlib.sha256('|'.join(map(str, r)).encode()).hexdigest()[:20]
                     for r in d[KEY].itertuples(index=False, name=None)]
    if d['match_id'].duplicated().any():
        raise ValueError('Colision de identificadores')
    return d


def squad_snapshots(availability):
    """One observation per club/snapshot, without imputing across a future season."""
    cm = pd.read_csv(DATA / 'players/club_map.csv')
    rows = []
    for entry in availability['snapshots']:
        year = entry['year']
        d = pd.read_csv(DATA / f'players/raw/{year}.csv', low_memory=False)
        if year == 2025:
            d = d[['club_name', 'club_position', 'overall_rating'] + list(SK_OLD)].rename(columns={'overall_rating': 'overall'})
        else:
            # Legacy CSVs also have an aggregate `dribbling`; select the individual
            # skill columns before renaming, otherwise duplicate labels are created.
            d = d[['club_name', 'club_position', 'overall'] + list(SK_OLD.values())].rename(columns={v: k for k, v in SK_OLD.items()})
        # The curated identity map is not a predictive variable or an imputation.
        mapping = cm[cm.season.eq(year)][['club', 'div', 'team']]
        d = d.merge(mapping, left_on='club_name', right_on='club', validate='many_to_one')
        for (div, team), g in d.groupby(['div', 'team']):
            xi = g[g.club_position.notna() & ~g.club_position.isin(['SUB', 'RES'])]
            xi = xi.nlargest(11, 'overall') if len(xi) > 11 else xi
            field, gk = xi[xi.club_position.ne('GK')], xi[xi.club_position.eq('GK')]
            r = dict(Div=div, Team=team, available_at=pd.Timestamp(entry['available_at']),
                     snapshot_year=year, SQ=g.overall.nlargest(11).mean(), XI=xi.overall.mean())
            for skill in SK_OLD:
                r['SK_' + skill] = (gk[skill].mean() if skill.startswith('gk_')
                                    else field[skill].nlargest(4).mean())
            rows.append(r)
    return pd.DataFrame(rows).sort_values(['available_at', 'Div', 'Team']).reset_index(drop=True)


def historical_features(matches, squads):
    """Read the state for a whole date before updating it with any of that date's scores."""
    hist = defaultdict(lambda: deque(maxlen=5))
    sums = defaultdict(lambda: np.zeros(len(STATS)))
    counts = defaultdict(lambda: np.zeros(len(STATS)))
    team_sum, team_n = defaultdict(float), defaultdict(int)
    elo, last_season = {}, {}
    ew, ewden = defaultdict(float), defaultdict(float)
    appearances, seen_seasons = defaultdict(list), defaultdict(set)
    current_squad = {}
    snapshot_rows = list(squads.itertuples(index=False))
    cursor = 0
    out = []
    sk_cols = [c for c in squads if c.startswith('SK_')]
    for date, day in matches.groupby('Date', sort=True):
        while cursor < len(snapshot_rows) and snapshot_rows[cursor].available_at < date:
            r = snapshot_rows[cursor]
            current_squad[(r.Div, r.Team)] = r
            cursor += 1
        league_sq = defaultdict(list)
        for (div, _), r in current_squad.items():
            league_sq[div].append(r.SQ)
        for r in day.itertuples():
            keys = [(r.Div, r.HomeTeam), (r.Div, r.AwayTeam)]
            f = dict(match_id=r.match_id)
            evs, ratings = [], []
            for side, key in zip(('H', 'A'), keys):
                old = list(hist[key])
                prior = np.divide(sums[r.Div], counts[r.Div],
                                  out=np.full(len(STATS), np.nan), where=counts[r.Div] > 0)
                values = np.array(old) if old else np.empty((0, len(STATS)))
                n = np.isfinite(values).sum(0)
                padded = np.nansum(values, axis=0) + (5 - n) * prior
                roll = np.where(n == 5, np.nansum(values, axis=0), padded) / 5
                for j, stat in enumerate(STATS):
                    f[f'{side}_{stat}_5'] = roll[j]
                prior_pts = prior[2] if np.isfinite(prior[2]) else 1.0
                f[f'{side}_HPPG'] = (team_sum[key] + 10 * prior_pts) / (team_n[key] + 10)
                seq = appearances[key]
                f[f'REST_{side}'] = min((date - seq[-1]).days, 14) if seq else 14
                f[f'CONG_{side}'] = sum((date - x).days <= 14 for x in seq[-8:])
                f[f'NEW_{side}'] = float(r.Season - 101 not in seen_seasons[key])
                rating = elo.get(key, 1500.)
                if key in last_season and last_season[key] != r.Season:
                    rating = .75 * rating + .25 * 1500
                ratings.append(rating)
                values_ew = {}
                for name in ('Pts', 'GD', 'GF', 'GA', 'Draw', 'Total', 'SOTD'):
                    for half in (5, 10):
                        k = (key, name, half)
                        values_ew[(name, half)] = ew[k] / ewden[k] if ewden[k] else np.nan
                evs.append(values_ew)
                sq = current_squad.get(key)
                f[f'{side}_snapshot_at'] = sq.available_at if sq else pd.NaT
                f[f'{side}_snapshot_age'] = (date - sq.available_at).days if sq else np.nan
                f[f'SQ_{side}'] = sq.SQ if sq else np.nan
                f[f'XI_{side}'] = sq.XI if sq else np.nan
                pool = league_sq[r.Div]
                sd = np.std(pool, ddof=1) if len(pool) > 1 else np.nan
                f[f'SQz_{side}'] = (sq.SQ - np.mean(pool)) / sd if sq and sd > 0 else np.nan
            for name in ('Pts', 'GD', 'SOTD'):
                for half in (5, 10):
                    f[f'EW_{name}{half}_DIF'] = evs[0][(name, half)] - evs[1][(name, half)]
            f['ELO_DIF'] = ratings[0] - ratings[1]
            for base in ('SQ', 'XI', 'SQz'):
                f[base + '_DIF'] = f[base + '_H'] - f[base + '_A']
            f['HPPG_DIF'] = f['H_HPPG'] - f['A_HPPG']
            f['DRAW_PROP'] = np.nanmean([evs[0][('Draw', 10)], evs[1][('Draw', 10)]])
            f['TOTG_PROP'] = np.nanmean([evs[0][('Total', 10)], evs[1][('Total', 10)]])
            f['XGD'] = (evs[0][('GF', 10)] + evs[1][('GA', 10)]
                        - evs[1][('GF', 10)] - evs[0][('GA', 10)]) / 2
            f['SEASON_WEEK'] = (date - pd.Timestamp(2000 + r.Season // 100, 7, 1)).days // 7
            for col in sk_cols:
                a, b = current_squad.get(keys[0]), current_squad.get(keys[1])
                f[col] = getattr(a, col) - getattr(b, col) if a and b else np.nan
            out.append(f)
        # No feature above depends on the following update, including other games today.
        for r in day.itertuples():
            h, a = (r.Div, r.HomeTeam), (r.Div, r.AwayTeam)
            rh, ra = elo.get(h, 1500.), elo.get(a, 1500.)
            if h in last_season and last_season[h] != r.Season: rh = .75 * rh + 375
            if a in last_season and last_season[a] != r.Season: ra = .75 * ra + 375
            expected = 1 / (1 + 10 ** (-(rh + 60 - ra) / 400))
            observed = 1 if r.FTR == 'H' else .5 if r.FTR == 'D' else 0
            delta = 20 * (observed - expected)
            elo[h], elo[a] = rh + delta, ra - delta
            for side, key in (('H', h), ('A', a)):
                gf, ga = (r.FTHG, r.FTAG) if side == 'H' else (r.FTAG, r.FTHG)
                pts = 1 if gf == ga else 3 if gf > ga else 0
                other = 'A' if side == 'H' else 'H'
                stat = [gf, ga, pts, float(ga == 0), getattr(r, side + 'S', np.nan),
                        getattr(r, side + 'ST', np.nan), getattr(r, side + 'C', np.nan),
                        getattr(r, side + 'F', np.nan),
                        getattr(r, side + 'Y', np.nan) + getattr(r, side + 'R', np.nan)]
                vals = np.asarray(stat, float)
                hist[key].append(vals)
                sums[r.Div] += np.nan_to_num(vals)
                counts[r.Div] += np.isfinite(vals)
                team_sum[key] += pts
                team_n[key] += 1
                raw_ew = dict(Pts=pts, GD=gf-ga, GF=gf, GA=ga, Draw=float(gf == ga),
                              Total=gf+ga, SOTD=getattr(r, side+'ST', np.nan)-getattr(r, other+'ST', np.nan))
                for name, value in raw_ew.items():
                    for half in (5, 10):
                        k = (key, name, half)
                        decay = .5 ** (1 / half)
                        ew[k] *= decay
                        ewden[k] *= decay
                        if np.isfinite(value): ew[k] += value; ewden[k] += 1
                appearances[key].append(date)
                seen_seasons[key].add(r.Season)
                last_season[key] = r.Season
    return pd.DataFrame(out)


def market_features(d):
    probabilities, margins = [], []
    for prefix in ('B365', 'BW', 'PS', 'WH'):
        odds = d.reindex(columns=[prefix+c for c in 'HDA']).to_numpy(float)
        valid = np.isfinite(odds).all(1) & (odds > 1).all(1)
        if prefix == 'PS': valid &= (d.Date < PIN_BAD_FROM).to_numpy()
        inv = np.divide(1, odds, out=np.full_like(odds, np.nan), where=odds > 1)
        total = inv.sum(1)
        p = inv / total[:, None]
        p[~valid] = np.nan
        probabilities.append(p)
        margins.append(np.where(valid, total-1, np.nan))
    probs = np.stack(probabilities)
    consensus = np.nanmean(probs, axis=0)
    pinok = np.isfinite(probs[2]).all(1)
    ref = np.where(pinok[:, None], probs[2], consensus)
    f = pd.DataFrame(index=d.index)
    for j, c in enumerate('HDA'): f['MKT_p'+c] = consensus[:, j]; f['REF_p'+c] = ref[:, j]
    f['MKT_logit'] = np.log(consensus[:, 0]/consensus[:, 2])
    f['REF_logit'] = np.log(ref[:, 0]/ref[:, 2])
    f['MKT_margin'] = np.nanmean(np.stack(margins), axis=0)
    lg = np.log(probs[:, :, 0] / probs[:, :, 2])
    f['MKT_STD'] = np.nanstd(lg, axis=0)
    f['PIN_GAP'] = np.where(pinok, lg[2] - np.nanmean(np.delete(lg, 2, axis=0), axis=0), 0)
    f['pinnacle_fiable'] = pinok
    f['n_casas'] = np.isfinite(probs[:, :, 0]).sum(0)
    return f


def prepare(out):
    _, availability = verify_inputs()
    d = load_matches()
    squads = squad_snapshots(availability)
    h = historical_features(d, squads)
    d = d.merge(h, on='match_id', validate='one_to_one')
    model = d[d.Season.ge(1415)].reset_index(drop=True)
    market = market_features(model)
    model = pd.concat([model, market], axis=1)
    keep = model.n_casas.gt(0)
    cohort = model[keep].copy().reset_index(drop=True)
    cohort['y'] = cohort.FTR.map({'H': 0, 'D': 1, 'A': 2}).astype(int)
    cohort['GD'] = cohort.FTHG-cohort.FTAG
    cohort['row_id'] = np.arange(len(cohort))
    history = [c for c in h.columns if c not in ['match_id'] and 'snapshot' not in c
               and not c.startswith(('SQ', 'XI', 'SK_'))]
    roster = ['SQ_H', 'SQ_DIF', 'XI_H', 'XI_DIF', 'SQz_H', 'SQz_DIF']
    skills = [c for c in h if c.startswith('SK_')]
    sets = dict(history=history, roster=roster, skills=skills,
                sin_cuotas=history+roster+skills,
                mercado=['REF_logit', 'REF_pD'], consenso=['MKT_logit', 'MKT_pD', 'MKT_margin'])
    sets['con_cuotas'] = sets['mercado'] + sets['consenso'] + ['MKT_STD', 'PIN_GAP'] + sets['sin_cuotas']
    out.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(out/'dataset.csv', index=False)
    model.loc[~keep, KEY].to_csv(out/'excluded_no_odds.csv', index=False)
    squads.to_csv(out/'squad_snapshots.csv', index=False)
    coverage = cohort.assign(coverage=cohort.SQ_H.notna() & cohort.SQ_A.notna()).groupby('Season').agg(
        n=('y', 'size'), cobertura_plantilla=('coverage', 'mean'),
        antiguedad_H=('H_snapshot_age', 'mean'), antiguedad_A=('A_snapshot_age', 'mean'),
        pinnacle_fiable=('pinnacle_fiable', 'mean'))
    coverage.to_csv(out/'coverage.csv')
    (out/'feature_sets.json').write_text(json.dumps(sets, indent=2))
    cohort[cohort.Season.isin(TEST)].to_csv(out/'test.csv', index=False)
    cohort[~cohort.Season.isin(TEST)].to_csv(out/'train_val.csv', index=False)
    return cohort, sets


def read_data(out):
    df = pd.read_csv(out/'dataset.csv', parse_dates=['Date', 'H_snapshot_at', 'A_snapshot_at'])
    if not df.Date.is_monotonic_increasing or df.match_id.duplicated().any():
        raise ValueError('Dataset no ordenado o identificadores duplicados')
    return df, json.loads((out/'feature_sets.json').read_text())
