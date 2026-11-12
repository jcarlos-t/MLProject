#!/usr/bin/env python3
"""Predict a fixture from prior results and verified snapshots, without a target."""
import argparse
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from mlcore.data import load_matches, historical_features, market_features
from mlcore.experiment import current


def predict_match(model, history, snapshots, *, date, league, home, away, quotes=None):
    date = pd.Timestamp(date)
    if home == away: raise ValueError('Los equipos deben ser distintos')
    if league not in set(history.Div): raise ValueError('Liga desconocida')
    # Targets of the requested day and all later days are structurally excluded.
    history = history[history.Date.lt(date)].copy()
    start = date.year if date.month >= 7 else date.year-1
    fixture = dict(match_id='fixture', Date=date, Div=league, HomeTeam=home, AwayTeam=away,
        Season=(start % 100)*100+(start+1) % 100, FTHG=np.nan, FTAG=np.nan, FTR=None)
    frame = pd.concat([history, pd.DataFrame([fixture])], ignore_index=True, sort=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        features = historical_features(frame, snapshots)
    row = features[features.match_id.eq('fixture')].reset_index(drop=True)
    if quotes is not None:
        observed = pd.DataFrame([{'Date': date}])
        for prefix, triple in quotes.items():
            if prefix not in ('B365', 'BW', 'PS', 'WH'): raise ValueError('Casa no admitida')
            a = np.asarray(triple, float)
            if a.shape != (3,): raise ValueError('Cada casa requiere H, D, A')
            for key, val in zip('HDA', a): observed[prefix+key] = val
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            market = market_features(observed)
        if market.n_casas.iloc[0] == 0: raise ValueError('No hay casas completas y fiables')
        row = pd.concat([row, market], axis=1)
    p = model.predict_proba(row)[0]
    return {'probabilities': dict(zip('HDA', p.tolist())),
            'fair_odds': dict(zip('HDA', (1/p).tolist())),
            'date': str(date.date()), 'home': home, 'away': away}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--branch', choices=['con', 'sin'], required=True)
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--league', required=True)
    ap.add_argument('--home', required=True)
    ap.add_argument('--away', required=True)
    ap.add_argument('--quotes', help='JSON: {"B365":[H,D,A],"BW":[H,D,A]}')
    args = ap.parse_args(); out = current()
    if args.branch == 'con' and not args.quotes: ap.error('El enfoque con cuotas requiere --quotes')
    if args.branch == 'sin' and args.quotes: ap.error('El predictor puro sin cuotas no recibe precios')
    model = joblib.load(out/f'modelo_{args.branch}_cuotas.joblib')
    snapshots = pd.read_csv(out/'squad_snapshots.csv', parse_dates=['available_at'])
    result = predict_match(model, load_matches(), snapshots, date=args.date, league=args.league,
        home=args.home, away=args.away, quotes=json.loads(args.quotes) if args.quotes else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
