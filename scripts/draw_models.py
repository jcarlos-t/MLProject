"""Pre-match goals and draw experiments, separate from the frozen main models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skellam
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from mlcore.models import normalize, preprocessing


def goal_probabilities(rates):
    """Independent Poisson scores; sum all scores using the Skellam law."""
    rates = np.asarray(rates, float)
    if rates.ndim != 2 or rates.shape[1] != 2 or not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError('Se requieren dos tasas de gol positivas y finitas')
    h, a = rates.T
    return normalize(np.c_[skellam.sf(0, h, a), skellam.pmf(0, h, a), skellam.cdf(-1, h, a)])


def goal_band(rates, width):
    """A band on EXPECTED goal difference, never the observed match score."""
    gap = np.asarray(rates)[:, 0] - np.asarray(rates)[:, 1]
    return np.where(np.abs(gap) <= width, 1, np.where(gap > 0, 0, 2))


def roster_rule(frame, p, width):
    pred = np.asarray(p).argmax(1)
    gap = frame.SQ_DIF.to_numpy(float)
    return np.where(np.isfinite(gap) & (np.abs(gap) <= width), 1, pred)


def boost_draw(p, multiplier):
    """Decision only: keep the original probability vector for scoring."""
    scores = np.asarray(p).copy(); scores[:, 1] *= multiplier
    return scores.argmax(1)


def replace_draw(p, q):
    p = normalize(p); q = np.asarray(q, float)
    if not np.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError('Probabilidad de empate invalida')
    nondraw = p[:, [0, 2]] / p[:, [0, 2]].sum(1, keepdims=True)
    return normalize(np.c_[(1-q)*nondraw[:, 0], q, (1-q)*nondraw[:, 1]])


def draw_threshold(p, threshold):
    p = np.asarray(p)
    return np.where(p[:, 1] >= threshold, 1, np.where(p[:, 0] >= p[:, 2], 0, 2))


def draw_features(frame, branch):
    """Fixed predictors known before kickoff; missing roster stays missing."""
    out = pd.DataFrame(index=frame.index)
    for name in ('SQ_DIF', 'XI_DIF', 'ELO_DIF', 'XGD'):
        out[name] = frame[name]
        out['ABS_'+name] = frame[name].abs()
    skills = [c for c in frame if c.startswith('SK_')]
    out['SKILLS_RMS'] = np.sqrt(frame[skills].pow(2).mean(axis=1))
    for name in ('TOTG_PROP', 'DRAW_PROP'):
        out[name] = frame[name]
    out['GOALS_FOR'] = frame.H_GF_5 + frame.A_GF_5
    out['GOALS_AGAINST'] = frame.H_GA_5 + frame.A_GA_5
    out['REST_GAP'] = (frame.REST_H-frame.REST_A).abs()
    out['ROSTER_MISSING'] = frame.SQ_DIF.isna().astype(float)
    if branch == 'con':
        for name in ('REF_pD', 'MKT_pD', 'MKT_margin'):
            out[name] = frame[name]
        out['ABS_REF_logit'] = frame.REF_logit.abs()
    elif branch != 'sin':
        raise ValueError('Rama desconocida')
    return out


class GoalsModel:
    def __init__(self, columns, alpha): self.columns, self.alpha = list(columns), alpha

    def fit(self, frame):
        self.pre = preprocessing()
        z = self.pre.fit_transform(frame[self.columns].to_numpy(float))
        self.models = []
        with threadpool_limits(limits=1):
            for target in ('FTHG', 'FTAG'):
                model = PoissonRegressor(alpha=self.alpha, max_iter=1000, tol=1e-7)
                model.fit(z, frame[target].to_numpy(float)); self.models.append(model)
        return self

    def predict_rates(self, frame):
        z = self.pre.transform(frame[self.columns].to_numpy(float))
        return np.clip(np.column_stack([m.predict(z) for m in self.models]), .01, 15.)

    def predict_proba(self, frame): return goal_probabilities(self.predict_rates(frame))


class DrawDetector:
    def __init__(self, branch, C): self.branch, self.C = branch, C

    def fit(self, frame):
        x = draw_features(frame, self.branch); self.columns = list(x)
        self.pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True),
                                      StandardScaler(), LogisticRegression(C=self.C, max_iter=3000))
        with threadpool_limits(limits=1): self.pipeline.fit(x, frame.y.eq(1).astype(int))
        return self

    def predict_draw(self, frame):
        return self.pipeline.predict_proba(draw_features(frame, self.branch))[:, 1]


class DrawPredictor:
    """Serializable extension: probabilities and decision policy are distinct."""
    def __init__(self, base, goals, detector, family, parameter):
        self.base, self.goals, self.detector = base, goals, detector
        self.family, self.parameter = family, parameter

    def predict_proba(self, frame):
        if self.family.startswith('poisson'): return self.goals.predict_proba(frame)
        p = self.base.predict_proba(frame)
        if self.family.startswith('detector'): return replace_draw(p, self.detector.predict_draw(frame))
        return p

    def predict(self, frame):
        p = self.predict_proba(frame)
        if self.family == 'boost': return boost_draw(p, self.parameter)
        if self.family == 'roster': return roster_rule(frame, p, self.parameter)
        if self.family.startswith('poisson_band'): return goal_band(self.goals.predict_rates(frame), self.parameter)
        if self.family == 'detector_threshold': return draw_threshold(p, self.parameter)
        return p.argmax(1)
