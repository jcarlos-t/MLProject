"""One implementation of every metric used by notebooks and deliverables."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, balanced_accuracy_score
from .models import normalize


def ece(p, observed, nb=10, equal_mass=True):
    p, o = np.asarray(p), np.asarray(observed)
    if np.ptp(p) < 1e-12: return float(abs(p.mean()-o.mean()))
    if equal_mass:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, nb+1)[1:-1]))
        b = np.digitize(p, edges)
    else: b = np.minimum((p*nb).astype(int), nb-1)
    return float(sum((b == k).mean()*abs(p[b == k].mean()-o[b == k].mean()) for k in np.unique(b)))


def losses(y, p):
    p = normalize(p)
    return -np.log(p[np.arange(len(y)), np.asarray(y, int)])


def metrics(y, p):
    p = normalize(p); y = np.asarray(y, int); onehot = np.eye(3)[y]
    nd = y != 1
    return dict(n=len(y), log_loss=log_loss(y, p, labels=[0, 1, 2]),
        RPS=float(np.mean(.5*np.sum((np.cumsum(p-onehot, axis=1)[:, :2])**2, axis=1))),
        Brier=float(np.mean(np.sum((p-onehot)**2, axis=1))),
        AUC_macro=float(np.mean([roc_auc_score(y == j, p[:, j]) for j in range(3)])),
        accuracy=accuracy_score(y, p.argmax(1)), balanced_accuracy=balanced_accuracy_score(y, p.argmax(1)),
        ECE_masa=float(np.mean([ece(p[:, j], y == j) for j in range(3)])),
        ECE_ancho=float(np.mean([ece(p[:, j], y == j, equal_mass=False) for j in range(3)])),
        acierto_ganador_condicional=float(((p[nd, 0] > p[nd, 2]) == (y[nd] == 0)).mean()),
        recall_empate=float((p[y == 1].argmax(1) == 1).mean()))


def block_codes(frame):
    # Include ISO year: COVID seasons may contain the same ISO week twice.
    iso = frame.Date.dt.isocalendar()
    return pd.factorize(frame.Season.astype(str)+'-'+iso.year.astype(str)+'-'+iso.week.astype(str))[0]


def compare(frame, a, b, B=5000, seed=2026):
    delta = np.asarray(a)-np.asarray(b)
    cod = block_codes(frame); K = cod.max()+1
    sums, counts = np.bincount(cod, weights=delta), np.bincount(cod)
    rng = np.random.default_rng(seed); ix = rng.integers(0, K, (B, K))
    draws = sums[ix].sum(1)/counts[ix].sum(1)
    per = pd.Series(delta).groupby(frame.Season.to_numpy()).mean()
    return dict(delta=float(delta.mean()), lo=float(np.quantile(draws, .025)), hi=float(np.quantile(draws, .975)),
                p=float(min(1., 2*min(((draws <= 0).sum()+1)/(B+1), ((draws >= 0).sum()+1)/(B+1)))),
                temporadas_mejora=f'{int((per < 0).sum())}/{len(per)}')


def bh(p):
    p = np.asarray(p, float); order = np.argsort(p); n = len(p)
    result = np.empty(n); result[order] = np.minimum.accumulate((p[order]*n/(np.arange(n)+1))[::-1])[::-1]
    return np.minimum(result, 1)


def backtest(frame, p, prefix='B365', tau=.02):
    odds = frame.reindex(columns=[prefix+c for c in 'HDA']).to_numpy(float)
    valid = np.isfinite(odds).all(1) & (odds > 1).all(1)
    if prefix == 'PS': valid &= frame.pinnacle_fiable.to_numpy(bool)
    value = np.where(valid[:, None], p*odds-1, -np.inf)
    side = value.argmax(1); bet = valid & (value.max(1) > tau)
    pnl = np.where(frame.y.to_numpy() == side, odds[np.arange(len(frame)), side]-1, -1.)
    result = dict(casa=prefix, tau=tau, apuestas=int(bet.sum()), ROI=np.nan, lo=np.nan, hi=np.nan,
                  EV_prometido=float(value.max(1)[bet].mean()) if bet.any() else np.nan)
    if bet.sum() >= 30:
        c = compare(frame.loc[bet].reset_index(drop=True), pnl[bet], np.zeros(bet.sum()))
        result.update(ROI=float(pnl[bet].mean()), lo=c['lo'], hi=c['hi'])
    return result
