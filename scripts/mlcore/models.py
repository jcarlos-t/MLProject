"""Fold-local transformations and serializable probabilistic predictors."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


def normalize(p):
    p = np.asarray(p, float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Probabilidades invalidas: se requieren tres columnas finitas no negativas')
    p = np.clip(p, 1e-12, 1)
    return p / p.sum(1, keepdims=True)


class CorrelationPruner(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=.95): self.threshold = threshold
    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with np.errstate(invalid='ignore', divide='ignore'):
            corr = np.nan_to_num(np.abs(np.corrcoef(X, rowvar=False)))
        self.indices_ = []
        for j in range(X.shape[1]):
            if not any(corr[j, k] > self.threshold for k in self.indices_): self.indices_.append(j)
        return self
    def transform(self, X): return np.asarray(X)[:, self.indices_]


class MISelector(TransformerMixin, BaseEstimator):
    def __init__(self, k=10): self.k = k
    def fit(self, X, y):
        # Feature selection never reads a validation row or its label.
        mi = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
        self.indices_ = np.argsort(mi)[::-1][:min(self.k, X.shape[1])]
        return self
    def transform(self, X): return np.asarray(X)[:, self.indices_]


def preprocessing(reduction='all'):
    steps = [SimpleImputer(strategy='median', keep_empty_features=True, add_indicator=True),
             StandardScaler(), CorrelationPruner()]
    if reduction.startswith('MI'): steps.append(MISelector(int(reduction[2:])))
    if reduction.startswith('PCA'):
        n = .95 if reduction == 'PCA95' else int(reduction[3:])
        steps.append(PCA(n_components=n, svd_solver='full'))
    return make_pipeline(*steps)


# Every grid goes from the most to the least regularised configuration: the
# tie-break in choose_parameter keeps the first one within the tolerance.
GRIDS = {
    'LR': [dict(C=c) for c in (.001, .05, 1.)],
    'HGB': [dict(max_depth=d, max_iter=n, learning_rate=.05) for d, n in ((1, 80), (2, 100), (3, 150))],
    'XGB': [dict(max_depth=d, n_estimators=n, learning_rate=.05) for d, n in ((1, 100), (2, 150), (3, 200))],
    'RF': [dict(min_samples_leaf=n, max_features=.5) for n in (200, 100, 30)],
    'MLP': [dict(hidden_layer_sizes=h, alpha=a, max_iter=100) for h, a in (((16,), 10.), ((16,), 1.), ((32,), 1.))],
    'SVM': [dict(C=c, gamma=g) for c, g in ((.1, .001), (1., .001), (1., .01))],
    'GD_LIN': [dict(dummy=0)],
    'GD_RF': [dict(min_samples_leaf=n) for n in (100, 50)],
    'GD_HGB': [dict(max_depth=d, max_iter=100) for d in (1, 2)],
    'GD_SVR': [dict(C=c) for c in (.1, 1.)],
    'GD_ALL': [dict(min_samples_leaf=n, max_depth=2, max_iter=100) for n in (100, 50)],
}
PRIORITY = {n: i for i, n in enumerate(['LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM',
                                       'GD_LIN', 'GD_HGB', 'GD_RF', 'GD_SVR', 'GD_ALL',
                                       'mean', 'weighted', 'stack'])}


@dataclass(frozen=True)
class Spec:
    branch: str
    name: str
    family: str
    columns: tuple[str, ...]
    reduction: str = 'all'
    @property
    def id(self): return self.branch + '/' + self.name


def specifications(sets):
    specs = []
    for branch, key in [('con', 'con_cuotas'), ('sin', 'sin_cuotas')]:
        for family in ('LR', 'HGB', 'XGB', 'RF', 'MLP', 'SVM'):
            specs.append(Spec(branch, family, family, tuple(sets[key])))
        for method in ('MI5', 'MI10', 'PCA5', 'PCA95'):
            specs.append(Spec(branch, 'LR_'+method, 'LR', tuple(sets[key]), method))
    for name, columns in [('LR_REF', sets['mercado']), ('LR_consenso', sets['consenso'])]:
        specs.append(Spec('con', name, 'LR', tuple(columns)))
    for name, columns in [('LR_historia', sets['history']),
                          ('LR_plantilla', sets['history']+sets['roster'])]:
        specs.append(Spec('sin', name, 'LR', tuple(columns)))
    for family in ('GD_LIN', 'GD_RF', 'GD_HGB', 'GD_SVR', 'GD_ALL'):
        specs.append(Spec('sin', family, family, tuple(sets['sin_cuotas'])))
    return specs


def estimator(family, params, seed=42):
    if family == 'LR': return LogisticRegression(**params, max_iter=3000)
    if family == 'HGB':
        return HistGradientBoostingClassifier(**params, min_samples_leaf=100,
                l2_regularization=1., early_stopping=False, random_state=seed)
    if family == 'XGB':
        return XGBClassifier(**params, objective='multi:softprob', tree_method='hist',
                min_child_weight=20, subsample=.8, colsample_bytree=.8, n_jobs=1, random_state=seed)
    if family == 'RF':
        return RandomForestClassifier(**params, n_estimators=150, n_jobs=1, random_state=seed)
    if family == 'MLP': return MLPClassifier(**params, early_stopping=False, random_state=seed)
    if family == 'SVM': return SVC(**params, decision_function_shape='ovr')
    if family == 'GD_LIN': return LinearRegression()
    if family == 'GD_RF': return RandomForestRegressor(**params, n_estimators=100, max_features=.5, n_jobs=1, random_state=seed)
    if family == 'GD_HGB':
        return HistGradientBoostingRegressor(**params, learning_rate=.05,
                min_samples_leaf=100, early_stopping=False, random_state=seed)
    if family == 'GD_SVR': return SVR(**params)
    raise ValueError(family)


class Predictor:
    """Input is a feature frame in match-date order, never positional arbitrary rows."""
    def __init__(self, spec, params): self.spec, self.params = spec, params

    def _fit_base(self, frame, y, weights=None):
        self.pre = preprocessing(self.spec.reduction)
        z = self.pre.fit_transform(frame[list(self.spec.columns)].to_numpy(float), y)
        families = ['GD_LIN', 'GD_RF', 'GD_HGB'] if self.spec.family == 'GD_ALL' else [self.spec.family]
        self.models = []
        for family in families:
            params = self.params
            if self.spec.family == 'GD_ALL':
                params = {} if family == 'GD_LIN' else ({'min_samples_leaf': params['min_samples_leaf']}
                          if family == 'GD_RF' else {k: params[k] for k in ('max_depth', 'max_iter')})
            seeds = (0, 1, 2) if family == 'MLP' else (42,)
            for seed in seeds:
                m = estimator(family, params, seed)
                target = frame.GD.to_numpy(float) if family.startswith('GD_') else y
                kwargs = {'sample_weight': weights} if weights is not None else {}
                with warnings.catch_warnings(), threadpool_limits(limits=1):
                    warnings.filterwarnings('ignore', category=ConvergenceWarning)
                    m.fit(z, target, **kwargs)
                self.models.append(m)

    def fit(self, frame, weights=None):
        original_index = frame.index
        if weights is not None and not original_index.is_unique:
            raise ValueError('Usar indices unicos para alinear pesos; remuestreo sin pesos adicionales')
        frame = frame.sort_values(['Date', 'match_id']).copy()
        if weights is not None:
            weights = pd.Series(np.asarray(weights, float), index=original_index).reindex(frame.index).to_numpy()
        dates = frame.Date
        self.train_ids_ = frame.match_id.to_list()
        self.train_end_ = str(dates.max())
        if self.spec.family in ('SVM', 'GD_SVR'):
            frame = frame.iloc[-9000:].copy()
            if weights is not None: weights = weights[-9000:]
        y = frame.y.to_numpy(int)
        if self.spec.family == 'SVM' or self.spec.family.startswith('GD_'):
            # Cut by date, so the same date can never appear on both sides.
            if self.spec.family.startswith('GD_'):
                cal_season = frame.Season.max()
                base = frame[frame.Season.lt(cal_season)]
                cal = frame[frame.Season.eq(cal_season)]
            else:
                cut = frame.Date.iloc[int(.8*len(frame))]
                base, cal = frame[frame.Date.lt(cut)], frame[frame.Date.ge(cut)]
            if len(base) < 200 or len(cal) < 100 or base.Date.max() >= cal.Date.min():
                raise ValueError('Separacion temporal de calibracion insuficiente')
            self.base_end_, self.cal_start_ = str(base.Date.max()), str(cal.Date.min())
            w_base = weights[frame.index.get_indexer(base.index)] if weights is not None else None
            self._fit_base(base, base.y.to_numpy(int), w_base)
            zc = self._cal_features(cal)
            self.calibrator = LogisticRegression(C=1., max_iter=3000).fit(zc, cal.y)
            if self.spec.family.startswith('GD_'):
                # Out-of-sample calibrator; the deployed regressor can use all past rows.
                self._fit_base(frame, y, weights)
        else:
            self._fit_base(frame, y, weights)
        return self

    def fit_weighted(self, frame, weights):
        return self.fit(frame, np.asarray(weights, float))

    def _cal_features(self, frame):
        z = self.pre.transform(frame[list(self.spec.columns)].to_numpy(float))
        if self.spec.family == 'SVM': return self.models[0].decision_function(z)
        pred = np.mean([m.predict(z) for m in self.models], axis=0)
        return pred[:, None]

    def predict_proba(self, frame):
        missing = set(self.spec.columns) - set(frame.columns)
        if missing: raise ValueError(f'Variables faltantes: {sorted(missing)}')
        if self.spec.family == 'SVM' or self.spec.family.startswith('GD_'):
            return normalize(self.calibrator.predict_proba(self._cal_features(frame)))
        z = self.pre.transform(frame[list(self.spec.columns)].to_numpy(float))
        return normalize(np.mean([m.predict_proba(z) for m in self.models], axis=0))


def odds_prob(odds):
    a = np.asarray(odds, float)
    if a.shape != (3,) or not np.isfinite(a).all() or not (a > 1).all():
        raise ValueError('Se requieren tres cuotas decimales finitas mayores que uno')
    a = 1/a
    return a/a.sum()


def predict_odds(bundle, pinnacle=None, others=None, as_of=None):
    """Public convenience API only for a selected market-only specification."""
    if bundle.spec.name not in ('LR_REF', 'LR_consenso'):
        raise ValueError('Este modelo requiere fundamentos; usar predict_proba con su esquema completo')
    from .data import PIN_BAD_FROM
    ps = None
    if pinnacle is not None and as_of is not None and pd.Timestamp(as_of) < PIN_BAD_FROM:
        try: ps = odds_prob(pinnacle)
        except ValueError: pass
    observed = []
    for o in others if others is not None else []:
        try: observed.append(odds_prob(o))
        except ValueError: continue
    if ps is None and not observed: raise ValueError('No hay ninguna casa completa y fiable')
    cons = np.mean(observed + ([ps] if ps is not None else []), axis=0)
    ref = ps if ps is not None else cons
    row = dict(REF_logit=np.log(ref[0]/ref[2]), REF_pD=ref[1],
               MKT_logit=np.log(cons[0]/cons[2]), MKT_pD=cons[1])
    if 'MKT_margin' in bundle.spec.columns:
        raise ValueError('Para el modelo con margen proporcionar las variables exactas de cada casa')
    p = bundle.predict_proba(pd.DataFrame([row]))[0]
    return dict(zip(('H', 'D', 'A'), p))
