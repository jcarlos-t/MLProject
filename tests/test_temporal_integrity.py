"""Regression tests for the actual failure modes identified in the audit."""
import sys
from pathlib import Path
import unittest
import warnings
import tempfile
import joblib

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mlcore.data import historical_features, market_features, PIN_BAD_FROM
from mlcore.models import Predictor, Spec, odds_prob, preprocessing
from mlcore.metrics import ece, metrics, block_codes


class TemporalTests(unittest.TestCase):
    def matches(self):
        rows = []
        for i, date in enumerate(pd.date_range('2020-08-01', periods=12, freq='7D')):
            rows.append(dict(match_id=str(i), Div='E0', Season=2021, Date=date,
                HomeTeam='A' if i % 2 else 'B', AwayTeam='B' if i % 2 else 'A',
                FTHG=i % 4, FTAG=i % 2, FTR='D' if i % 3 else 'H', HS=10, AS=8,
                HST=4, AST=3, HC=5, AC=4, HF=10, AF=10, HY=1, AY=1, HR=0, AR=0))
        return pd.DataFrame(rows)

    def squads(self):
        return pd.DataFrame([dict(Div='E0', Team=t, available_at=pd.Timestamp('2020-08-15'),
              snapshot_year=2021, SQ=v, XI=v, SK_finishing=v) for t, v in [('A', 80.), ('B', 70.)]])

    def features(self, m, s):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            return historical_features(m, s).sort_values('match_id').reset_index(drop=True)

    def test_future_scores_cannot_change_past_features(self):
        m = self.matches(); a = self.features(m, self.squads())
        changed = m.copy(); changed.loc[changed.index >= 8, ['FTHG', 'FTAG', 'FTR']] = [9, 0, 'H']
        b = self.features(changed, self.squads())
        ids = set(m.iloc[:9].match_id)  # own match outcome is also excluded
        assert_frame_equal(a[a.match_id.isin(ids)], b[b.match_id.isin(ids)])

    def test_future_roster_cannot_change_past_features(self):
        m, s = self.matches(), self.squads(); a = self.features(m, s)
        later = s.copy(); later['available_at'] = pd.Timestamp('2020-10-01'); later['SQ'] = 99
        b = self.features(m, pd.concat([s, later]).sort_values('available_at'))
        ids = set(m.loc[m.Date.le('2020-10-01'), 'match_id'])
        assert_frame_equal(a[a.match_id.isin(ids)], b[b.match_id.isin(ids)])

    def test_snapshot_is_strictly_earlier(self):
        m = self.matches(); f = self.features(m, self.squads()).set_index('match_id')
        self.assertTrue(pd.isna(f.loc['2', 'SQ_H']))  # snapshot same day is unavailable
        self.assertTrue(np.isfinite(f.loc['3', 'SQ_H']))

    def test_future_rows_do_not_change_normalization(self):
        m = self.matches(); a = self.features(m.iloc[:8], self.squads())
        b = self.features(m, self.squads())
        assert_frame_equal(a, b[b.match_id.isin(a.match_id)].reset_index(drop=True))

    def test_league_reference_only_holds_clubs_of_that_season(self):
        m = self.matches()
        later = m.iloc[:2].assign(match_id=['90', '91'], Season=2122, HomeTeam=['A', 'C'], AwayTeam=['C', 'A'],
                                  Date=pd.to_datetime(['2021-08-07', '2021-08-14']))
        s = pd.DataFrame([dict(Div='E0', Team=t, available_at=pd.Timestamp('2020-07-01'), snapshot_year=2021,
                               SQ=v, XI=v, SK_finishing=v) for t, v in [('A', 80.), ('B', 70.), ('C', 76.)]])
        f = self.features(pd.concat([m, later], ignore_index=True), s).set_index('match_id')
        # 2020/21 is played by A and B; 2021/22 by A and C. B's old snapshot must not count there.
        self.assertAlmostEqual(f.loc['0', 'SQz_A'], (80-75)/np.std([80, 70], ddof=1))
        self.assertAlmostEqual(f.loc['90', 'SQz_H'], (80-78)/np.std([80, 76], ddof=1))

    def test_fixture_api_rejects_dates_the_model_was_trained_on(self):
        from predict import predict_match
        for date in ('2014-08-01', '2020-08-01'):
            with self.subTest(date=date), self.assertRaisesRegex(ValueError, 'retrospectiva'):
                predict_match(None, self.matches(), self.squads(), date=date, league='E0', home='A', away='B',
                              trained_until='2020-08-01')

    def test_fixture_api_rejects_misspelled_teams(self):
        from predict import predict_match
        args = dict(date='2020-09-10', league='E0', home='A', away='Bb', trained_until='2020-08-01')
        with self.assertRaisesRegex(ValueError, 'Equipo desconocido'):
            predict_match(None, self.matches(), self.squads(), **args)

    def test_future_fixture_uses_full_last_known_league_for_normalization(self):
        from predict import predict_match
        class Recorder:
            def predict_proba(self, frame):
                self.features = frame.copy()
                return np.array([[.4, .3, .3]])
        history = self.matches()
        history.loc[0, 'HomeTeam'] = 'C'
        squads = pd.concat([self.squads(), self.squads().iloc[:1].assign(Team='C', SQ=60.)])
        model = Recorder()
        predict_match(model, history, squads, date='2021-08-01', league='E0', home='A', away='B',
                      trained_until='2021-07-01')
        self.assertAlmostEqual(model.features.loc[0, 'SQz_H'], 1.)
        self.assertAlmostEqual(model.features.loc[0, 'SQz_A'], 0.)

    def test_market_quality_cutoff_and_partial_books(self):
        d = pd.DataFrame({'Date': [PIN_BAD_FROM-pd.Timedelta(days=1), PIN_BAD_FROM]})
        for prefix, val in [('PS', 2.), ('B365', 3.), ('BW', 3.), ('WH', 3.)]:
            for c in 'HDA': d[prefix+c] = val
        d['PSA'] = 8.
        f = market_features(d)
        self.assertEqual(f.pinnacle_fiable.tolist(), [True, False])
        self.assertAlmostEqual(f.loc[1, 'REF_pD'], 1/3)
        self.assertEqual(f.n_casas.tolist(), [4, 3])

    def test_svm_calibration_after_base_training(self):
        rng = np.random.default_rng(10); n = 700
        d = pd.DataFrame(dict(Date=pd.date_range('2018-01-01', periods=n), match_id=[str(i) for i in range(n)],
            Season=1819, y=np.arange(n) % 3, GD=0, a=rng.normal(size=n), b=rng.normal(size=n)))
        m = Predictor(Spec('sin', 'SVM', 'SVM', ('a', 'b')), {'C': .1, 'gamma': .001}).fit(d)
        self.assertLess(pd.Timestamp(m.base_end_), pd.Timestamp(m.cal_start_))
        p = m.predict_proba(d.iloc[-10:]); np.testing.assert_allclose(p.sum(1), 1)

    def test_imputer_only_learns_training_rows(self):
        a = np.array([[1, np.nan], [3, 2], [5, 6]], float)
        pre = preprocessing(); pre.fit(a, np.array([0, 1, 2]))
        np.testing.assert_allclose(pre[0].statistics_, [3, 4])
        pre.transform(np.array([[10000, 10000]]))
        np.testing.assert_allclose(pre[0].statistics_, [3, 4])

    def test_weights_follow_rows_after_chronological_sort(self):
        rng=np.random.default_rng(11); n=120
        d=pd.DataFrame(dict(Date=pd.date_range('2018-01-01',periods=n),match_id=[str(i) for i in range(n)],
             Season=1819,y=np.arange(n)%3,GD=0,a=rng.normal(size=n),b=rng.normal(size=n)))
        weights=pd.Series(np.where(d.y.eq(1),3.,1.),index=d.index)
        spec=Spec('sin','LR','LR',('a','b'))
        a=Predictor(spec,{'C':.05}).fit_weighted(d,weights)
        shuffled=d.sample(frac=1,random_state=1)
        b=Predictor(spec,{'C':.05}).fit_weighted(shuffled,weights.loc[shuffled.index])
        np.testing.assert_allclose(a.predict_proba(d),b.predict_proba(d),atol=1e-12)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.joblib';joblib.dump(a,path)
            np.testing.assert_allclose(a.predict_proba(d),joblib.load(path).predict_proba(d))

    def test_public_fixture_api_ignores_same_day_and_future_targets(self):
        from predict import predict_match
        class Recorder:
            def predict_proba(self, frame):
                self.features=frame.copy()
                return np.array([[.4,.3,.3]])
        a,b=Recorder(),Recorder();history=self.matches()
        args=dict(date='2020-09-10',league='E0',home='A',away='B',trained_until='2020-08-01')
        predict_match(a,history,self.squads(),**args)
        altered=history.copy();altered.loc[altered.Date.ge(args['date']),['FTHG','FTAG','FTR']]=[8,0,'H']
        predict_match(b,altered,self.squads(),**args)
        assert_frame_equal(a.features,b.features)

    def test_equal_mass_ece_handles_constant_probabilities(self):
        self.assertAlmostEqual(ece(np.full(100, .4), np.r_[np.ones(60), np.zeros(40)]), .2)

    def test_invalid_odds_fail_clearly(self):
        for odds in [(0, 3, 4), (1, 2, 3), (np.nan, 3, 4), (2, 3)]:
            with self.assertRaises(ValueError): odds_prob(odds)

    def test_bootstrap_week_includes_iso_year(self):
        d = pd.DataFrame({'Date': pd.to_datetime(['2019-08-26', '2020-08-24']), 'Season': [1920, 1920]})
        self.assertEqual(len(np.unique(block_codes(d))), 2)


if __name__ == '__main__': unittest.main()
