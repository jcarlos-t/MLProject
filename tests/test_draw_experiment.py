"""Checks for decision/probability separation and the new temporal experiment."""
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from draw_models import (goal_probabilities, goal_band, roster_rule, boost_draw,
                         replace_draw, draw_features, GoalsModel)
from experiment_draws import label_metrics, confusion, select_past


class DrawExperimentTests(unittest.TestCase):
    def test_poisson_symmetry_and_equal_means_do_not_force_draw(self):
        rates = np.array([[1., 1.], [3., .5], [.5, 3.]])
        p = goal_probabilities(rates)
        np.testing.assert_allclose(p.sum(1), 1.)
        self.assertAlmostEqual(p[0, 0], p[0, 2])
        self.assertGreater(p[0, 0], p[0, 1])
        np.testing.assert_allclose(p[1], p[2, ::-1])
        with self.assertRaises(ValueError): goal_probabilities([[0., 1.]])

    def test_goal_band_is_absolute_and_includes_boundary(self):
        rates = np.array([[2., 1.], [1., 2.], [3., 1.], [1., 3.]])
        np.testing.assert_array_equal(goal_band(rates, 1.), [1, 1, 0, 2])

    def test_decision_boost_does_not_mutate_probabilities(self):
        p = np.array([[.4, .3, .3], [.2, .25, .55]])
        original = p.copy()
        np.testing.assert_array_equal(boost_draw(p, 2.), [1, 2])
        np.testing.assert_array_equal(p, original)

    def test_roster_rule_falls_back_when_rating_is_missing(self):
        f = pd.DataFrame({'SQ_DIF': [np.nan, -2., 4.]})
        p = np.tile([.5, .3, .2], (3, 1))
        np.testing.assert_array_equal(roster_rule(f, p, 2.), [0, 1, 0])

    def test_replace_draw_preserves_home_away_odds_ratio(self):
        p = np.array([[.4, .3, .3], [.2, .25, .55]])
        q = np.array([.5, .1]); replaced = replace_draw(p, q)
        np.testing.assert_allclose(replaced[:, 1], q)
        np.testing.assert_allclose(replaced.sum(1), 1.)
        np.testing.assert_allclose(replaced[:, 0]/replaced[:, 2], p[:, 0]/p[:, 2])

    def test_metrics_match_sklearn_including_absent_draw_predictions(self):
        y, pred = np.array([0, 1, 2, 1, 0, 2]), np.array([0, 0, 2, 2, 0, 0])
        m = label_metrics(confusion(y, pred))
        self.assertAlmostEqual(m['F1_macro'], f1_score(y, pred, average='macro'))
        self.assertAlmostEqual(m['F1_weighted'], f1_score(y, pred, average='weighted'))
        self.assertAlmostEqual(m['accuracy'], accuracy_score(y, pred))
        self.assertEqual(m['F1_D'], 0.)

    def test_tuning_window_cannot_read_current_or_future_season(self):
        f = pd.DataFrame({'Season': [1718, 1819, 1920, 2021], 'y': [0, 1, 2, 1]})
        a = select_past(f, 1920)
        f.loc[f.Season.ge(1920), 'y'] = 0
        pd.testing.assert_frame_equal(a, select_past(f, 1920))
        self.assertEqual(select_past(f, 2021).Season.tolist(), [1819, 1920])

    def test_pure_draw_features_ignore_targets_and_market(self):
        names = ['SQ_DIF', 'XI_DIF', 'ELO_DIF', 'XGD', 'TOTG_PROP', 'DRAW_PROP',
                 'H_GF_5', 'A_GF_5', 'H_GA_5', 'A_GA_5', 'REST_H', 'REST_A', 'SK_finishing']
        f = pd.DataFrame({name: [1., 2.] for name in names})
        f['FTHG'], f['FTAG'], f['y'], f['REF_pD'] = 3, 1, 0, .3
        a = draw_features(f, 'sin')
        f[['FTHG', 'FTAG', 'y', 'REF_pD']] = [0, 0, 1, .9]
        pd.testing.assert_frame_equal(a, draw_features(f, 'sin'))

    def test_poisson_imputer_uses_training_and_not_evaluation_values(self):
        f = pd.DataFrame({'x': [1., 2., np.nan, 4.], 'FTHG': [1, 2, 0, 1], 'FTAG': [0, 1, 2, 1]})
        model = GoalsModel(['x'], 1.).fit(f)
        self.assertEqual(model.pre[0].statistics_[0], 2.)
        p = model.predict_proba(pd.DataFrame({'x': [1000., np.nan]}))
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(model.pre[0].statistics_[0], 2.)


if __name__ == '__main__': unittest.main()
