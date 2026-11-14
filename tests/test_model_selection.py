"""Protect the regularization preference used for near-tied CV losses."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from mlcore.experiment import choose_parameter
from mlcore.models import Spec


class ModelSelectionTests(unittest.TestCase):
    def choose(self, family, preferred, excess):
        frame = pd.DataFrame({'Season': np.repeat([1617, 1718, 1819], 3),
                              'y': [0, 1, 2] * 3})
        spec = Spec('sin', family, family, ('feature',))

        def prediction(df, spec, parameter, season, out):
            self.assertIn(season, (1617, 1718))
            is_preferred = all(parameter[k] == v for k, v in preferred.items())
            # Give the simpler configuration a slightly worse loss, so this
            # exercises the tolerance rather than a plain minimum or exact tie.
            correct = np.exp(-(1.0 + (excess if is_preferred else 0.0)))
            probabilities = np.full((3, 3), (1 - correct) / 2)
            np.fill_diagonal(probabilities, correct)
            return probabilities

        with patch('mlcore.experiment.cached_prediction', side_effect=prediction):
            return choose_parameter(frame, spec, 1819, Path('/unused'))[0]

    def test_near_ties_prefer_regularized_forests_and_small_penalized_network(self):
        for family, expected in (
            ('RF', {'min_samples_leaf': 200}),
            ('GD_RF', {'min_samples_leaf': 100}),
            ('GD_ALL', {'min_samples_leaf': 100}),
            ('MLP', {'hidden_layer_sizes': (16,), 'alpha': 10.0}),
        ):
            with self.subTest(family=family):
                actual = self.choose(family, expected, excess=0.0004)
                self.assertEqual({key: actual[key] for key in expected}, expected)

    def test_regularization_does_not_override_loss_outside_tolerance(self):
        actual = self.choose('RF', {'min_samples_leaf': 200}, excess=0.0006)
        self.assertEqual(actual['min_samples_leaf'], 100)


if __name__ == '__main__':
    unittest.main()
