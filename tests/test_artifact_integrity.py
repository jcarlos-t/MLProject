"""Reject corrupted cached results and keep report templates reproducible."""
import sys
import json
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mlcore.data import sha
from mlcore.experiment import evaluate


class ArtifactIntegrityTests(unittest.TestCase):
    def fixture(self, out):
        for name in ('protocol.json', 'oof_candidates.csv', 'model_comparison.csv', 'test_predictions.csv'):
            (out/name).write_text('{}')
        (out/'final_metrics.csv').write_text('candidate,log_loss\nexample,1.0\n')
        frozen = {key: sha(out/name) for key, name in (
            ('protocol_sha', 'protocol.json'), ('oof_sha', 'oof_candidates.csv'),
            ('comparison_sha', 'model_comparison.csv'))}
        (out/'frozen.json').write_text(json.dumps(frozen))
        (out/'evaluation_complete.json').write_text(json.dumps({
            'frozen_sha': sha(out/'frozen.json'),
            'test_predictions_sha': sha(out/'test_predictions.csv'),
            'outputs_sha': {'final_metrics.csv': sha(out/'final_metrics.csv')}}))

    def test_cached_evaluation_rejects_modified_predictions_and_protocol(self):
        for name in ('test_predictions.csv', 'protocol.json', 'final_metrics.csv'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                out = Path(folder); self.fixture(out)
                self.assertEqual(len(evaluate(out)), 1)
                (out/name).write_text('changed')
                with self.assertRaises(RuntimeError):
                    evaluate(out)

    def test_cached_evaluation_rejects_unlisted_and_unfingerprinted_results(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); self.fixture(out)
            (out/'backtest.csv').write_text('appeared later')
            with self.assertRaises(RuntimeError): evaluate(out)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); self.fixture(out)
            marker = json.loads((out/'evaluation_complete.json').read_text()); del marker['outputs_sha']
            (out/'evaluation_complete.json').write_text(json.dumps(marker))
            with self.assertRaises(RuntimeError): evaluate(out)

    def test_report_templates_are_self_contained(self):
        root = Path(__file__).resolve().parents[1]
        preamble = (root/'scripts/report/templates/preamble.tex').read_text()
        bibliography = (root/'scripts/report/templates/bibliography.tex').read_text()
        self.assertIn(r'\begin{document}', preamble)
        self.assertIn(r'\bibitem{footballdata}', bibliography)
        self.assertIn('consultado en noviembre de 2026', bibliography)
        self.assertNotIn('\x08', bibliography)


if __name__ == '__main__':
    unittest.main()
