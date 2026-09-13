"""Pruebas de limites y ejecucion reducida; no producen resultados de la tarea."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from target_config import (
    PRICE_DIRECTION_CLASSES,
    price_direction_display_labels,
    price_direction_labels,
)


def module(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class PipelineTests(unittest.TestCase):
    def test_thresholds(self):
        current = pd.Series([100.] * 11 + [0., 100.])
        future = pd.Series([80., 90., 90.01, 96.99, 97., 100., 103., 103.01, 109.99, 110., 120., 10., np.nan])
        result = price_direction_labels(current, future)
        self.assertEqual(result.iloc[:11].tolist(), [PRICE_DIRECTION_CLASSES[i] for i in [0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4]])
        self.assertTrue(result.iloc[11:].isna().all())
        self.assertEqual(
            price_direction_display_labels(PRICE_DIRECTION_CLASSES),
            ["Baja++", "Baja", "Neutro", "Sube", "Sube++"],
        )

    def test_end_to_end(self):
        import dnn_config as cfg
        import torch
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            clean, artifacts, results = root / 'clean', root / 'clean' / 'artifacts', root / 'results'
            clean.mkdir()
            artifacts.mkdir()
            results.mkdir()
            rows = []
            for day, date in enumerate(pd.date_range('2023-01-01', periods=120)):
                for label, change in enumerate([-20, -5, 0, 5, 20]):
                    rows.append({'product_id': f'p{label}', 'timestamp': date.isoformat(),
                                 'platform': 'amazon', 'price': 100., 'target_price_7d': 100. + change,
                                 'target_price_30d': 110., 'target_price_direction_7d': 0,
                                 **{f'feature_{i}': day % (i + 2) + label for i in range(15)}})
            frame = pd.DataFrame(rows)
            files = {}
            for split, indices in zip(('train', 'val', 'test'), np.array_split(np.arange(len(frame)), 3)):
                piece = frame.iloc[indices]
                path = root / f'{split}.csv'
                piece.to_csv(path, index=False)
                files[split] = str(path)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cleaning = module('01_limpieza_datos.py')
                cleaning.SPLIT_FILES, cleaning.CLEAN_DIR = files, str(clean)
                cleaning.DATASET_ROWS = 600
                cleaning.main()
                metadata = json.loads((clean / 'split_metadata.json').read_text())
                for left, right in [('train', 'val'), ('val', 'test')]:
                    self.assertLess(pd.Timestamp(metadata['splits'][left]['end']) + pd.Timedelta(days=7),
                                    pd.Timestamp(metadata['splits'][right]['start']))
                features = module('02_feature_engineering.py')
                features.CLEAN_DIR, features.ARTIFACT_DIR = str(clean), str(artifacts)
                features.INPUT_FILES = {s: str(clean / f'{s}.parquet') for s in files}
                features.OUTPUT_FILES = {s: str(clean / f'{s}_features.parquet') for s in files}
                features.main()
                manifest = json.loads((artifacts / 'dataset_manifest.json').read_text())
                self.assertGreaterEqual(manifest['feature_count'], 15)
                self.assertNotIn('target_price_7d', manifest['features'])
                self.assertNotIn('timestamp', manifest['features'])
                for split in files:
                    self.assertEqual(set(manifest['splits'][split]['classes']), set(PRICE_DIRECTION_CLASSES))
                settings = dict(CLEAN_DIR=str(clean), ARTIFACT_DIR=str(artifacts), RESULTS_DIR=str(results),
                                INPUT_FILES=features.OUTPUT_FILES, EPOCHS=1, BATCH_SIZE=64,
                                HIDDEN_LAYERS=[8], DEVICE=torch.device('cpu'))
                with patch.multiple(cfg, **settings):
                    baseline = module('03_baselines.py')
                    baseline.CLEAN_DIR, baseline.ARTIFACT_DIR, baseline.RESULTS_DIR = str(clean), str(artifacts), str(results)
                    baseline.LEARNING_CURVE_SAMPLE = 200
                    baseline.SVM_RBF_SAMPLE = 200
                    baseline.main()
                    dnn = module('04_dnn.py')
                    dnn.train()
                    automl = module('05_automl.py')
                    automl.TIME_BUDGET, automl.MAX_ITER, automl.ESTIMATORS = 3, 1, ['lrl2']
                    automl.main()
                    nas = module('06_nas.py')
                    nas.N_TRIALS = nas.NAS_EPOCHS_PER_TRIAL = nas.NAS_FINAL_EPOCHS = 1
                    nas.main()
                    comparison = module('07_comparacion.py')
                    comparison.main()
                    combined = pd.read_csv(results / 'comparacion_modelos.csv')
                    self.assertEqual(len(combined), 4)
                    self.assertEqual(combined.train_rows.nunique(), 1)
                    self.assertTrue((combined.tiempo_s > 0).all())
                    nas_metrics = json.loads((results / 'metrics_nas.json').read_text())
                    self.assertAlmostEqual(nas_metrics['tiempo_s'], nas_metrics['search_time_s'] + nas_metrics['fit_time_s'])
                    old = json.loads((results / 'metrics_dnn.json').read_text())
                    old['run_id'] = 'old-run'
                    (results / 'metrics_dnn.json').write_text(json.dumps(old))
                    with self.assertRaises(ValueError):
                        comparison.load_results()


if __name__ == '__main__':
    unittest.main()
