import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from inference import CLASS_NAMES, classify_image, load_metrics, preprocess_image


class FakeClassifier:
    def predict(self, batch, verbose=0):
        self.batch_shape = batch.shape
        return np.array([[0.1, 0.2, 0.6, 0.1]], dtype=np.float32)


class InferenceTests(unittest.TestCase):
    def test_preprocess_image_returns_rgb_normalized_batch_item(self):
        image = Image.new("L", (20, 10), color=255)
        result = preprocess_image(image)

        self.assertEqual(result.shape, (224, 224, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result, 1.0))

    def test_classify_image_returns_all_categories_and_probabilities(self):
        model = FakeClassifier()
        result = classify_image(model, Image.new("RGB", (10, 10)))

        self.assertEqual([item["category"] for item in result], list(CLASS_NAMES))
        self.assertAlmostEqual(result[2]["probability"], 0.6)
        self.assertEqual(model.batch_shape, (1, 224, 224, 3))

    def test_load_metrics_reads_numeric_json_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps({"mAP50": 0.8}), encoding="utf-8")

            self.assertEqual(load_metrics(path), {"mAP50": 0.8})


if __name__ == "__main__":
    unittest.main()
