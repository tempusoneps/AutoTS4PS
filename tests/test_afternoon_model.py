import unittest
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.afternoon_autots import build_afternoon_model

class TestAfternoonModelBuilder(unittest.TestCase):
    def test_model_initialization(self):
        model = build_afternoon_model(fast_mode=True)
        self.assertEqual(model.forecast_length, 1)
        self.assertEqual(model.prediction_interval, 0.90)
        self.assertIn('dwae_weighting', model.metric_weighting)
        self.assertEqual(model.metric_weighting['dwae_weighting'], 3.5)

    def test_whitelist_transformers(self):
        model = build_afternoon_model(fast_mode=True)
        self.assertIsInstance(model.transformer_list, dict)
        self.assertIn("DifferencedTransformer", model.transformer_list)
        # Verify forbidden transformers are absent
        self.assertNotIn("PCA", model.transformer_list)
        self.assertNotIn("HPFilter", model.transformer_list)

    def test_fast_mode(self):
        model_fast = build_afternoon_model(fast_mode=True)
        self.assertEqual(model_fast.max_generations, 5)
        self.assertEqual(model_fast.num_validations, 1)

        model_full = build_afternoon_model(fast_mode=False)
        self.assertEqual(model_full.max_generations, 15)
        self.assertEqual(model_full.num_validations, 2)

    def test_template_mode_with_nonexistent_file(self):
        # If file does not exist, fall back to normal fast mode
        model = build_afternoon_model(fast_mode=True, template_path="nonexistent_template.json")
        self.assertEqual(model.max_generations, 5)

if __name__ == '__main__':
    unittest.main()
