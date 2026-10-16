import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Ensure src and examples are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
for p in [SRC_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from afternoon_session_prediction import (
    compute_forecast_metrics,
    run_advisor,
    run_backtest,
    run_evaluate,
    run_predict,
)


def _make_synthetic_5m_df(day_list):
    dates = []
    for d in day_list:
        dates.extend(pd.date_range(f"{d} 09:00", f"{d} 11:30", freq="5min"))
        dates.extend(pd.date_range(f"{d} 13:00", f"{d} 14:45", freq="5min"))
    n = len(dates)
    return pd.DataFrame({
        "Date": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dates],
        "Open": 2000.0 + np.sin(np.linspace(0, 10, n)) * 5,
        "High": 2005.0 + np.sin(np.linspace(0, 10, n)) * 5,
        "Low": 1995.0 + np.sin(np.linspace(0, 10, n)) * 5,
        "Close": 2001.0 + np.sin(np.linspace(0, 10, n)) * 5,
        "Volume": np.random.randint(1000, 5000, size=n),
    })


class TestAfternoonScriptForecast(unittest.TestCase):
    def setUp(self):
        df_train = _make_synthetic_5m_df(["2026-06-22", "2026-06-23", "2026-06-24"])
        df_test = _make_synthetic_5m_df(["2026-06-25", "2026-06-26"])

        self.temp_train_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_train.to_csv(self.temp_train_csv.name, index=False)

        self.temp_test_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_test.to_csv(self.temp_test_csv.name, index=False)

        self.temp_out_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        self.temp_out_csv.close()

    def tearDown(self):
        for f in [self.temp_train_csv.name, self.temp_test_csv.name, self.temp_out_csv.name]:
            if os.path.exists(f):
                os.remove(f)

    def test_compute_forecast_metrics_exact(self):
        df_res = pd.DataFrame([
            {
                "day": "2026-06-25",
                "morning_close": 2000.0,
                "actual_close": 2010.0,
                "pred_close": 2008.0,
                "lower_bound": 1995.0,
                "upper_bound": 2015.0,
            },
            {
                "day": "2026-06-26",
                "morning_close": 2010.0,
                "actual_close": 2000.0,
                "pred_close": 2014.0,
                "lower_bound": 2005.0,
                "upper_bound": 2025.0,
            },
        ])
        metrics = compute_forecast_metrics(df_res)
        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["mae"], 8.0, places=5)
        self.assertAlmostEqual(metrics["rmse"], 10.0, places=5)
        self.assertAlmostEqual(metrics["oda"], 50.0, places=5)
        self.assertAlmostEqual(metrics["coverage_90"], 50.0, places=5)
        self.assertAlmostEqual(metrics["mean_interval_width"], 20.0, places=5)
        self.assertGreater(metrics["mape"], 0.0)
        self.assertGreater(metrics["smape"], 0.0)

    @patch("afternoon_session_prediction.build_afternoon_model")
    def test_run_predict_pure_forecast_without_long_short(self, mock_build_model):
        mock_model = MagicMock()
        mock_model.best_model_name = "Mocked_RollingRegression"
        mock_pred = MagicMock()
        mock_pred.forecast = pd.DataFrame({"VN30F1M": [2005.0]})
        mock_pred.upper_forecast = pd.DataFrame({"VN30F1M": [2015.0]})
        mock_pred.lower_forecast = pd.DataFrame({"VN30F1M": [1995.0]})
        mock_model.predict.return_value = mock_pred
        mock_model.fit.return_value = mock_model
        mock_build_model.return_value = mock_model

        buf = io.StringIO()
        with redirect_stdout(buf):
            res = run_predict(self.temp_train_csv.name, target_date="2026-06-23", fast_mode=True)
        output = buf.getvalue()

        self.assertTrue(mock_model.fit.called)
        self.assertTrue(mock_model.predict.called)
        self.assertNotIn("STRONG LONG", output)
        self.assertNotIn("STRONG SHORT", output)
        self.assertNotIn("Take-Profit", output)
        self.assertNotIn("Stop-Loss", output)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["pred_close"], 2005.0)
        self.assertIn("abs_error", res)
        self.assertIn("oda_correct", res)

    @patch("afternoon_session_prediction.build_afternoon_model")
    def test_run_evaluate_on_test_set_with_mock(self, mock_build_model):
        mock_model = MagicMock()
        mock_model.best_model_name = "Mocked_VECM"
        mock_pred = MagicMock()
        mock_pred.forecast = pd.DataFrame({"VN30F1M": [2008.0]})
        mock_pred.upper_forecast = pd.DataFrame({"VN30F1M": [2018.0]})
        mock_pred.lower_forecast = pd.DataFrame({"VN30F1M": [1998.0]})
        mock_model.predict.return_value = mock_pred
        mock_model.fit.return_value = mock_model
        mock_build_model.return_value = mock_model

        buf = io.StringIO()
        with redirect_stdout(buf):
            df_res, metrics = run_evaluate(
                train_csv_path=self.temp_train_csv.name,
                test_csv_path=self.temp_test_csv.name,
                fast_mode=True,
                output_csv=self.temp_out_csv.name,
            )
        output = buf.getvalue()

        # 2 test days in temp_test_csv
        self.assertEqual(len(df_res), 2)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("smape", metrics)
        self.assertIn("oda", metrics)
        self.assertIn("coverage_90", metrics)
        self.assertNotIn("STRONG LONG", output)
        self.assertNotIn("STRONG SHORT", output)

        # Verify output CSV saved properly
        df_saved = pd.read_csv(self.temp_out_csv.name)
        self.assertEqual(len(df_saved), 2)


if __name__ == "__main__":
    unittest.main()
