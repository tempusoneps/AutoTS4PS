import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import tempfile
import os
import sys

# Ensure src and examples are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
for p in [SRC_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from afternoon_session_prediction import run_advisor, run_backtest

class TestAfternoonScriptMocked(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic 5-minute CSV (3 days)
        dates = []
        for d in ["2026-06-24", "2026-06-25", "2026-06-26"]:
            dates.extend(pd.date_range(f"{d} 09:00", f"{d} 11:30", freq="5min"))
            dates.extend(pd.date_range(f"{d} 13:00", f"{d} 14:45", freq="5min"))

        n = len(dates)
        df_raw = pd.DataFrame({
            "Date": [dt.strftime("%Y-%m-%d %H:%M:%S") for dt in dates],
            "Open": 2000.0 + np.sin(np.linspace(0, 10, n)) * 5,
            "High": 2005.0 + np.sin(np.linspace(0, 10, n)) * 5,
            "Low": 1995.0 + np.sin(np.linspace(0, 10, n)) * 5,
            "Close": 2001.0 + np.sin(np.linspace(0, 10, n)) * 5,
            "Volume": np.random.randint(1000, 5000, size=n),
        })
        self.temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_raw.to_csv(self.temp_csv.name, index=False)

    def tearDown(self):
        if os.path.exists(self.temp_csv.name):
            os.remove(self.temp_csv.name)

    @patch("afternoon_session_prediction.build_afternoon_model")
    def test_run_advisor_syntax_and_flow_with_mock(self, mock_build_model):
        mock_model = MagicMock()
        mock_model.best_model_name = "Mocked_RollingRegression"
        mock_pred = MagicMock()
        mock_pred.forecast = pd.DataFrame({'VN30F1M': [2005.0]})
        mock_pred.upper_forecast = pd.DataFrame({'VN30F1M': [2015.0]})
        mock_pred.lower_forecast = pd.DataFrame({'VN30F1M': [1995.0]})
        mock_model.predict.return_value = mock_pred
        mock_model.fit.return_value = mock_model
        mock_build_model.return_value = mock_model

        # Run advisor mode with mock: should complete in < 0.1s
        run_advisor(self.temp_csv.name, target_date="2026-06-25", fast_mode=True)
        self.assertTrue(mock_model.fit.called)
        self.assertTrue(mock_model.predict.called)

    @patch("afternoon_session_prediction.build_afternoon_model")
    def test_run_backtest_syntax_and_flow_with_mock(self, mock_build_model):
        mock_model = MagicMock()
        mock_model.best_model_name = "Mocked_VECM"
        mock_pred = MagicMock()
        mock_pred.forecast = pd.DataFrame({'VN30F1M': [2008.0]})
        mock_pred.upper_forecast = pd.DataFrame({'VN30F1M': [2018.0]})
        mock_pred.lower_forecast = pd.DataFrame({'VN30F1M': [1998.0]})
        mock_model.predict.return_value = mock_pred
        mock_model.fit.return_value = mock_model
        mock_build_model.return_value = mock_model

        # Run backtest mode for 2 days with mock: completes instantly
        run_backtest(self.temp_csv.name, n_days=2, fast_mode=True)
        self.assertEqual(mock_model.fit.call_count, 2)
        self.assertEqual(mock_model.predict.call_count, 2)

if __name__ == "__main__":
    unittest.main()
