import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from tools.halfday_data import load_and_resample_halfday

class TestAfternoonData(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 2-day 5-minute CSV
        dates_day1_morning = pd.date_range("2026-06-25 09:00", "2026-06-25 11:30", freq="5min")
        dates_day1_afternoon = pd.date_range("2026-06-25 13:00", "2026-06-25 14:45", freq="5min")
        dates_day2_morning = pd.date_range("2026-06-26 09:00", "2026-06-26 11:30", freq="5min")
        dates_day2_afternoon = pd.date_range("2026-06-26 13:00", "2026-06-26 14:45", freq="5min")

        all_dates = list(dates_day1_morning) + list(dates_day1_afternoon) + list(dates_day2_morning) + list(dates_day2_afternoon)
        n = len(all_dates)
        df_raw = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d %H:%M:%S") for d in all_dates],
            "Open": 1000.0 + np.sin(np.linspace(0, 10, n)) * 10,
            "High": 1005.0 + np.sin(np.linspace(0, 10, n)) * 10,
            "Low": 995.0 + np.sin(np.linspace(0, 10, n)) * 10,
            "Close": 1001.0 + np.sin(np.linspace(0, 10, n)) * 10,
            "Volume": np.random.randint(500, 3000, size=n),
        })
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        df_raw.to_csv(self.temp_file.name, index=False)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_resample_output_shape_and_columns(self):
        df_halfday, df_features = load_and_resample_halfday(self.temp_file.name)
        # 2 days * 2 bars = 4 rows
        self.assertEqual(len(df_halfday), 4)
        self.assertEqual(len(df_features), 4)
        for col in ['VN30F1M', 'ret_morning', 'norm_atr', 'vwap_dev', 'vol_ratio']:
            self.assertIn(col, df_features.columns)

    def test_no_nan_in_features(self):
        df_halfday, df_features = load_and_resample_halfday(self.temp_file.name)
        self.assertFalse(df_features.isna().any().any())

if __name__ == '__main__':
    unittest.main()
