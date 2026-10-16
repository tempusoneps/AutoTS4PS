import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from autots.tools.halfday_data import load_and_resample_halfday
except ImportError:
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
        expected_cols = [
            'VN30F1M',
            'ret_morning',
            'norm_atr',
            'vwap_dev',
            'vol_ratio',
            'ret_last30m',
            'morning_clv',
            'overnight_gap',
            'prev_afternoon_ret',
            'prev_ret_morning',
        ]
        for col in expected_cols:
            self.assertIn(col, df_features.columns)

    def test_new_features_calculation_and_causality(self):
        df_halfday, df_features = load_and_resample_halfday(self.temp_file.name)
        # First day (rows 0 and 1) should have 0.0 for inter-day lag features
        self.assertAlmostEqual(df_features['overnight_gap'].iloc[0], 0.0, places=6)
        self.assertAlmostEqual(df_features['prev_afternoon_ret'].iloc[0], 0.0, places=6)
        self.assertAlmostEqual(df_features['prev_ret_morning'].iloc[0], 0.0, places=6)

        # Second day (rows 2 and 3): prev_ret_morning should equal day 1's ret_morning
        self.assertAlmostEqual(
            df_features['prev_ret_morning'].iloc[2],
            df_features['ret_morning'].iloc[0],
            places=6,
        )

        # morning_clv must be bounded in [-1, 1]
        self.assertTrue((df_features['morning_clv'] >= -1.0001).all())
        self.assertTrue((df_features['morning_clv'] <= 1.0001).all())

        # Forward-fill within each day: morning bar (row 2) and afternoon bar (row 3) share morning features
        for col in [
            'ret_morning', 'norm_atr', 'vwap_dev', 'vol_ratio',
            'ret_last30m', 'morning_clv', 'overnight_gap',
            'prev_afternoon_ret', 'prev_ret_morning'
        ]:
            self.assertAlmostEqual(df_features[col].iloc[2], df_features[col].iloc[3], places=6)

    def test_no_nan_in_features(self):
        df_halfday, df_features = load_and_resample_halfday(self.temp_file.name)
        self.assertFalse(df_features.isna().any().any())

    def test_resample_multiple_csv_files(self):
        df_halfday, df_features = load_and_resample_halfday([self.temp_file.name, self.temp_file.name])
        # Duplicate days get deduplicated/grouped cleanly into 2 valid days (4 rows)
        self.assertEqual(len(df_halfday), 4)
        self.assertEqual(len(df_features), 4)

if __name__ == '__main__':
    unittest.main()
