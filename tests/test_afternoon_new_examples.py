import os
import sys
import unittest
import numpy as np
import pandas as pd

# Ensure src and examples are on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples")
for p in [SRC_DIR, EXAMPLES_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from afternoon_sl_tp_event_risk import (
    simulate_first_passage_path,
    evaluate_first_passage_risk,
)
from afternoon_extremes_volatility import (
    classify_regime,
    AfternoonExtremesModel,
    generate_strategy_advisory,
)


class TestAfternoonNewExamples(unittest.TestCase):
    def test_first_passage_simulation_long(self):
        # Synthetic afternoon bars with morning close = 2000.0
        # Bar 1 reaches high +5, low -2
        # Bar 2 reaches high +8, low -3 (hits TP +7)
        bars = pd.DataFrame([
            {"Time_str": "13:00:00", "delta_high": 5.0, "delta_low": -2.0, "delta_close": 4.0},
            {"Time_str": "13:05:00", "delta_high": 8.0, "delta_low": -3.0, "delta_close": 7.5},
        ])
        res = simulate_first_passage_path(bars, "LONG", tp_pts=7.0, sl_pts=4.0)
        self.assertEqual(res["outcome"], "HIT_TP")
        self.assertEqual(res["exit_type"], "TP")
        self.assertEqual(res["pnl"], 7.0)
        self.assertEqual(res["step"], 2)

    def test_first_passage_simulation_short(self):
        # Bar 1 drops to low -5 (hits TP 4.0 for SHORT)
        bars = pd.DataFrame([
            {"Time_str": "13:00:00", "delta_high": 1.0, "delta_low": -5.0, "delta_close": -4.5},
        ])
        res = simulate_first_passage_path(bars, "SHORT", tp_pts=4.0, sl_pts=3.0)
        self.assertEqual(res["outcome"], "HIT_TP")
        self.assertEqual(res["pnl"], 4.0)

    def test_first_passage_simulation_sl(self):
        # Long position gets stopped out at low -5.0 when sl=4.0
        bars = pd.DataFrame([
            {"Time_str": "13:00:00", "delta_high": 2.0, "delta_low": -5.0, "delta_close": -4.0},
        ])
        res = simulate_first_passage_path(bars, "LONG", tp_pts=8.0, sl_pts=4.0)
        self.assertEqual(res["outcome"], "HIT_SL")
        self.assertEqual(res["pnl"], -4.0)

    def test_classify_regime(self):
        self.assertEqual(classify_regime(22.0), "EXPANSION / BREAKOUT")
        self.assertEqual(classify_regime(14.0), "NORMAL VOLATILITY")
        self.assertEqual(classify_regime(6.5), "COMPRESSION / SIDEWAY")

    def test_extremes_model_fit_and_predict(self):
        # Create synthetic data with 50 samples
        np.random.seed(42)
        X = np.random.randn(50, 9)
        dh = np.random.uniform(2.0, 15.0, size=50)
        dl = -np.random.uniform(2.0, 15.0, size=50)

        model = AfternoonExtremesModel(n_estimators=10)
        model.fit(X, dh, dl)

        morning_close = 2000.0
        x_test = np.random.randn(9)
        pred = model.predict(x_test, morning_close)

        self.assertGreaterEqual(pred["pred_high"], morning_close)
        self.assertLessEqual(pred["pred_low"], morning_close)
        self.assertGreater(pred["pred_range"], 0.0)
        self.assertGreater(pred["high_upper"], pred["pred_high"])
        self.assertLess(pred["low_lower"], pred["pred_low"])
        self.assertIn(pred["regime"], ["EXPANSION / BREAKOUT", "NORMAL VOLATILITY", "COMPRESSION / SIDEWAY"])

    def test_generate_strategy_advisory(self):
        pred = {
            "regime": "EXPANSION / BREAKOUT",
            "pred_high": 2025.0,
            "pred_low": 2005.0,
            "pred_range": 20.0,
            "high_upper": 2035.0,
            "low_lower": 1995.0,
        }
        morning_row = pd.Series({
            "high": 2018.0,
            "low": 2002.0,
            "close": 2010.0,
            "ret_last30m": 0.5,
            "morning_clv": 0.8,
        })
        advisory = generate_strategy_advisory(pred, morning_row)
        self.assertIn("BREAKOUT", advisory["action"])
        self.assertIn("Long Stop", advisory["details"])


if __name__ == "__main__":
    unittest.main()
