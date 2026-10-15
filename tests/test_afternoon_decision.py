import unittest
import sys
import os

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from autots.tools.decision_engine import evaluate_decision
except ImportError:
    from tools.decision_engine import evaluate_decision

class TestDecisionEngine(unittest.TestCase):
    def setUp(self):
        self.morning_info = {
            'close': 1000.0,
            'high': 1005.0,
            'low': 995.0,
            'atr': 10.0,
            'vwap': 999.0
        }

    def test_strong_long_trigger(self):
        # High upper bound, high forecast point
        res = evaluate_decision(
            forecast_point=1010.0,
            upper_bound=1015.0,
            lower_bound=1005.0,
            morning_info=self.morning_info,
            friction=0.8,
            min_prob=0.60
        )
        self.assertEqual(res['action'], "STRONG LONG")
        self.assertGreaterEqual(res['confidence'], 0.60)
        self.assertGreater(res['target'], res['entry'])
        self.assertLess(res['stop_loss'], res['entry'])

    def test_strong_short_trigger(self):
        # Low lower bound, low forecast point
        res = evaluate_decision(
            forecast_point=990.0,
            upper_bound=995.0,
            lower_bound=985.0,
            morning_info=self.morning_info,
            friction=0.8,
            min_prob=0.60
        )
        self.assertEqual(res['action'], "STRONG SHORT")
        self.assertGreaterEqual(res['confidence'], 0.60)
        self.assertLess(res['target'], res['entry'])
        self.assertGreater(res['stop_loss'], res['entry'])

    def test_wait_on_choppy_market(self):
        # Bounds wrap around close, low expectation
        res = evaluate_decision(
            forecast_point=1000.2,
            upper_bound=1008.0,
            lower_bound=992.0,
            morning_info=self.morning_info,
            friction=0.8,
            min_prob=0.60
        )
        self.assertEqual(res['action'], "WAIT / NO TRADE")

    def test_schema_conformance(self):
        res = evaluate_decision(
            forecast_point=1010.0,
            upper_bound=1015.0,
            lower_bound=1005.0,
            morning_info=self.morning_info
        )
        required_keys = [
            "action", "confidence", "entry", "target", "stop_loss",
            "risk_reward", "sigma", "dynamic_band", "z_long", "z_short",
            "prob_long", "prob_short", "reason"
        ]
        for key in required_keys:
            self.assertIn(key, res)

if __name__ == '__main__':
    unittest.main()
