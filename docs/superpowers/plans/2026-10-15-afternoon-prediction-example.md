# Afternoon Session Prediction Example Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone, production-ready example script `examples/afternoon_session_prediction.py` for predicting VN30F1M afternoon trading sessions using AutoTS4PS, with data resampling, causal morning features, directional metric optimization, and a probabilistic decision engine with dynamic thresholds.

**Architecture:** Resample 5-minute VN30F1M bars into Morning (09:00–11:30) and Afternoon (13:00–14:45) bars with sequential gapless index. Train an AutoTS model using causal whitelist transformers and directional metrics (`dwae`, `oda`, `spl`). Feed the 90% prediction interval and Hinge midpoint into a probabilistic decision engine that calculates Z-score net of friction (0.8 pts) and dynamic ATR breakout thresholds, providing actionable signals (LONG/SHORT/WAIT) with dynamic SL/TP.

**Tech Stack:** Python 3.12, AutoTS (autots4ps), Pandas, NumPy, SciPy (`scipy.stats.norm`), Argparse.

## Global Constraints

- Target script path: `examples/afternoon_session_prediction.py`
- Dataset path: `datasets/VN30F1M_5m.csv`
- Python version floor: `>=3.12`
- Friction cost: `0.8` points per round-trip (brokerage + exchange + tax + slippage)
- Prediction interval: `0.90` (quantile 0.05 and 0.95)
- Whitelist transformers: `DifferencedTransformer`, `PctChangeTransformer`, `RobustScaler`, `StandardScaler`, `EWMAFilter`, `AlignLastValue`
- Excluded transformers: `PCA`, `FastICA`, `HPFilter`, `STLFilter`, `KalmanSmoothing`
- Metric weightings: `dwae: 3.5`, `oda: 2.0`, `spl: 2.0`, `mae: 0.8`, `rmse: 0.4`, `smape: 0.2`

---

### Task 1: Package Environment & Import Resolution

**Files:**
- Modify: `pyproject.toml`
- Create / Symlink: `src/autots` (symlink to `.` inside `src/`)
- Copy: `src/datasets/` from `/home/v002353/Desktop/Git-clone/AutoTS/autots/datasets/`
- Test: `tests/test_package_imports.py`

**Interfaces:**
- Consumes: None (system level)
- Produces: `import autots`, `from autots import AutoTS`, `from autots.tools.transform import GeneralTransformer`

- [ ] **Step 1: Write test for package imports**

Create `tests/test_package_imports.py`:
```python
import unittest
import sys
import os

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

class TestPackageImports(unittest.TestCase):
    def test_autots_import(self):
        import autots
        self.assertIsNotNone(autots.__file__)

    def test_autots_core_classes(self):
        from autots import AutoTS, GeneralTransformer
        self.assertTrue(callable(AutoTS))
        self.assertTrue(callable(GeneralTransformer))

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests/test_package_imports.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'autots'`

- [ ] **Step 3: Fix package configuration and symlink**

1. Copy datasets:
```bash
cp -r /home/v002353/Desktop/Git-clone/AutoTS/autots/datasets src/datasets
```
2. Create symlink `src/autots -> .`:
```bash
ln -sf . src/autots
```
3. Update `pyproject.toml` to declare packages find or package-dir:
```toml
[tool.setuptools.package-dir]
"autots" = "src"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests/test_package_imports.py`
Expected: PASS (2 tests pass)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/autots src/datasets tests/test_package_imports.py
git commit -m "fix(pkg): configure autots package alias and datasets module"
```

---

### Task 2: Data Resampling & Feature Engineering

**Files:**
- Create: `src/tools/halfday_data.py`
- Test: `tests/test_afternoon_data.py`

**Interfaces:**
- Consumes: CSV file path (`datasets/VN30F1M_5m.csv`)
- Produces: `load_and_resample_halfday(csv_path: str, max_days: int = None) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Returns `(df_halfday, df_features)`
  - `df_halfday`: DataFrame with columns `['open', 'high', 'low', 'close', 'volume', 'session', 'day_str']` indexed by sequential datetime/integer.
  - `df_features`: DataFrame formatted for AutoTS with columns `['VN30F1M', 'ret_morning', 'norm_atr', 'vwap_dev', 'vol_ratio']`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_afternoon_data.py`:
```python
import unittest
import numpy as np
import pandas as pd
import tempfile
import os

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests/test_afternoon_data.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.halfday_data'`

- [ ] **Step 3: Implement `src/tools/halfday_data.py`**

Write `src/tools/halfday_data.py`:
```python
"""
Data Resampling & Causal Feature Engineering for VN30F1M Half-Day Trading.
"""
import pandas as pd
import numpy as np

def load_and_resample_halfday(csv_path: str, max_days: int = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads 5-minute VN30F1M CSV, splits into Morning (09:00-11:30) and Afternoon (13:00-14:45),
    aggregates OHLCV, and extracts causal stationary features.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Parse timestamp
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Extract time components
    df['time_str'] = df['date'].dt.strftime('%H:%M:%S')
    df['day_str'] = df['date'].dt.strftime('%Y-%m-%d')
    
    # Tag sessions
    # Morning: 09:00:00 to 11:30:00
    # Afternoon: 13:00:00 to 14:45:00
    is_morning = (df['time_str'] >= '09:00:00') & (df['time_str'] <= '11:30:00')
    is_afternoon = (df['time_str'] >= '13:00:00') & (df['time_str'] <= '14:45:00')
    
    df = df[is_morning | is_afternoon].copy()
    df['session'] = np.where(is_morning[df.index], 'morning', 'afternoon')
    
    # Calculate VWAP component
    df['vol_price'] = df['close'] * df['volume']
    
    # Aggregate by day and session
    grouped = df.groupby(['day_str', 'session'], sort=False, as_index=False)
    
    records = []
    for (day_str, session), grp in grouped:
        first_row = grp.iloc[0]
        last_row = grp.iloc[-1]
        
        open_val = first_row['open']
        high_val = grp['high'].max()
        low_val = grp['low'].min()
        close_val = last_row['close']
        vol_sum = grp['volume'].sum()
        vol_price_sum = grp['vol_price'].sum()
        vwap_val = vol_price_sum / (vol_sum + 1e-9)
        
        records.append({
            'day_str': day_str,
            'session': session,
            'timestamp': last_row['date'],
            'open': open_val,
            'high': high_val,
            'low': low_val,
            'close': close_val,
            'volume': vol_sum,
            'vwap': vwap_val,
        })
        
    df_halfday = pd.DataFrame(records)
    
    # Keep only days that have BOTH morning and afternoon
    day_counts = df_halfday.groupby('day_str')['session'].count()
    valid_days = day_counts[day_counts == 2].index
    df_halfday = df_halfday[df_halfday['day_str'].isin(valid_days)].copy()
    df_halfday = df_halfday.sort_values(['day_str', 'session']).reset_index(drop=True)
    
    if max_days is not None and max_days > 0:
        unique_days = df_halfday['day_str'].unique()
        if len(unique_days) > max_days:
            chosen_days = unique_days[-max_days:]
            df_halfday = df_halfday[df_halfday['day_str'].isin(chosen_days)].reset_index(drop=True)
            
    # Calculate Causal Morning Features
    # Morning Return
    df_halfday['ret_morning'] = (df_halfday['close'] - df_halfday['open']) / (df_halfday['open'] + 1e-9)
    # Morning Normalized Range
    df_halfday['norm_atr'] = (df_halfday['high'] - df_halfday['low']) / (df_halfday['close'] + 1e-9)
    # VWAP deviation
    df_halfday['vwap_dev'] = (df_halfday['close'] - df_halfday['vwap']) / (df_halfday['vwap'] + 1e-9)
    # Volume MA 10
    morning_mask = df_halfday['session'] == 'morning'
    morning_vols = df_halfday.loc[morning_mask, 'volume']
    vol_ma10 = morning_vols.rolling(10, min_periods=1).mean()
    df_halfday.loc[morning_mask, 'vol_ratio'] = morning_vols / (vol_ma10 + 1e-9)
    
    # Forward fill morning features to afternoon bar of the same day so afternoon has morning's context
    for feat in ['ret_morning', 'norm_atr', 'vwap_dev', 'vol_ratio']:
        df_halfday[feat] = df_halfday[feat].ffill().bfill()
        
    # Build AutoTS-ready sequential DataFrame
    # Generate sequential step index to eliminate gap
    n_bars = len(df_halfday)
    step_dates = pd.date_range("2020-01-01", periods=n_bars, freq="12h")
    
    df_features = pd.DataFrame(index=step_dates)
    df_features['VN30F1M'] = df_halfday['close'].values
    df_features['ret_morning'] = df_halfday['ret_morning'].values
    df_features['norm_atr'] = df_halfday['norm_atr'].values
    df_features['vwap_dev'] = df_halfday['vwap_dev'].values
    df_features['vol_ratio'] = df_halfday['vol_ratio'].values
    
    return df_halfday, df_features
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests/test_afternoon_data.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/halfday_data.py tests/test_afternoon_data.py
git commit -m "feat: add halfday data resampling and causal feature engineering"
```

---

### Task 3: Probabilistic Decision & Risk Engine

**Files:**
- Create: `src/tools/decision_engine.py`
- Test: `tests/test_afternoon_decision.py`

**Interfaces:**
- Consumes:
  - `forecast_point`: float
  - `upper_bound`: float
  - `lower_bound`: float
  - `morning_info`: dict (`close`, `high`, `low`, `atr`, `vwap`)
  - `friction`: float (default 0.8)
  - `min_prob`: float (default 0.60)
- Produces: `evaluate_decision(...) -> dict`
  - Output dict schema:
    ```python
    {
        "action": "STRONG LONG" | "STRONG SHORT" | "WAIT / NO TRADE",
        "confidence": float,
        "entry": float,
        "target": float,
        "stop_loss": float,
        "risk_reward": float,
        "sigma": float,
        "dynamic_band": float,
        "z_long": float,
        "z_short": float,
        "prob_long": float,
        "prob_short": float,
        "reason": str
    }
    ```

- [ ] **Step 1: Write the failing test**

Create `tests/test_afternoon_decision.py`:
```python
import unittest
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

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests/test_afternoon_decision.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.decision_engine'`

- [ ] **Step 3: Implement `src/tools/decision_engine.py`**

Write `src/tools/decision_engine.py`:
```python
"""
Probabilistic Decision & Risk Engine for VN30F1M Afternoon Sessions.
"""
import numpy as np
from scipy.stats import norm

def evaluate_decision(
    forecast_point: float,
    upper_bound: float,
    lower_bound: float,
    morning_info: dict,
    friction: float = 0.8,
    min_prob: float = 0.60
) -> dict:
    """
    Evaluates net profit edge, Z-scores, dynamic ATR breakout band,
    and returns a structured trading advisory with dynamic SL/TP.
    """
    close_now = float(morning_info['close'])
    high_morning = float(morning_info.get('high', close_now))
    low_morning = float(morning_info.get('low', close_now))
    atr_morning = float(morning_info.get('atr', high_morning - low_morning))
    
    # 1. Hinge midpoint and forecast sigma
    hinge_point = (upper_bound + lower_bound) / 2.0
    # 90% prediction interval covers 2 * 1.645 sigma
    sigma = (upper_bound - lower_bound) / (2.0 * 1.645)
    sigma = max(sigma, 0.5)  # Avoid division by zero
    
    # 2. Net profit edge after friction
    edge_long = (hinge_point - close_now) - friction
    edge_short = (close_now - hinge_point) - friction
    
    # 3. Z-scores and win probabilities
    z_long = edge_long / sigma
    z_short = edge_short / sigma
    
    prob_long = float(norm.cdf(z_long))
    prob_short = float(norm.cdf(z_short))
    
    # 4. Dynamic breakout threshold
    dynamic_band = max(2.0 * friction, 0.25 * atr_morning)
    
    # 5. Classify action
    if prob_long >= min_prob and (hinge_point - close_now) > dynamic_band:
        action = "STRONG LONG"
        confidence = prob_long
        entry = close_now
        target = float(upper_bound)
        stop_loss = close_now - min(1.5 * sigma, max(atr_morning * 0.75, 2.0))
        reward = abs(target - entry)
        risk = abs(entry - stop_loss)
        risk_reward = round(reward / (risk + 1e-9), 2)
        reason = f"Xác suất Long đạt {prob_long * 100:.1f}% (>= {min_prob * 100:.0f}%) và biên độ kỳ vọng vượt ngưỡng động {dynamic_band:.2f} điểm."
    elif prob_short >= min_prob and (close_now - hinge_point) > dynamic_band:
        action = "STRONG SHORT"
        confidence = prob_short
        entry = close_now
        target = float(lower_bound)
        stop_loss = close_now + min(1.5 * sigma, max(atr_morning * 0.75, 2.0))
        reward = abs(entry - target)
        risk = abs(stop_loss - entry)
        risk_reward = round(reward / (risk + 1e-9), 2)
        reason = f"Xác suất Short đạt {prob_short * 100:.1f}% (>= {min_prob * 100:.0f}%) và biên độ kỳ vọng vượt ngưỡng động {dynamic_band:.2f} điểm."
    else:
        action = "WAIT / NO TRADE"
        confidence = max(prob_long, prob_short)
        entry = close_now
        target = hinge_point
        stop_loss = close_now
        risk_reward = 0.0
        reason = f"Kỳ vọng ròng chưa đủ lớn so với chi phí ma sát ({friction:.1f} điểm) hoặc dải dao động hẹp."

    return {
        "action": action,
        "confidence": confidence,
        "entry": entry,
        "target": target,
        "stop_loss": stop_loss,
        "risk_reward": risk_reward,
        "hinge": hinge_point,
        "sigma": sigma,
        "dynamic_band": dynamic_band,
        "z_long": z_long,
        "z_short": z_short,
        "prob_long": prob_long,
        "prob_short": prob_short,
        "reason": reason
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests/test_afternoon_decision.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/tools/decision_engine.py tests/test_afternoon_decision.py
git commit -m "feat: add probabilistic decision and risk engine"
```

---

### Task 4: AutoTS Model Builder Configuration

**Files:**
- Create: `src/models/afternoon_autots.py`
- Test: `tests/test_afternoon_model.py`

**Interfaces:**
- Consumes: `fast_mode: bool = False`
- Produces: `build_afternoon_model(fast_mode: bool = False) -> AutoTS`
  - Returns initialized AutoTS instance with whitelist transformers, financial metric weights, and fast parallel models.

- [ ] **Step 1: Write the failing test**

Create `tests/test_afternoon_model.py`:
```python
import unittest
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

if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests/test_afternoon_model.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'models.afternoon_autots'`

- [ ] **Step 3: Implement `src/models/afternoon_autots.py`**

Write `src/models/afternoon_autots.py`:
```python
"""
Custom AutoTS Configuration for VN30F1M Afternoon Sessions.
"""
from autots import AutoTS

def build_afternoon_model(fast_mode: bool = False) -> AutoTS:
    """
    Initializes AutoTS with directional metrics and causal whitelist transformers.
    """
    financial_transformer_list = {
        "DifferencedTransformer": 0.30,
        "PctChangeTransformer": 0.20,
        "RobustScaler": 0.15,
        "StandardScaler": 0.10,
        "EWMAFilter": 0.15,
        "AlignLastValue": 0.10,
    }

    metric_weighting = {
        'dwae_weighting': 3.5,     # Directional Weighted Absolute Error
        'oda_weighting': 2.0,      # Origin Direction Accuracy
        'spl_weighting': 2.0,      # Scaled Pinball Loss (prediction interval coverage)
        'contour_weighting': 1.0,  # Slope and curvature
        'mae_weighting': 0.8,
        'rmse_weighting': 0.4,
        'smape_weighting': 0.2,
        'runtime_weighting': 0.05,
    }

    model_candidates = [
        "RollingRegression",
        "WindowRegression",
        "VAR",
        "VECM",
        "BallTreeMultivariateMotif",
    ]

    max_generations = 5 if fast_mode else 15
    num_validations = 1 if fast_mode else 2

    return AutoTS(
        forecast_length=1,
        frequency="infer",
        prediction_interval=0.90,
        ensemble='simple',
        model_list=model_candidates,
        transformer_list=financial_transformer_list,
        transformer_max_depth=2,
        metric_weighting=metric_weighting,
        max_generations=max_generations,
        num_validations=num_validations,
        validation_method="backwards",
        n_jobs="auto",
        verbose=0
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests/test_afternoon_model.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/afternoon_autots.py tests/test_afternoon_model.py
git commit -m "feat: add specialized autots model configuration for afternoon sessions"
```

---

### Task 5: End-to-End Afternoon Prediction Script (`examples/afternoon_session_prediction.py`)

**Files:**
- Create: `examples/afternoon_session_prediction.py`
- Test: CLI execution with `--mode advisor --fast` and `--mode backtest --days 5 --fast`

**Interfaces:**
- Consumes:
  - CLI Arguments: `--mode`, `--csv`, `--days`, `--date`, `--fast`
  - Modules from Tasks 2, 3, 4
- Produces:
  - Formatted Advisory Report for midday trader
  - Comprehensive Backtest Summary Table with Win Rate, ODA, Net PnL, and Max Drawdown

- [ ] **Step 1: Implement `examples/afternoon_session_prediction.py`**

Write `examples/afternoon_session_prediction.py`:
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Afternoon Session Prediction & Advisory for VN30F1M using AutoTS4PS.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np

# Ensure src is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from tools.halfday_data import load_and_resample_halfday
from tools.decision_engine import evaluate_decision
from models.afternoon_autots import build_afternoon_model

def print_banner(title: str):
    print("=" * 80)
    print(title.center(80))
    print("=" * 80)

def run_advisor(csv_path: str, target_date: str = None, fast_mode: bool = False):
    """
    Simulates midday 11:30 decision for a specific date (or the latest available date).
    """
    print_banner("BẢN TIN KHUYẾN NGHỊ GIAO DỊCH PHIÊN CHIỀU (LUNCHTIME ADVISOR)")
    print("Loading and preparing half-day dataset...")
    df_halfday, df_features = load_and_resample_halfday(csv_path)

    # Find candidate morning bar
    if target_date:
        matching = df_halfday[(df_halfday['day_str'] == target_date) & (df_halfday['session'] == 'morning')]
        if matching.empty:
            print(f"Error: Target date {target_date} morning session not found in dataset.")
            return
        target_idx = matching.index[0]
    else:
        # Default to the latest morning bar
        morning_bars = df_halfday[df_halfday['session'] == 'morning']
        target_idx = morning_bars.index[-1]
        target_date = df_halfday.loc[target_idx, 'day_str']

    morning_bar = df_halfday.loc[target_idx]
    actual_afternoon_bar = None
    if target_idx + 1 < len(df_halfday) and df_halfday.loc[target_idx + 1, 'session'] == 'afternoon':
        actual_afternoon_bar = df_halfday.loc[target_idx + 1]

    # Training slice: all bars up to and including this morning bar
    train_features = df_features.iloc[:target_idx + 1]

    print(f"\n📅 Ngày phân tích: {target_date} | Thời điểm: 11:30:00")
    print(f"📊 Nến Sáng: Close = {morning_bar['close']:.2f} | High = {morning_bar['high']:.2f} | Low = {morning_bar['low']:.2f} | Biên độ = {morning_bar['high'] - morning_bar['low']:.2f} điểm")
    print(f"⚡ VWAP Sáng: {morning_bar['vwap']:.2f} (Độ lệch: {morning_bar['vwap_dev']*100:.2f}%) | Vol Sáng: {int(morning_bar['volume']):,} HĐ")
    
    print("\n⏳ AutoTS đang huấn luyện & tìm kiếm mô hình tối ưu (Fast Mode = {})...".format(fast_mode))
    model = build_afternoon_model(fast_mode=fast_mode)
    
    series_weights = {
        'VN30F1M': 100.0,
        'ret_morning': 0.0,
        'norm_atr': 0.0,
        'vwap_dev': 0.0,
        'vol_ratio': 0.0,
    }
    
    # Train
    model = model.fit(train_features, weights=series_weights)
    prediction = model.predict()
    
    pred_point = float(prediction.forecast['VN30F1M'].iloc[0])
    upper_bound = float(prediction.upper_forecast['VN30F1M'].iloc[0])
    lower_bound = float(prediction.lower_forecast['VN30F1M'].iloc[0])
    
    # Evaluate decision
    morning_info = {
        'close': morning_bar['close'],
        'high': morning_bar['high'],
        'low': morning_bar['low'],
        'atr': morning_bar['high'] - morning_bar['low'],
        'vwap': morning_bar['vwap']
    }
    decision = evaluate_decision(pred_point, upper_bound, lower_bound, morning_info)
    
    print("-" * 80)
    print("🤖 DỰ BÁO TỪ AUTOTS4PS:")
    print(f"   - Mô hình tối ưu: {model.best_model_name}")
    print(f"   - Dự báo Hinge Midpoint: {decision['hinge']:.2f} điểm (Kỳ vọng: {decision['hinge'] - morning_bar['close']:+.2f} điểm)")
    print(f"   - Dải tin cậy 90%: [{lower_bound:.2f}  --->  {upper_bound:.2f}]")
    print(f"   - Độ biến động ước lượng (Sigma): {decision['sigma']:.2f} điểm")
    print(f"   - Ngưỡng Breakout động: ±{decision['dynamic_band']:.2f} điểm")
    print("-" * 80)
    print("📈 PHÂN TÍCH XÁC SUẤT ĐỘNG (Sau trừ phí ma sát 0.8 điểm):")
    print(f"   - Xác suất LONG có lãi: {decision['prob_long'] * 100:.1f}%  (Z-Score = {decision['z_long']:+.2f})")
    print(f"   - Xác suất SHORT có lãi: {decision['prob_short'] * 100:.1f}%  (Z-Score = {decision['z_short']:+.2f})")
    print("-" * 80)
    print("🎯 KHUYẾN NGHỊ HÀNH ĐỘNG LÚC 13:00:")
    print(f"   👉 TÍN HIỆU: [{decision['action']}] (Độ tin cậy: {decision['confidence'] * 100:.1f}%)")
    if decision['action'] != "WAIT / NO TRADE":
        print(f"   📍 Điểm vào lệnh: Mở vị thế quanh {decision['entry']:.2f} lúc 13:00 - 13:05")
        print(f"   🎯 Chốt lời (Take-Profit): {decision['target']:.2f} ({decision['target'] - decision['entry']:+.2f} điểm)")
        print(f"   🛑 Cắt lỗ (Stop-Loss): {decision['stop_loss']:.2f} ({decision['stop_loss'] - decision['entry']:+.2f} điểm)")
        print(f"   ⚖️ Tỷ lệ Risk / Reward: 1 : {decision['risk_reward']}")
    print(f"   💡 Lý do: {decision['reason']}")
    
    if actual_afternoon_bar is not None:
        print("-" * 80)
        actual_close = actual_afternoon_bar['close']
        actual_diff = actual_close - morning_bar['close']
        print("🔍 KẾT QUẢ THỰC TẾ PHIÊN CHIỀU:")
        print(f"   - Giá đóng cửa chiều thực tế: {actual_close:.2f} ({actual_diff:+.2f} điểm)")
        if decision['action'] == "STRONG LONG":
            pnl = actual_diff - 0.8
            print(f"   - Kết quả lệnh LONG: {pnl:+.2f} điểm ({'THẮNG ✅' if pnl > 0 else 'THUA ❌'})")
        elif decision['action'] == "STRONG SHORT":
            pnl = -actual_diff - 0.8
            print(f"   - Kết quả lệnh SHORT: {pnl:+.2f} điểm ({'THẮNG ✅' if pnl > 0 else 'THUA ❌'})")
        else:
            print("   - Đứng ngoài bảo toàn vốn.")
    print("=" * 80)

def run_backtest(csv_path: str, n_days: int = 15, fast_mode: bool = True):
    """
    Runs walk-forward backtest across the last N test days.
    """
    print_banner(f"CHẠY WALK-FORWARD BACKTEST {n_days} PHIÊN GẦN NHẤT")
    df_halfday, df_features = load_and_resample_halfday(csv_path)
    
    unique_days = df_halfday['day_str'].unique()
    if len(unique_days) < n_days + 10:
        print(f"Error: Dataset too small for {n_days} backtest days.")
        return
        
    test_days = unique_days[-n_days:]
    print(f"Thời gian kiểm thử: {test_days[0]} -> {test_days[-1]} ({n_days} phiên)")
    
    results = []
    
    series_weights = {
        'VN30F1M': 100.0,
        'ret_morning': 0.0,
        'norm_atr': 0.0,
        'vwap_dev': 0.0,
        'vol_ratio': 0.0,
    }
    
    for i, day in enumerate(test_days):
        print(f"\n[{i+1}/{n_days}] Kiểm thử ngày {day}...", end=" ")
        morning_idx = df_halfday[(df_halfday['day_str'] == day) & (df_halfday['session'] == 'morning')].index[0]
        afternoon_idx = df_halfday[(df_halfday['day_str'] == day) & (df_halfday['session'] == 'afternoon')].index[0]
        
        train_features = df_features.iloc[:morning_idx + 1]
        morning_bar = df_halfday.loc[morning_idx]
        afternoon_bar = df_halfday.loc[afternoon_idx]
        
        model = build_afternoon_model(fast_mode=fast_mode)
        try:
            model = model.fit(train_features, weights=series_weights)
            prediction = model.predict()
            
            pred_point = float(prediction.forecast['VN30F1M'].iloc[0])
            upper_bound = float(prediction.upper_forecast['VN30F1M'].iloc[0])
            lower_bound = float(prediction.lower_forecast['VN30F1M'].iloc[0])
            
            morning_info = {
                'close': morning_bar['close'],
                'high': morning_bar['high'],
                'low': morning_bar['low'],
                'atr': morning_bar['high'] - morning_bar['low'],
                'vwap': morning_bar['vwap']
            }
            decision = evaluate_decision(pred_point, upper_bound, lower_bound, morning_info)
            
            actual_ret = afternoon_bar['close'] - morning_bar['close']
            
            # PnL net of friction
            pnl = 0.0
            is_win = False
            if decision['action'] == "STRONG LONG":
                pnl = actual_ret - 0.8
                is_win = pnl > 0
            elif decision['action'] == "STRONG SHORT":
                pnl = -actual_ret - 0.8
                is_win = pnl > 0
                
            oda_correct = (decision['hinge'] > morning_bar['close'] and actual_ret > 0) or (decision['hinge'] < morning_bar['close'] and actual_ret < 0)
            
            results.append({
                'day': day,
                'action': decision['action'],
                'pnl': pnl,
                'is_win': is_win,
                'oda_correct': oda_correct,
                'actual_ret': actual_ret,
                'confidence': decision['confidence']
            })
            print(f"Tín hiệu: {decision['action']:<15} | PnL: {pnl:+6.2f} pts")
        except Exception as e:
            print(f"Lỗi: {e}")
            
    # Summary
    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("Không có kết quả nào được tạo.")
        return
        
    trades = df_res[df_res['action'] != 'WAIT / NO TRADE']
    n_trades = len(trades)
    n_waits = len(df_res) - n_trades
    
    total_pnl = trades['pnl'].sum() if n_trades > 0 else 0.0
    win_rate = (trades['is_win'].sum() / n_trades * 100.0) if n_trades > 0 else 0.0
    oda_rate = df_res['oda_correct'].mean() * 100.0
    
    # Cumulative PnL and Max DD
    cum_pnl = trades['pnl'].cumsum() if n_trades > 0 else pd.Series([0.0])
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = drawdowns.min() if not drawdowns.empty else 0.0
    
    print("\n" + "=" * 80)
    print(f"TỔNG KẾT BACKTEST {len(df_res)} PHIÊN".center(80))
    print("=" * 80)
    print(f"- Tổng số phiên kiểm thử: {len(df_res)}")
    print(f"- Số phiên có tín hiệu (Trades): {n_trades} ({n_trades/len(df_res)*100:.1f}%)")
    print(f"- Số phiên ĐỨNG NGOÀI (Wait): {n_waits} ({n_waits/len(df_res)*100:.1f}%)")
    print(f"- Tỷ lệ dự báo đúng hướng chiều (ODA): {oda_rate:.1f}%")
    print(f"- Tỷ lệ lệnh thắng (Win Rate sau phí 0.8): {win_rate:.1f}%")
    print(f"- Tổng PnL ròng (Net Points): {total_pnl:+.2f} điểm")
    if n_trades > 0:
        print(f"- Lợi nhuận trung bình / lệnh: {total_pnl / n_trades:+.2f} điểm")
    print(f"- Maximum Drawdown (MDD): {max_dd:.2f} điểm")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="AutoTS4PS Afternoon Session Prediction Example")
    parser.add_argument("--mode", choices=["advisor", "backtest"], default="advisor", help="Execution mode: advisor or backtest")
    parser.add_argument("--csv", default=os.path.join(PROJECT_ROOT, "datasets", "VN30F1M_5m.csv"), help="Path to 5m CSV dataset")
    parser.add_argument("--date", default=None, help="Specific date for advisor mode (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=15, help="Number of test days for backtest mode")
    parser.add_argument("--fast", action="store_true", help="Enable fast mode for quick training")
    
    args = parser.parse_args()
    
    if args.mode == "advisor":
        run_advisor(args.csv, target_date=args.date, fast_mode=args.fast)
    elif args.mode == "backtest":
        run_backtest(args.csv, n_days=args.days, fast_mode=args.fast)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run script in advisor mode with `--fast`**

Run: `uv run python examples/afternoon_session_prediction.py --mode advisor --fast`
Expected: Outputs complete advisory report for the latest morning session with no runtime error.

- [ ] **Step 3: Run script in backtest mode with `--days 3 --fast`**

Run: `uv run python examples/afternoon_session_prediction.py --mode backtest --days 3 --fast`
Expected: Completes 3-day walk-forward backtest and prints summary table.

- [ ] **Step 4: Commit**

```bash
git add examples/afternoon_session_prediction.py
git commit -m "feat: add afternoon session prediction and lunchtime advisory example"
```
