#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2: Afternoon Extremes (High/Low) & Volatility Regime Forecasting for VN30F1M.

Instead of predicting only the single closing price at 14:45, this script forecasts:
  1. Afternoon High (Point forecast & 90% Upper Bound)
  2. Afternoon Low (Point forecast & 90% Lower Bound)
  3. Afternoon Range (Biên độ dao động High - Low)
  4. Volatility Regime Classification:
     - EXPANSION / BREAKOUT (High Volatility: Range >= 18.0 pts)
     - NORMAL VOLATILITY (8.5 <= Range < 18.0 pts)
     - COMPRESSION / SIDEWAY (Low Volatility: Range < 8.5 pts)
  5. Operational strategy generation (Breakout triggers vs Mean-reversion grid bounds)
  6. Standard quantitative evaluation: MAE/RMSE of High/Low/Range, Envelope Coverage 90%, and Regime Accuracy.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Ensure src is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from autots.tools.halfday_data import load_and_resample_halfday
except ImportError:
    try:
        from tools.halfday_data import load_and_resample_halfday
    except ImportError:
        AUTOTS_DIR = os.path.join(SRC_DIR, "autots")
        if AUTOTS_DIR not in sys.path:
            sys.path.insert(0, AUTOTS_DIR)
        from tools.halfday_data import load_and_resample_halfday

FEATURE_COLS = [
    "ret_morning",
    "norm_atr",
    "vwap_dev",
    "vol_ratio",
    "ret_last30m",
    "morning_clv",
    "overnight_gap",
    "prev_afternoon_ret",
    "prev_ret_morning",
]

# Regime thresholds based on VN30F1M empirical distribution
THRESHOLD_EXPANSION = 18.0   # Top quartile: strong trending / expansion
THRESHOLD_COMPRESSION = 8.5  # Bottom quartile: tight chop / compression


def resolve_csv_path(path: str) -> str:
    if not path:
        return path
    if os.path.exists(path):
        return path
    candidates = [
        os.path.join(PROJECT_ROOT, path),
        os.path.join(PROJECT_ROOT, "datasets", os.path.basename(path)),
        os.path.join(PROJECT_ROOT, "datasets", "train.csv") if "train" in path else "",
        os.path.join(PROJECT_ROOT, "datasets", "test.csv") if "test" in path else "",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return path


def print_banner(title: str):
    print("=" * 85)
    print(title.center(85))
    print("=" * 85)


def classify_regime(range_val: float) -> str:
    """Classifies afternoon session into volatility regimes."""
    if range_val >= THRESHOLD_EXPANSION:
        return "EXPANSION / BREAKOUT"
    elif range_val < THRESHOLD_COMPRESSION:
        return "COMPRESSION / SIDEWAY"
    else:
        return "NORMAL VOLATILITY"


class AfternoonExtremesModel:
    """
    Causal regression engine predicting Afternoon High, Low, and Range,
    with 90% probabilistic envelope bounds using quantile loss.
    """

    def __init__(self, n_estimators: int = 60, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()

        # Point models (least squares)
        self.model_high_point = GradientBoostingRegressor(
            loss="squared_error", n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.model_low_point = GradientBoostingRegressor(
            loss="squared_error", n_estimators=self.n_estimators, random_state=self.random_state
        )

        # 90% Envelope models (Quantile loss: 95th percentile for high, 5th percentile for low)
        self.model_high_upper = GradientBoostingRegressor(
            loss="quantile", alpha=0.95, n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.model_low_lower = GradientBoostingRegressor(
            loss="quantile", alpha=0.05, n_estimators=self.n_estimators, random_state=self.random_state
        )

    def fit(self, X: np.ndarray, delta_high: np.ndarray, delta_low: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model_high_point.fit(X_scaled, delta_high)
        self.model_low_point.fit(X_scaled, delta_low)
        self.model_high_upper.fit(X_scaled, delta_high)
        self.model_low_lower.fit(X_scaled, delta_low)
        return self

    def predict(self, x: np.ndarray, morning_close: float) -> dict:
        x_scaled = self.scaler.transform(x.reshape(1, -1))

        # Raw deltas
        dh_point = float(self.model_high_point.predict(x_scaled)[0])
        dl_point = float(self.model_low_point.predict(x_scaled)[0])
        dh_upper = float(self.model_high_upper.predict(x_scaled)[0])
        dl_lower = float(self.model_low_lower.predict(x_scaled)[0])

        # Logical constraints: Delta High >= 0, Delta Low <= 0
        dh_point = max(dh_point, 0.5)
        dl_point = min(dl_point, -0.5)
        dh_upper = max(dh_upper, dh_point + 1.0)
        dl_lower = min(dl_lower, dl_point - 1.0)

        pred_high = morning_close + dh_point
        pred_low = morning_close + dl_point
        high_upper = morning_close + dh_upper
        low_lower = morning_close + dl_lower

        pred_range = pred_high - pred_low
        regime = classify_regime(pred_range)

        return {
            "morning_close": morning_close,
            "pred_high": pred_high,
            "pred_low": pred_low,
            "pred_range": pred_range,
            "high_upper": high_upper,
            "low_lower": low_lower,
            "dh_point": dh_point,
            "dl_point": dl_point,
            "regime": regime,
        }


def prepare_extremes_dataset(df_half: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairs morning features with afternoon extremes (High, Low, Range).
    """
    morning_mask = df_half["session"] == "morning"
    afternoon_mask = df_half["session"] == "afternoon"

    df_morn = df_half[morning_mask].reset_index(drop=True)
    df_aft = df_half[afternoon_mask].reset_index(drop=True)

    df_paired = df_morn.copy()
    df_paired["aft_high"] = df_aft["high"].values
    df_paired["aft_low"] = df_aft["low"].values
    df_paired["aft_range"] = (df_aft["high"] - df_aft["low"]).values
    df_paired["delta_high"] = (df_aft["high"] - df_morn["close"]).values
    df_paired["delta_low"] = (df_aft["low"] - df_morn["close"]).values

    return df_morn, df_paired


def generate_strategy_advisory(pred: dict, morning_row: pd.Series) -> dict:
    """
    Generates actionable trade setups based on predicted extremes and volatility regime.
    """
    regime = pred["regime"]
    m_high = float(morning_row["high"])
    m_low = float(morning_row["low"])
    m_close = float(morning_row["close"])
    pred_high = pred["pred_high"]
    pred_low = pred["pred_low"]
    pred_range = pred["pred_range"]

    if regime == "EXPANSION / BREAKOUT":
        # Trending / expansion strategy: Place breakout stops outside morning box
        long_trigger = m_high + 0.5
        long_target = pred_high
        long_sl = m_close

        short_trigger = m_low - 0.5
        short_target = pred_low
        short_sl = m_close

        action = "BREAKOUT EXPANSION (BUY STOP / SELL STOP)"
        details = (
            f"Kỳ vọng chiều biến động mạnh (Range={pred_range:.1f} pts >= {THRESHOLD_EXPANSION} pts). "
            f"Kích hoạt chiến lược Breakout hai đầu hộp sáng:\n"
            f"      - Lệnh Long Stop: Đặt Mua tại {long_trigger:.2f} (vượt Đỉnh sáng {m_high:.2f}), "
            f"Mục tiêu: {long_target:.2f}, Cắt lỗ: {long_sl:.2f}\n"
            f"      - Lệnh Short Stop: Đặt Bán tại {short_trigger:.2f} (thủng Đáy sáng {m_low:.2f}), "
            f"Mục tiêu: {short_target:.2f}, Cắt lỗ: {short_sl:.2f}"
        )
    elif regime == "COMPRESSION / SIDEWAY":
        # Range-bound chop: Mean-reversion fade at envelope boundaries
        action = "MEAN-REVERSION GRID (FADE EXTREMES)"
        details = (
            f"Kỳ vọng chiều tích lũy biên hẹp (Range={pred_range:.1f} pts < {THRESHOLD_COMPRESSION} pts). "
            f"Kích hoạt chiến lược Đánh biên Sideway:\n"
            f"      - Mua gom hỗ trợ: Limit Buy tại {pred_low:.2f}, Chốt lời: {m_close:.2f}, Cắt lỗ: {pred['low_lower']:.2f}\n"
            f"      - Bán tỉa kháng cự: Limit Sell tại {pred_high:.2f}, Chốt lời: {m_close:.2f}, Cắt lỗ: {pred['high_upper']:.2f}"
        )
    else:
        # Normal volatility: Follow momentum direction
        momentum = float(morning_row["ret_last30m"])
        clv = float(morning_row["morning_clv"])
        bias = "LONG" if (momentum > 0 and clv > 0) else ("SHORT" if (momentum < 0 and clv < 0) else "NEUTRAL")

        action = f"TREND-MOMENTUM ({bias})"
        details = (
            f"Biên độ chiều ở mức trung bình ({pred_range:.1f} pts). "
            f"Định hướng theo đà 30p cuối sáng ({momentum:+.2f}%, CLV={clv:+.2f}) -> Thiên hướng {bias}."
        )

    return {"action": action, "details": details}


def run_predict(
    csv_path: str = None,
    train_csv: str = None,
    target_date: str = None,
) -> dict:
    """Runs Extremes & Volatility Regime prediction for a single target date."""
    print_banner("DỰ BÁO CỰC TRỊ PHIÊN CHIỀU (HIGH/LOW) & CHẾ ĐỘ BIẾN ĐỘNG VN30F1M")
    train_p = resolve_csv_path(train_csv or "datasets/train.csv")
    test_p = resolve_csv_path(csv_path or "datasets/test.csv")

    print(f"Đang nạp dữ liệu: {os.path.basename(train_p)} & {os.path.basename(test_p)}...")
    df_half, _ = load_and_resample_halfday([train_p, test_p])
    df_morn, df_paired = prepare_extremes_dataset(df_half)

    if target_date:
        match = df_morn[df_morn["day_str"] == target_date]
        if match.empty:
            print(f"Lỗi: Không tìm thấy ngày {target_date} trong dữ liệu.")
            return None
        target_idx = match.index[0]
    else:
        target_idx = len(df_morn) - 1
        target_date = df_morn.loc[target_idx, "day_str"]

    target_row = df_morn.loc[target_idx]
    morning_close = float(target_row["close"])

    print(f"\n📅 Ngày dự báo: {target_date} | Chốt dữ liệu sáng: 11:30:00")
    print(
        f"📊 Phiên Sáng: Open = {target_row['open']:.2f} | High = {target_row['high']:.2f} "
        f"| Low = {target_row['low']:.2f} | Close = {morning_close:.2f} "
        f"(Biên độ Sáng = {target_row['high'] - target_row['low']:.2f} pts)"
    )
    print(
        f"⚡ Causal Indicators: Norm ATR={target_row['norm_atr']:.2f}% | "
        f"VWAP Dev={target_row['vwap_dev']:+.2f}% | Momentum 30p={target_row['ret_last30m']:+.2f}% | "
        f"CLV={target_row['morning_clv']:+.2f}"
    )

    # Strictly causal fit on all historical sessions before target_idx
    train_slice = df_paired.iloc[:target_idx].copy()
    X_train = train_slice[FEATURE_COLS].fillna(0).values
    dh_train = train_slice["delta_high"].values
    dl_train = train_slice["delta_low"].values

    model = AfternoonExtremesModel()
    model.fit(X_train, dh_train, dl_train)

    x_target = target_row[FEATURE_COLS].fillna(0).values
    pred = model.predict(x_target, morning_close)
    advisory = generate_strategy_advisory(pred, target_row)

    print("\n" + "-" * 85)
    print("📈 KẾT QUẢ DỰ BÁO CỰC TRỊ VÀ BIÊN ĐỘ PHIÊN CHIỀU:")
    print(f"   - Đỉnh chiều dự kiến (High)    : {pred['pred_high']:.2f} điểm (Biên độ kỳ vọng: +{pred['dh_point']:.2f} pts)")
    print(f"   - Đáy chiều dự kiến (Low)      : {pred['pred_low']:.2f} điểm (Biên độ kỳ vọng: {pred['dl_point']:.2f} pts)")
    print(f"   - Biên độ chiều dự báo (Range) : {pred['pred_range']:.2f} điểm")
    print(f"   - Khung bao phủ 90% (Envelope) : [{pred['low_lower']:.2f}  --->  {pred['high_upper']:.2f}]")
    print(f"   - Phân loại Chế độ Biến động   : 【 {pred['regime']} 】")

    print("-" * 85)
    print(f"🧭 CHIẾN LƯỢC GIAO DỊCH GỢI Ý: {advisory['action']}")
    print(f"   {advisory['details']}")

    # Verification against actual afternoon session if present
    if target_idx < len(df_paired):
        act_row = df_paired.loc[target_idx]
        act_high = float(act_row["aft_high"])
        act_low = float(act_row["aft_low"])
        act_range = float(act_row["aft_range"])
        act_regime = classify_regime(act_range)

        err_high = pred["pred_high"] - act_high
        err_low = pred["pred_low"] - act_low
        err_range = pred["pred_range"] - act_range
        in_envelope = bool(act_low >= pred["low_lower"] and act_high <= pred["high_upper"])
        regime_match = bool(pred["regime"] == act_regime)

        print("-" * 85)
        print("🔍 ĐỐI CHIẾU KẾT QUẢ CỰC TRỊ THỰC TẾ TRONG PHIÊN CHIỀU:")
        print(f"   - Cực trị thực tế              : High = {act_high:.2f} | Low = {act_low:.2f} | Range = {act_range:.2f} pts")
        print(f"   - Sai số Đỉnh High (|Error|)   : {abs(err_high):.2f} pts (Dự báo: {pred['pred_high']:.2f} vs Thực tế: {act_high:.2f})")
        print(f"   - Sai số Đáy Low (|Error|)     : {abs(err_low):.2f} pts (Dự báo: {pred['pred_low']:.2f} vs Thực tế: {act_low:.2f})")
        print(f"   - Sai số Biên độ Range (|Error|): {abs(err_range):.2f} pts (Dự báo: {pred['pred_range']:.2f} vs Thực tế: {act_range:.2f})")
        print(f"   - Nằm trọn trong Khung 90%     : {'CÓ ✅ (Thực tế nằm gọn trong Envelope)' if in_envelope else 'KHÔNG ❌ (Vượt khung bao phủ)'}")
        print(f"   - Phân loại Chế độ Biến động   : {'ĐÚNG ✅' if regime_match else 'SAI ❌'} (Dự báo: {pred['regime']} vs Thực tế: {act_regime})")
    print("=" * 85)
    return pred


def run_evaluate(
    train_csv: str = None,
    test_csv: str = None,
    n_days: int = None,
    output_csv: str = None,
) -> tuple[pd.DataFrame, dict]:
    """Runs walk-forward evaluation on afternoon extremes across the test dataset."""
    train_p = resolve_csv_path(train_csv or "datasets/train.csv")
    test_p = resolve_csv_path(test_csv or "datasets/test.csv")

    print_banner(f"KIỂM ĐỊNH WALK-FORWARD CỰC TRỊ & BIÊN ĐỘ ({os.path.basename(test_p)})")
    print(f"Đang nạp dữ liệu: {os.path.basename(train_p)} & {os.path.basename(test_p)}...")
    df_half, _ = load_and_resample_halfday([train_p, test_p])
    df_morn, df_paired = prepare_extremes_dataset(df_half)

    df_test_only, _ = load_and_resample_halfday(test_p)
    test_days = list(df_test_only["day_str"].unique())

    if n_days is not None and n_days > 0 and len(test_days) > n_days:
        test_days = test_days[-n_days:]

    total_test = len(test_days)
    print(f"Khoảng thời gian kiểm định: {test_days[0]} -> {test_days[-1]} ({total_test} phiên chiều)\n")

    results = []
    for i, day in enumerate(test_days):
        matching = df_morn[df_morn["day_str"] == day]
        if matching.empty:
            continue
        idx = matching.index[0]

        train_slice = df_paired.iloc[:idx].copy()
        X_train = train_slice[FEATURE_COLS].fillna(0).values
        dh_train = train_slice["delta_high"].values
        dl_train = train_slice["delta_low"].values

        model = AfternoonExtremesModel()
        model.fit(X_train, dh_train, dl_train)

        target_row = df_morn.loc[idx]
        morning_close = float(target_row["close"])
        x_target = target_row[FEATURE_COLS].fillna(0).values
        pred = model.predict(x_target, morning_close)

        act_row = df_paired.loc[idx]
        act_high = float(act_row["aft_high"])
        act_low = float(act_row["aft_low"])
        act_range = float(act_row["aft_range"])
        act_regime = classify_regime(act_range)

        err_high = pred["pred_high"] - act_high
        err_low = pred["pred_low"] - act_low
        err_range = pred["pred_range"] - act_range
        in_envelope = bool(act_low >= pred["low_lower"] and act_high <= pred["high_upper"])
        regime_correct = bool(pred["regime"] == act_regime)

        results.append({
            "day": day,
            "morning_close": morning_close,
            "pred_high": pred["pred_high"],
            "act_high": act_high,
            "err_high": err_high,
            "abs_err_high": abs(err_high),
            "pred_low": pred["pred_low"],
            "act_low": act_low,
            "err_low": err_low,
            "abs_err_low": abs(err_low),
            "pred_range": pred["pred_range"],
            "act_range": act_range,
            "err_range": err_range,
            "abs_err_range": abs(err_range),
            "low_lower": pred["low_lower"],
            "high_upper": pred["high_upper"],
            "in_envelope_90": in_envelope,
            "pred_regime": pred["regime"],
            "act_regime": act_regime,
            "regime_correct": regime_correct,
        })

        print(
            f"[{i+1:3d}/{total_test:3d}] {day} | High: {pred['pred_high']:7.2f} (tt:{act_high:7.2f}) "
            f"| Low: {pred['pred_low']:7.2f} (tt:{act_low:7.2f}) | Range: {pred['pred_range']:5.1f} (tt:{act_range:5.1f}) "
            f"| Env90: {'CÓ ' if in_envelope else 'KHÔNG'} | Chế độ: {'ĐÚNG' if regime_correct else 'SAI '}"
        )

    df_res = pd.DataFrame(results)
    if df_res.empty:
        return df_res, {}

    mae_high = float(df_res["abs_err_high"].mean())
    rmse_high = float(np.sqrt((df_res["err_high"] ** 2).mean()))

    mae_low = float(df_res["abs_err_low"].mean())
    rmse_low = float(np.sqrt((df_res["err_low"] ** 2).mean()))

    mae_range = float(df_res["abs_err_range"].mean())
    rmse_range = float(np.sqrt((df_res["err_range"] ** 2).mean()))

    coverage_envelope = float(df_res["in_envelope_90"].mean() * 100.0)
    regime_acc = float(df_res["regime_correct"].mean() * 100.0)

    metrics = {
        "count": len(df_res),
        "mae_high": mae_high,
        "rmse_high": rmse_high,
        "mae_low": mae_low,
        "rmse_low": rmse_low,
        "mae_range": mae_range,
        "rmse_range": rmse_range,
        "envelope_coverage_90": coverage_envelope,
        "regime_accuracy": regime_acc,
    }

    print("\n" + "=" * 85)
    print("BẢNG TỔNG HỢP CHỈ SỐ ĐÁNH GIÁ CỰC TRỊ & BIÊN ĐỘ TRÊN TẬP TEST".center(85))
    print("=" * 85)
    print(f"  1. Số phiên kiểm thử (Test Sessions)          : {metrics['count']} phiên")
    print(f"  2. MAE Cực trị Đỉnh (High MAE)                : {metrics['mae_high']:.2f} điểm (RMSE: {metrics['rmse_high']:.2f})")
    print(f"  3. MAE Cực trị Đáy (Low MAE)                  : {metrics['mae_low']:.2f} điểm (RMSE: {metrics['rmse_low']:.2f})")
    print(f"  4. MAE Biên độ dao động (Range MAE)           : {metrics['mae_range']:.2f} điểm (RMSE: {metrics['rmse_range']:.2f})")
    print(f"  5. Tỷ lệ Bao phủ Khung 90% (Envelope Coverage): {metrics['envelope_coverage_90']:.2f}% (Cả High & Low nằm gọn)")
    print(f"  6. Độ chính xác nhận diện Chế độ Biến động    : {metrics['regime_accuracy']:.2f}% (High vs Normal vs Low Volatility)")
    print("=" * 85)

    if output_csv:
        df_res.to_csv(output_csv, index=False)
        print(f"💾 Đã xuất bảng kết quả chi tiết ra: {output_csv}")

    return df_res, metrics


def main():
    parser = argparse.ArgumentParser(
        description="AutoTS4PS: Afternoon Extremes (High/Low) & Volatility Regime Forecasting"
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "evaluate"],
        default="predict",
        help="Chế độ thực thi: 'predict' (1 ngày cụ thể) hoặc 'evaluate' (walk-forward tập test)",
    )
    parser.add_argument(
        "--csv",
        default="datasets/test.csv",
        help="Đường dẫn file dữ liệu 5m cho predict mode",
    )
    parser.add_argument(
        "--train-csv",
        default="datasets/train.csv",
        help="Đường dẫn file tập train",
    )
    parser.add_argument(
        "--test-csv",
        default="datasets/test.csv",
        help="Đường dẫn file tập test",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Ngày dự báo (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Giới hạn N ngày cuối của tập test khi đánh giá",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(SCRIPT_DIR, "afternoon_extremes_predictions.csv"),
        help="File lưu kết quả chi tiết kiểm định",
    )

    args = parser.parse_args()

    if args.mode == "predict":
        run_predict(
            csv_path=args.csv,
            train_csv=args.train_csv,
            target_date=args.date,
        )
    else:
        run_evaluate(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            n_days=args.days,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()
