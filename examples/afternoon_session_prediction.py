#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Afternoon Session Time-Series Prediction & Test Set Evaluation for VN30F1M using AutoTS4PS.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

# Ensure src is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from autots.tools.halfday_data import load_and_resample_halfday
    from autots.models.afternoon_autots import build_afternoon_model
except ImportError:
    try:
        from tools.halfday_data import load_and_resample_halfday
        from models.afternoon_autots import build_afternoon_model
    except ImportError:
        AUTOTS_DIR = os.path.join(SRC_DIR, "autots")
        if AUTOTS_DIR not in sys.path:
            sys.path.insert(0, AUTOTS_DIR)
        from tools.halfday_data import load_and_resample_halfday
        from models.afternoon_autots import build_afternoon_model

SERIES_WEIGHTS = {
    "VN30F1M": 100.0,
    "ret_morning": 0.0,
    "norm_atr": 0.0,
    "vwap_dev": 0.0,
    "vol_ratio": 0.0,
    "ret_last30m": 0.0,
    "morning_clv": 0.0,
    "overnight_gap": 0.0,
    "prev_afternoon_ret": 0.0,
    "prev_ret_morning": 0.0,
}


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


def compute_forecast_metrics(df_res: pd.DataFrame) -> dict:
    """
    Computes standard time-series forecasting metrics on afternoon session predictions.
    Expected columns in df_res:
      - morning_close
      - actual_close
      - pred_close
      - lower_bound
      - upper_bound
    """
    if df_res is None or df_res.empty:
        return {
            "count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "mape": 0.0,
            "smape": 0.0,
            "oda": 0.0,
            "coverage_90": 0.0,
            "mean_interval_width": 0.0,
        }

    actual = df_res["actual_close"].to_numpy(dtype=float)
    pred = df_res["pred_close"].to_numpy(dtype=float)
    morning = df_res["morning_close"].to_numpy(dtype=float)
    lower = df_res["lower_bound"].to_numpy(dtype=float)
    upper = df_res["upper_bound"].to_numpy(dtype=float)

    errors = pred - actual
    abs_errors = np.abs(errors)
    sq_errors = errors ** 2

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(sq_errors)))
    mape = float(np.mean(abs_errors / (np.abs(actual) + 1e-9)) * 100.0)
    smape = float(
        np.mean(2.0 * abs_errors / (np.abs(pred) + np.abs(actual) + 1e-9)) * 100.0
    )

    pred_delta = pred - morning
    actual_delta = actual - morning
    oda_matches = np.sign(pred_delta) == np.sign(actual_delta)
    oda = float(np.mean(oda_matches) * 100.0)

    in_band = (actual >= lower) & (actual <= upper)
    coverage_90 = float(np.mean(in_band) * 100.0)
    mean_width = float(np.mean(upper - lower))

    return {
        "count": int(len(df_res)),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "oda": oda,
        "coverage_90": coverage_90,
        "mean_interval_width": mean_width,
    }


def run_predict(csv_path: str, target_date: str = None, fast_mode: bool = False) -> dict:
    """
    Runs standard afternoon closing price forecast at 11:30 for a target date (or latest available date).
    """
    csv_path = resolve_csv_path(csv_path)
    print_banner("DỰ BÁO GIÁ ĐÓNG CỬA PHIÊN CHIỀU VN30F1M (AFTERNOON SESSION FORECAST)")
    print(f"Đang nạp và xử lý dữ liệu half-day từ: {os.path.basename(csv_path)}...")
    df_halfday, df_features = load_and_resample_halfday(csv_path)

    if target_date:
        matching = df_halfday[
            (df_halfday["day_str"] == target_date) & (df_halfday["session"] == "morning")
        ]
        if matching.empty:
            print(f"Lỗi: Không tìm thấy phiên sáng của ngày {target_date} trong dữ liệu.")
            return None
        target_idx = matching.index[0]
    else:
        morning_bars = df_halfday[df_halfday["session"] == "morning"]
        target_idx = morning_bars.index[-1]
        target_date = df_halfday.loc[target_idx, "day_str"]

    morning_bar = df_halfday.loc[target_idx]
    actual_afternoon_bar = None
    if (
        target_idx + 1 < len(df_halfday)
        and df_halfday.loc[target_idx + 1, "session"] == "afternoon"
    ):
        actual_afternoon_bar = df_halfday.loc[target_idx + 1]

    train_features = df_features.iloc[: target_idx + 1]
    morning_close = float(morning_bar["close"])

    print(f"\n📅 Ngày dự báo: {target_date} | Thời điểm chốt dữ liệu sáng: 11:30:00")
    print(
        f"📊 Phiên Sáng: Open = {morning_bar['open']:.2f} | High = {morning_bar['high']:.2f} "
        f"| Low = {morning_bar['low']:.2f} | Close = {morning_close:.2f}"
    )
    print(
        f"⚡ VWAP Sáng: {morning_bar['vwap']:.2f} (Lệch: {morning_bar['vwap_dev']:+.2f}%) "
        f"| Khối lượng Sáng: {int(morning_bar['volume']):,}"
    )
    print(
        f"🧭 Động lượng & Cấu trúc: Ret 30p cuối = {morning_bar['ret_last30m']:+.2f}% "
        f"| CLV Sáng = {morning_bar['morning_clv']:+.2f} "
        f"| Gap Qua Đêm = {morning_bar['overnight_gap']:+.2f}% "
        f"| Chiều Hôm Trước = {morning_bar['prev_afternoon_ret']:+.2f}% "
        f"| Sáng Hôm Trước = {morning_bar['prev_ret_morning']:+.2f}%"
    )

    default_template_path = os.path.join(SCRIPT_DIR, "best_afternoon_template.json")
    if os.path.exists(default_template_path):
        print(f"\n⚡ Nạp template mô hình tối ưu: {os.path.basename(default_template_path)}")
        model = build_afternoon_model(template_path=default_template_path)
    else:
        print(f"\n⏳ Đang huấn luyện AutoTS tìm mô hình tối ưu (Fast Mode = {fast_mode})...")
        model = build_afternoon_model(fast_mode=fast_mode)

    model = model.fit(train_features, weights=SERIES_WEIGHTS)
    prediction = model.predict()

    if not os.path.exists(default_template_path):
        try:
            model.export_template(default_template_path, models="best", n=1)
            print(f"💾 Đã lưu template mô hình tối ưu vào: {os.path.basename(default_template_path)}")
        except Exception:
            pass

    pred_close = float(prediction.forecast["VN30F1M"].iloc[0])
    upper_bound = float(prediction.upper_forecast["VN30F1M"].iloc[0])
    lower_bound = float(prediction.lower_forecast["VN30F1M"].iloc[0])
    pred_delta = pred_close - morning_close
    pred_pct = (pred_delta / (morning_close + 1e-9)) * 100.0
    interval_width = upper_bound - lower_bound

    print("-" * 85)
    print("🤖 KẾT QUẢ DỰ BÁO PHIÊN CHIỀU (14:45):")
    print(f"   - Mô hình sử dụng       : {model.best_model_name}")
    print(f"   - Giá đóng cửa dự báo   : {pred_close:.2f} điểm")
    print(f"   - Biến động kỳ vọng     : {pred_delta:+.2f} điểm ({pred_pct:+.2f}% so với 11:30)")
    print(f"   - Khoảng tin cậy 90%    : [{lower_bound:.2f}  --->  {upper_bound:.2f}] (Độ rộng: {interval_width:.2f} điểm)")

    result = {
        "day": target_date,
        "morning_close": morning_close,
        "pred_close": pred_close,
        "pred_delta": pred_delta,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "actual_close": None,
        "actual_delta": None,
        "error": None,
        "abs_error": None,
        "ape_pct": None,
        "oda_correct": None,
        "in_band_90": None,
    }

    if actual_afternoon_bar is not None:
        actual_close = float(actual_afternoon_bar["close"])
        actual_delta = actual_close - morning_close
        error = pred_close - actual_close
        abs_error = abs(error)
        ape_pct = (abs_error / (abs(actual_close) + 1e-9)) * 100.0
        oda_correct = bool(np.sign(pred_delta) == np.sign(actual_delta))
        in_band_90 = bool(lower_bound <= actual_close <= upper_bound)

        result.update({
            "actual_close": actual_close,
            "actual_delta": actual_delta,
            "error": error,
            "abs_error": abs_error,
            "ape_pct": ape_pct,
            "oda_correct": oda_correct,
            "in_band_90": in_band_90,
        })

        print("-" * 85)
        print("🔍 ĐỐI CHIẾU KẾT QUẢ THỰC TẾ PHIÊN CHIỀU:")
        print(f"   - Giá đóng cửa thực tế  : {actual_close:.2f} điểm ({actual_delta:+.2f} điểm so với 11:30)")
        print(f"   - Sai số dự báo (Error) : {error:+.2f} điểm (|Error| = {abs_error:.2f} điểm | APE = {ape_pct:.2f}%)")
        print(f"   - Dự báo hướng (ODA)    : {'ĐÚNG HƯỚNG ✅' if oda_correct else 'SAI HƯỚNG ❌'}")
        print(f"   - Nằm trong dải 90%     : {'CÓ ✅' if in_band_90 else 'KHÔNG ❌'}")
    print("=" * 85)
    return result


def run_evaluate(
    train_csv_path: str = None,
    test_csv_path: str = None,
    n_days: int = None,
    fast_mode: bool = True,
    output_csv: str = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Runs walk-forward evaluation on the test dataset (datasets/test.csv),
    using history from datasets/train.csv, and reports standard forecasting metrics.
    """
    # Support legacy call where first argument was a single CSV path and test_csv_path was omitted
    if train_csv_path and test_csv_path is None and not os.path.exists(
        os.path.join(PROJECT_ROOT, "datasets", "test.csv")
    ):
        resolved_train = resolve_csv_path(train_csv_path)
        resolved_test = None
    else:
        resolved_train = resolve_csv_path(
            train_csv_path or os.path.join(PROJECT_ROOT, "datasets", "train.csv")
        )
        resolved_test = (
            resolve_csv_path(test_csv_path)
            if test_csv_path
            else resolve_csv_path(os.path.join(PROJECT_ROOT, "datasets", "test.csv"))
        )

    # Determine test days and combined half-day features
    if resolved_test and os.path.exists(resolved_test) and resolved_train != resolved_test:
        print_banner(
            f"ĐÁNH GIÁ CHỈ SỐ DỰ BÁO TRÊN TẬP TEST ({os.path.basename(resolved_test)})"
        )
        print(
            f"Đang nạp dữ liệu Train ({os.path.basename(resolved_train)}) "
            f"+ Test ({os.path.basename(resolved_test)})..."
        )
        df_test_only, _ = load_and_resample_halfday(resolved_test)
        test_days = list(df_test_only["day_str"].unique())
        df_halfday, df_features = load_and_resample_halfday([resolved_train, resolved_test])
    else:
        single_csv = resolved_train or resolved_test
        print_banner(
            f"ĐÁNH GIÁ CHỈ SỐ DỰ BÁO WALK-FORWARD ({os.path.basename(single_csv)})"
        )
        df_halfday, df_features = load_and_resample_halfday(single_csv)
        all_days = list(df_halfday["day_str"].unique())
        eval_count = n_days if (n_days is not None and n_days > 0) else 15
        if len(all_days) <= eval_count:
            print(f"Lỗi: Tập dữ liệu quá nhỏ để kiểm thử {eval_count} phiên.")
            return pd.DataFrame(), {}
        test_days = all_days[-eval_count:]

    if n_days is not None and n_days > 0 and len(test_days) > n_days:
        test_days = test_days[-n_days:]

    total_test_days = len(test_days)
    print(
        f"Khoảng thời gian tập Test: {test_days[0]} -> {test_days[-1]} ({total_test_days} phiên chiều)"
    )

    default_template_path = os.path.join(SCRIPT_DIR, "best_afternoon_template.json")
    if os.path.exists(default_template_path):
        print(
            f"⚡ Đã phát hiện template: {os.path.basename(default_template_path)} "
            f"-> Chạy Fast Walk-Forward Evaluation trên {total_test_days} phiên test."
        )
    else:
        first_test_morning_idx = df_halfday[
            (df_halfday["day_str"] == test_days[0]) & (df_halfday["session"] == "morning")
        ].index[0]
        if first_test_morning_idx >= 4:
            print(
                f"⏳ Chưa có template -> Huấn luyện mô hình trên tập Train "
                f"({first_test_morning_idx} thanh half-day trước {test_days[0]})..."
            )
            init_model = build_afternoon_model(fast_mode=fast_mode)
            init_model = init_model.fit(
                df_features.iloc[:first_test_morning_idx], weights=SERIES_WEIGHTS
            )
            try:
                init_model.export_template(default_template_path, models="best")
                print(f"💾 Đã xuất template tối ưu vào: {os.path.basename(default_template_path)}")
            except Exception:
                pass

    results = []
    for i, day in enumerate(test_days):
        morning_matches = df_halfday[
            (df_halfday["day_str"] == day) & (df_halfday["session"] == "morning")
        ]
        afternoon_matches = df_halfday[
            (df_halfday["day_str"] == day) & (df_halfday["session"] == "afternoon")
        ]
        if morning_matches.empty or afternoon_matches.empty:
            continue

        morning_idx = morning_matches.index[0]
        afternoon_idx = afternoon_matches.index[0]

        train_slice = df_features.iloc[: morning_idx + 1]
        morning_bar = df_halfday.loc[morning_idx]
        afternoon_bar = df_halfday.loc[afternoon_idx]

        if os.path.exists(default_template_path):
            model = build_afternoon_model(template_path=default_template_path)
        else:
            model = build_afternoon_model(fast_mode=fast_mode)

        try:
            model = model.fit(train_slice, weights=SERIES_WEIGHTS)
            prediction = model.predict()

            pred_close = float(prediction.forecast["VN30F1M"].iloc[0])
            upper_bound = float(prediction.upper_forecast["VN30F1M"].iloc[0])
            lower_bound = float(prediction.lower_forecast["VN30F1M"].iloc[0])

            morning_close = float(morning_bar["close"])
            actual_close = float(afternoon_bar["close"])

            pred_delta = pred_close - morning_close
            actual_delta = actual_close - morning_close
            error = pred_close - actual_close
            abs_error = abs(error)
            ape_pct = (abs_error / (abs(actual_close) + 1e-9)) * 100.0
            oda_correct = bool(np.sign(pred_delta) == np.sign(actual_delta))
            in_band_90 = bool(lower_bound <= actual_close <= upper_bound)

            results.append({
                "day": day,
                "morning_close": morning_close,
                "actual_close": actual_close,
                "pred_close": pred_close,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "pred_delta": pred_delta,
                "actual_delta": actual_delta,
                "error": error,
                "abs_error": abs_error,
                "ape_pct": ape_pct,
                "oda_correct": oda_correct,
                "in_band_90": in_band_90,
            })

            print(
                f"[{i+1:3d}/{total_test_days:3d}] {day} | Sáng: {morning_close:7.2f} "
                f"| Dự báo: {pred_close:7.2f} [{lower_bound:7.2f}, {upper_bound:7.2f}] "
                f"| Thực tế: {actual_close:7.2f} | |Err|: {abs_error:5.2f} "
                f"| Hướng: {'ĐÚNG' if oda_correct else 'SAI '}"
            )
        except Exception as e:
            print(f"[{i+1:3d}/{total_test_days:3d}] {day} | Lỗi: {e}")

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("Không có kết quả dự báo nào được tạo.")
        return df_res, {}

    metrics = compute_forecast_metrics(df_res)

    print("\n" + "=" * 85)
    print(f"BẢNG TỔNG HỢP CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP TEST ({metrics['count']} PHIÊN)".center(85))
    print("=" * 85)
    print(f"  1. Số phiên kiểm thử (Test Sessions)         : {metrics['count']} phiên")
    print(f"  2. MAE  (Mean Absolute Error)                : {metrics['mae']:.2f} điểm")
    print(f"  3. RMSE (Root Mean Squared Error)            : {metrics['rmse']:.2f} điểm")
    print(f"  4. MAPE (Mean Absolute Percentage Error)     : {metrics['mape']:.3f}%")
    print(f"  5. sMAPE (Symmetric MAPE)                    : {metrics['smape']:.3f}%")
    print(f"  6. ODA  (Origin Direction Accuracy - Đúng hướng): {metrics['oda']:.2f}%")
    print(f"  7. Coverage 90% (Tỷ lệ nằm trong dải 90%)    : {metrics['coverage_90']:.2f}%")
    print(f"  8. Độ rộng dải tin cậy 90% trung bình        : {metrics['mean_interval_width']:.2f} điểm")
    print("=" * 85)

    if output_csv:
        df_res.to_csv(output_csv, index=False)
        print(f"💾 Đã xuất bảng chi tiết kết quả dự báo tập test ra: {output_csv}")

    return df_res, metrics


# Backward-compatible function aliases
run_advisor = run_predict


def run_backtest(csv_path: str = None, n_days: int = None, fast_mode: bool = True):
    return run_evaluate(
        train_csv_path=csv_path,
        test_csv_path=None,
        n_days=n_days,
        fast_mode=fast_mode,
    )


def main():
    parser = argparse.ArgumentParser(
        description="AutoTS4PS Afternoon Session Forecasting & Test Evaluation"
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "evaluate", "advisor", "backtest"],
        default="evaluate",
        help="Execution mode: 'predict' (single day forecast) or 'evaluate' (test set evaluation)",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join(PROJECT_ROOT, "datasets", "test.csv"),
        help="Path to 5m CSV dataset for single-day predict mode",
    )
    parser.add_argument(
        "--train-csv",
        default=os.path.join(PROJECT_ROOT, "datasets", "train.csv"),
        help="Path to training 5m CSV dataset for evaluate mode",
    )
    parser.add_argument(
        "--test-csv",
        default=os.path.join(PROJECT_ROOT, "datasets", "test.csv"),
        help="Path to test 5m CSV dataset for evaluate mode",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Specific date for predict mode (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Optional: limit evaluation to the last N days of the test set (default: all test days)",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(SCRIPT_DIR, "afternoon_test_predictions.csv"),
        help="Path to save detailed test set predictions CSV",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode if training from scratch",
    )

    args = parser.parse_args()

    if args.mode in ("predict", "advisor"):
        run_predict(args.csv, target_date=args.date, fast_mode=args.fast)
    else:
        run_evaluate(
            train_csv_path=args.train_csv,
            test_csv_path=args.test_csv,
            n_days=args.days,
            fast_mode=args.fast,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()
