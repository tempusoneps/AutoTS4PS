#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1: Afternoon Intraday Trajectory & First-Passage SL/TP Event Risk Forecasting for VN30F1M.

Instead of predicting a single closing point at 14:45, this script forecasts the multi-step intraday
trajectory of the afternoon session (13:00-14:45) using nearest-neighbor motif trajectories on causal
morning features. It evaluates the first-passage probabilities of hitting Take Profit (TP) vs Stop Loss (SL)
before market close, computes Expected Net Value (EV) after friction, and delivers an actionable trading signal.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

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


def load_raw_5m_data(csv_paths) -> pd.DataFrame:
    """Loads and standardizes raw 5m candlestick data."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    dfs = []
    for p in csv_paths:
        resolved = resolve_csv_path(p)
        if resolved and os.path.exists(resolved):
            dfs.append(pd.read_csv(resolved))
    if not dfs:
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {csv_paths}")
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.columns = [c.strip().capitalize() for c in df_all.columns]
    df_all["Date"] = pd.to_datetime(df_all["Date"])
    df_all = (
        df_all.sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    df_all["Day_str"] = df_all["Date"].dt.strftime("%Y-%m-%d")
    df_all["Time_str"] = df_all["Date"].dt.strftime("%H:%M:%S")
    return df_all


def extract_afternoon_trajectory(df_day_5m: pd.DataFrame, morning_close: float) -> pd.DataFrame:
    """
    Extracts the afternoon session 5m candles relative to the 11:30 morning close.
    """
    aft_bars = df_day_5m[
        (df_day_5m["Time_str"] >= "13:00:00") & (df_day_5m["Time_str"] <= "14:45:00")
    ].copy()
    if aft_bars.empty:
        return pd.DataFrame()
    aft_bars["delta_high"] = aft_bars["High"] - morning_close
    aft_bars["delta_low"] = aft_bars["Low"] - morning_close
    aft_bars["delta_close"] = aft_bars["Close"] - morning_close
    return aft_bars


def simulate_first_passage_path(
    aft_bars: pd.DataFrame,
    direction: str,
    tp_pts: float,
    sl_pts: float,
) -> dict:
    """
    Determines whether Take Profit (TP) or Stop Loss (SL) is hit first along an intraday path.
    For LONG: TP is +tp_pts, SL is -sl_pts.
    For SHORT: TP is -tp_pts, SL is +sl_pts.
    """
    if aft_bars is None or aft_bars.empty:
        return {"outcome": "NO_DATA", "pnl": 0.0, "step": -1, "exit_type": "NONE"}

    for step, (_, bar) in enumerate(aft_bars.iterrows(), start=1):
        d_high = bar["delta_high"]
        d_low = bar["delta_low"]

        if direction == "LONG":
            hit_tp = d_high >= tp_pts
            hit_sl = d_low <= -sl_pts

            if hit_tp and hit_sl:
                # Conservative intra-bar collision assumption: stop loss hit first
                return {
                    "outcome": "HIT_SL",
                    "pnl": -sl_pts,
                    "step": step,
                    "time": bar["Time_str"],
                    "exit_type": "SL",
                }
            elif hit_tp:
                return {
                    "outcome": "HIT_TP",
                    "pnl": tp_pts,
                    "step": step,
                    "time": bar["Time_str"],
                    "exit_type": "TP",
                }
            elif hit_sl:
                return {
                    "outcome": "HIT_SL",
                    "pnl": -sl_pts,
                    "step": step,
                    "time": bar["Time_str"],
                    "exit_type": "SL",
                }

        elif direction == "SHORT":
            hit_tp = d_low <= -tp_pts
            hit_sl = d_high >= sl_pts

            if hit_tp and hit_sl:
                return {
                    "outcome": "HIT_SL",
                    "pnl": -sl_pts,
                    "step": step,
                    "time": bar["Time_str"],
                    "exit_type": "SL",
                }
            elif hit_tp:
                return {
                    "outcome": "HIT_TP",
                    "pnl": tp_pts,
                    "step": step,
                    "time": bar["Time_str"],
                    "exit_type": "TP",
                }
            elif hit_sl:
                return {
                    "outcome": "HIT_SL",
                    "pnl": -sl_pts,
                    "step": step,
                    "time": bar["Time_str"],
                    "exit_type": "SL",
                }

    # Neither hit before market close (exit at ATC close)
    last_bar = aft_bars.iloc[-1]
    final_delta = last_bar["delta_close"]
    atc_pnl = final_delta if direction == "LONG" else -final_delta
    outcome = "ATC_PROFIT" if atc_pnl > 0 else "ATC_LOSS"
    return {
        "outcome": outcome,
        "pnl": float(atc_pnl),
        "step": len(aft_bars),
        "time": last_bar["Time_str"],
        "exit_type": "ATC",
    }


def evaluate_first_passage_risk(
    df_mornings: pd.DataFrame,
    df_all_5m: pd.DataFrame,
    target_idx: int,
    k_neighbors: int = 30,
    tp_pts: float = 7.0,
    sl_pts: float = 4.0,
    friction: float = 0.8,
    trend_filter: bool = True,
    strict_fp: bool = True,
) -> dict:
    """
    Finds K nearest historical morning analogs, evaluates first-passage paths for LONG & SHORT,
    and returns probabilistic risk scores and expected payoffs.
    """
    target_morning = df_mornings.loc[target_idx]
    target_day = target_morning["day_str"]
    morning_close = float(target_morning["close"])

    # Strict causal slicing: use history strictly prior to target_idx
    hist_mornings = df_mornings.iloc[:target_idx].copy()
    if len(hist_mornings) < k_neighbors:
        k_neighbors = max(5, len(hist_mornings))

    # Scale causal features and fit Nearest Neighbors
    scaler = StandardScaler()
    X_hist = scaler.fit_transform(hist_mornings[FEATURE_COLS].fillna(0).values)
    x_target = scaler.transform(df_mornings.loc[target_idx:target_idx, FEATURE_COLS].fillna(0).values)

    nn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
    nn.fit(X_hist)
    distances, indices = nn.kneighbors(x_target)

    long_sims = []
    short_sims = []
    analog_days = []

    for dist, idx in zip(distances[0], indices[0]):
        analog_day = hist_mornings.iloc[idx]["day_str"]
        analog_close = float(hist_mornings.iloc[idx]["close"])
        analog_days.append({"day": analog_day, "distance": dist})

        df_day_5m = df_all_5m[df_all_5m["Day_str"] == analog_day]
        aft_bars = extract_afternoon_trajectory(df_day_5m, analog_close)

        res_long = simulate_first_passage_path(aft_bars, "LONG", tp_pts, sl_pts)
        res_short = simulate_first_passage_path(aft_bars, "SHORT", tp_pts, sl_pts)

        long_sims.append(res_long)
        short_sims.append(res_short)

    df_long = pd.DataFrame(long_sims)
    df_short = pd.DataFrame(short_sims)

    p_hit_tp_long = float((df_long["outcome"] == "HIT_TP").mean())
    p_hit_sl_long = float((df_long["outcome"] == "HIT_SL").mean())
    p_win_long = float((df_long["pnl"] > 0).mean())
    ev_long = float(df_long["pnl"].mean() - friction)

    p_hit_tp_short = float((df_short["outcome"] == "HIT_TP").mean())
    p_hit_sl_short = float((df_short["outcome"] == "HIT_SL").mean())
    p_win_short = float((df_short["pnl"] > 0).mean())
    ev_short = float(df_short["pnl"].mean() - friction)

    # Directional momentum filter against strong morning trends
    clv = float(target_morning.get("morning_clv", 0.0))
    ret_m = float(target_morning.get("ret_morning", 0.0))
    allow_long = not (trend_filter and (clv < -0.5 and ret_m < -0.25))
    allow_short = not (trend_filter and (clv > 0.5 and ret_m > 0.25))

    # Strict first-passage dominance: TP first must outpace SL first
    fp_dom_long = (
        (p_hit_tp_long >= p_hit_sl_long) and (p_hit_tp_long / (p_hit_sl_long + 1e-9) >= 1.15)
        if strict_fp
        else True
    )
    fp_dom_short = (
        (p_hit_tp_short >= p_hit_sl_short) and (p_hit_tp_short / (p_hit_sl_short + 1e-9) >= 1.15)
        if strict_fp
        else True
    )

    # Decision policy: Require minimum win probability >= 55% and positive EV edge
    min_prob = 0.55
    min_edge = 0.5  # points net expected gain

    if allow_long and fp_dom_long and ev_long > ev_short and ev_long >= min_edge and p_win_long >= min_prob:
        action = "STRONG LONG"
        chosen_side = "LONG"
        confidence = p_win_long
        chosen_ev = ev_long
        target_price = morning_close + tp_pts
        stop_price = morning_close - sl_pts
        reason = (
            f"Xác suất Win Long ({p_win_long*100:.1f}%) >= {min_prob*100:.0f}%, "
            f"P(Hit TP First)={p_hit_tp_long*100:.1f}% >= P(Hit SL First)={p_hit_sl_long*100:.1f}%, EV ròng={ev_long:+.2f} pts."
        )
    elif allow_short and fp_dom_short and ev_short > ev_long and ev_short >= min_edge and p_win_short >= min_prob:
        action = "STRONG SHORT"
        chosen_side = "SHORT"
        confidence = p_win_short
        chosen_ev = ev_short
        target_price = morning_close - tp_pts
        stop_price = morning_close + sl_pts
        reason = (
            f"Xác suất Win Short ({p_win_short*100:.1f}%) >= {min_prob*100:.0f}%, "
            f"P(Hit TP First)={p_hit_tp_short*100:.1f}% >= P(Hit SL First)={p_hit_sl_short*100:.1f}%, EV ròng={ev_short:+.2f} pts."
        )
    else:
        action = "WAIT / NO TRADE"
        chosen_side = "NONE"
        confidence = max(p_win_long, p_win_short)
        chosen_ev = max(ev_long, ev_short)
        target_price = morning_close
        stop_price = morning_close
        if not allow_short and ev_short > ev_long:
            reason = f"Đứng ngoài: Phiên sáng tăng mạnh (Ret={ret_m:+.2f}%, CLV={clv:+.2f}) -> Chặn lệnh Short ngược xu hướng."
        elif not allow_long and ev_long > ev_short:
            reason = f"Đứng ngoài: Phiên sáng giảm mạnh (Ret={ret_m:+.2f}%, CLV={clv:+.2f}) -> Chặn lệnh Long ngược xu hướng."
        elif (not fp_dom_long and ev_long >= ev_short) or (not fp_dom_short and ev_short > ev_long):
            reason = "Đứng ngoài: Tỷ lệ chạm TP trước chưa đủ áp đảo so với chạm SL trước (rủi ro quét 2 đầu cao)."
        else:
            reason = (
                f"Kỳ vọng ròng chưa đủ lớn (EV Long={ev_long:+.2f}, EV Short={ev_short:+.2f}) "
                f"hoặc xác suất Win chưa đạt ngưỡng an toàn {min_prob*100:.0f}%."
            )

    return {
        "target_day": target_day,
        "morning_close": morning_close,
        "k_neighbors": k_neighbors,
        "tp_pts": tp_pts,
        "sl_pts": sl_pts,
        "friction": friction,
        "action": action,
        "chosen_side": chosen_side,
        "confidence": confidence,
        "chosen_ev": chosen_ev,
        "target_price": target_price,
        "stop_price": stop_price,
        "reason": reason,
        "long_metrics": {
            "p_hit_tp": p_hit_tp_long,
            "p_hit_sl": p_hit_sl_long,
            "p_win": p_win_long,
            "ev": ev_long,
        },
        "short_metrics": {
            "p_hit_tp": p_hit_tp_short,
            "p_hit_sl": p_hit_sl_short,
            "p_win": p_win_short,
            "ev": ev_short,
        },
        "analog_days": analog_days[:5],
    }


def run_predict(
    csv_path: str = None,
    train_csv: str = None,
    target_date: str = None,
    tp: float = None,
    sl: float = None,
    use_atr: bool = True,
    trend_filter: bool = True,
    strict_fp: bool = True,
    k: int = 30,
    friction: float = 0.8,
) -> dict:
    """Runs First-Passage SL/TP Event Risk forecast for a single date."""
    print_banner("DỰ BÁO XÁC SUẤT CHẠM SL / TP ĐẦU TIÊN PHIÊN CHIỀU (FIRST-PASSAGE RISK)")
    train_p = resolve_csv_path(train_csv or "datasets/train.csv")
    test_p = resolve_csv_path(csv_path or "datasets/test.csv")

    print(f"Đang nạp dữ liệu: {os.path.basename(train_p)} & {os.path.basename(test_p)}...")
    df_half, _ = load_and_resample_halfday([train_p, test_p])
    df_all_5m = load_raw_5m_data([train_p, test_p])

    morning_mask = df_half["session"] == "morning"
    df_mornings = df_half[morning_mask].reset_index(drop=True)

    if target_date:
        match = df_mornings[df_mornings["day_str"] == target_date]
        if match.empty:
            print(f"Lỗi: Không tìm thấy ngày {target_date} trong dữ liệu.")
            return None
        target_idx = match.index[0]
    else:
        target_idx = len(df_mornings) - 1
        target_date = df_mornings.loc[target_idx, "day_str"]

    target_morning = df_mornings.loc[target_idx]
    morning_close = float(target_morning["close"])
    morning_atr = float(target_morning["high"] - target_morning["low"])

    # Determine dynamic ATR vs static TP/SL
    if use_atr and (tp is None or sl is None):
        eff_sl = max(6.0, round(morning_atr * 0.65, 1))
        eff_tp = max(8.0, round(eff_sl * 1.25, 1))
        barrier_mode = f"Động theo ATR ({morning_atr:.1f} pts) -> TP={eff_tp:.1f}, SL={eff_sl:.1f}"
    else:
        eff_sl = sl if sl is not None else 4.0
        eff_tp = tp if tp is not None else 7.0
        barrier_mode = f"Cố định -> TP={eff_tp:.1f}, SL={eff_sl:.1f}"

    print(f"\n📅 Ngày dự báo: {target_date} | Chốt dữ liệu sáng: 11:30:00")
    print(
        f"📊 Giá chốt sáng: {morning_close:.2f} | Open: {target_morning['open']:.2f} "
        f"| High: {target_morning['high']:.2f} | Low: {target_morning['low']:.2f} "
        f"(ATR Sáng = {morning_atr:.1f} pts)"
    )
    print(
        f"⚡ Causal Fingerprint: Ret={target_morning['ret_morning']:+.2f}% | "
        f"VWAP Dev={target_morning['vwap_dev']:+.2f}% | Momentum 30p={target_morning['ret_last30m']:+.2f}% | "
        f"CLV={target_morning['morning_clv']:+.2f}"
    )
    print(f"⚙️  Cấu hình dải rủi ro: {barrier_mode} | Bộ lọc Xu hướng: {'BẬT' if trend_filter else 'TẮT'}")

    eval_res = evaluate_first_passage_risk(
        df_mornings=df_mornings,
        df_all_5m=df_all_5m,
        target_idx=target_idx,
        k_neighbors=k,
        tp_pts=eff_tp,
        sl_pts=eff_sl,
        friction=friction,
        trend_filter=trend_filter,
        strict_fp=strict_fp,
    )

    lm = eval_res["long_metrics"]
    sm = eval_res["short_metrics"]

    print("\n" + "-" * 85)
    print(f"🎲 KẾT QUẢ MÔ PHỎNG QUỸ ĐẠO TỪ {eval_res['k_neighbors']} MẪU HÌNH TƯƠNG ĐỒNG NHẤT:")
    print(
        f"   🟢 Kịch bản LONG  (TP = +{eff_tp:.1f} pts, SL = -{eff_sl:.1f} pts)  : "
        f"P(Hit TP First) = {lm['p_hit_tp']*100:5.1f}% | P(Hit SL First) = {lm['p_hit_sl']*100:5.1f}% "
        f"| P(Win) = {lm['p_win']*100:5.1f}% | EV = {lm['ev']:+5.2f} pts"
    )
    print(
        f"   🔴 Kịch bản SHORT (TP = -{eff_tp:.1f} pts, SL = +{eff_sl:.1f} pts)  : "
        f"P(Hit TP First) = {sm['p_hit_tp']*100:5.1f}% | P(Hit SL First) = {sm['p_hit_sl']*100:5.1f}% "
        f"| P(Win) = {sm['p_win']*100:5.1f}% | EV = {sm['ev']:+5.2f} pts"
    )

    print("-" * 85)
    print("🎯 KHUYẾN NGHỊ GIAO DỊCH PHIÊN CHIỀU:")
    print(f"   - Hành động           : {eval_res['action']}")
    print(f"   - Mức giá vào (Entry) : {morning_close:.2f} (Đầu phiên chiều 13:00)")
    if eval_res["action"] != "WAIT / NO TRADE":
        print(f"   - Mục tiêu Chốt lời   : {eval_res['target_price']:.2f} ({eval_res['tp_pts']:+.1f} pts)")
        print(f"   - Mức dừng Cắt lỗ     : {eval_res['stop_price']:.2f} ({eval_res['sl_pts']:+.1f} pts)")
        print(f"   - Xác suất Thắng kỳ vọng: {eval_res['confidence']*100:.1f}%")
        print(f"   - Kỳ vọng lợi nhuận EV: {eval_res['chosen_ev']:+.2f} điểm (đã trừ {friction:.1f} phí/trượt giá)")
    print(f"   - Lý giải             : {eval_res['reason']}")

    # Check actual afternoon execution if available
    df_target_5m = df_all_5m[df_all_5m["Day_str"] == target_date]
    aft_bars = extract_afternoon_trajectory(df_target_5m, morning_close)

    if not aft_bars.empty:
        print("-" * 85)
        print("🔍 ĐỐI CHIẾU DIỄN BIẾN THỰC TẾ TRONG PHIÊN CHIỀU:")
        act_long = simulate_first_passage_path(aft_bars, "LONG", eff_tp, eff_sl)
        act_short = simulate_first_passage_path(aft_bars, "SHORT", eff_tp, eff_sl)

        last_bar = aft_bars.iloc[-1]
        print(
            f"   - Điểm số chiều thực tế: High = {aft_bars['High'].max():.2f} (+{aft_bars['delta_high'].max():.2f} pts) "
            f"| Low = {aft_bars['Low'].min():.2f} ({aft_bars['delta_low'].min():.2f} pts) "
            f"| Close = {last_bar['Close']:.2f} ({last_bar['delta_close']:+.2f} pts)"
        )
        if eval_res["chosen_side"] == "LONG":
            is_win = act_long["pnl"] > 0
            print(
                f"   - Vị thế LONG thực tế  : {act_long['outcome']} tại {act_long['time']} "
                f"-> PnL = {act_long['pnl']:+.2f} pts [{'THẮNG ✅' if is_win else 'THUA ❌'}]"
            )
        elif eval_res["chosen_side"] == "SHORT":
            is_win = act_short["pnl"] > 0
            print(
                f"   - Vị thế SHORT thực tế : {act_short['outcome']} tại {act_short['time']} "
                f"-> PnL = {act_short['pnl']:+.2f} pts [{'THẮNG ✅' if is_win else 'THUA ❌'}]"
            )
        else:
            print("   - Khuyến nghị WAIT -> Giữ an toàn vốn, không chịu rủi ro.")
    print("=" * 85)
    return eval_res


def run_evaluate(
    train_csv: str = None,
    test_csv: str = None,
    n_days: int = None,
    tp: float = None,
    sl: float = None,
    use_atr: bool = True,
    trend_filter: bool = True,
    strict_fp: bool = True,
    k: int = 30,
    friction: float = 0.8,
    output_csv: str = None,
) -> tuple[pd.DataFrame, dict]:
    """Runs walk-forward backtest across test days."""
    train_p = resolve_csv_path(train_csv or "datasets/train.csv")
    test_p = resolve_csv_path(test_csv or "datasets/test.csv")

    print_banner(f"KIỂM ĐỊNH WALK-FORWARD FIRST-PASSAGE RISK ({os.path.basename(test_p)})")
    print(f"Đang nạp dữ liệu: {os.path.basename(train_p)} & {os.path.basename(test_p)}...")
    df_half, _ = load_and_resample_halfday([train_p, test_p])
    df_all_5m = load_raw_5m_data([train_p, test_p])

    morning_mask = df_half["session"] == "morning"
    df_mornings = df_half[morning_mask].reset_index(drop=True)

    df_test_only, _ = load_and_resample_halfday(test_p)
    test_days = list(df_test_only["day_str"].unique())

    if n_days is not None and n_days > 0 and len(test_days) > n_days:
        test_days = test_days[-n_days:]

    total_test = len(test_days)
    mode_str = "Dynamic ATR (0.65x ATR SL, 1.25x TP)" if (use_atr and (tp is None or sl is None)) else f"Fixed TP={tp}, SL={sl}"
    print(f"Kiểm thử trên {total_test} phiên: {test_days[0]} -> {test_days[-1]}")
    print(f"Cấu hình: {mode_str} | K = {k} | Trend Filter: {'BẬT' if trend_filter else 'TẮT'} | Ma sát = {friction:.1f} pts\n")

    records = []
    for i, day in enumerate(test_days):
        matching = df_mornings[df_mornings["day_str"] == day]
        if matching.empty:
            continue
        idx = matching.index[0]
        target_row = df_mornings.loc[idx]
        morning_close = float(target_row["close"])
        morning_atr = float(target_row["high"] - target_row["low"])

        if use_atr and (tp is None or sl is None):
            eff_sl = max(6.0, round(morning_atr * 0.65, 1))
            eff_tp = max(8.0, round(eff_sl * 1.25, 1))
        else:
            eff_sl = sl if sl is not None else 4.0
            eff_tp = tp if tp is not None else 7.0

        pred = evaluate_first_passage_risk(
            df_mornings=df_mornings,
            df_all_5m=df_all_5m,
            target_idx=idx,
            k_neighbors=k,
            tp_pts=eff_tp,
            sl_pts=eff_sl,
            friction=friction,
            trend_filter=trend_filter,
            strict_fp=strict_fp,
        )

        df_day_5m = df_all_5m[df_all_5m["Day_str"] == day]
        aft_bars = extract_afternoon_trajectory(df_day_5m, morning_close)

        actual_outcome = "NO_DATA"
        actual_pnl = 0.0
        net_pnl = 0.0

        if not aft_bars.empty:
            if pred["chosen_side"] == "LONG":
                act_res = simulate_first_passage_path(aft_bars, "LONG", eff_tp, eff_sl)
                actual_outcome = act_res["outcome"]
                actual_pnl = act_res["pnl"]
                net_pnl = actual_pnl - friction
            elif pred["chosen_side"] == "SHORT":
                act_res = simulate_first_passage_path(aft_bars, "SHORT", eff_tp, eff_sl)
                actual_outcome = act_res["outcome"]
                actual_pnl = act_res["pnl"]
                net_pnl = actual_pnl - friction
            else:
                actual_outcome = "SKIPPED"
                actual_pnl = 0.0
                net_pnl = 0.0

        is_win = bool(net_pnl > 0)
        records.append({
            "day": day,
            "morning_close": morning_close,
            "tp_pts": eff_tp,
            "sl_pts": eff_sl,
            "action": pred["action"],
            "chosen_side": pred["chosen_side"],
            "confidence": pred["confidence"],
            "pred_ev": pred["chosen_ev"],
            "actual_outcome": actual_outcome,
            "gross_pnl": actual_pnl,
            "net_pnl": net_pnl,
            "is_win": is_win,
        })

        tag = "WAIT "
        if pred["chosen_side"] == "LONG":
            tag = "LONG "
        elif pred["chosen_side"] == "SHORT":
            tag = "SHORT"

        print(
            f"[{i+1:3d}/{total_test:3d}] {day} | Tín hiệu: {tag} "
            f"| P(Win): {pred['confidence']*100:4.1f}% | EV: {pred['chosen_ev']:+5.2f} pts "
            f"| Thực tế: {actual_outcome:10s} | Net PnL: {net_pnl:+6.2f} pts "
            f"| {'WIN ✅' if is_win else ('WAIT ⚪' if tag == 'WAIT ' else 'LOSS ❌')}"
        )

    df_res = pd.DataFrame(records)
    if df_res.empty:
        return df_res, {}

    # Compute summary statistics
    traded = df_res[df_res["chosen_side"] != "NONE"]
    total_trades = len(traded)
    skipped_count = len(df_res) - total_trades

    if total_trades > 0:
        win_count = int(traded["is_win"].sum())
        win_rate = (win_count / total_trades) * 100.0
        total_pnl = float(traded["net_pnl"].sum())
        avg_pnl = float(traded["net_pnl"].mean())

        wins = traded[traded["net_pnl"] > 0]["net_pnl"]
        losses = traded[traded["net_pnl"] <= 0]["net_pnl"]
        gross_win = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = float(abs(losses.sum())) if not losses.empty else 1e-9
        profit_factor = gross_win / gross_loss

        # Calculate max drawdown in points
        cum_pnl = traded["net_pnl"].cumsum()
        peak = cum_pnl.cummax()
        drawdown = peak - cum_pnl
        max_dd = float(drawdown.max())
    else:
        win_count = 0
        win_rate = 0.0
        total_pnl = 0.0
        avg_pnl = 0.0
        profit_factor = 0.0
        max_dd = 0.0

    metrics = {
        "total_sessions": len(df_res),
        "traded_sessions": total_trades,
        "skipped_sessions": skipped_count,
        "win_rate": win_rate,
        "total_net_pnl": total_pnl,
        "avg_trade_pnl": avg_pnl,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }

    print("\n" + "=" * 85)
    print("BẢNG TỔNG HỢP HIỆU SUẤT CHIẾN LƯỢC FIRST-PASSAGE RISK TRÊN TẬP TEST".center(85))
    print("=" * 85)
    print(f"  1. Tổng số phiên kiểm thử                 : {metrics['total_sessions']} phiên")
    print(f"  2. Số phiên kích hoạt lệnh (Traded)       : {metrics['traded_sessions']} phiên ({metrics['traded_sessions']/len(df_res)*100:.1f}%)")
    print(f"  3. Số phiên đứng ngoài an toàn (Skipped)  : {metrics['skipped_sessions']} phiên ({metrics['skipped_sessions']/len(df_res)*100:.1f}%)")
    print(f"  4. Tỷ lệ Thắng (Win Rate)                 : {metrics['win_rate']:.2f}% ({win_count}/{total_trades})")
    print(f"  5. Tổng Lợi nhuận Ròng (Total Net PnL)   : {metrics['total_net_pnl']:+.2f} điểm VN30F1M")
    print(f"  6. Lợi nhuận trung bình mỗi lệnh          : {metrics['avg_trade_pnl']:+.2f} điểm")
    print(f"  7. Hệ số Lợi nhuận (Profit Factor)        : {metrics['profit_factor']:.2f}")
    print(f"  8. Sụt giảm tài sản cực đại (Max Drawdown): {metrics['max_drawdown']:.2f} điểm")
    print("=" * 85)

    if output_csv:
        df_res.to_csv(output_csv, index=False)
        print(f"💾 Đã xuất kết quả kiểm định ra file: {output_csv}")

    return df_res, metrics


def main():
    parser = argparse.ArgumentParser(
        description="AutoTS4PS: Afternoon Intraday Trajectory & First-Passage SL/TP Event Risk Forecasting"
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
        help="Đường dẫn file dữ liệu 5m cho chế độ predict",
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
        "--tp",
        type=float,
        default=None,
        help="Ngưỡng Chốt lời (Take Profit) tính theo điểm. Nếu bỏ trống, tự động tính theo Dynamic ATR.",
    )
    parser.add_argument(
        "--sl",
        type=float,
        default=None,
        help="Ngưỡng Cắt lỗ (Stop Loss) tính theo điểm. Nếu bỏ trống, tự động tính theo Dynamic ATR.",
    )
    parser.add_argument(
        "--no-atr",
        action="store_true",
        help="Tắt tính toán động theo ATR, ép buộc sử dụng giá trị cố định (mặc định TP=7.0, SL=4.0)",
    )
    parser.add_argument(
        "--no-trend-filter",
        action="store_true",
        help="Tắt bộ lọc xu hướng sáng (cho phép bắt đỉnh/đáy ngược xu hướng)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=30,
        help="Số lượng mẫu hình tương đồng lịch sử gần nhất (mặc định 30)",
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.8,
        help="Chi phí ma sát (phí giao dịch + trượt giá) theo điểm (mặc định 0.8)",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(SCRIPT_DIR, "afternoon_sl_tp_predictions.csv"),
        help="File lưu kết quả chi tiết kiểm định",
    )

    args = parser.parse_args()

    use_atr = not args.no_atr and (args.tp is None or args.sl is None)
    trend_filter = not args.no_trend_filter

    if args.mode == "predict":
        run_predict(
            csv_path=args.csv,
            train_csv=args.train_csv,
            target_date=args.date,
            tp=args.tp,
            sl=args.sl,
            use_atr=use_atr,
            trend_filter=trend_filter,
            k=args.k,
            friction=args.friction,
        )
    else:
        run_evaluate(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            n_days=args.days,
            tp=args.tp,
            sl=args.sl,
            use_atr=use_atr,
            trend_filter=trend_filter,
            k=args.k,
            friction=args.friction,
            output_csv=args.output_csv,
        )


if __name__ == "__main__":
    main()

