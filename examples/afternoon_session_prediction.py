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
    if len(unique_days) <= n_days:
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
