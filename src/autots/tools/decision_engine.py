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
