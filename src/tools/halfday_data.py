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

    # Order sessions chronologically: morning before afternoon within each day
    session_order = {'morning': 0, 'afternoon': 1}
    df_halfday['session_order'] = df_halfday['session'].map(session_order)
    df_halfday = df_halfday.sort_values(['day_str', 'session_order']).drop(columns=['session_order']).reset_index(drop=True)
    
    if max_days is not None and max_days > 0:
        unique_days = df_halfday['day_str'].unique()
        if len(unique_days) > max_days:
            chosen_days = unique_days[-max_days:]
            df_halfday = df_halfday[df_halfday['day_str'].isin(chosen_days)].reset_index(drop=True)
            
    # Calculate Causal Morning Features
    morning_mask = df_halfday['session'] == 'morning'

    # Morning Return
    df_halfday.loc[morning_mask, 'ret_morning'] = (
        (df_halfday.loc[morning_mask, 'close'] - df_halfday.loc[morning_mask, 'open'])
        / (df_halfday.loc[morning_mask, 'open'] + 1e-9)
    )
    # Morning Normalized Range
    df_halfday.loc[morning_mask, 'norm_atr'] = (
        (df_halfday.loc[morning_mask, 'high'] - df_halfday.loc[morning_mask, 'low'])
        / (df_halfday.loc[morning_mask, 'close'] + 1e-9)
    )
    # VWAP deviation
    df_halfday.loc[morning_mask, 'vwap_dev'] = (
        (df_halfday.loc[morning_mask, 'close'] - df_halfday.loc[morning_mask, 'vwap'])
        / (df_halfday.loc[morning_mask, 'vwap'] + 1e-9)
    )
    # Volume MA 10
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
