import pandas as pd
import numpy as np


# --- Add_lagged Features ---

def add_lagged(df, column, no_max_lags):
    for lag in range(1, no_max_lags + 1):
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

def add_lagged_returns(df, column, no_max_lags):
    
    for lag in range(0, no_max_lags + 1):
        if lag == 0:     
            df[f"{column}_return"] = df[column].pct_change()
        else: 
            df[f"{column}_return_lag_{lag}"] = df[column].pct_change().shift(lag)
    return df

def add_lagged_log_returns(df, column, no_max_lags):
    
    for lag in range(0, no_max_lags + 1):

        if lag == 0:
            df[f"{column}_log_return"] = np.log(df[column] / df[column].shift(1))
        else:  
            df[f"{column}_log_return_lag_{lag}"] = np.log(df[column] / df[column].shift(1)).shift(lag)
    return df


# --- Trend, Momentum indicators ---    

def calculate_sma(series, window, lag=1):
    return series.rolling(window=window).mean().shift(lag)

def calculate_ema(series, span, lag=1):
    return series.ewm(span=span, adjust=False).mean().shift(lag)

def calculate_rsi(series, window, lag=1):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100/(1 +rs))
    return rsi.shift(lag)

def caculate_macd(series, span_short=12, span_long=26, signal_span=9, lag=1):

    ema_short = calculate_ema(series, span_short, lag=0)

    ema_long = calculate_ema(series, span_long, lag=0)
    
    macd_line = ema_short - ema_long
    
    signal_line = calculate_ema(macd_line, signal_span, lag=0)
    return signal_line.shift(lag), macd_line.shift(lag)

def calculate_obv(close, volume, lag=1):
    obv = np.where(close.diff() > 0, volume, 
                   np.where(close.diff() < 0, -volume, 0))
    obv = pd.Series(obv).cumsum()
    return obv.shift(lag)

# --- VOLATILITY & STATS (Bollinger, Volatility, ZScore) ---

def calculate_bollinger(series, window=20, num_std=2, lag=1):
    
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper.shift(lag), lower.shift(lag)

def calculate_volatility(series, window, lag=1):
    return series.pct_change().rolling(window=window).std().shift(lag)

def calculate_zscore(series, window, lag=1):
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    z_score =  (series - mean) / std
    return z_score.shift(lag)



def add_features(data: pd.DataFrame, features_pool: list) -> pd.DataFrame:
    df = data.copy()
    
    # Map common variations to standard internal names
    # This ensures df['Close'] works even if your data has df['close']
    col_map = {c.lower(): c for c in df.columns}
    
    # Helper to safely get column (case-insensitive)
    def get_col(name):
        return df[col_map.get(name.lower(), name)]

    for feature_req in features_pool:
        # Standardize request: "ma_60" -> "MA_60"
        parts = feature_req.upper().split('_')
        f_type = parts[0]
        
        # Safely extract parameter
        param = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        
        try:
            # --- Moving Averages ---
            if f_type == 'MA':
                p = param if param else 20
                df[f"MA_{p}"] = calculate_sma(get_col('Close'), window=p)
                
            elif f_type == 'EMA':
                p = param if param else 20
                df[f"EMA_{p}"] = calculate_ema(get_col('Close'), span=p)

            # --- RSI ---
            elif f_type == 'RSI':
                p = param if param else 14
                df[f"RSI_{p}"] = calculate_rsi(get_col('Close'), window=p)

            # --- MACD ---
            elif f_type == 'MACD':
                # Default values
                fast = 12
                slow = 26
                signal = 9
                
                # Check if we have extra parts in the string (e.g., parts = ['MACD', '12', '26', '9'])
                if len(parts) >= 4:
                    if parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit():
                        fast = int(parts[1])
                        slow = int(parts[2])
                        signal = int(parts[3])
                
                # Calculate
                s_line, m_line = caculate_macd(
                    get_col('Close'), 
                    span_short=fast, 
                    span_long=slow, 
                    signal_span=signal
                )
                
                suffix = f"{fast}_{slow}_{signal}"
                df[f'MACD_Line_{suffix}'] = m_line
                df[f'MACD_Signal_{suffix}'] = s_line

            # --- Bollinger Bands ---
            elif f_type == 'BOLLINGER':
                p = param if param else 20
                upper, lower = calculate_bollinger(get_col('Close'), window=p)
                
                # FIX: Include param in name so Bollinger_20 doesn't overwrite Bollinger_50
                df[f'BB_Upper_{p}'] = upper
                df[f'BB_Lower_{p}'] = lower

            # --- Volatility & ZScore ---
            elif f_type == 'VOLATILITY':
                p = param if param else 20
                df[f"Volatility_{p}"] = calculate_volatility(get_col('Close'), window=p)
                
            elif f_type == 'ZSCORE':
                p = param if param else 20
                df[f"ZScore_{p}"] = calculate_zscore(get_col('Close'), window=p)

            # --- OBV ---
            elif f_type == 'OBV':
                df['OBV'] = calculate_obv(get_col('Close'), get_col('Volume'))

            else:
                print(f" Warning: Feature '{f_type}' not recognized.")

        except Exception as e:
            print(f"Error processing '{feature_req}': {e}")

    return df



