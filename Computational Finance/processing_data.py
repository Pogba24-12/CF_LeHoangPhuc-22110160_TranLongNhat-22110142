import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# add shift(1) in rolling functions to avoid lookahead bias
# ========== Data Processing Functions ================================================================================

# Load data 

def load_data(csv_path):

    df = pd.read_csv(
        csv_path,
        usecols=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
        dtype={
            "open": np.float64,
            "high": np.float64,
            "low": np.float64,
            "close": np.float64,
            "adj_close": np.float64,
            "volume": np.float64,
        },
    )
    df["date"] = pd.to_datetime(df["date"])

    return df


# Sort & deduplicate

def sort_and_deduplicate(df):

    df = df.sort_values(["date"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    return df

# Check OHLCV sanity

def check_ohlcv_sanity(df):
    cols_to_check = ["open", "high", "low", "close", "volume"]
    
    cond_negative = (df[cols_to_check] < 0).any(axis=1)

    cond_price_logic = (
        (df["low"] > df["open"]) | 
        (df["low"] > df["close"]) | 
        (df["high"] < df["open"]) | 
        (df["high"] < df["close"]) |
        (df["low"] > df["high"])  
    )

    errors = df[cond_negative | cond_price_logic]
    
    return errors

# Check missing value

import pandas as pd

def check_missing_values(df):

    missing_count = df.isnull().sum()
    
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    missing_data = pd.concat([missing_count, missing_percent], axis=1, keys=['Total Missing', 'Percent (%)'])
    
    missing_data = missing_data[missing_data['Total Missing'] > 0]
    
    missing_data = missing_data.sort_values(by='Total Missing', ascending=False)
    
    if missing_data.empty:
        print("No Missing value")
        return None
    else:
        print("Missing value")
        return missing_data

# Missing dates report

def report_missing_dates(df):

    # create a complete date range from the min to max date in the dataframe
    full_date_range = pd.date_range(start=df["date"].min(), end=df["date"].max())  
    # find missing dates
    missing_dates = full_date_range.difference(df["date"])

    #return len(missing_dates)

    print(f"Missing Dates Report: {len(missing_dates)} missing dates found.")

def add_missing_dates(df):

    df = df.set_index("date")
    full_date_range  = pd.date_range(start=df.index.min(), end=df.index.max())
    df_filled = df.reindex(full_date_range, method='ffill').reset_index().rename(columns={"index": "date"})

    return df_filled

# ========= Adjust Cooperate Actions ==================================================================================
def adjust_for_splits_and_dividends(df):

    df["adj_factor"] = df["adj_close"] / df["close"]

    df["open"] = df["open"] * df["adj_factor"]
    df["high"] = df["high"] * df["adj_factor"]
    df["low"] = df["low"] * df["adj_factor"]
    df["close"] = df["close"] * df["adj_factor"]
    df["volume"] = df["volume"] / df["adj_factor"]

    df = df.drop(columns=["adj_factor"])

    df = df.sort_values("date").reset_index(drop=True)

    return df



# ======Visualize Data ================================================================================================


def plot_column(df: pd.DataFrame, column: str, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df[column], label=column)
    ax.set_title(f"{column}")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()


def plot_time_series(df, column="close", ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df["date"], df[column], label=column)
    ax.set_xlabel("Date")
    ax.set_ylabel(column)
    ax.set_title(f"Time Series of {column}")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()


def histogram_distribution(df, column="volume", bins=50, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[column].dropna(), bins=bins, alpha=0.7)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {column}")
    ax.grid(True, linestyle='--', alpha=0.5)


