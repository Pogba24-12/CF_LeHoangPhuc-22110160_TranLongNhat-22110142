import pandas as pd
import numpy as np

# ---- 1) Load & standardize ---------------------------------------------------
def load_raw(csv_path, symbol):
    df = pd.read_csv(
        csv_path,
        usecols=["timestamp","open","high","low","close","volume"],
        dtype={"open":"float64","high":"float64","low":"float64","close":"float64","volume":"float64"},
    )
    # Parse timestamps; assume source is exchange local time, then localize & convert
    df["ts"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    # EXAMPLE: source in 'America/New_York' -> convert to UTC
    df["ts"] = (df["ts"]
                .dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
                .dt.tz_convert("UTC"))
    df = (df.drop(columns=["timestamp"])
            .assign(symbol=symbol)
            .rename(columns=str.lower)
            [["ts","open","high","low","close","volume","symbol"]])
    return df
# ---- 2) Sort & de-duplicate --------------------------------------------------
def sort_dedup(df):
    df = df.sort_values(["symbol","ts"])
    # Keep last within duplicate timestamp (e.g., vendor correction)
    df = df.drop_duplicates(subset=["symbol","ts"], keep="last")
    return df
# ---- 3) OHLCV sanity checks --------------------------------------------------
def ohlcv_checks(df):
    bad = (
        (df["low"] > df[["open","close"]].min(axis=1)) |
        (df["high"] < df[["open","close"]].max(axis=1)) |
        (df["low"] > df["high"]) |
        (df["volume"] < 0)
    )
    return df.loc[bad]
def fix_bad_bars(df):
    # Soft-fix: enforce bounds where possible; otherwise mark NaN for review
    df["low"]  = np.minimum(df["low"],  df[["open","close","high"]].min(axis=1))
    df["high"] = np.maximum(df["high"], df[["open","close","low"]].max(axis=1))
    df.loc[df["low"] > df["high"], ["low","high"]] = np.nan
    df.loc[df["volume"] < 0, "volume"] = np.nan
    return df
# ---- 4) Missing data policy --------------------------------------------------
def report_missing(df, freq="1min"):
    # Build expected UTC index for each symbol, then compare
    out = []
    for sym, g in df.groupby("symbol", group_keys=False):
        rng = pd.date_range(g["ts"].min().floor(freq), g["ts"].max().ceil(freq), freq=freq, tz="UTC")
        missing = pd.Index(rng).difference(g["ts"])
        out.append({"symbol": sym, "missing_count": len(missing)})
    return pd.DataFrame(out)
def regularize(df, freq="1min"):
    def _reindex(g):
        rng = pd.date_range(g["ts"].min().floor(freq), g["ts"].max().ceil(freq), freq=freq, tz="UTC")
        g = g.set_index("ts").reindex(rng)
        g.index.name = "ts"
        return g.reset_index()
    return (df.groupby("symbol", group_keys=False)
              .apply(_reindex)
              .rename(columns=str.lower))
def fill_policy(df, method="ffill", max_gap="5min"):
    # Constrain forward-fill to avoid look-ahead leakage across large gaps
    df = df.sort_values(["symbol","ts"])
    def _fill(g):
        g = g.set_index("ts")
        if method == "ffill":
            # mask fills across big gaps
            dt = g.index.to_series().diff()
            mask = (dt is not None) & (dt > pd.to_timedelta(max_gap))
            # start a new block after big gap
            block = mask.cumsum().values
            g[["open","high","low","close","volume"]] = (
                g.groupby(block)[["open","high","low","close","volume"]].ffill()
            )
        elif method == "interpolate":
            g[["open","high","low","close","volume"]] = (
                g[["open","high","low","close","volume"]].interpolate(limit_area="inside")
            )
        return g.reset_index()
    return df.groupby("symbol", group_keys=False).apply(_fill)
# ---- 5) Outlier detection (robust) -------------------------------------------
def cap_outliers(df, col="close", window=50, z=6.0):
    # Robust z using rolling median & MAD
    def _cap(g):
        x = g[col]
        med = x.rolling(window, min_periods=20).median()
        mad = (x - med).abs().rolling(window, min_periods=20).median()
        robust_z = (x - med) / (1.4826 * mad.replace(0, np.nan))
        # Cap to median ± z * MAD
        upper = med + z * 1.4826 * mad
        lower = med - z * 1.4826 * mad
        g[col] = np.where(robust_z > z, upper, np.where(robust_z < -z, lower, x))
        return g
    return df.groupby("symbol", group_keys=False).apply(_cap)
# ---- 6) Corporate actions (conceptual) ---------------------------------------
# Keep adjusted & raw; below is a placeholder for applying precomputed factors
def apply_split_dividend_adjustments(df, factors):
    # factors: DataFrame with columns ["date","symbol","factor_close","factor_volume", ...]
    # Merge by date/symbol; multiply prices by factor_close, divide volume if needed
    return df  # implement with your vendor's factor table
# ---- 7) Multi-asset alignment ------------------------------------------------
def align_symbols(dfs, freq="1min"):
    # dfs: dict(symbol -> cleaned df)
    # Outer-join on time, then suffix columns; or return dicts aligned to a shared index
    aligned = {}
    # build union index
    idx = None
    for sym, g in dfs.items():
        sidx = pd.DatetimeIndex(g["ts"]).tz_convert("UTC")
        idx = sidx if idx is None else idx.union(sidx)
    idx = idx.sort_values()
    for sym, g in dfs.items():
        gg = g.set_index("ts").reindex(idx)
        gg.index.name = "ts"
        aligned[sym] = gg.reset_index()
    return aligned
# ---- Example end-to-end ------------------------------------------------------
def clean_market_csv(csv_path, symbol, bar_freq="1min"):
    df = load_raw(csv_path, symbol)
    df = sort_dedup(df)
    bad = ohlcv_checks(df)
    if not bad.empty:
        df = fix_bad_bars(df)
    # Regularize to grid, then apply deliberate fill policy
    df = regularize(df, freq=bar_freq)
    df = fill_policy(df, method="ffill", max_gap="10min")
    df = cap_outliers(df, col="close", window=100, z=8.0)
    # Final assertions
    assert df["ts"].is_monotonic_increasing or df.sort_values("ts") is not None
    assert not df.duplicated(subset=["symbol","ts"]).any()
    return df