import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


import pandas as pd
import numpy as np


# ============================================
# 1) OUTLIER DETECTION METHODS
# ============================================

def detect_outliers_iqr(df, column="volume", k=1.5, window_size=30,
                        lower_quartile=0.25, upper_quartile=0.75):

    df2 = df.copy()

    q1 = df2[column].rolling(window_size, min_periods=window_size//2).quantile(lower_quartile).shift()
    q3 = df2[column].rolling(window_size, min_periods=window_size//2).quantile(upper_quartile).shift()

    iqr = q3 - q1
    df2["lower_bound"] = q1 - k * iqr
    df2["upper_bound"] = q3 + k * iqr

    mask = (df2[column] < df2["lower_bound"]) | (df2[column] > df2["upper_bound"])
    return df2.loc[mask]   


def detect_outliers_zscore(df, column="volume", threshold=3, window_size=30):
    mean = df[column].rolling(window_size, min_periods=window_size//2).mean().shift()
    std = df[column].rolling(window_size, min_periods=window_size//2).std().shift().replace(0, 1e-9)

    z = (df[column] - mean) / std
    mask = z.abs() > threshold

    out = df.loc[mask].copy()
    out["lower_bound"] = mean - threshold * std
    out["upper_bound"] = mean + threshold * std
    return out


def detect_outliers_mad(df, column="volume", threshold=3, window_size=30):
    med = df[column].rolling(window_size, min_periods=window_size//2).median().shift()
    rolling = df[column].rolling(window_size, min_periods=window_size//2)
    mad = rolling.apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True).shift().replace(0, 1e-9)

    zmad = 0.6745 * (df[column] - med) / mad
    mask = zmad.abs() > threshold

    out = df.loc[mask].copy()
    out["lower_bound"] = med - threshold * mad
    out["upper_bound"] = med + threshold * mad
    return out


# ----- Plotly Visualization for Outliers -----
def visualize_outliers(df, column="volume", method=None, **kwargs):

    outliers = method(df, column=column, **kwargs)
    plt.figure(figsize=(12,6))
    plt.plot(df["date"], df[column], label="Data", color="blue")
    if not outliers.empty:
        plt.scatter(outliers["date"], outliers[column], color="red", label="Outliers")
    else:
        print("No outliers detected.")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.title(f"Outlier Detection in {column}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()




def visualize_outliers_plotly(df, column="volume", method=None, **kwargs):
    if method is None:
        print("Error: Please provide a detection method.")
        return

    # Detect outliers using the passed method and kwargs
    outliers = method(df, column=column, **kwargs)

    # Create the figure
    fig = go.Figure()

    # Trace 1: The Main Line (Normal Data)
    fig.add_trace(go.Scatter(
        x=df["date"], 
        y=df[column],
        mode='lines',
        name='Normal Data',
        line=dict(color='rgba(0, 255, 255, 1)') # cyan line
    ))

    # Trace 2: The Outliers (Scatter Points)
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers["date"], 
            y=outliers[column],
            mode='markers',
            name='Outliers',
            marker=dict(size=8, color='rgba(255, 20, 147, 1)') # cyan pink
        ))

    # Update Layout
    fig.update_layout(
        title=f"Outlier Detection: {column} (Window={kwargs.get('window_size', 'N/A')})",
        xaxis_title="Date",
        yaxis_title=column,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
        # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
   )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(128,128,128,0.5)'
  )
    fig.show()

# Handle Outliers Methods

def handle_outliers(df, column="volume", method=None, **kwargs):
    """
    Handle outliers by capping them to lower/upper bounds.

    Parameters:
        df : pd.DataFrame
        column : str, column to clean
        method : function, should return DataFrame with columns ['lower_bound', 'upper_bound']
        **kwargs : extra arguments to method
    """
    df_clean = df.copy()
    
    # Detect outliers
    detected = method(df_clean, column=column, **kwargs)
    if detected.empty:
        return df_clean

    idx_out = detected.index

    # Cap outliers to bounds
    df_clean.loc[idx_out, column] = detected.apply(
        lambda row: row["lower_bound"] if row[column] < row["lower_bound"] 
                    else row["upper_bound"],
        axis=1
    )

    return df_clean

def remove_outlier_mad(df, column, threshold=3):
    median = df[column].median()
    mad = (df[column] - median).abs().median()
    if mad == 0:
        return df[column]
    z = 1.4826 * (df[column] - median) / mad
    return df[column].where(abs(z) < threshold, median)


def winsorize(df, column, lower_q=0.01, upper_q=0.99):
    lower = df[column].quantile(lower_q)
    upper = df[column].quantile(upper_q)
    df[column] = df[column].clip(lower, upper)
    return df