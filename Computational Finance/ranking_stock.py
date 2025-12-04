import pandas as pd
import numpy as np

def _calculate_group_z_scores(df:pd.DataFrame, 
                              factor_col:str, 
                              group_col:str,
                              direction:float,
                              min_count:int
                              )->pd.Series:
    """
    Internal function to calculate Z-scores within a specific group.
    
    Parameters:
    - direction: 1 for 'Higher is Better' (ROE), -1 for 'Lower is Better' (P/E).
    - min_count: Minimum stocks required in a group to calculate a valid Z-score.
    """
    df = df.copy()

    # 1. Calculate Group Stats
    # Transform('count') allows to filter group 
    grouper = df.groupby(group_col)[factor_col]

    # 2. Calculate Stats
    # Note: 'count' excludes NaNs by default
    group_counts = grouper.transform("count")
    group_means = grouper.transform("mean")
    group_stds = grouper.transform("std", ddof=1) # Explicit sample std

    # 2. Calculate Z-score within each group_col
    z_scores = (df[factor_col] - group_means) / group_stds.replace(0, np.nan)

    # 3. Apply direction

    z_scores = z_scores * direction

    # 4. Filter Small Sample Sizes (Safety Net)

    z_scores = z_scores.where(group_counts > min_count, np.nan)

    return z_scores

def rank_by_sector(df:pd.DataFrame, factor_col:str, **kwargs):
    """
    Ranking Stocks by Sector
    i.e: Compares 'Apple' vs 'Orange'

    """
    return _calculate_group_z_scores(df, factor_col, "sector", **kwargs)

def rank_by_industry_group(df:pd.DataFrame, factor_col:str, **kwargs):
    """
    Ranking Stocks by Industry_Group
    i.e: Compares 'Apple' vs 'Apple'

    """
    return _calculate_group_z_scores(df, factor_col, "industry_group", **kwargs)

def rank_by_industry(df:pd.DataFrame, factor_col:str, **kwargs):
    """
    Ranking Stocks by Industry
    i.e: Compares 'red Apple' vs 'green Apple'

    """
    return _calculate_group_z_scores(df, factor_col, "industry", **kwargs)

def rank_by_sub_industry(df:pd.DataFrame, factor_col:str, **kwargs):
    """
    Ranking Stocks by Sub_Industry
    i.e: Compares 'red Apple' vs 'red Apple'

    """
    return _calculate_group_z_scores(df, factor_col, "sub_industry", **kwargs)



import pandas as pd
import numpy as np

def rank_stocks_multi_factor(df, factor_config, group_col='sector'):
    """
    Ranks stocks based on a weighted composite of multiple factors.
    
    Parameters:
    - df: DataFrame with factor columns.
    - factor_config: Dictionary defining factors.
      Format: {
        'col_name': {'weight': 0.4, 'direction': -1}, 
        ...
      }
    - group_col: Column to group by (e.g., 'sector' or 'industry').
                 Set to None for Global Ranking.
    """
    df = df.copy()
    
    # 1. Initialize Total Score
    df['composite_z_score'] = 0
    total_weight = 0
    
    # 2. Loop through each factor
    for col, config in factor_config.items():
        weight = config.get('weight', 1.0)
        direction = config.get('direction', 1) # 1 = High is Good, -1 = Low is Good
        
        # A. Calculate Z-Score (Sector Neutral or Global)
        if group_col:
            group_means = df.groupby(group_col)[col].transform('mean')
            group_stds = df.groupby(group_col)[col].transform('std')
            z_score = (df[col] - group_means) / group_stds.replace(0, 1)
        else:
            z_score = (df[col] - df[col].mean()) / df[col].std()
            
        # B. Winsorize Outliers (Clip at +/- 3)
        # Prevents one extreme stock from distorting the whole ranking
        z_score = z_score.clip(-3, 3)
        
        # C. Add to Composite (Direction * Weight * Z-Score)
        # Note: We fill NaN Z-scores with 0 (Neutral) so we don't lose the stock entirely
        df['composite_z_score'] += z_score.fillna(0) * direction * weight
        total_weight += weight

    # 3. Normalize Composite Score (Optional, but good for readability)
    if total_weight > 0:
        df['composite_z_score'] /= total_weight
        
    # 4. Generate Final Percentile Rank (0 to 1)
    # We rank based on the Composite Score. 1.0 = Best Stock.
    if group_col:
        df['final_rank'] = df.groupby(group_col)['composite_z_score'].rank(pct=True)
    else:
        df['final_rank'] = df['composite_z_score'].rank(pct=True)
        
    return df.sort_values('final_rank', ascending=False)

