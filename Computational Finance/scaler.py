import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

class TimeSeriesScaler:
    def __init__(self, method: str = 'robust', columns: list = None):
        """
        Wrapper for Sklearn scalers to handle DataFrames safely.
        
        Args:
            method: 'standard', 'minmax', or 'robust' (Recommended for finance)
            columns: List of column names to scale. If None, scales all numeric cols.
        """
        self.method = method.lower()
        self.columns = columns
        self.scaler = None
        
        # Initialize the correct scaler
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'standard', 'minmax', or 'robust'.")

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler on the TRAINING data.
        """
        # If columns not specified, select all numeric float/int columns
        if self.columns is None:
            # Exclude likely targets or metadata if they are numeric (optional logic)
            # For now, we take all numeric types
            self.columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Safe removal of typical non-feature columns if they exist
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close', 'date', 'target']
            # Only keep columns that are actually generated features (optional strictness)
            # self.columns = [c for c in self.columns if c not in exclude_cols]

        self.scaler.fit(df[self.columns])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the fitted scaler.
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Call .fit() on training data first.")
        
        df_scaled = df.copy()
        
        # Apply scaling
        df_scaled[self.columns] = self.scaler.transform(df[self.columns])
        
        return df_scaled

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reverses the scaling (useful for visualizing predictions vs real values).
        """
        df_inv = df.copy()
        df_inv[self.columns] = self.scaler.inverse_transform(df[self.columns])
        return df_inv


# --- HELPER FUNCTION FOR PIPELINE ---

def scale_datasets(train_df, test_df, columns_to_scale, val_df=None, method='robust'):
    """
    Orchestrates the scaling process across splits.
    
    1. Fits on TRAIN
    2. Transforms TRAIN
    3. Transforms VAL (using Train's stats)
    4. Transforms TEST (using Train's stats)
    """
    scaler = TimeSeriesScaler(method=method, columns=columns_to_scale)
    
    # 1. Fit on Train
    scaler.fit(train_df)
    
    # 2. Transform All
    train_scaled = scaler.transform(train_df)
    val_scaled = scaler.transform(val_df) if val_df is not None else None
    test_scaled = scaler.transform(test_df)
    
    print(f"Scaler ({method}) fitted on Train. Transformed Train, Val, Test.")
    
    return (train_scaled,
           val_scaled, 
           test_scaled, scaler
          ) 