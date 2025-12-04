import pandas as pd
import torch

# =====Split Data Methods=====

import pandas as pd

def vanilla_split_data(df: pd.DataFrame, 
                       train_size: float = 0.7, 
                       val_size: float = 0.15, 
                       test_size: float = 0.15,
                       purge_size: int = 0) -> tuple:
    """
    Standard time split with PURGING to prevent feature leakage.
    purge_size: Should equal your largest rolling window (e.g., 50).
    """
    assert train_size + val_size + test_size == 1.0
    
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Slice with Purge
    # Train ends at T. Val starts at T + Purge.
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end + purge_size : val_end]
    test_df = df.iloc[val_end + purge_size :]
    
    return train_df, val_df, test_df

class PurgedWalkForwardCV:
    """
    Rolling Walk-Forward split that respects time and prevents leakage.
    Logic: Expanding Window Training.
    """
    def __init__(self, n_splits: int = 5, purge_size: int = 0):
        self.n_splits = n_splits
        self.purge_size = purge_size

    def split(self, df: pd.DataFrame):
        n = len(df)
        
        # FIX: Subtract purge_size from n before dividing.
        # This guarantees the last fold (with the gap) fits inside the dataframe.
        fold_size = (n - self.purge_size) // (self.n_splits + 1)
        
        indices = []
        
        for i in range(self.n_splits):
            # Train grows: 0 to (End of Fold i)
            train_end = fold_size * (i + 1)
            
            # Test is the NEXT fold
            test_start = train_end + self.purge_size
            test_end = test_start + fold_size
            
            # Stop if we run out of data
            if test_end > n:
                break
                
            train_idx = slice(0, train_end)
            test_idx = slice(test_start, test_end)
            
            yield df.iloc[train_idx], df.iloc[test_idx]


# Tranform to torch tensor

def tranform_to_tensor(train_set: pd.DataFrame, 
                       val_set: pd.DataFrame, 
                       test_set: pd.DataFrame, 
                       input_features: list, 
                       targets: list) -> tuple:
    
    # 1. Convert to Tensors
    X_train = torch.tensor(train_set[input_features].to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(train_set[targets].to_numpy(), dtype=torch.float32)
    
    X_val = torch.tensor(val_set[input_features].to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(val_set[targets].to_numpy(), dtype=torch.float32)
    
    X_test = torch.tensor(test_set[input_features].to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(test_set[targets].to_numpy(), dtype=torch.float32)

    # 2. Return the variables
    return (X_train, y_train, X_val, y_val, X_test, y_test)




    