import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union

from .utils import list_identity_columns

def apply_negative_downsampling(
    df: pd.DataFrame,
    target_col: str = 'target',
    identity_cols: Optional[List[str]] = None,
    first_epoch: bool = True,
    drop_percentage: float = 0.5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Apply negative downsampling according to the 3rd-place strategy:
        - Epoch 1: drop 50% of rows where target < 0.2 AND identity_columns.sum() == 0
        - Epoch 2+: restore half of the dropped examples
    
    Args:
        df: DataFrame to apply downsampling to
        target_col: Column name for target labels
        identity_cols: List of identity column names
        first_epoch: Whether this is the first epoch
        drop_percentage: Percentage of negative examples to drop
        random_state: Random seed for reproducibility
        
    Returns:
        Downsampled DataFrame
    """
    # Return original dataframe if not first epoch and no identity columns
    if not first_epoch or identity_cols is None:
        return df
    
    # Set default identity columns if none provided
    if identity_cols is None:
        identity_cols = list_identity_columns()
    
    # Ensure all identity columns exist
    for col in identity_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Calculate identity sum
    df['identity_sum'] = df[identity_cols].sum(axis=1)
    
    # Find rows to potentially drop: negative examples without identity markers
    mask_to_drop = (df[target_col] < 0.2) & (df['identity_sum'] == 0)
    drop_indices = df[mask_to_drop].index.tolist()
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Randomly select indices to drop
    if first_epoch:
        # Drop 50% of eligible rows in first epoch
        num_to_drop = int(len(drop_indices) * drop_percentage)
        drop_indices = np.random.choice(drop_indices, num_to_drop, replace=False)
    else:
        # In subsequent epochs, keep more examples (drop only 25%)
        num_to_drop = int(len(drop_indices) * drop_percentage / 2)
        drop_indices = np.random.choice(drop_indices, num_to_drop, replace=False)
    
    # Drop the selected indices
    df_downsampled = df.drop(drop_indices).reset_index(drop=True)
    
    # Remove temporary identity_sum column if we created it
    if 'identity_sum' in df_downsampled.columns and 'identity_sum' not in df.columns:
        df_downsampled = df_downsampled.drop('identity_sum', axis=1)
    
    print(f"Applied negative downsampling: {len(df)} -> {len(df_downsampled)} examples")
    return df_downsampled


def get_sample_weights(
    df: pd.DataFrame,
    target_col: str = 'target',
    identity_cols: Optional[List[str]] = None,
    base_weight: float = 1.0,
    identity_weight: float = 3.0,
    target_weight: float = 8.0,
    annotator_weight: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate sample weights for each example according to the strategy:
        - w = 1
        - w += 3 * identity_sum
        - w += 8 * target
        - w += 2 * log(toxicity_annotator_count + 2)  # if annotator_weight=True
        - w /= w.max()
    
    Args:
        df: DataFrame to calculate weights for
        target_col: Column name for target labels
        identity_cols: List of identity column names
        base_weight: Base weight for all samples
        identity_weight: Weight multiplier for identity columns
        target_weight: Weight multiplier for target value
        annotator_weight: Whether to apply annotator count weight term
        normalize: Whether to normalize weights to max value of 1.0
        
    Returns:
        Array of sample weights
    """
    # Set default identity columns if none provided
    if identity_cols is None:
        identity_cols = list_identity_columns()
    
    # Ensure all identity columns exist
    for col in identity_cols:
        if col not in df.columns:
            df[col] = 0
    
    # Calculate identity sum if it doesn't exist
    if 'identity_sum' not in df.columns:
        df['identity_sum'] = df[identity_cols].sum(axis=1)
    
    # Initialize weights with base value
    weights = np.ones(len(df)) * base_weight
    
    # Add weight for identity columns
    weights += identity_weight * df['identity_sum'].values
    
    # Add weight for target value
    weights += target_weight * df[target_col].values
    
    # Add weight based on annotator count (3rd place solution trick)
    if annotator_weight and 'toxicity_annotator_count' in df.columns:
        weights += 2 * np.log(df['toxicity_annotator_count'].values + 2)
    
    # Normalize weights if requested
    if normalize:
        weights = weights / weights.max()
    
    return weights 