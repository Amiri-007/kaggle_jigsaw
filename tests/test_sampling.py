import pandas as pd
import numpy as np
import pytest
from src.data.sampling import apply_negative_downsampling, get_sample_weights, list_identity_columns

def test_list_identity_columns():
    """Test that list_identity_columns returns the expected columns"""
    identity_cols = list_identity_columns()
    assert len(identity_cols) == 15
    assert 'male' in identity_cols
    assert 'female' in identity_cols
    assert 'transgender' in identity_cols

def test_negative_downsampling_first_epoch():
    """Test that negative downsampling drops approximately 50% of eligible rows in the first epoch"""
    # Create a test DataFrame with 100 rows
    # - 60 rows with target < 0.2 and no identity markers (eligible for dropping)
    # - 20 rows with target < 0.2 but with identity markers (not eligible)
    # - 20 rows with target >= 0.2 (not eligible)
    
    np.random.seed(42)
    
    identity_cols = list_identity_columns()
    n_rows = 100
    
    # Initialize DataFrame with zeros
    df = pd.DataFrame(0, index=range(n_rows), columns=['target'] + identity_cols)
    
    # Set first 60 rows as eligible for dropping (target < 0.2, no identity)
    df.loc[:59, 'target'] = np.random.uniform(0, 0.19, 60)
    
    # Set next 20 rows with identity markers but still target < 0.2
    df.loc[60:79, 'target'] = np.random.uniform(0, 0.19, 20)
    for i in range(60, 80):
        # Randomly set 1-3 identity columns to 1
        num_identities = np.random.randint(1, 4)
        identity_indices = np.random.choice(len(identity_cols), num_identities, replace=False)
        for idx in identity_indices:
            df.loc[i, identity_cols[idx]] = 1
    
    # Set last 20 rows with target >= 0.2
    df.loc[80:, 'target'] = np.random.uniform(0.2, 1.0, 20)
    
    # Apply downsampling
    df_downsampled = apply_negative_downsampling(
        df, 
        target_col='target',
        identity_cols=identity_cols,
        first_epoch=True,
        random_state=42
    )
    
    # Calculate expected number of rows after downsampling
    # We expect to drop 50% of the 60 eligible rows = 30 rows
    expected_rows = n_rows - 30
    
    # Assert that we have the expected number of rows
    assert len(df_downsampled) == expected_rows, \
        f"Expected {expected_rows} rows, got {len(df_downsampled)}"
    
    # Calculate how many eligible rows were actually dropped
    eligible_before = len(df[(df['target'] < 0.2) & (df[identity_cols].sum(axis=1) == 0)])
    eligible_after = len(df_downsampled[(df_downsampled['target'] < 0.2) & 
                                         (df_downsampled[identity_cols].sum(axis=1) == 0)])
    dropped = eligible_before - eligible_after
    
    # Assert we dropped approximately 50% of eligible rows
    assert dropped == 30, f"Expected to drop 30 rows, dropped {dropped}"
    
    # Assert that rows with identity markers were not dropped
    identity_before = len(df[(df[identity_cols].sum(axis=1) > 0)])
    identity_after = len(df_downsampled[(df_downsampled[identity_cols].sum(axis=1) > 0)])
    assert identity_before == identity_after, \
        f"Expected {identity_before} rows with identity markers, got {identity_after}"

def test_negative_downsampling_later_epoch():
    """Test that negative downsampling drops fewer rows in later epochs"""
    np.random.seed(42)
    
    identity_cols = list_identity_columns()
    n_rows = 100
    
    # Initialize DataFrame with zeros
    df = pd.DataFrame(0, index=range(n_rows), columns=['target'] + identity_cols)
    
    # Set first 60 rows as eligible for dropping (target < 0.2, no identity)
    df.loc[:59, 'target'] = np.random.uniform(0, 0.19, 60)
    
    # Apply downsampling with first_epoch=False
    df_downsampled = apply_negative_downsampling(
        df, 
        target_col='target',
        identity_cols=identity_cols,
        first_epoch=False,
        random_state=42
    )
    
    # Assert that no rows were dropped since first_epoch=False
    assert len(df_downsampled) == n_rows, \
        f"Expected {n_rows} rows (no dropping), got {len(df_downsampled)}"

def test_sample_weights():
    """Test that sample weights are calculated correctly"""
    np.random.seed(42)
    
    identity_cols = list_identity_columns()
    n_rows = 100
    
    # Initialize DataFrame with zeros
    df = pd.DataFrame(0, index=range(n_rows), columns=['target'] + identity_cols)
    
    # Set random target values
    df['target'] = np.random.uniform(0, 1, n_rows)
    
    # Set random identity values for some rows
    for i in range(n_rows):
        if np.random.random() > 0.7:  # 30% chance of having identity markers
            num_identities = np.random.randint(1, 4)
            identity_indices = np.random.choice(len(identity_cols), num_identities, replace=False)
            for idx in identity_indices:
                df.loc[i, identity_cols[idx]] = 1
    
    # Calculate sample weights
    weights = get_sample_weights(
        df,
        target_col='target',
        identity_cols=identity_cols,
        base_weight=1.0,
        identity_weight=3.0,
        target_weight=8.0,
        normalize=True
    )
    
    # Assert weights have the correct length
    assert len(weights) == n_rows
    
    # Assert all weights are between 0 and 1
    assert np.all(weights >= 0) and np.all(weights <= 1)
    
    # Assert at least one weight is exactly 1.0 (due to normalization)
    assert np.isclose(np.max(weights), 1.0)
    
    # Test a specific example
    # Create a synthetic row
    test_df = pd.DataFrame({
        'target': [0.5],
        **{col: [0] for col in identity_cols}
    })
    
    # Set two identity columns to 1
    test_df['male'] = 1
    test_df['christian'] = 1
    
    # Expected weight before normalization: 1 + 3*2 + 8*0.5 = 11
    # But since we'd need to normalize with max weight, we'll just check the formula
    
    test_weights = get_sample_weights(
        test_df,
        target_col='target',
        identity_cols=identity_cols,
        base_weight=1.0,
        identity_weight=3.0,
        target_weight=8.0,
        normalize=False
    )
    
    expected_weight = 1.0 + 3.0*2 + 8.0*0.5
    assert np.isclose(test_weights[0], expected_weight)

if __name__ == "__main__":
    test_list_identity_columns()
    test_negative_downsampling_first_epoch()
    test_negative_downsampling_later_epoch()
    test_sample_weights()
    print("All tests passed!") 