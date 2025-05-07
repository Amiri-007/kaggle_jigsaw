import pandas as pd
import numpy as np
import pytest
from src.data.sampling import get_sample_weights, list_identity_columns

def test_annotator_count_weight():
    """Test that the annotator count weight is applied correctly"""
    identity_cols = list_identity_columns()
    
    # Create test DataFrame with varying annotator counts
    df = pd.DataFrame({
        'target': [0.5] * 5,
        'toxicity_annotator_count': [1, 5, 10, 20, 50],
        **{col: [0] * 5 for col in identity_cols}
    })
    
    # Get weights with annotator_weight enabled
    weights_with_annotator = get_sample_weights(
        df,
        identity_cols=identity_cols,
        annotator_weight=True,
        normalize=False
    )
    
    # Get weights with annotator_weight disabled
    weights_without_annotator = get_sample_weights(
        df,
        identity_cols=identity_cols,
        annotator_weight=False,
        normalize=False
    )
    
    # The weights should be higher when annotator_weight is enabled
    assert np.all(weights_with_annotator > weights_without_annotator)
    
    # The weight difference should be log(annotator_count + 2) * 2
    expected_diffs = 2 * np.log(df['toxicity_annotator_count'] + 2)
    actual_diffs = weights_with_annotator - weights_without_annotator
    
    assert np.allclose(actual_diffs, expected_diffs)
    
    # Weights should increase with higher annotator counts
    assert np.all(np.diff(weights_with_annotator) > 0)
    
    # Verify the exact value for a specific case
    # For annotator_count=10:
    # Base weight: 1 + 8*0.5 = 5
    # With annotator: 5 + 2*log(10+2) = 5 + 2*log(12) ≈ 5 + 2*2.48 ≈ 5 + 4.96 ≈ 9.96
    expected_weight = 1.0 + 8.0*0.5 + 2.0*np.log(12)
    assert np.isclose(weights_with_annotator[2], expected_weight)

def test_weight_normalization():
    """Test that weight normalization works correctly with annotator weights"""
    identity_cols = list_identity_columns()
    
    # Create test DataFrame with varying annotator counts
    df = pd.DataFrame({
        'target': [0.3, 0.6, 0.9],
        'toxicity_annotator_count': [5, 15, 30],
        **{col: [0, 0, 0] for col in identity_cols}
    })
    
    # Set one row to have identity markers
    df.loc[1, 'male'] = 1
    df.loc[1, 'christian'] = 1
    
    # Get weights with normalization
    normalized_weights = get_sample_weights(
        df,
        identity_cols=identity_cols,
        annotator_weight=True,
        normalize=True
    )
    
    # Get weights without normalization
    unnormalized_weights = get_sample_weights(
        df,
        identity_cols=identity_cols,
        annotator_weight=True,
        normalize=False
    )
    
    # The normalized weights should have a maximum of 1.0
    assert np.isclose(np.max(normalized_weights), 1.0)
    
    # The relative proportions should be preserved after normalization
    normalized_ratios = normalized_weights[1:] / normalized_weights[0]
    unnormalized_ratios = unnormalized_weights[1:] / unnormalized_weights[0]
    
    assert np.allclose(normalized_ratios, unnormalized_ratios)

if __name__ == "__main__":
    test_annotator_count_weight()
    test_weight_normalization()
    print("All weight tests passed!") 