#!/usr/bin/env python3
"""
Unit tests for the enhanced metrics_v2 module.

These tests verify that the new vectorized implementations produce 
the same results as the original functions in bias_metrics.py.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import just the metrics_v2 module directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import metrics_v2

# Import original metrics if available
try:
    import bias_metrics
    HAS_ORIGINAL_METRICS = True
except ImportError:
    # Create mock bias_metrics module
    class MockBiasMetrics:
        @staticmethod
        def get_subgroup_auc(*args, **kwargs):
            return None
        
        @staticmethod
        def get_bpsn_auc(*args, **kwargs):
            return None
        
        @staticmethod
        def get_bnsp_auc(*args, **kwargs):
            return None
    
    bias_metrics = MockBiasMetrics()
    HAS_ORIGINAL_METRICS = False


def create_synthetic_data(n_samples=1000, n_identities=5, random_seed=42):
    """
    Create a synthetic dataset for testing bias metrics.
    
    Args:
        n_samples: Number of samples to generate
        n_identities: Number of identity subgroups
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (df, y_true, y_pred, identity_masks)
    """
    np.random.seed(random_seed)
    
    # Generate true labels
    y_true = np.random.randint(0, 2, size=n_samples)
    
    # Generate predictions with some bias
    # Make predictions slightly worse for some subgroups
    y_pred_base = y_true.copy().astype(float)
    # Add noise to make it non-perfect
    y_pred_base += np.random.normal(0, 0.3, size=n_samples)
    
    # Create identity columns
    identity_cols = [f"identity_{i}" for i in range(n_identities)]
    identity_data = np.random.randint(0, 2, size=(n_samples, n_identities))
    
    # Create pandas DataFrame
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": np.clip(y_pred_base, 0, 1)  # Ensure predictions are in [0,1]
    })
    
    # Add identity columns
    for i, col in enumerate(identity_cols):
        df[col] = identity_data[:, i]
    
    # Create identity masks dict for vectorized functions
    identity_masks = {
        col: df[col].values.astype(bool) for col in identity_cols
    }
    
    return df, y_true, df["y_pred"].values, identity_masks, identity_cols


class TestBiasMetrics:
    """Tests for the bias metrics implementations."""
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Create test data fixture for use in multiple tests."""
        return create_synthetic_data(n_samples=2000, n_identities=5, random_seed=42)
    
    def test_subgroup_auc(self, test_data):
        """Test that subgroup_auc matches the original implementation."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        for identity in identity_cols:
            # Calculate using new vectorized function
            new_auc = metrics_v2.subgroup_auc(y_true, y_pred, identity_masks[identity])
            
            if HAS_ORIGINAL_METRICS:
                # Calculate using original function
                orig_auc = bias_metrics.get_subgroup_auc(df, identity, "y_true", "y_pred")
                
                # Check that results match
                if orig_auc is not None and not np.isnan(new_auc):
                    assert np.isclose(orig_auc, new_auc, rtol=1e-5), \
                        f"Subgroup AUC mismatch for {identity}: {orig_auc} vs {new_auc}"
                else:
                    # Both should be None/NaN
                    assert (orig_auc is None or np.isnan(orig_auc)) and np.isnan(new_auc)
            else:
                # Just check that new_auc is a valid value
                assert isinstance(new_auc, (float, np.float64)) or np.isnan(new_auc)
    
    def test_bpsn_auc(self, test_data):
        """Test that bpsn_auc matches the original implementation."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        for identity in identity_cols:
            # Calculate using new vectorized function
            new_auc = metrics_v2.bpsn_auc(y_true, y_pred, identity_masks[identity])
            
            if HAS_ORIGINAL_METRICS:
                # Calculate using original function
                orig_auc = bias_metrics.get_bpsn_auc(df, identity, "y_true", "y_pred")
                
                # Check that results match
                if orig_auc is not None and not np.isnan(new_auc):
                    assert np.isclose(orig_auc, new_auc, rtol=1e-5), \
                        f"BPSN AUC mismatch for {identity}: {orig_auc} vs {new_auc}"
                else:
                    # Both should be None/NaN
                    assert (orig_auc is None or np.isnan(orig_auc)) and np.isnan(new_auc)
            else:
                # Just check that new_auc is a valid value
                assert isinstance(new_auc, (float, np.float64)) or np.isnan(new_auc)
    
    def test_bnsp_auc(self, test_data):
        """Test that bnsp_auc matches the original implementation."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        for identity in identity_cols:
            # Calculate using new vectorized function
            new_auc = metrics_v2.bnsp_auc(y_true, y_pred, identity_masks[identity])
            
            if HAS_ORIGINAL_METRICS:
                # Calculate using original function
                orig_auc = bias_metrics.get_bnsp_auc(df, identity, "y_true", "y_pred")
                
                # Check that results match
                if orig_auc is not None and not np.isnan(new_auc):
                    assert np.isclose(orig_auc, new_auc, rtol=1e-5), \
                        f"BNSP AUC mismatch for {identity}: {orig_auc} vs {new_auc}"
                else:
                    # Both should be None/NaN
                    assert (orig_auc is None or np.isnan(orig_auc)) and np.isnan(new_auc)
            else:
                # Just check that new_auc is a valid value
                assert isinstance(new_auc, (float, np.float64)) or np.isnan(new_auc)
    
    def test_empty_subgroup(self):
        """Test that metrics correctly handle empty subgroups and return np.nan."""
        # Create simple test data with one empty subgroup
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        
        # Empty subgroup mask
        empty_mask = np.zeros(len(y_true), dtype=bool)
        
        # Test all three metric functions
        assert np.isnan(metrics_v2.subgroup_auc(y_true, y_pred, empty_mask))
        assert np.isnan(metrics_v2.bpsn_auc(y_true, y_pred, empty_mask))
        assert np.isnan(metrics_v2.bnsp_auc(y_true, y_pred, empty_mask))
                
    def test_power_mean(self):
        """Test the generalized power mean function."""
        test_values = [0.6, 0.7, 0.8, 0.9]
        
        # Test with p = -5 (emphasizes minimum)
        power_mean = metrics_v2.generalized_power_mean(test_values, p=-5)
        # Should be closer to the minimum than arithmetic mean
        assert power_mean < np.mean(test_values)
        assert power_mean > min(test_values)
        
        # Test with p = 1 (arithmetic mean)
        power_mean = metrics_v2.generalized_power_mean(test_values, p=1)
        assert np.isclose(power_mean, np.mean(test_values))
        
        # Test with p = -1 (harmonic mean)
        harmonic_mean = len(test_values) / sum(1/x for x in test_values)
        power_mean = metrics_v2.generalized_power_mean(test_values, p=-1)
        assert np.isclose(power_mean, harmonic_mean)
        
        # Test with missing values
        test_values_with_nan = [0.6, np.nan, 0.8, 0.9]
        power_mean = metrics_v2.generalized_power_mean(test_values_with_nan, p=-5)
        # Should ignore NaN values
        assert not np.isnan(power_mean)
        
        # Test with empty valid values
        power_mean = metrics_v2.generalized_power_mean([np.nan, np.nan], p=-5)
        assert np.isnan(power_mean)
        
    def test_final_bias_score(self, test_data):
        """Test the final bias score calculation."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        # Calculate metrics using the new implementation
        subgroup_aucs = []
        bpsn_aucs = []
        bnsp_aucs = []
        
        for identity in identity_cols:
            subgroup_aucs.append(metrics_v2.subgroup_auc(y_true, y_pred, identity_masks[identity]))
            bpsn_aucs.append(metrics_v2.bpsn_auc(y_true, y_pred, identity_masks[identity]))
            bnsp_aucs.append(metrics_v2.bnsp_auc(y_true, y_pred, identity_masks[identity]))
        
        # Calculate overall AUC
        overall_auc = roc_auc_score(y_true, y_pred)
        
        # Calculate final score using our implementation
        bias_dict = {
            "subgroup_auc": subgroup_aucs,
            "bpsn_auc": bpsn_aucs,
            "bnsp_auc": bnsp_aucs
        }
        
        final_score, detailed_metrics = metrics_v2.final_bias_score(
            overall_auc, bias_dict, power=-5, weight_overall=0.25
        )
        
        # Verify weight is applied correctly (0.25 * overall_auc + 0.75 * bias_score)
        expected_bias_score = np.mean([
            metrics_v2.generalized_power_mean(subgroup_aucs, p=-5),
            metrics_v2.generalized_power_mean(bpsn_aucs, p=-5),
            metrics_v2.generalized_power_mean(bnsp_aucs, p=-5)
        ])
        expected_final_score = 0.25 * overall_auc + 0.75 * expected_bias_score
        
        assert np.isclose(final_score, expected_final_score, rtol=1e-5)
        assert np.isclose(detailed_metrics["bias_score"], expected_bias_score, rtol=1e-5)
    
    def test_compute_all_metrics(self, test_data):
        """Test the computation of all metrics."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        # Compute all metrics
        result = metrics_v2.compute_all_metrics(
            y_true, y_pred, identity_masks, model_name="test_model"
        )
        
        # Check that the result is a BiasReport object
        assert isinstance(result, metrics_v2.BiasReport)
        assert result.model_name == "test_model"
        assert isinstance(result.metrics, pd.DataFrame)
        assert isinstance(result.overall_auc, float)
        assert isinstance(result.final_score, float)
        
        # Check that all subgroups are in the metrics DataFrame
        assert set(result.metrics["subgroup_name"].values) == set(identity_cols)
        
        # Check that all required metrics are present
        for required_col in ["subgroup_auc", "bpsn_auc", "bnsp_auc", "subgroup_size"]:
            assert required_col in result.metrics.columns
    
    def test_bias_report_final_score(self):
        """Test that BiasReport.final_score matches hand-computed example."""
        # Create a simple example with known values for verification
        # Using a larger dataset to avoid NaN values
        n_samples = 50
        np.random.seed(42)
        
        # Generate binary labels and probabilities
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.clip(y_true + np.random.normal(0, 0.3, size=n_samples), 0, 1)
        
        # Create two balanced subgroups
        subgroup1 = np.zeros(n_samples, dtype=bool)
        subgroup1[:n_samples//2] = True
        subgroup2 = ~subgroup1
        
        # Compute results with the metrics module
        result = metrics_v2.compute_all_metrics(
            y_true, y_pred, {"group1": subgroup1, "group2": subgroup2}, 
            model_name="hand_computed_test", power=-5, weight_overall=0.25
        )
        
        # Get the individual metric values from the result metrics DataFrame
        metrics_df = result.metrics
        
        # Extract values for hand calculation
        subgroup1_values = metrics_df[metrics_df['subgroup_name'] == 'group1']
        subgroup2_values = metrics_df[metrics_df['subgroup_name'] == 'group2']
        
        subgroup1_auc = subgroup1_values['subgroup_auc'].values[0]
        subgroup1_bpsn = subgroup1_values['bpsn_auc'].values[0]
        subgroup1_bnsp = subgroup1_values['bnsp_auc'].values[0]
        
        subgroup2_auc = subgroup2_values['subgroup_auc'].values[0]
        subgroup2_bpsn = subgroup2_values['bpsn_auc'].values[0]
        subgroup2_bnsp = subgroup2_values['bnsp_auc'].values[0]
        
        # Hand compute the expected final score
        power_mean_subgroup_auc = metrics_v2.generalized_power_mean([subgroup1_auc, subgroup2_auc], p=-5)
        power_mean_bpsn_auc = metrics_v2.generalized_power_mean([subgroup1_bpsn, subgroup2_bpsn], p=-5)
        power_mean_bnsp_auc = metrics_v2.generalized_power_mean([subgroup1_bnsp, subgroup2_bnsp], p=-5)
        
        # Calculate bias score - only use non-NaN values
        valid_means = [m for m in [power_mean_subgroup_auc, power_mean_bpsn_auc, power_mean_bnsp_auc] 
                       if not np.isnan(m)]
        
        if valid_means:
            bias_score = np.mean(valid_means)
            expected_final_score = 0.25 * result.overall_auc + 0.75 * bias_score
            
            # Verify final score matches
            assert np.isclose(result.final_score, expected_final_score, rtol=1e-5)
        else:
            # If all metrics are NaN, the final score should also be NaN
            assert np.isnan(result.final_score)
    
    def test_list_identity_columns(self):
        """Test the auto-detection of identity columns."""
        # Create a test DataFrame with identity columns
        df = pd.DataFrame({
            # Identity columns (float type)
            'male': np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float),
            'female': np.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=float),
            'black': np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=float),
            
            # Non-float columns with identity names (should be ignored)
            'white': np.array([1, 1, 0, 0, 1], dtype=int),
            
            # Non-identity columns
            'id': [1, 2, 3, 4, 5],
            'text': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0],
            'pred': [0.1, 0.9, 0.3, 0.8, 0.2]
        })
        
        # Get identity columns
        identity_cols = metrics_v2.list_identity_columns(df)
        
        # Check that only float identity columns are included
        assert 'male' in identity_cols
        assert 'female' in identity_cols
        assert 'black' in identity_cols
        assert 'white' not in identity_cols  # int type, should be excluded
        
        # Check that non-identity columns are excluded
        assert 'id' not in identity_cols
        assert 'text' not in identity_cols
        assert 'target' not in identity_cols
        assert 'pred' not in identity_cols


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 