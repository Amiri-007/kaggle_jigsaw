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

# Import both the original and new metrics implementations
import src.metrics_v2 as metrics_v2
import bias_metrics


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
            # Calculate using original function
            orig_auc = bias_metrics.get_subgroup_auc(df, identity, "y_true", "y_pred")
            
            # Calculate using new vectorized function
            new_auc = metrics_v2.subgroup_auc(y_true, y_pred, identity_masks[identity])
            
            # Check that results match
            if orig_auc is not None and not np.isnan(new_auc):
                assert np.isclose(orig_auc, new_auc, rtol=1e-5), \
                    f"Subgroup AUC mismatch for {identity}: {orig_auc} vs {new_auc}"
            else:
                # Both should be None/NaN
                assert (orig_auc is None or np.isnan(orig_auc)) and np.isnan(new_auc)
    
    def test_bpsn_auc(self, test_data):
        """Test that bpsn_auc matches the original implementation."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        for identity in identity_cols:
            # Calculate using original function
            orig_auc = bias_metrics.get_bpsn_auc(df, identity, "y_true", "y_pred")
            
            # Calculate using new vectorized function
            new_auc = metrics_v2.bpsn_auc(y_true, y_pred, identity_masks[identity])
            
            # Check that results match
            if orig_auc is not None and not np.isnan(new_auc):
                assert np.isclose(orig_auc, new_auc, rtol=1e-5), \
                    f"BPSN AUC mismatch for {identity}: {orig_auc} vs {new_auc}"
            else:
                # Both should be None/NaN
                assert (orig_auc is None or np.isnan(orig_auc)) and np.isnan(new_auc)
    
    def test_bnsp_auc(self, test_data):
        """Test that bnsp_auc matches the original implementation."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        for identity in identity_cols:
            # Calculate using original function
            orig_auc = bias_metrics.get_bnsp_auc(df, identity, "y_true", "y_pred")
            
            # Calculate using new vectorized function
            new_auc = metrics_v2.bnsp_auc(y_true, y_pred, identity_masks[identity])
            
            # Check that results match
            if orig_auc is not None and not np.isnan(new_auc):
                assert np.isclose(orig_auc, new_auc, rtol=1e-5), \
                    f"BNSP AUC mismatch for {identity}: {orig_auc} vs {new_auc}"
            else:
                # Both should be None/NaN
                assert (orig_auc is None or np.isnan(orig_auc)) and np.isnan(new_auc)
                
    def test_power_mean(self):
        """Test the generalized power mean function."""
        test_values = [0.6, 0.7, 0.8, 0.9]
        
        # Test with p = -5 (emphasizes minimum)
        power_mean = metrics_v2.generalised_power_mean(test_values, p=-5)
        # Should be closer to the minimum than arithmetic mean
        assert power_mean < np.mean(test_values)
        assert power_mean > min(test_values)
        
        # Test with p = 1 (arithmetic mean)
        power_mean = metrics_v2.generalised_power_mean(test_values, p=1)
        assert np.isclose(power_mean, np.mean(test_values))
        
        # Test with p = -1 (harmonic mean)
        harmonic_mean = len(test_values) / sum(1/x for x in test_values)
        power_mean = metrics_v2.generalised_power_mean(test_values, p=-1)
        assert np.isclose(power_mean, harmonic_mean)
        
        # Test with missing values
        test_values_with_nan = [0.6, np.nan, 0.8, 0.9]
        power_mean = metrics_v2.generalised_power_mean(test_values_with_nan, p=-5)
        # Should ignore NaN values
        assert not np.isnan(power_mean)
        
        # Test with empty valid values
        power_mean = metrics_v2.generalised_power_mean([np.nan, np.nan], p=-5)
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
            metrics_v2.generalised_power_mean(subgroup_aucs, p=-5),
            metrics_v2.generalised_power_mean(bpsn_aucs, p=-5),
            metrics_v2.generalised_power_mean(bnsp_aucs, p=-5)
        ])
        expected_final_score = 0.25 * overall_auc + 0.75 * expected_bias_score
        
        assert np.isclose(final_score, expected_final_score, rtol=1e-5)
        assert np.isclose(detailed_metrics["bias_score"], expected_bias_score, rtol=1e-5)
        
    def test_compute_all_metrics(self, test_data):
        """Test the compute_all_metrics function."""
        df, y_true, y_pred, identity_masks, identity_cols = test_data
        
        # Get results from the comprehensive function
        results = metrics_v2.compute_all_metrics(y_true, y_pred, identity_masks)
        
        # Check structure of results
        assert "overall" in results
        assert "subgroup_metrics" in results
        assert "bias_metrics" in results
        
        # Check overall AUC
        assert np.isclose(results["overall"]["auc"], roc_auc_score(y_true, y_pred))
        
        # Verify each subgroup's metrics exist
        assert len(results["subgroup_metrics"]) == len(identity_masks)
        
        # Check a specific subgroup's metrics
        for subgroup_metrics in results["subgroup_metrics"]:
            assert "subgroup_name" in subgroup_metrics
            assert "subgroup_size" in subgroup_metrics
            assert "subgroup_auc" in subgroup_metrics
            assert "bpsn_auc" in subgroup_metrics
            assert "bnsp_auc" in subgroup_metrics


if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 