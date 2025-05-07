#!/usr/bin/env python
"""
Unit tests for the figure utility functions.
"""

import os
import numpy as np
import pandas as pd
import pytest
from src import figure_utils, threshold_utils
import matplotlib.pyplot as plt


# Fixture for synthetic dataset
@pytest.fixture
def synthetic_data():
    # Create synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Generate random labels and predictions
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.random(n_samples)
    
    # Generate synthetic identity groups
    identity_cols = ['identity_1', 'identity_2', 'identity_3']
    identity_arrays = [np.random.randint(0, 2, n_samples) for _ in range(len(identity_cols))]
    
    # Create synthetic metrics DataFrame
    metrics_data = []
    
    for i, col in enumerate(identity_cols):
        for metric in ['subgroup_auc', 'bpsn_auc', 'bnsp_auc', 'power_diff']:
            value = np.random.random()
            metrics_data.append({
                'identity_group': col,
                'metric_name': metric,
                'value': value
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create a synthetic training DataFrame
    train_data = {
        'text': ['text'] * n_samples,
        'target': y_true
    }
    
    for i, col in enumerate(identity_cols):
        train_data[col] = identity_arrays[i]
    
    train_df = pd.DataFrame(train_data)
    
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'identity_cols': identity_cols,
        'identity_arrays': identity_arrays,
        'metrics_df': metrics_df,
        'train_df': train_df
    }


def test_identity_prevalence(synthetic_data):
    """Test identity prevalence figure generation."""
    # Check if function runs without errors
    fig = figure_utils.identity_prevalence(
        synthetic_data['train_df'], 
        synthetic_data['identity_cols']
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/identity_prevalence.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_roc_curve_figure(synthetic_data):
    """Test ROC curve figure generation."""
    # Check if function runs without errors
    fig = figure_utils.roc_curve_figure(
        synthetic_data['y_true'], 
        synthetic_data['y_pred'], 
        'test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/overall_roc_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_fairness_heatmap(synthetic_data):
    """Test fairness heatmap figure generation."""
    # Check if function runs without errors
    fig = figure_utils.fairness_heatmap(
        synthetic_data['metrics_df'], 
        'test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/fairness_heatmap_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_power_mean_bar(synthetic_data):
    """Test power mean bar figure generation."""
    # Check if function runs without errors
    fig = figure_utils.power_mean_bar(
        synthetic_data['metrics_df'], 
        'test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/power_mean_bar_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_threshold_sweep(synthetic_data):
    """Test threshold sweep figure generation."""
    # Check if function runs without errors
    fig = figure_utils.threshold_sweep(
        synthetic_data['y_true'], 
        synthetic_data['y_pred'], 
        synthetic_data['identity_arrays'], 
        'test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/threshold_sweep_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_worst_k_table(synthetic_data):
    """Test worst k table figure generation."""
    # Check if function runs without errors
    fig = figure_utils.worst_k_table(
        synthetic_data['metrics_df'], 
        k=3, 
        model_name='test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/worst_k_table_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_before_vs_after_scatter(synthetic_data):
    """Test before vs after scatter figure generation."""
    # Create a copy of the metrics DataFrame with slightly different values for the "improved" model
    improved_df = synthetic_data['metrics_df'].copy()
    improved_df['value'] = improved_df['value'] * 1.1
    
    # Check if function runs without errors
    fig = figure_utils.before_vs_after_scatter(
        synthetic_data['metrics_df'], 
        improved_df, 
        synthetic_data['identity_cols']
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/before_vs_after_scatter.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_sweep_thresholds():
    """Test sweep thresholds function."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 500
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.random(n_samples)
    identity_mask = np.random.randint(0, 2, n_samples).astype(bool)
    
    # Run the function
    results = threshold_utils.sweep_thresholds(y_true, y_pred, identity_mask, n_steps=11)
    
    # Check if the function returns a DataFrame with expected columns
    assert isinstance(results, pd.DataFrame)
    assert 'threshold' in results.columns
    assert 'fpr_gap' in results.columns
    assert 'fnr_gap' in results.columns
    assert 'final_score' in results.columns
    
    # Check if the thresholds span the range [0, 1]
    assert results['threshold'].min() == 0
    assert results['threshold'].max() == 1
    
    # Check if the number of rows matches n_steps
    assert len(results) == 11


def test_worst_k_bar(synthetic_data):
    """Test worst k bar figure generation."""
    # Check if function runs without errors
    fig = figure_utils.worst_k_bar(
        synthetic_data['metrics_df'], 
        model_name='test_model',
        k=3
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/worst_k_bar_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_add_confusion_mosaic(synthetic_data):
    """Test confusion mosaic figure generation."""
    # Set up test data
    y_true = synthetic_data['y_true']
    y_pred = (synthetic_data['y_pred'] > 0.5).astype(int)  # Convert to binary predictions
    identity_mask = synthetic_data['identity_arrays'][0].astype(bool)
    
    # Check if function runs without errors
    fig = figure_utils.add_confusion_mosaic(
        y_true, 
        y_pred, 
        identity_mask, 
        identity_name='test_identity',
        model_name='test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/confusion_test_identity_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_error_gap_heatmap(synthetic_data):
    """Test error gap heatmap figure generation."""
    # Create sample gaps dictionary
    gaps_dict = {
        'identity_1': (0.1, -0.05),
        'identity_2': (-0.15, 0.2),
        'identity_3': (0.02, 0.03)
    }
    
    # Check if function runs without errors
    fig = figure_utils.error_gap_heatmap(
        model_name='test_model',
        gaps_dict=gaps_dict
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/error_gap_heatmap_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_threshold_gap_curve(synthetic_data):
    """Test threshold gap curve figure generation."""
    # Create sample sweep DataFrame
    thresholds = np.linspace(0, 1, 11)
    sweep_data = []
    for t in thresholds:
        sweep_data.append({
            'threshold': t,
            'fpr_gap': np.sin(t * np.pi) * 0.1,
            'fnr_gap': np.cos(t * np.pi) * 0.1,
            'abs_fpr_gap': abs(np.sin(t * np.pi) * 0.1),
            'abs_fnr_gap': abs(np.cos(t * np.pi) * 0.1),
            'mean_gap': (abs(np.sin(t * np.pi) * 0.1) + abs(np.cos(t * np.pi) * 0.1)) / 2
        })
    df_sweep = pd.DataFrame(sweep_data)
    
    # Check if function runs without errors
    fig = figure_utils.threshold_gap_curve(
        df_sweep,
        identity_name='test_identity',
        model_name='test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/threshold_gap_curve_test_identity_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_grouped_bar_by_identity(synthetic_data):
    """Test grouped bar by identity figure generation."""
    # Check if function runs without errors
    fig = figure_utils.grouped_bar_by_identity(
        synthetic_data['metrics_df'], 
        model_name='test_model'
    )
    
    # Check if figure was saved
    assert os.path.exists('figs/grouped_bar_test_model.png')
    
    # Check if the function returns a figure
    assert fig is not None


def test_fairness_heatmap_colorbar_settings(synthetic_data):
    """Test that fairness heatmap uses the correct vmin/vmax settings."""
    # Generate the heatmap
    fig = figure_utils.fairness_heatmap(
        synthetic_data['metrics_df'], 
        'test_model'
    )
    
    # Get the axes from the figure
    ax = fig.axes[0]
    
    # Get the heatmap object (first collection in the axes)
    heatmap = ax.collections[0]
    
    # Check colormap, vmin, vmax settings
    assert heatmap.cmap.name == 'RdYlGn_r'
    assert heatmap.norm.vmin == 0.5
    assert heatmap.norm.vmax == 1.0
    
    # Check that colorbar exists and has the correct label
    colorbar = heatmap.colorbar
    assert colorbar is not None
    assert colorbar.ax.get_ylabel() == 'AUC'
    
    # Clean up
    plt.close(fig)


def test_error_rate_gaps():
    """Test error_rate_gaps function."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.random(n_samples)
    identity_mask = np.random.randint(0, 2, n_samples).astype(bool)
    
    # Run the function
    fpr_gap, fnr_gap = threshold_utils.error_rate_gaps(y_true, y_prob, identity_mask)
    
    # Check that the function returns the expected types
    assert isinstance(fpr_gap, float)
    assert isinstance(fnr_gap, float)
    
    # Check that the function returns values within reasonable bounds
    assert -1 <= fpr_gap <= 1
    assert -1 <= fnr_gap <= 1 