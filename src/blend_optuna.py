import os
import sys
import argparse
import logging
import json
import glob
import optuna
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from optuna.samplers import TPESampler

# Try to import fairness metrics from the RDS repository
try:
    from fairness.metrics_v2 import final_score, BiasReport
except ImportError:
    # Fallback to metrics module if metrics_v2 is not available
    try:
        from src.metrics_v2 import final_score, BiasReport
    except ImportError:
        print("Warning: Could not import fairness metrics. Please ensure the fairness metrics module is available.")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_predictions(preds_dir: str = 'output/preds') -> Tuple[List[str], List[pd.DataFrame]]:
    """
    Load all prediction CSV files from the specified directory.
    
    Args:
        preds_dir: Directory containing prediction CSV files
        
    Returns:
        List of model names and list of prediction DataFrames
    """
    # Find all CSV files
    pred_files = glob.glob(os.path.join(preds_dir, '*.csv'))
    
    if not pred_files:
        logger.error(f"No prediction files found in {preds_dir}")
        raise FileNotFoundError(f"No prediction files found in {preds_dir}")
    
    # Load each file
    model_names = []
    pred_dfs = []
    
    for pred_file in pred_files:
        model_name = os.path.basename(pred_file).split('.')[0]
        df = pd.read_csv(pred_file)
        
        # Ensure DataFrame has 'id' and 'prediction' columns
        if 'id' not in df.columns or 'prediction' not in df.columns:
            logger.warning(f"Skipping {pred_file}: Missing required columns (id, prediction)")
            continue
        
        model_names.append(model_name)
        pred_dfs.append(df)
        logger.info(f"Loaded predictions from {pred_file}: {len(df)} examples")
    
    return model_names, pred_dfs

def load_ground_truth(data_path: str) -> pd.DataFrame:
    """
    Load ground truth data with identity columns for fairness evaluation.
    
    Args:
        data_path: Path to ground truth data CSV file
        
    Returns:
        DataFrame containing ground truth data
    """
    if not os.path.exists(data_path):
        logger.error(f"Ground truth file not found: {data_path}")
        raise FileNotFoundError(f"Ground truth file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_cols = ['id', 'target']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in ground truth data")
            raise ValueError(f"Required column '{col}' not found in ground truth data")
    
    logger.info(f"Loaded ground truth data: {len(df)} examples")
    return df

def blend_predictions(
    pred_dfs: List[pd.DataFrame],
    weights: List[float]
) -> pd.DataFrame:
    """
    Blend predictions using given weights.
    
    Args:
        pred_dfs: List of prediction DataFrames
        weights: List of weights for each model
        
    Returns:
        DataFrame with blended predictions
    """
    if len(pred_dfs) != len(weights):
        logger.error(f"Number of models ({len(pred_dfs)}) does not match number of weights ({len(weights)})")
        raise ValueError(f"Number of models ({len(pred_dfs)}) does not match number of weights ({len(weights)})")
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Start with the first DataFrame
    blended_df = pred_dfs[0].copy()
    blended_df['prediction'] = blended_df['prediction'] * weights[0]
    
    # Add weighted predictions from other models
    for i in range(1, len(pred_dfs)):
        blended_df['prediction'] += pred_dfs[i]['prediction'] * weights[i]
    
    return blended_df

def evaluate_fairness(
    preds_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    identity_cols: Optional[List[str]] = None
) -> float:
    """
    Evaluate fairness of predictions using the final_score metric.
    
    Args:
        preds_df: DataFrame with predictions
        ground_truth_df: DataFrame with ground truth data
        identity_cols: List of identity columns to consider for fairness
        
    Returns:
        Fairness score (higher is better)
    """
    # Set default identity columns if none provided
    if identity_cols is None:
        identity_cols = [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness',
            'asian', 'hindu', 'buddhist', 'atheist', 'bisexual', 'transgender'
        ]
    
    # Merge predictions with ground truth
    merged_df = ground_truth_df.merge(preds_df, on='id', how='inner')
    
    # Calculate fairness score using BiasReport
    bias_report = BiasReport(
        df=merged_df,
        identity_cols=identity_cols,
        label_col='target',
        pred_col='prediction'
    )
    
    # Get final score
    score = final_score(bias_report)
    
    return score

def objective(
    trial: optuna.Trial,
    model_names: List[str],
    pred_dfs: List[pd.DataFrame],
    ground_truth_df: pd.DataFrame,
    identity_cols: Optional[List[str]] = None
) -> float:
    """
    Optuna objective function for optimizing blending weights.
    
    Args:
        trial: Optuna trial
        model_names: List of model names
        pred_dfs: List of prediction DataFrames
        ground_truth_df: DataFrame with ground truth data
        identity_cols: List of identity columns for fairness evaluation
        
    Returns:
        Fairness score (higher is better)
    """
    # Sample weights for each model
    weights = []
    for i, model_name in enumerate(model_names):
        # Sample weight between 0 and 1
        weight = trial.suggest_float(f"weight_{i}_{model_name}", 0.0, 1.0)
        weights.append(weight)
    
    # Blend predictions
    blended_df = blend_predictions(pred_dfs, weights)
    
    # Evaluate fairness
    fairness_score = evaluate_fairness(
        preds_df=blended_df,
        ground_truth_df=ground_truth_df,
        identity_cols=identity_cols
    )
    
    return fairness_score

def optimize_weights(
    model_names: List[str],
    pred_dfs: List[pd.DataFrame],
    ground_truth_df: pd.DataFrame,
    identity_cols: Optional[List[str]] = None,
    n_trials: int = 200,
    output_dir: str = 'output',
    random_state: int = 42
) -> Dict[str, float]:
    """
    Optimize blending weights using Optuna.
    
    Args:
        model_names: List of model names
        pred_dfs: List of prediction DataFrames
        ground_truth_df: DataFrame with ground truth data
        identity_cols: List of identity columns for fairness evaluation
        n_trials: Number of Optuna trials
        output_dir: Directory to save results
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of optimized weights for each model
    """
    # Create Optuna study
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction='maximize',  # Maximize fairness score
        sampler=sampler
    )
    
    # Run optimization
    objective_func = lambda trial: objective(
        trial=trial,
        model_names=model_names,
        pred_dfs=pred_dfs,
        ground_truth_df=ground_truth_df,
        identity_cols=identity_cols
    )
    
    logger.info(f"Starting optimization with {n_trials} trials")
    study.optimize(objective_func, n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best fairness score: {best_value:.4f}")
    logger.info(f"Best parameters: {best_params}")
    
    # Extract weights
    weights = {}
    for i, model_name in enumerate(model_names):
        param_name = f"weight_{i}_{model_name}"
        weights[model_name] = best_params[param_name]
    
    # Normalize weights
    total_weight = sum(weights.values())
    weights = {model: weight / total_weight for model, weight in weights.items()}
    
    # Save weights to JSON
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, 'blend_weights.json')
    with open(weights_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    logger.info(f"Saved weights to {weights_path}")
    
    # Generate and save blended predictions
    weight_list = [weights[model_name] for model_name in model_names]
    blended_df = blend_predictions(pred_dfs, weight_list)
    
    # Save blended predictions
    blended_path = os.path.join(output_dir, 'preds', 'blended.csv')
    os.makedirs(os.path.dirname(blended_path), exist_ok=True)
    blended_df.to_csv(blended_path, index=False)
    
    logger.info(f"Saved blended predictions to {blended_path}")
    
    return weights

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Optimize ensemble weights using Optuna")
    parser.add_argument('--preds_dir', type=str, default='output/preds',
                        help="Directory containing prediction CSV files")
    parser.add_argument('--ground_truth', type=str, required=True,
                        help="Path to ground truth data with identity columns")
    parser.add_argument('--output_dir', type=str, default='output',
                        help="Directory to save results")
    parser.add_argument('--n_trials', type=int, default=200,
                        help="Number of Optuna trials")
    parser.add_argument('--random_state', type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Load predictions
    try:
        model_names, pred_dfs = load_predictions(args.preds_dir)
        logger.info(f"Loaded predictions for {len(model_names)} models")
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        sys.exit(1)
    
    # Load ground truth data
    try:
        ground_truth_df = load_ground_truth(args.ground_truth)
    except Exception as e:
        logger.error(f"Error loading ground truth data: {e}")
        sys.exit(1)
    
    # Optimize weights
    try:
        weights = optimize_weights(
            model_names=model_names,
            pred_dfs=pred_dfs,
            ground_truth_df=ground_truth_df,
            n_trials=args.n_trials,
            output_dir=args.output_dir,
            random_state=args.random_state
        )
        
        logger.info("Optimization complete")
        logger.info("Optimized weights:")
        for model_name, weight in weights.items():
            logger.info(f"  {model_name}: {weight:.4f}")
            
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise

if __name__ == '__main__':
    main() 