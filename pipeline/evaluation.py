"""
Evaluation module for computing metrics and aggregating results.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }

    # Add ROC-AUC if probabilities provided
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


def evaluate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate model results and compute metrics.

    Args:
        results_df: DataFrame with y_test, y_pred, y_proba columns

    Returns:
        DataFrame with computed metrics
    """
    print("Computing evaluation metrics...")

    eval_results = []

    # Group by model and feature set
    for (model, feature_set), group in results_df.groupby(['model', 'feature_set']):
        # Concatenate all predictions across folds
        y_test_all = np.concatenate([np.array(y) if isinstance(y, (list, np.ndarray)) else [y] for y in group['y_test']])
        y_pred_all = np.concatenate([np.array(y) if isinstance(y, (list, np.ndarray)) else [y] for y in group['y_pred']])

        # Handle y_proba
        try:
            y_proba_all = np.concatenate([np.array(y) if isinstance(y, (list, np.ndarray)) else [y] for y in group['y_proba']])
        except:
            y_proba_all = None

        # Compute metrics
        metrics = compute_metrics(y_test_all, y_pred_all, y_proba_all)
        metrics['model'] = model
        metrics['feature_set'] = feature_set
        metrics['n_folds'] = len(group)
        metrics['n_samples'] = len(y_test_all)

        eval_results.append(metrics)

    eval_df = pd.DataFrame(eval_results)

    # Sort by ROC-AUC descending
    if 'roc_auc' in eval_df.columns:
        eval_df = eval_df.sort_values('roc_auc', ascending=False)

    print(f"Evaluation complete: {len(eval_df)} model-feature combinations")

    return eval_df


def aggregate_patient_level(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate predictions at patient level.

    Args:
        results_df: DataFrame with patient-level results

    Returns:
        DataFrame with patient-level aggregated predictions
    """
    print("Aggregating patient-level predictions...")

    patient_results = []

    for (model, feature_set), group in results_df.groupby(['model', 'feature_set']):
        # Group by patient
        for patient, patient_group in group.groupby('patient'):
            # Average probabilities
            try:
                mean_proba = np.mean([p for p in patient_group['y_proba']])
                pred = 1 if mean_proba > 0.5 else 0
            except:
                pred = patient_group['y_pred'].mode()[0] if len(patient_group) > 0 else 0
                mean_proba = np.nan

            # Get true label
            true_label = patient_group['y_test'].iloc[0] if len(patient_group) > 0 else np.nan

            patient_results.append({
                'model': model,
                'feature_set': feature_set,
                'patient': patient,
                'y_true': true_label,
                'y_pred': pred,
                'y_proba': mean_proba,
                'n_samples': len(patient_group)
            })

    patient_df = pd.DataFrame(patient_results)

    print(f"Patient-level aggregation complete: {len(patient_df)} patient predictions")

    return patient_df


def get_best_model(eval_df: pd.DataFrame, metric: str = 'roc_auc') -> Dict:
    """
    Get best model based on specified metric.

    Args:
        eval_df: Evaluation DataFrame
        metric: Metric to use for selection

    Returns:
        Dictionary with best model information
    """
    if metric not in eval_df.columns:
        raise ValueError(f"Metric {metric} not found in evaluation results")

    best_idx = eval_df[metric].idxmax()
    best_row = eval_df.loc[best_idx]

    return {
        'model': best_row['model'],
        'feature_set': best_row['feature_set'],
        metric: best_row[metric],
        'accuracy': best_row.get('accuracy', np.nan),
        'f1': best_row.get('f1', np.nan)
    }


def print_evaluation_summary(eval_df: pd.DataFrame, top_n: int = 5) -> None:
    """
    Print evaluation summary.

    Args:
        eval_df: Evaluation DataFrame
        top_n: Number of top models to display
    """
    print("\n" + "=" * 80)
    print(f"TOP {top_n} MODELS BY ROC-AUC")
    print("=" * 80)

    # Select columns to display
    display_cols = ['model', 'feature_set', 'roc_auc', 'accuracy', 'f1', 'precision', 'recall']
    display_cols = [col for col in display_cols if col in eval_df.columns]

    top_models = eval_df.head(top_n)[display_cols]

    # Format numbers
    for col in ['roc_auc', 'accuracy', 'f1', 'precision', 'recall']:
        if col in top_models.columns:
            top_models[col] = top_models[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")

    print(top_models.to_string(index=False))
    print("=" * 80)
