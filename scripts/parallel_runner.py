#!/usr/bin/env python3
"""
Ray-based distributed execution coordinator for VM-level parallelization.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    print("Warning: Ray not installed. Install with: pip install ray")

from pipeline.modeling import train_traditional_ml
from pipeline.utils import ConfigManager


@ray.remote(num_cpus=4) if HAS_RAY else None
def train_fold_remote(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_set_name: str,
    config: dict
) -> dict:
    """
    Ray remote function for distributed LOPO fold processing.

    Args:
        X: Feature matrix
        y: Labels
        groups: Patient groups
        fold_idx: Fold index
        train_idx: Training indices
        test_idx: Test indices
        feature_set_name: Name of feature set
        config: Configuration dictionary

    Returns:
        Dictionary with fold results
    """
    from pipeline.modeling import train_single_fold, get_default_models, get_default_param_grids

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    models = get_default_models()
    param_grids = get_default_param_grids()

    fold_results = []

    for model_name, model in models.items():
        if model_name not in param_grids:
            continue

        result = train_single_fold(
            model,
            param_grids[model_name],
            X_train,
            y_train,
            X_test,
            y_test,
            use_random_search=config.get('use_random_search', True),
            n_iter=config.get('n_iter', 15)
        )

        fold_results.append({
            'model': model_name,
            'feature_set': feature_set_name,
            'fold': fold_idx,
            'patient': groups[test_idx[0]],
            'best_params': str(result['best_params']),
            'y_test': result['y_test'],
            'y_pred': result['y_pred'],
            'y_proba': result['y_proba']
        })

    return {'fold': fold_idx, 'results': fold_results}


def train_parallel_ray(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_set_name: str,
    config: dict,
    ray_address: str = None
) -> list:
    """
    Train models using Ray for distributed execution.

    Args:
        X: Feature matrix
        y: Labels
        groups: Patient groups
        feature_set_name: Name of feature set
        config: Configuration dictionary
        ray_address: Ray cluster address

    Returns:
        List of results
    """
    if not HAS_RAY:
        raise RuntimeError("Ray not installed. Install with: pip install ray")

    # Initialize Ray
    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            ray.init(ignore_reinit_error=True)

    print(f"Ray cluster resources: {ray.cluster_resources()}")

    # Generate LOPO folds
    from sklearn.model_selection import LeaveOneGroupOut
    logo = LeaveOneGroupOut()
    folds = list(logo.split(X, y, groups))

    print(f"Distributing {len(folds)} LOPO folds across Ray cluster...")

    # Distribute folds across cluster
    futures = [
        train_fold_remote.remote(
            X, y, groups, fold_idx, train_idx, test_idx, feature_set_name, config
        )
        for fold_idx, (train_idx, test_idx) in enumerate(folds)
    ]

    # Collect results
    print("Collecting results from distributed workers...")
    fold_results = ray.get(futures)

    # Flatten results
    all_results = []
    for fold_result in fold_results:
        all_results.extend(fold_result['results'])

    print(f"Completed {len(folds)} folds")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Distributed pipeline execution with Ray")
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address (auto-detects if None)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="config/pipeline_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["basic", "gene_enhanced", "tcr_enhanced", "comprehensive"],
        default="comprehensive",
        help="Feature set to use"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to preprocessed adata file"
    )

    args = parser.parse_args()

    if not HAS_RAY:
        print("Error: Ray not installed. Install with: pip install ray")
        sys.exit(1)

    # Load configuration
    print("=" * 80)
    print("DISTRIBUTED PIPELINE EXECUTION WITH RAY")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Ray address: {args.ray_address or 'auto-detect'}")

    config = ConfigManager(args.config)

    # Load data
    print(f"\nLoading data from {args.data_path}...")
    import scanpy as sc
    adata = sc.read_h5ad(args.data_path)

    # Create supervised mask and prepare data
    from pipeline.modeling import create_supervised_mask, prepare_ml_data
    from pipeline.feature_engineering import create_feature_sets

    supervised_mask = create_supervised_mask(adata)
    feature_sets = create_feature_sets(adata, supervised_mask, use_cache=False, cache_manager=None)
    feature_sets, y, groups = prepare_ml_data(adata, feature_sets, supervised_mask)

    # Get feature set
    X = feature_sets[args.feature_set]
    print(f"Feature set: {args.feature_set}, shape: {X.shape}")

    # Train using Ray
    ml_config = {
        'use_random_search': config.get('pipeline.steps.modeling.traditional_ml.use_random_search', True),
        'n_iter': config.get('pipeline.steps.modeling.traditional_ml.n_iter_search', 15)
    }

    results = train_parallel_ray(
        X, y, groups, args.feature_set, ml_config, ray_address=args.ray_address
    )

    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results)
    output_path = Path("./Output") / f"ray_results_{args.feature_set}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")
    print("=" * 80)

    # Optionally shutdown Ray
    # ray.shutdown()


if __name__ == "__main__":
    main()
