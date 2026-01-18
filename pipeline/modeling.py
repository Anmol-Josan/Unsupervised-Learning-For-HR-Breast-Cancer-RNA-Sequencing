"""
Modeling module for traditional ML and deep learning with LOPO cross-validation.
"""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from pipeline.utils import CacheManager, compute_data_hash


def create_supervised_mask(adata: AnnData, response_col: str = 'response') -> np.ndarray:
    """
    Create mask for supervised learning samples (those with response labels).

    Args:
        adata: AnnData object
        response_col: Column name for response labels

    Returns:
        Boolean mask array
    """
    supervised_mask = adata.obs[response_col].isin(['Responder', 'Non-Responder'])
    print(f"Supervised samples: {supervised_mask.sum()} / {len(supervised_mask)}")
    return supervised_mask.values


def prepare_ml_data(
    adata: AnnData,
    feature_sets: Dict[str, np.ndarray],
    supervised_mask: np.ndarray,
    response_col: str = 'response',
    patient_col: str = 'patient_id'
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning.

    Args:
        adata: AnnData object
        feature_sets: Dictionary of feature sets
        supervised_mask: Boolean mask for supervised samples
        response_col: Column name for response labels
        patient_col: Column name for patient IDs

    Returns:
        Tuple of (feature_sets, y_encoded, groups)
    """
    print("Preparing ML data...")

    # Get labels
    y = adata.obs[response_col][supervised_mask]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Get patient groups for LOPO CV
    groups = np.array(adata.obs[patient_col][supervised_mask])

    print(f"  Samples: {len(y_encoded)}")
    print(f"  Classes: {dict(zip(le.classes_, [np.sum(y_encoded == i) for i in range(len(le.classes_))]))}")
    print(f"  Patients: {len(np.unique(groups))}")

    return feature_sets, y_encoded, groups


def get_default_param_grids() -> Dict[str, Dict]:
    """Get default hyperparameter grids for traditional ML models."""
    param_grids = {
        'LogisticRegression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs'],
            'max_iter': [1000]
        },
        'DecisionTree': {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    }

    if HAS_XGBOOST:
        param_grids['XGBoost'] = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0]
        }

    return param_grids


def get_default_models() -> Dict[str, Any]:
    """Get default models."""
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=1)
    }

    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(random_state=42, n_jobs=1, eval_metric='logloss')

    return models


def train_single_fold(
    model: Any,
    param_grid: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    use_random_search: bool = True,
    n_iter: int = 15,
    cv_splits: int = 3
) -> Dict[str, Any]:
    """
    Train a single fold with hyperparameter search.

    Args:
        model: Sklearn model
        param_grid: Hyperparameter grid
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        use_random_search: Whether to use RandomizedSearchCV
        n_iter: Number of iterations for random search
        cv_splits: Number of CV splits for hyperparameter search

    Returns:
        Dictionary with results
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter search
    grid_size = np.prod([len(v) for v in param_grid.values()])

    if use_random_search and grid_size > n_iter:
        search = RandomizedSearchCV(
            model,
            param_grid,
            n_iter=n_iter,
            cv=min(cv_splits, len(np.unique(y_train))),
            random_state=42,
            n_jobs=-1
        )
    else:
        search = GridSearchCV(
            model,
            param_grid,
            cv=min(cv_splits, len(np.unique(y_train))),
            n_jobs=-1
        )

    # Fit
    search.fit(X_train_scaled, y_train)

    # Predict
    y_pred = search.predict(X_test_scaled)
    y_proba = search.predict_proba(X_test_scaled)[:, 1] if hasattr(search, 'predict_proba') else y_pred

    return {
        'best_params': search.best_params_,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test
    }


def train_traditional_ml(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_set_name: str,
    models: Optional[Dict[str, Any]] = None,
    param_grids: Optional[Dict[str, Dict]] = None,
    use_random_search: bool = True,
    n_iter: int = 15,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> pd.DataFrame:
    """
    Train traditional ML models with LOPO cross-validation.

    Args:
        X: Feature matrix
        y: Labels
        groups: Patient groups for LOPO
        feature_set_name: Name of feature set
        models: Dictionary of models
        param_grids: Dictionary of parameter grids
        use_random_search: Whether to use RandomizedSearchCV
        n_iter: Number of iterations for random search
        use_cache: Whether to use cached results
        cache_manager: Cache manager instance

    Returns:
        DataFrame with results
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "train_traditional_ml",
            {
                "feature_set": feature_set_name,
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "use_random_search": use_random_search,
                "n_iter": n_iter
            },
            data_hash=compute_data_hash(X)
        )
        cached_data = cache_manager.load_cache(cache_key, "modeling", format="pt")
        if cached_data is not None:
            print(f"Loaded traditional ML results from cache")
            return cached_data

    if models is None:
        models = get_default_models()
    if param_grids is None:
        param_grids = get_default_param_grids()

    print(f"Training traditional ML models on {feature_set_name} features...")
    print(f"  Feature shape: {X.shape}")
    print(f"  Models: {list(models.keys())}")

    # LOPO cross-validation
    logo = LeaveOneGroupOut()
    results_list = []

    for model_name, model in models.items():
        if model_name not in param_grids:
            print(f"  Skipping {model_name} (no param grid)")
            continue

        print(f"  Training {model_name}...")
        param_grid = param_grids[model_name]

        for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            result = train_single_fold(
                model,
                param_grid,
                X_train,
                y_train,
                X_test,
                y_test,
                use_random_search=use_random_search,
                n_iter=n_iter
            )

            # Store results
            results_list.append({
                'model': model_name,
                'feature_set': feature_set_name,
                'fold': fold_idx,
                'patient': groups[test_idx[0]],
                'best_params': str(result['best_params']),
                'y_test': result['y_test'],
                'y_pred': result['y_pred'],
                'y_proba': result['y_proba']
            })

        print(f"    Completed {fold_idx + 1} folds")

    results_df = pd.DataFrame(results_list)

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(results_df, cache_key, "modeling", format="pt")

    return results_df


def train_all_feature_sets(
    feature_sets: Dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    models: Optional[Dict[str, Any]] = None,
    param_grids: Optional[Dict[str, Dict]] = None,
    use_random_search: bool = True,
    n_iter: int = 15,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> pd.DataFrame:
    """
    Train models on all feature sets.

    Args:
        feature_sets: Dictionary of feature sets
        y: Labels
        groups: Patient groups
        models: Dictionary of models
        param_grids: Dictionary of parameter grids
        use_random_search: Whether to use RandomizedSearchCV
        n_iter: Number of iterations
        use_cache: Whether to use cached results
        cache_manager: Cache manager instance

    Returns:
        Combined DataFrame with all results
    """
    print("=" * 60)
    print("Training models on all feature sets")
    print("=" * 60)

    all_results = []

    for feature_set_name, X in feature_sets.items():
        print(f"\nFeature set: {feature_set_name}")
        print("-" * 60)

        results_df = train_traditional_ml(
            X,
            y,
            groups,
            feature_set_name,
            models=models,
            param_grids=param_grids,
            use_random_search=use_random_search,
            n_iter=n_iter,
            use_cache=use_cache,
            cache_manager=cache_manager
        )

        all_results.append(results_df)

    combined_results = pd.concat(all_results, ignore_index=True)

    print("=" * 60)
    print(f"Training complete: {len(combined_results)} total runs")
    print("=" * 60)

    return combined_results
