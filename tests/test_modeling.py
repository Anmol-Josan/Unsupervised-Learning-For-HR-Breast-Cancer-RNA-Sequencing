"""
Unit tests for pipeline.modeling module.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pipeline.modeling import (
    create_supervised_mask,
    get_default_models,
    get_default_param_grids,
    prepare_ml_data,
    train_single_fold,
)


class TestSupervisedMask:
    """Tests for supervised mask creation."""

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData."""
        obs = pd.DataFrame({
            "response": ["Responder", "Non-Responder", "Unknown", "Responder", "Non-Responder"]
        })
        return AnnData(X=np.random.rand(5, 10), obs=obs)

    def test_create_supervised_mask(self, mock_adata):
        """Test supervised mask creation."""
        mask = create_supervised_mask(mock_adata, response_col="response")

        # Should be True for Responder/Non-Responder, False for Unknown
        expected = np.array([True, True, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_supervised_mask_count(self, mock_adata):
        """Test supervised sample count."""
        mask = create_supervised_mask(mock_adata)
        assert mask.sum() == 4  # 4 labeled samples


class TestModelConfiguration:
    """Tests for model and parameter grid configuration."""

    def test_get_default_models(self):
        """Test default model retrieval."""
        models = get_default_models()

        assert "LogisticRegression" in models
        assert "DecisionTree" in models
        assert "RandomForest" in models

    def test_get_default_param_grids(self):
        """Test default parameter grids."""
        param_grids = get_default_param_grids()

        assert "LogisticRegression" in param_grids
        assert "DecisionTree" in param_grids
        assert "RandomForest" in param_grids

        # Check LogisticRegression params
        assert "C" in param_grids["LogisticRegression"]
        assert isinstance(param_grids["LogisticRegression"]["C"], list)


class TestDataPreparation:
    """Tests for ML data preparation."""

    @pytest.fixture
    def mock_setup(self):
        """Create mock data setup."""
        n_samples = 50
        n_features = 20

        # Create mock AnnData
        obs = pd.DataFrame({
            "response": ["Responder"] * 25 + ["Non-Responder"] * 25,
            "patient_id": [f"P{i//10}" for i in range(n_samples)]
        })
        adata = AnnData(X=np.random.rand(n_samples, 100), obs=obs)

        # Create mock feature sets
        feature_sets = {
            "basic": np.random.rand(n_samples, n_features),
            "enhanced": np.random.rand(n_samples, n_features * 2)
        }

        supervised_mask = np.ones(n_samples, dtype=bool)

        return adata, feature_sets, supervised_mask

    def test_prepare_ml_data(self, mock_setup):
        """Test ML data preparation."""
        adata, feature_sets, supervised_mask = mock_setup

        prepared_features, y, groups = prepare_ml_data(
            adata,
            feature_sets,
            supervised_mask,
            response_col="response",
            patient_col="patient_id"
        )

        # Check outputs
        assert isinstance(prepared_features, dict)
        assert isinstance(y, np.ndarray)
        assert isinstance(groups, np.ndarray)

        # Check labels are encoded as 0/1
        assert set(y) == {0, 1}

        # Check groups
        assert len(groups) == supervised_mask.sum()


class TestModelTraining:
    """Tests for model training."""

    @pytest.fixture
    def training_data(self):
        """Create training data."""
        np.random.seed(42)

        X_train = np.random.rand(80, 20)
        X_test = np.random.rand(20, 20)
        y_train = np.random.randint(0, 2, 80)
        y_test = np.random.randint(0, 2, 20)

        return X_train, X_test, y_train, y_test

    def test_train_single_fold(self, training_data):
        """Test single fold training."""
        X_train, X_test, y_train, y_test = training_data

        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42)
        param_grid = {"C": [0.1, 1.0, 10.0]}

        result = train_single_fold(
            model,
            param_grid,
            X_train,
            y_train,
            X_test,
            y_test,
            use_random_search=False,
            n_iter=5,
            cv_splits=3
        )

        # Check outputs
        assert "best_params" in result
        assert "y_pred" in result
        assert "y_proba" in result
        assert "y_test" in result

        assert len(result["y_pred"]) == len(y_test)
        assert len(result["y_proba"]) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
