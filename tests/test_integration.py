"""
Integration tests for the full pipeline.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pipeline.clustering import cluster_pipeline
from pipeline.evaluation import compute_metrics, evaluate_results
from pipeline.feature_engineering import (
    create_feature_sets,
    encode_gene_expression_patterns,
    encode_tcr_sequences,
)
from pipeline.modeling import create_supervised_mask, prepare_ml_data, train_traditional_ml
from pipeline.preprocessing import preprocess_pipeline
from pipeline.utils import CacheManager, setup_random_seeds


@pytest.fixture
def mock_adata_full():
    """Create a comprehensive mock AnnData object."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 200

    # Create expression matrix
    X = np.random.poisson(5, size=(n_obs, n_vars)).astype(float)

    # Create observations
    obs = pd.DataFrame({
        "sample_id": [f"S{i%10}" for i in range(n_obs)],
        "patient_id": [f"P{i//20}" for i in range(n_obs)],
        "response": (["Responder"] * 50 + ["Non-Responder"] * 40 + ["Unknown"] * 10),
        "cdr3_TRA": ["ACDEFGHIKLMN"] * n_obs,
        "cdr3_TRB": ["PQRSTVWY"] * n_obs,
    })

    # Create variables
    var = pd.DataFrame(
        {"gene_names": [f"Gene_{i}" for i in range(n_vars)]},
        index=[f"Gene_{i}" for i in range(n_vars)]
    )

    adata = AnnData(X=X, obs=obs, var=var)
    return adata


class TestEndToEndPipeline:
    """Test end-to-end pipeline execution."""

    def test_preprocessing_integration(self, mock_adata_full):
        """Test preprocessing pipeline."""
        setup_random_seeds(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir)

            adata = preprocess_pipeline(
                mock_adata_full.copy(),
                min_genes=10,
                min_counts=50,
                max_pct_mt=50.0,
                min_cells=3,
                target_sum=10000,
                n_top_genes=100,
                use_cache=False,
                cache_manager=None
            )

            # Check preprocessing results
            assert adata.shape[0] > 0  # Some cells should remain
            assert adata.shape[1] > 0  # Some genes should remain
            assert "highly_variable" in adata.var

    def test_feature_engineering_integration(self, mock_adata_full):
        """Test feature engineering pipeline."""
        setup_random_seeds(42)

        # Preprocess first
        adata = preprocess_pipeline(
            mock_adata_full.copy(),
            min_genes=10,
            min_counts=50,
            use_cache=False,
            cache_manager=None
        )

        # Gene encoding
        adata = encode_gene_expression_patterns(
            adata,
            n_pca_components=20,
            n_svd_components=20,
            compute_umap=False,
            use_cache=False,
            cache_manager=None
        )

        # Check gene encodings
        assert "X_gene_pca" in adata.obsm
        assert "X_gene_svd" in adata.obsm
        assert adata.obsm["X_gene_pca"].shape[1] <= 20

        # TCR encoding
        adata = encode_tcr_sequences(
            adata,
            kmer_k=3,
            n_svd_components=50,
            use_cache=False,
            cache_manager=None
        )

        # Check TCR encodings
        assert "X_tcr_tra_kmer" in adata.obsm
        assert "X_tcr_trb_kmer" in adata.obsm

    def test_clustering_integration(self, mock_adata_full):
        """Test clustering pipeline."""
        setup_random_seeds(42)

        # Preprocess
        adata = preprocess_pipeline(
            mock_adata_full.copy(),
            min_genes=10,
            use_cache=False,
            cache_manager=None
        )

        # Encode
        adata = encode_gene_expression_patterns(
            adata,
            n_pca_components=20,
            use_cache=False,
            cache_manager=None
        )

        # Cluster
        adata = cluster_pipeline(
            adata,
            n_neighbors=10,
            n_pcs=20,
            leiden_resolutions=[0.5],
            kmeans_n_clusters=3,
            use_cache=False,
            cache_manager=None
        )

        # Check clustering results
        assert "leiden_0.5" in adata.obs
        assert len(adata.obs["leiden_0.5"].unique()) > 1

    def test_modeling_integration(self, mock_adata_full):
        """Test modeling pipeline."""
        setup_random_seeds(42)

        # Preprocess and encode
        adata = preprocess_pipeline(
            mock_adata_full.copy(),
            min_genes=10,
            n_top_genes=50,
            use_cache=False,
            cache_manager=None
        )

        adata = encode_gene_expression_patterns(
            adata,
            n_pca_components=20,
            n_svd_components=20,
            use_cache=False,
            cache_manager=None
        )

        adata = encode_tcr_sequences(
            adata,
            kmer_k=3,
            n_svd_components=50,
            use_cache=False,
            cache_manager=None
        )

        # Create supervised mask and feature sets
        supervised_mask = create_supervised_mask(adata)
        feature_sets = create_feature_sets(adata, supervised_mask, use_cache=False, cache_manager=None)

        # Prepare ML data
        feature_sets, y, groups = prepare_ml_data(
            adata,
            feature_sets,
            supervised_mask
        )

        # Train only on basic feature set for speed
        X = feature_sets["basic"]

        # Use only LogisticRegression for speed
        from sklearn.linear_model import LogisticRegression
        models = {"LogisticRegression": LogisticRegression(random_state=42)}
        param_grids = {"LogisticRegression": {"C": [1.0]}}

        results_df = train_traditional_ml(
            X,
            y,
            groups,
            "basic",
            models=models,
            param_grids=param_grids,
            use_random_search=False,
            use_cache=False,
            cache_manager=None
        )

        # Check results
        assert len(results_df) > 0
        assert "y_pred" in results_df.columns
        assert "y_test" in results_df.columns

    def test_evaluation_integration(self):
        """Test evaluation pipeline."""
        # Create mock results
        results_df = pd.DataFrame({
            "model": ["LogisticRegression"] * 5,
            "feature_set": ["basic"] * 5,
            "fold": range(5),
            "patient": [f"P{i}" for i in range(5)],
            "y_test": [[0, 1, 0, 1]] * 5,
            "y_pred": [[0, 1, 1, 1]] * 5,
            "y_proba": [[0.2, 0.8, 0.6, 0.9]] * 5
        })

        eval_df = evaluate_results(results_df)

        # Check evaluation results
        assert len(eval_df) > 0
        assert "accuracy" in eval_df.columns
        assert "roc_auc" in eval_df.columns
        assert "f1" in eval_df.columns


class TestCachingIntegration:
    """Test caching across pipeline."""

    def test_cache_persistence(self, mock_adata_full):
        """Test that cache persists across runs."""
        setup_random_seeds(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir)

            # First run - should compute
            adata1 = preprocess_pipeline(
                mock_adata_full.copy(),
                min_genes=10,
                use_cache=True,
                cache_manager=cache_manager
            )

            # Second run - should load from cache
            adata2 = preprocess_pipeline(
                mock_adata_full.copy(),
                min_genes=10,
                use_cache=True,
                cache_manager=cache_manager
            )

            # Results should be identical
            assert adata1.shape == adata2.shape
            np.testing.assert_array_almost_equal(adata1.X, adata2.X)

    def test_cache_invalidation_on_params(self, mock_adata_full):
        """Test that cache invalidates when parameters change."""
        setup_random_seeds(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir)

            # Run with one parameter
            adata1 = preprocess_pipeline(
                mock_adata_full.copy(),
                min_genes=10,
                n_top_genes=50,
                use_cache=True,
                cache_manager=cache_manager
            )

            # Run with different parameter - should recompute
            adata2 = preprocess_pipeline(
                mock_adata_full.copy(),
                min_genes=10,
                n_top_genes=100,  # Different
                use_cache=True,
                cache_manager=cache_manager
            )

            # Results should differ
            assert adata1.var["highly_variable"].sum() != adata2.var["highly_variable"].sum()


class TestReproducibility:
    """Test reproducibility across runs."""

    def test_reproducible_preprocessing(self, mock_adata_full):
        """Test that preprocessing is reproducible."""
        setup_random_seeds(42)
        adata1 = preprocess_pipeline(
            mock_adata_full.copy(),
            min_genes=10,
            use_cache=False
        )

        setup_random_seeds(42)
        adata2 = preprocess_pipeline(
            mock_adata_full.copy(),
            min_genes=10,
            use_cache=False
        )

        # Should produce identical results
        assert adata1.shape == adata2.shape
        np.testing.assert_array_almost_equal(adata1.X, adata2.X)

    def test_reproducible_modeling(self):
        """Test that modeling is reproducible."""
        setup_random_seeds(42)

        # Create simple training data
        X = np.random.rand(50, 10)
        y = np.random.randint(0, 2, 50)
        groups = np.repeat([f"P{i}" for i in range(5)], 10)

        from sklearn.linear_model import LogisticRegression
        models = {"LogisticRegression": LogisticRegression(random_state=42)}
        param_grids = {"LogisticRegression": {"C": [1.0]}}

        results1 = train_traditional_ml(
            X, y, groups, "test",
            models=models,
            param_grids=param_grids,
            use_random_search=False,
            use_cache=False
        )

        setup_random_seeds(42)
        results2 = train_traditional_ml(
            X, y, groups, "test",
            models=models,
            param_grids=param_grids,
            use_random_search=False,
            use_cache=False
        )

        # Results should be identical
        pd.testing.assert_frame_equal(
            results1.drop(columns=["best_params"], errors="ignore"),
            results2.drop(columns=["best_params"], errors="ignore")
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
