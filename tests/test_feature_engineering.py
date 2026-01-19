"""
Unit tests for pipeline.feature_engineering module.
"""

import numpy as np
import pytest
from anndata import AnnData

from pipeline.feature_engineering import (
    _clean_seq,
    create_feature_sets,
    kmer_encode_sequence,
    one_hot_encode_sequence,
    physicochemical_features,
)


class TestSequenceCleaning:
    """Tests for sequence cleaning."""

    def test_clean_seq_valid(self):
        """Test cleaning valid sequences."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        cleaned = _clean_seq(seq)
        assert cleaned == "ACDEFGHIKLMNPQRSTVWY"

    def test_clean_seq_lowercase(self):
        """Test cleaning lowercase sequences."""
        seq = "acdefghiklmnpqrstvwy"
        cleaned = _clean_seq(seq)
        assert cleaned == "ACDEFGHIKLMNPQRSTVWY"

    def test_clean_seq_invalid_chars(self):
        """Test removing invalid characters."""
        seq = "ACT123XYZ"
        cleaned = _clean_seq(seq)
        assert cleaned == "ACTXYZ"  # Should keep valid amino acids

    def test_clean_seq_empty(self):
        """Test cleaning empty sequences."""
        assert _clean_seq("") == ""
        assert _clean_seq(None) == ""


class TestPhysicochemicalFeatures:
    """Tests for physicochemical feature computation."""

    def test_physicochemical_basic(self):
        """Test basic physicochemical feature computation."""
        seq = "ACDEFG"
        features = physicochemical_features(seq)

        assert "length" in features
        assert "molecular_weight" in features
        assert "hydrophobicity" in features
        assert "charge" in features
        assert "polarity" in features

        assert features["length"] == 6

    def test_physicochemical_empty(self):
        """Test physicochemical features for empty sequence."""
        features = physicochemical_features("")

        assert features["length"] == 0
        assert features["molecular_weight"] == 0.0
        assert features["hydrophobicity"] == 0.0

    def test_physicochemical_reproducible(self):
        """Test that features are reproducible."""
        seq = "ACDEFGHIKLMNP"
        features1 = physicochemical_features(seq)
        features2 = physicochemical_features(seq)

        assert features1 == features2


class TestOneHotEncoding:
    """Tests for one-hot encoding."""

    def test_onehot_basic(self):
        """Test basic one-hot encoding."""
        seq = "ACT"
        max_length = 10
        encoded = one_hot_encode_sequence(seq, max_length=max_length)

        # Should be flattened: max_length * 20 (amino acids)
        assert encoded.shape == (max_length * 20,)
        assert encoded.sum() == 3  # Three amino acids

    def test_onehot_padding(self):
        """Test that short sequences are padded."""
        seq = "A"
        max_length = 5
        encoded = one_hot_encode_sequence(seq, max_length=max_length)

        assert encoded.shape == (max_length * 20,)
        assert encoded.sum() == 1  # Only one amino acid

    def test_onehot_truncation(self):
        """Test that long sequences are truncated."""
        seq = "A" * 100
        max_length = 10
        encoded = one_hot_encode_sequence(seq, max_length=max_length)

        assert encoded.shape == (max_length * 20,)
        assert encoded.sum() == 10  # Truncated to 10


class TestKmerEncoding:
    """Tests for k-mer encoding."""

    def test_kmer_basic(self):
        """Test basic k-mer extraction."""
        seq = "ACDEFG"
        kmers = kmer_encode_sequence(seq, k=3)

        # Should have 4 3-mers: ACD, CDE, DEF, EFG
        assert len(kmers) == 4
        assert "ACD" in kmers
        assert "CDE" in kmers

    def test_kmer_short_sequence(self):
        """Test k-mer extraction from short sequence."""
        seq = "AC"
        kmers = kmer_encode_sequence(seq, k=3)

        # Too short for 3-mers
        assert len(kmers) == 0

    def test_kmer_different_k(self):
        """Test different k values."""
        seq = "ACDEFGH"

        kmers_2 = kmer_encode_sequence(seq, k=2)
        kmers_3 = kmer_encode_sequence(seq, k=3)

        # 2-mers: AC, CD, DE, EF, FG, GH = 6
        # 3-mers: ACD, CDE, DEF, EFG, FGH = 5
        assert len(kmers_2) == 6
        assert len(kmers_3) == 5


class TestFeatureSetCreation:
    """Tests for feature set creation."""

    @pytest.fixture
    def mock_adata(self):
        """Create mock AnnData object for testing."""
        n_obs = 100
        n_vars = 50

        # Create mock data
        X = np.random.rand(n_obs, n_vars)
        obs = {
            "response": ["Responder"] * 50 + ["Non-Responder"] * 50,
            "patient_id": [f"P{i//10}" for i in range(n_obs)],
            "n_genes_by_counts": np.random.randint(1000, 5000, n_obs),
            "total_counts": np.random.randint(10000, 50000, n_obs),
            "pct_counts_mt": np.random.uniform(0, 10, n_obs),
            "tra_molecular_weight": np.random.uniform(1000, 2000, n_obs),
            "tra_hydrophobicity": np.random.uniform(-1, 1, n_obs),
            "trb_molecular_weight": np.random.uniform(1000, 2000, n_obs),
            "trb_hydrophobicity": np.random.uniform(-1, 1, n_obs),
        }

        adata = AnnData(X=X, obs=obs)

        # Add mock encodings
        adata.obsm["X_gene_pca"] = np.random.rand(n_obs, 50)
        adata.obsm["X_gene_svd"] = np.random.rand(n_obs, 50)
        adata.obsm["X_tcr_tra_kmer"] = np.random.rand(n_obs, 200)
        adata.obsm["X_tcr_trb_kmer"] = np.random.rand(n_obs, 200)

        return adata

    def test_create_feature_sets_basic(self, mock_adata):
        """Test basic feature set creation."""
        supervised_mask = mock_adata.obs["response"].isin(["Responder", "Non-Responder"]).values

        feature_sets = create_feature_sets(
            mock_adata,
            supervised_mask,
            use_cache=False,
            cache_manager=None
        )

        # Check all feature sets are created
        assert "basic" in feature_sets
        assert "gene_enhanced" in feature_sets
        assert "tcr_enhanced" in feature_sets
        assert "comprehensive" in feature_sets

        # Check shapes
        n_supervised = supervised_mask.sum()
        assert feature_sets["basic"].shape[0] == n_supervised
        assert feature_sets["gene_enhanced"].shape[0] == n_supervised
        assert feature_sets["tcr_enhanced"].shape[0] == n_supervised
        assert feature_sets["comprehensive"].shape[0] == n_supervised

        # Check feature dimensions are reasonable
        assert feature_sets["basic"].shape[1] > 20  # At least 20+ features
        assert feature_sets["gene_enhanced"].shape[1] > 80  # ~100 features
        assert feature_sets["tcr_enhanced"].shape[1] > 400  # ~400+ features
        assert feature_sets["comprehensive"].shape[1] > 450  # ~450+ features

    def test_feature_sets_no_nans(self, mock_adata):
        """Test that feature sets don't contain NaNs."""
        supervised_mask = mock_adata.obs["response"].isin(["Responder", "Non-Responder"]).values

        feature_sets = create_feature_sets(
            mock_adata,
            supervised_mask,
            use_cache=False,
            cache_manager=None
        )

        for name, features in feature_sets.items():
            assert not np.isnan(features).any(), f"Feature set {name} contains NaNs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
