"""
Unit tests for pipeline.utils module.
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from pipeline.utils import (
    CacheManager,
    ConfigManager,
    compute_data_hash,
    detect_environment,
    get_parallel_strategy,
    setup_random_seeds,
)


class TestCacheManager:
    """Tests for CacheManager class."""

    def test_cache_manager_init(self):
        """Test cache manager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir, enabled=True)
            assert cache_manager.cache_dir == Path(tmpdir)
            assert cache_manager.enabled is True
            assert cache_manager.cache_dir.exists()

    def test_generate_cache_key(self):
        """Test cache key generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir)

            key1 = cache_manager.generate_cache_key(
                "test_func",
                {"param1": 1, "param2": "test"},
                data_hash="abc123"
            )
            key2 = cache_manager.generate_cache_key(
                "test_func",
                {"param1": 1, "param2": "test"},
                data_hash="abc123"
            )

            # Same inputs should produce same key
            assert key1 == key2

            # Different params should produce different key
            key3 = cache_manager.generate_cache_key(
                "test_func",
                {"param1": 2, "param2": "test"},
                data_hash="abc123"
            )
            assert key1 != key3

    def test_save_load_cache_numpy(self):
        """Test saving and loading numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir)

            # Create test data
            data = np.random.rand(10, 5)

            # Save
            cache_manager.save_cache(data, "test_key", "test_section", format="pt")

            # Load
            loaded_data = cache_manager.load_cache("test_key", "test_section", format="pt")

            assert loaded_data is not None
            np.testing.assert_array_equal(data, loaded_data)

    def test_save_load_cache_dataframe(self):
        """Test saving and loading pandas DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir)

            # Create test data
            data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

            # Save
            cache_manager.save_cache(data, "test_key", "test_section", format="pt")

            # Load
            loaded_data = cache_manager.load_cache("test_key", "test_section", format="pt")

            assert loaded_data is not None
            pd.testing.assert_frame_equal(data, loaded_data)

    def test_clear_section(self):
        """Test clearing cache section."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir)

            # Save some data
            data = np.array([1, 2, 3])
            cache_manager.save_cache(data, "test_key", "test_section")

            # Verify it exists
            loaded = cache_manager.load_cache("test_key", "test_section")
            assert loaded is not None

            # Clear section
            cache_manager.clear_section("test_section")

            # Verify it's gone
            loaded = cache_manager.load_cache("test_key", "test_section")
            assert loaded is None

    def test_cache_disabled(self):
        """Test that caching can be disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = CacheManager(tmpdir, enabled=False)

            data = np.array([1, 2, 3])
            cache_manager.save_cache(data, "test_key", "test_section")

            # Should return None when disabled
            loaded = cache_manager.load_cache("test_key", "test_section")
            assert loaded is None


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_config_manager_load(self):
        """Test loading configuration from YAML."""
        # Use the actual config file
        config_path = Path("config/pipeline_config.yaml")
        if config_path.exists():
            config_manager = ConfigManager(config_path)
            assert config_manager.config is not None
            assert "pipeline" in config_manager.config

    def test_config_get_nested(self):
        """Test getting nested configuration values."""
        config_path = Path("config/pipeline_config.yaml")
        if config_path.exists():
            config_manager = ConfigManager(config_path)

            # Test nested access
            value = config_manager.get("pipeline.random_seed")
            assert value == 42

            # Test default value
            value = config_manager.get("nonexistent.key", default="default_value")
            assert value == "default_value"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_data_hash_numpy(self):
        """Test hash computation for numpy arrays."""
        data1 = np.array([[1, 2, 3], [4, 5, 6]])
        data2 = np.array([[1, 2, 3], [4, 5, 6]])
        data3 = np.array([[1, 2, 3], [4, 5, 7]])

        hash1 = compute_data_hash(data1)
        hash2 = compute_data_hash(data2)
        hash3 = compute_data_hash(data3)

        # Same data should produce same hash
        assert hash1 == hash2
        # Different data should produce different hash
        assert hash1 != hash3

    def test_compute_data_hash_dataframe(self):
        """Test hash computation for DataFrames."""
        df1 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        hash1 = compute_data_hash(df1)
        hash2 = compute_data_hash(df2)

        assert hash1 == hash2

    def test_detect_environment(self):
        """Test environment detection."""
        env = detect_environment()

        assert "is_kaggle" in env
        assert "platform" in env
        assert "data_dir" in env
        assert "output_dir" in env
        assert isinstance(env["is_kaggle"], bool)

    def test_setup_random_seeds(self):
        """Test random seed setup."""
        setup_random_seeds(42)

        # Generate some random numbers
        np_random1 = np.random.rand(5)

        setup_random_seeds(42)
        np_random2 = np.random.rand(5)

        # Should be identical after resetting seed
        np.testing.assert_array_equal(np_random1, np_random2)

    def test_get_parallel_strategy(self):
        """Test parallel strategy selection."""
        # Single VM, 8 cores
        strategy = get_parallel_strategy(n_vms=1, n_local_cores=8)
        assert strategy == "multiprocessing"

        # Multiple VMs
        strategy = get_parallel_strategy(n_vms=4, n_local_cores=8)
        assert strategy == "ray"

        # Few cores
        strategy = get_parallel_strategy(n_vms=1, n_local_cores=2)
        assert strategy == "sequential"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
