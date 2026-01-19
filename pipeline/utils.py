"""
Utility functions for caching, configuration, and parallelization.
"""

import hashlib
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from anndata import AnnData


class CacheManager:
    """Manages caching of pipeline intermediate results."""

    def __init__(self, cache_dir: Union[str, Path], enabled: bool = True):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for storing cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_cache_key(
        self,
        function_name: str,
        params: Dict[str, Any],
        data_hash: Optional[str] = None,
        version: str = "1.0"
    ) -> str:
        """
        Generate deterministic cache key from function name, parameters, and data hash.

        Args:
            function_name: Name of the function being cached
            params: Dictionary of parameters used
            data_hash: Optional hash of input data
            version: Code version for cache invalidation

        Returns:
            Cache key string
        """
        # Create a deterministic representation of parameters
        param_str = json.dumps(params, sort_keys=True, default=str)

        # Combine all components
        key_components = [function_name, param_str, version]
        if data_hash:
            key_components.append(data_hash)

        # Hash the combined string
        key_string = "|".join(key_components)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()[:16]

        return f"{function_name}.{cache_key}"

    def _get_cache_path(self, cache_key: str, section: str, format: str = "pt") -> Path:
        """Get full path to cache file."""
        section_dir = self.cache_dir / section
        section_dir.mkdir(parents=True, exist_ok=True)
        return section_dir / f"{cache_key}.{format}"

    def save_cache(
        self,
        data: Any,
        cache_key: str,
        section: str,
        format: str = "auto",
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Save data to cache.

        Args:
            data: Data to cache (numpy array, DataFrame, AnnData, etc.)
            cache_key: Cache key identifier
            section: Cache section (e.g., 'data_loading', 'feature_engineering')
            format: Cache format ('auto', 'pt', 'h5ad', 'pkl')
            metadata: Optional metadata dictionary
        """
        if not self.enabled:
            return

        # Auto-detect format
        if format == "auto":
            if isinstance(data, AnnData):
                format = "h5ad"
            elif isinstance(data, (np.ndarray, pd.DataFrame, dict)):
                format = "pt"
            else:
                format = "pkl"

        cache_path = self._get_cache_path(cache_key, section, format)

        try:
            if format == "pt":
                # Use PyTorch for efficient numpy/tensor storage
                torch.save(data, cache_path)
            elif format == "h5ad":
                # Use AnnData native format
                data.write_h5ad(cache_path)
            elif format == "pkl":
                # Fallback to pickle
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            if metadata:
                metadata_path = cache_path.with_suffix('.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

        except Exception as e:
            warnings.warn(f"Failed to save cache {cache_key}: {e}")

    def load_cache(
        self,
        cache_key: str,
        section: str,
        format: str = "auto"
    ) -> Optional[Any]:
        """
        Load data from cache.

        Args:
            cache_key: Cache key identifier
            section: Cache section
            format: Cache format ('auto', 'pt', 'h5ad', 'pkl')

        Returns:
            Cached data or None if not found
        """
        if not self.enabled:
            return None

        # Try different formats if auto
        if format == "auto":
            formats_to_try = ["pt", "h5ad", "pkl"]
        else:
            formats_to_try = [format]

        for fmt in formats_to_try:
            cache_path = self._get_cache_path(cache_key, section, fmt)

            if cache_path.exists():
                try:
                    if fmt == "pt":
                        return torch.load(cache_path)
                    elif fmt == "h5ad":
                        from anndata import read_h5ad
                        return read_h5ad(cache_path)
                    elif fmt == "pkl":
                        with open(cache_path, 'rb') as f:
                            return pickle.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load cache {cache_key}: {e}")
                    continue

        return None

    def clear_section(self, section: str) -> None:
        """Clear all cache files in a section."""
        section_dir = self.cache_dir / section
        if section_dir.exists():
            import shutil
            shutil.rmtree(section_dir)
            print(f"Cleared cache section: {section}")

    def clear_all(self) -> None:
        """Clear entire cache directory."""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("Cleared all cache")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cache usage."""
        if not self.cache_dir.exists():
            return {"total_size_mb": 0, "sections": {}}

        info = {"sections": {}}
        total_size = 0

        for section_dir in self.cache_dir.iterdir():
            if section_dir.is_dir():
                section_size = sum(f.stat().st_size for f in section_dir.rglob('*') if f.is_file())
                info["sections"][section_dir.name] = {
                    "size_mb": section_size / (1024 * 1024),
                    "num_files": len(list(section_dir.rglob('*.*')))
                }
                total_size += section_size

        info["total_size_mb"] = total_size / (1024 * 1024)
        return info


class ConfigManager:
    """Manages pipeline configuration from YAML files."""

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated key path (e.g., 'pipeline.steps.modeling.enabled')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to config."""
        return self.config[key]


def compute_data_hash(data: Any) -> str:
    """
    Compute hash of data for cache invalidation.

    Args:
        data: Data to hash (numpy array, DataFrame, etc.)

    Returns:
        Hash string
    """
    if isinstance(data, np.ndarray):
        # Hash array shape and first/last elements
        hash_input = f"{data.shape}|{data.flat[0] if data.size > 0 else ''}|{data.flat[-1] if data.size > 0 else ''}"
    elif isinstance(data, pd.DataFrame):
        # Hash DataFrame shape and columns
        hash_input = f"{data.shape}|{list(data.columns)}"
    elif isinstance(data, AnnData):
        # Hash AnnData shape
        hash_input = f"{data.shape}|{data.n_obs}|{data.n_vars}"
    elif isinstance(data, (str, Path)):
        # Hash file path and modification time
        path = Path(data)
        if path.exists():
            hash_input = f"{path}|{path.stat().st_mtime}"
        else:
            hash_input = str(path)
    else:
        hash_input = str(data)

    return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def detect_environment() -> Dict[str, Any]:
    """
    Detect execution environment (Kaggle, local, etc.).

    Returns:
        Dictionary with environment info
    """
    import os

    is_kaggle = os.path.exists('/kaggle/input') or os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

    return {
        'is_kaggle': is_kaggle,
        'platform': 'kaggle' if is_kaggle else 'local',
        'data_dir': '/kaggle/working/Data' if is_kaggle else './Data',
        'output_dir': '/kaggle/working/Output' if is_kaggle else './Output'
    }


def setup_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    import os

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # Set deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_parallel_strategy(n_vms: int = 1, n_local_cores: int = None) -> str:
    """
    Auto-select best parallelization strategy.

    Args:
        n_vms: Number of available VM instances
        n_local_cores: Number of local CPU cores (auto-detected if None)

    Returns:
        Parallelization strategy: 'ray', 'multiprocessing', or 'sequential'
    """
    import os

    if n_local_cores is None:
        n_local_cores = os.cpu_count() or 1

    if n_vms > 1:
        return "ray"
    elif n_local_cores >= 4:
        return "multiprocessing"
    else:
        return "sequential"
