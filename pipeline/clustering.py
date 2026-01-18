"""
Clustering module for unsupervised analysis using Leiden and hierarchical clustering.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import scanpy as sc
from anndata import AnnData
from sklearn.cluster import KMeans

from pipeline.utils import CacheManager, compute_data_hash


def compute_neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Compute nearest neighbor graph.

    Args:
        adata: AnnData object
        n_neighbors: Number of neighbors
        n_pcs: Number of PCs to use
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with neighbor graph computed
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "compute_neighbors",
            {"n_neighbors": n_neighbors, "n_pcs": n_pcs, "shape": adata.shape},
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "clustering", format="h5ad")
        if cached_data is not None:
            print(f"Loaded neighbor graph from cache")
            return cached_data

    print(f"Computing neighbor graph (n_neighbors={n_neighbors}, n_pcs={n_pcs})...")

    # Compute neighbors
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "clustering", format="h5ad")

    return adata


def leiden_clustering(
    adata: AnnData,
    resolutions: List[float] = [0.1, 0.2, 0.5, 1.0],
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Perform Leiden clustering at multiple resolutions.

    Args:
        adata: AnnData object with neighbor graph
        resolutions: List of resolution parameters
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with Leiden clustering results in obs
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "leiden_clustering",
            {"resolutions": resolutions, "shape": adata.shape},
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "clustering", format="h5ad")
        if cached_data is not None:
            print(f"Loaded Leiden clustering from cache")
            return cached_data

    print(f"Performing Leiden clustering at {len(resolutions)} resolutions...")

    for resolution in resolutions:
        key = f'leiden_{resolution}'
        print(f"  Resolution {resolution}...")
        sc.tl.leiden(adata, resolution=resolution, key_added=key)
        n_clusters = len(adata.obs[key].unique())
        print(f"    {n_clusters} clusters")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "clustering", format="h5ad")

    return adata


def kmeans_clustering_tcr(
    adata: AnnData,
    n_clusters: int = 5,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Perform K-means clustering on TCR k-mer features.

    Args:
        adata: AnnData object with TCR k-mer features
        n_clusters: Number of clusters
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with K-means cluster labels in obs
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "kmeans_clustering_tcr",
            {"n_clusters": n_clusters, "shape": adata.shape},
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "clustering", format="h5ad")
        if cached_data is not None:
            print(f"Loaded K-means TCR clustering from cache")
            return cached_data

    print(f"Performing K-means clustering on TCR features (n_clusters={n_clusters})...")

    # Cluster TRA k-mers
    if 'X_tcr_tra_kmer' in adata.obsm:
        X_tra = adata.obsm['X_tcr_tra_kmer']
        kmeans_tra = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        adata.obs['tra_kmer_clusters'] = kmeans_tra.fit_predict(X_tra).astype(str)
        print(f"  TRA clusters: {len(adata.obs['tra_kmer_clusters'].unique())}")

    # Cluster TRB k-mers
    if 'X_tcr_trb_kmer' in adata.obsm:
        X_trb = adata.obsm['X_tcr_trb_kmer']
        kmeans_trb = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        adata.obs['trb_kmer_clusters'] = kmeans_trb.fit_predict(X_trb).astype(str)
        print(f"  TRB clusters: {len(adata.obs['trb_kmer_clusters'].unique())}")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "clustering", format="h5ad")

    return adata


def cluster_pipeline(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    leiden_resolutions: List[float] = [0.1, 0.2, 0.5, 1.0],
    kmeans_n_clusters: int = 5,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Run complete clustering pipeline.

    Args:
        adata: Preprocessed AnnData object
        n_neighbors: Number of neighbors for graph
        n_pcs: Number of PCs to use
        leiden_resolutions: Resolutions for Leiden clustering
        kmeans_n_clusters: Number of clusters for K-means
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with clustering results
    """
    print("=" * 60)
    print("Running clustering pipeline")
    print("=" * 60)

    # Compute neighbors
    adata = compute_neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_cache=use_cache,
        cache_manager=cache_manager
    )

    # Leiden clustering
    adata = leiden_clustering(
        adata,
        resolutions=leiden_resolutions,
        use_cache=use_cache,
        cache_manager=cache_manager
    )

    # K-means clustering on TCR features
    adata = kmeans_clustering_tcr(
        adata,
        n_clusters=kmeans_n_clusters,
        use_cache=use_cache,
        cache_manager=cache_manager
    )

    print("=" * 60)
    print("Clustering complete")
    print("=" * 60)

    return adata
