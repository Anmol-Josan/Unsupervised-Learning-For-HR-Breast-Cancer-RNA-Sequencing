"""
Clustering module for unsupervised analysis using Leiden and hierarchical clustering.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
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
    target_clusters: Optional[int] = None,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Perform Leiden clustering at multiple resolutions and optionally select best resolution.

    Args:
        adata: AnnData object with neighbor graph
        resolutions: List of resolution parameters (matching notebook: 26 resolutions)
        target_clusters: Target number of clusters for auto-selection (default: None, uses all resolutions)
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with Leiden clustering results in obs
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "leiden_clustering",
            {"resolutions": resolutions, "target_clusters": target_clusters, "shape": adata.shape},
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "clustering", format="h5ad")
        if cached_data is not None:
            print(f"Loaded Leiden clustering from cache")
            return cached_data

    print(f"Performing Leiden clustering at {len(resolutions)} resolutions...")

    best_res = resolutions[0] if resolutions else 0.1
    best_diff = float('inf')

    for resolution in resolutions:
        key = f'leiden_{resolution}'
        try:
            print(f"  Resolution {resolution}...")
            sc.tl.leiden(adata, resolution=resolution, key_added=key, random_state=42)
            n_clusters = len(adata.obs[key].unique())
            print(f"    {n_clusters} clusters")
            
            # Auto-select best resolution if target_clusters specified
            if target_clusters is not None:
                diff = abs(n_clusters - target_clusters)
                if diff < best_diff:
                    best_diff = diff
                    best_res = resolution
        except Exception as e:
            print(f"    Leiden failed for resolution {resolution}: {e}")
            # Fallback to louvain if leiden not installed
            try:
                sc.tl.louvain(adata, resolution=resolution, key_added=key, random_state=42)
                n_clusters = len(adata.obs[key].unique())
                print(f"    Resolution {resolution} (Louvain): {n_clusters} clusters")
                if target_clusters is not None:
                    diff = abs(n_clusters - target_clusters)
                    if diff < best_diff:
                        best_diff = diff
                        best_res = resolution
            except Exception:
                pass

    # Set best clustering if target_clusters was specified
    if target_clusters is not None:
        print(f"Selected resolution: {best_res} (closest to {target_clusters} clusters)")
        best_key = f'leiden_{best_res}'
        if best_key in adata.obs:
            adata.obs['leiden'] = adata.obs[best_key]
        else:
            print("Warning: Best resolution clustering not found. Using default.")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "clustering", format="h5ad")

    return adata


def kmeans_clustering_tcr(
    adata: AnnData,
    n_clusters: int = 6,  # Changed from 5 to 6 to match notebook
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
        # Scale before clustering (matching notebook)
        from sklearn.preprocessing import StandardScaler
        scaler_tra = StandardScaler()
        tra_kmer_scaled = scaler_tra.fit_transform(X_tra)
        kmeans_tra = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        adata.obs['tra_kmer_clusters'] = pd.Categorical(kmeans_tra.fit_predict(tra_kmer_scaled))
        print(f"  TRA clusters: {len(adata.obs['tra_kmer_clusters'].unique())}")

    # Cluster TRB k-mers
    if 'X_tcr_trb_kmer' in adata.obsm:
        X_trb = adata.obsm['X_tcr_trb_kmer']
        # Scale before clustering (matching notebook)
        from sklearn.preprocessing import StandardScaler
        scaler_trb = StandardScaler()
        trb_kmer_scaled = scaler_trb.fit_transform(X_trb)
        kmeans_trb = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        adata.obs['trb_kmer_clusters'] = pd.Categorical(kmeans_trb.fit_predict(trb_kmer_scaled))
        print(f"  TRB clusters: {len(adata.obs['trb_kmer_clusters'].unique())}")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "clustering", format="h5ad")

    return adata


def cluster_pipeline(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    leiden_resolutions: List[float] = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5],
    target_clusters: Optional[int] = 7,
    kmeans_n_clusters: int = 6,
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
        target_clusters=target_clusters,
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

    # Gene Expression Module Discovery (matching notebook cell 40)
    print("Discovering gene expression modules...")
    if 'X_gene_pca' in adata.obsm:
        gene_pca = adata.obsm['X_gene_pca']
    elif 'X_pca' in adata.obsm:
        gene_pca = adata.obsm['X_pca']
    else:
        gene_pca = None

    if gene_pca is not None:
        gene_kmeans = KMeans(n_clusters=8, random_state=42, n_init=20)
        gene_expression_modules = gene_kmeans.fit_predict(gene_pca)
        adata.obs['gene_expression_modules'] = pd.Categorical(gene_expression_modules)
        print(f"  Gene expression modules: {len(adata.obs['gene_expression_modules'].unique())} modules")

    print("=" * 60)
    print("Clustering complete")
    print("=" * 60)

    return adata
