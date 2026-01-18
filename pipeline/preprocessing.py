"""
Preprocessing module for normalization, QC filtering, and feature selection.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData

from pipeline.utils import CacheManager, compute_data_hash


def calculate_qc_metrics(
    adata: AnnData,
    mt_pattern: str = '^MT-'
) -> AnnData:
    """
    Calculate QC metrics for cells and genes.

    Args:
        adata: AnnData object
        mt_pattern: Pattern to identify mitochondrial genes

    Returns:
        AnnData with QC metrics added to obs/var
    """
    print("Calculating QC metrics...")

    # Mark mitochondrial genes
    adata.var['mt'] = adata.var_names.str.startswith(mt_pattern)

    # Calculate QC metrics
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=['mt'],
        percent_top=None,
        log1p=False,
        inplace=True
    )

    print(f"QC metrics calculated:")
    print(f"  Median genes per cell: {np.median(adata.obs['n_genes_by_counts']):.0f}")
    print(f"  Median UMIs per cell: {np.median(adata.obs['total_counts']):.0f}")
    print(f"  Median MT%: {np.median(adata.obs['pct_counts_mt']):.2f}%")

    return adata


def filter_cells(
    adata: AnnData,
    min_genes: int = 200,
    min_counts: int = 500,
    max_pct_mt: float = 20.0
) -> AnnData:
    """
    Filter cells based on QC metrics.

    Args:
        adata: AnnData object with QC metrics
        min_genes: Minimum number of genes expressed per cell
        min_counts: Minimum total counts per cell
        max_pct_mt: Maximum mitochondrial percentage

    Returns:
        Filtered AnnData object
    """
    n_cells_before = adata.shape[0]

    print(f"Filtering cells (before: {n_cells_before} cells)...")

    # Filter cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)

    # Filter by mitochondrial percentage
    adata = adata[adata.obs['pct_counts_mt'] < max_pct_mt, :].copy()

    n_cells_after = adata.shape[0]
    n_cells_removed = n_cells_before - n_cells_after

    print(f"Filtering complete:")
    print(f"  Cells removed: {n_cells_removed} ({n_cells_removed/n_cells_before*100:.1f}%)")
    print(f"  Cells remaining: {n_cells_after}")

    return adata


def filter_genes(
    adata: AnnData,
    min_cells: int = 3
) -> AnnData:
    """
    Filter genes based on minimum cells expressing.

    Args:
        adata: AnnData object
        min_cells: Minimum number of cells expressing gene

    Returns:
        Filtered AnnData object
    """
    n_genes_before = adata.shape[1]

    print(f"Filtering genes (before: {n_genes_before} genes)...")

    # Filter genes
    sc.pp.filter_genes(adata, min_cells=min_cells)

    n_genes_after = adata.shape[1]
    n_genes_removed = n_genes_before - n_genes_after

    print(f"Filtering complete:")
    print(f"  Genes removed: {n_genes_removed} ({n_genes_removed/n_genes_before*100:.1f}%)")
    print(f"  Genes remaining: {n_genes_after}")

    return adata


def normalize_data(
    adata: AnnData,
    target_sum: float = 1e4,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Normalize gene expression data.

    Args:
        adata: AnnData object
        target_sum: Target sum for normalization (default: 10,000)
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        Normalized AnnData object
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "normalize_data",
            {"target_sum": target_sum, "shape": adata.shape},
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "preprocessing", format="h5ad")
        if cached_data is not None:
            print(f"Loaded normalized data from cache")
            return cached_data

    print(f"Normalizing data (target_sum={target_sum})...")

    # Store raw counts
    adata.raw = adata.copy()

    # Normalize to target sum
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Log transform
    sc.pp.log1p(adata)

    print("Normalization complete")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "preprocessing", format="h5ad")

    return adata


def select_highly_variable_genes(
    adata: AnnData,
    n_top_genes: int = 2000,
    flavor: str = 'seurat',
    min_mean: Optional[float] = None,
    max_mean: Optional[float] = None,
    min_disp: Optional[float] = None,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Select highly variable genes.

    Args:
        adata: Normalized AnnData object
        n_top_genes: Number of highly variable genes to select (used if flavor='seurat')
        flavor: Method to use ('seurat', 'cell_ranger', 'seurat_v3')
        min_mean: Minimum mean expression (matching notebook Cell 40)
        max_mean: Maximum mean expression (matching notebook Cell 40)
        min_disp: Minimum dispersion (matching notebook Cell 40)
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with HVG selection in var
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "select_hvg",
            {
                "n_top_genes": n_top_genes, 
                "flavor": flavor,
                "min_mean": min_mean,
                "max_mean": max_mean,
                "min_disp": min_disp,
                "shape": adata.shape
            },
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "preprocessing", format="h5ad")
        if cached_data is not None:
            print(f"Loaded HVG selection from cache")
            return cached_data

    # Use notebook parameters if provided, otherwise use defaults
    if min_mean is not None or max_mean is not None or min_disp is not None:
        print(f"Selecting highly variable genes with custom parameters (min_mean={min_mean}, max_mean={max_mean}, min_disp={min_disp})...")
        sc.pp.highly_variable_genes(
            adata,
            min_mean=min_mean if min_mean is not None else 0.0125,
            max_mean=max_mean if max_mean is not None else 3,
            min_disp=min_disp if min_disp is not None else 0.5,
            subset=False  # Don't subset yet, keep all genes
        )
    else:
        print(f"Selecting {n_top_genes} highly variable genes (flavor={flavor})...")
        # Identify highly variable genes
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=n_top_genes,
            flavor=flavor,
            subset=False  # Don't subset yet, keep all genes
        )

    n_hvg = adata.var['highly_variable'].sum()
    print(f"Selected {n_hvg} highly variable genes")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "preprocessing", format="h5ad")

    return adata


def scale_data(
    adata: AnnData,
    max_value: float = 10.0,
    zero_center: bool = True,
    use_hvg_only: bool = True
) -> AnnData:
    """
    Scale gene expression data.

    Args:
        adata: Normalized AnnData object
        max_value: Maximum value after scaling (clipping)
        zero_center: Whether to center data to zero mean
        use_hvg_only: Whether to scale only highly variable genes

    Returns:
        AnnData with scaled data
    """
    print("Scaling data...")

    if use_hvg_only and 'highly_variable' in adata.var:
        # Scale only HVGs
        adata_subset = adata[:, adata.var['highly_variable']].copy()
        sc.pp.scale(adata_subset, max_value=max_value, zero_center=zero_center)
        adata.X = adata_subset.X
        print(f"Scaled {adata.shape[1]} highly variable genes")
    else:
        # Scale all genes
        sc.pp.scale(adata, max_value=max_value, zero_center=zero_center)
        print(f"Scaled {adata.shape[1]} genes")

    return adata


def preprocess_pipeline(
    adata: AnnData,
    min_genes: int = 200,
    min_counts: int = 500,
    max_pct_mt: float = 20.0,
    min_cells: int = 3,
    target_sum: float = 1e4,
    n_top_genes: int = 2000,
    hvg_flavor: str = 'seurat',
    hvg_min_mean: Optional[float] = None,
    hvg_max_mean: Optional[float] = None,
    hvg_min_disp: Optional[float] = None,
    scale_max_value: float = 10.0,
    filter_tcr_cells: bool = True,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Run complete preprocessing pipeline.

    Args:
        adata: Raw AnnData object
        min_genes: Minimum genes per cell
        min_counts: Minimum counts per cell
        max_pct_mt: Maximum mitochondrial percentage
        min_cells: Minimum cells expressing gene
        target_sum: Normalization target sum
        n_top_genes: Number of HVGs to select
        hvg_flavor: HVG selection method
        hvg_min_mean: Minimum mean expression for HVG (matching notebook Cell 40)
        hvg_max_mean: Maximum mean expression for HVG (matching notebook Cell 40)
        hvg_min_disp: Minimum dispersion for HVG (matching notebook Cell 40)
        scale_max_value: Maximum value after scaling
        filter_tcr_cells: Filter cells without high-confidence TCR data
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        Preprocessed AnnData object
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "preprocess_pipeline",
            {
                "min_genes": min_genes,
                "min_counts": min_counts,
                "max_pct_mt": max_pct_mt,
                "min_cells": min_cells,
                "target_sum": target_sum,
                "n_top_genes": n_top_genes,
                "hvg_flavor": hvg_flavor,
                "hvg_min_mean": hvg_min_mean,
                "hvg_max_mean": hvg_max_mean,
                "hvg_min_disp": hvg_min_disp,
                "scale_max_value": scale_max_value,
                "filter_tcr_cells": filter_tcr_cells
            },
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "preprocessing", format="h5ad")
        if cached_data is not None:
            print(f"Loaded preprocessed data from cache")
            return cached_data

    print("=" * 60)
    print("Running preprocessing pipeline")
    print("=" * 60)

    # Calculate QC metrics
    adata = calculate_qc_metrics(adata)

    # Filter cells and genes
    adata = filter_cells(adata, min_genes=min_genes, min_counts=min_counts, max_pct_mt=max_pct_mt)
    adata = filter_genes(adata, min_cells=min_cells)

    # Filter for cells with high-confidence TCR data (matching notebook)
    if filter_tcr_cells and 'v_gene_TRA' in adata.obs.columns:
        initial_cells = adata.n_obs
        adata = adata[~adata.obs['v_gene_TRA'].isna()].copy()
        print(f"Filtered from {initial_cells} to {adata.n_obs} cells based on having high-confidence TCR data.")

    # Normalize
    adata = normalize_data(adata, target_sum=target_sum, use_cache=False, cache_manager=None)

    # Select HVGs
    adata = select_highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor=hvg_flavor,
        min_mean=hvg_min_mean,
        max_mean=hvg_max_mean,
        min_disp=hvg_min_disp,
        use_cache=False,
        cache_manager=None
    )

    # Scale data (matching notebook Cell 40)
    adata = scale_data(adata, max_value=scale_max_value, use_hvg_only=True)

    print("=" * 60)
    print("Preprocessing complete")
    print(f"Final shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print("=" * 60)

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "preprocessing", format="h5ad")

    return adata
