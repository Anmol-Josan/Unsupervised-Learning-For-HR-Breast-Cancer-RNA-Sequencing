"""
Feature engineering module for gene expression and TCR sequence encoding.
Includes PCA, SVD, UMAP, k-mer encoding, and physicochemical features.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

from pipeline.utils import CacheManager, compute_data_hash


# Amino acid property tables (Kyte-Doolittle hydrophobicity, etc.)
HYDROPHOBICITY_KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
}

MOLECULAR_WEIGHT = {
    'A': 89.1, 'R': 174.2, 'N': 132.1, 'D': 133.1, 'C': 121.2,
    'Q': 146.2, 'E': 147.1, 'G': 75.1, 'H': 155.2, 'I': 131.2,
    'L': 131.2, 'K': 146.2, 'M': 149.2, 'F': 165.2, 'P': 115.1,
    'S': 105.1, 'T': 119.1, 'W': 204.2, 'Y': 181.2, 'V': 117.1
}

POLARITY = {
    'A': 8.1, 'R': 10.5, 'N': 11.6, 'D': 13.0, 'C': 5.5,
    'Q': 10.5, 'E': 12.3, 'G': 9.0, 'H': 10.4, 'I': 5.2,
    'L': 4.9, 'K': 11.3, 'M': 5.7, 'F': 5.2, 'P': 8.0,
    'S': 9.2, 'T': 8.6, 'W': 5.4, 'Y': 6.2, 'V': 5.9
}


def _clean_seq(seq: str) -> str:
    """
    Clean TCR sequence by removing invalid characters.

    Args:
        seq: TCR sequence string

    Returns:
        Cleaned sequence
    """
    if pd.isna(seq) or seq == '':
        return ''

    # Keep only standard amino acids
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    cleaned = ''.join([aa for aa in str(seq).upper() if aa in valid_aas])

    return cleaned


def physicochemical_features(sequence: str) -> Dict[str, float]:
    """
    Compute physicochemical properties of a TCR CDR3 sequence.

    Args:
        sequence: TCR CDR3 amino acid sequence

    Returns:
        Dictionary of physicochemical features
    """
    sequence = _clean_seq(sequence)

    if len(sequence) == 0:
        return {
            'length': 0,
            'molecular_weight': 0.0,
            'hydrophobicity': 0.0,
            'charge': 0.0,
            'polarity': 0.0
        }

    # Calculate features
    length = len(sequence)
    molecular_weight = sum(MOLECULAR_WEIGHT.get(aa, 0) for aa in sequence)
    hydrophobicity = sum(HYDROPHOBICITY_KD.get(aa, 0) for aa in sequence) / length
    charge = sum(CHARGE.get(aa, 0) for aa in sequence)
    polarity = sum(POLARITY.get(aa, 0) for aa in sequence) / length

    return {
        'length': length,
        'molecular_weight': molecular_weight,
        'hydrophobicity': hydrophobicity,
        'charge': charge,
        'polarity': polarity
    }


def one_hot_encode_sequence(
    sequence: str,
    max_length: int = 50,
    alphabet: str = 'ACDEFGHIKLMNPQRSTVWY'
) -> np.ndarray:
    """
    One-hot encode a TCR sequence.

    Args:
        sequence: TCR amino acid sequence
        max_length: Maximum sequence length
        alphabet: Amino acid alphabet

    Returns:
        One-hot encoded array of shape (max_length * len(alphabet),)
    """
    sequence = _clean_seq(sequence)

    # Pad or truncate sequence
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    else:
        sequence = sequence.ljust(max_length, ' ')

    # Create one-hot encoding
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
    encoding = np.zeros((max_length, len(alphabet)))

    for pos, aa in enumerate(sequence[:max_length]):
        if aa in aa_to_idx:
            encoding[pos, aa_to_idx[aa]] = 1

    # Flatten to 1D
    return encoding.flatten()


def kmer_encode_sequence(
    sequence: str,
    k: int = 3,
    alphabet: str = 'ACDEFGHIKLMNPQRSTVWY'
) -> List[str]:
    """
    Extract k-mers from a TCR sequence.

    Args:
        sequence: TCR amino acid sequence
        k: K-mer length
        alphabet: Amino acid alphabet

    Returns:
        List of k-mers
    """
    sequence = _clean_seq(sequence)

    if len(sequence) < k:
        return []

    # Extract k-mers
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    # Filter to valid k-mers only
    valid_aas = set(alphabet)
    kmers = [kmer for kmer in kmers if all(aa in valid_aas for aa in kmer)]

    return kmers


def encode_gene_expression_patterns(
    adata: AnnData,
    n_pca_components: int = 50,
    n_svd_components: int = 50,
    compute_umap: bool = False,
    n_umap_components: int = 20,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Encode gene expression using PCA, SVD, and optionally UMAP.

    Args:
        adata: Preprocessed AnnData object
        n_pca_components: Number of PCA components
        n_svd_components: Number of SVD components
        compute_umap: Whether to compute UMAP
        n_umap_components: Number of UMAP components
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with encodings added to obsm
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "encode_gene_expression",
            {
                "n_pca": n_pca_components,
                "n_svd": n_svd_components,
                "compute_umap": compute_umap,
                "n_umap": n_umap_components,
                "shape": adata.shape
            },
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "feature_engineering", format="h5ad")
        if cached_data is not None:
            print(f"Loaded gene expression encodings from cache")
            return cached_data

    print(f"Encoding gene expression patterns...")

    # Ensure we have scaled data
    if adata.X is None or issparse(adata.X):
        print("Scaling data for dimensionality reduction...")
        # Use highly variable genes if available
        if 'highly_variable' in adata.var:
            adata_hvg = adata[:, adata.var['highly_variable']].copy()
        else:
            adata_hvg = adata.copy()

        # Scale
        sc.pp.scale(adata_hvg, max_value=10)
        X_scaled = adata_hvg.X
    else:
        X_scaled = adata.X

    # Convert sparse to dense if needed
    if issparse(X_scaled):
        X_scaled = X_scaled.toarray()

    # PCA
    print(f"  Computing PCA ({n_pca_components} components)...")
    pca = PCA(n_components=min(n_pca_components, X_scaled.shape[1]), random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    adata.obsm['X_gene_pca'] = X_pca
    print(f"    Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # SVD (alternative dimensionality reduction)
    print(f"  Computing SVD ({n_svd_components} components)...")
    svd = TruncatedSVD(n_components=min(n_svd_components, X_scaled.shape[1] - 1), random_state=42)
    X_svd = svd.fit_transform(X_scaled)
    adata.obsm['X_gene_svd'] = X_svd
    print(f"    Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

    # UMAP (optional, expensive)
    if compute_umap:
        print(f"  Computing UMAP ({n_umap_components} components)...")
        import umap
        reducer = umap.UMAP(
            n_components=n_umap_components,
            random_state=42,
            n_neighbors=15,
            min_dist=0.1
        )
        X_umap = reducer.fit_transform(X_scaled)
        adata.obsm['X_gene_umap'] = X_umap

    print("Gene expression encoding complete")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "feature_engineering", format="h5ad")

    return adata


def encode_tcr_sequences(
    adata: AnnData,
    kmer_k: int = 3,
    n_svd_components: int = 200,
    max_onehot_length: int = 50,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Encode TCR sequences using k-mer, one-hot, and physicochemical features.

    Args:
        adata: AnnData object with TCR data in obs
        kmer_k: K-mer length
        n_svd_components: Number of SVD components for k-mer reduction
        max_onehot_length: Maximum sequence length for one-hot encoding
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        AnnData with TCR encodings added to obsm
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "encode_tcr_sequences",
            {
                "kmer_k": kmer_k,
                "n_svd": n_svd_components,
                "max_onehot_length": max_onehot_length,
                "shape": adata.shape
            },
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "feature_engineering", format="h5ad")
        if cached_data is not None:
            print(f"Loaded TCR encodings from cache")
            return cached_data

    print("Encoding TCR sequences...")

    # Ensure we have CDR3 sequences
    if 'cdr3_TRA' not in adata.obs or 'cdr3_TRB' not in adata.obs:
        print("Warning: No TCR sequences found, skipping TCR encoding")
        return adata

    # Clean sequences
    adata.obs['cdr3_TRA_clean'] = adata.obs['cdr3_TRA'].fillna('').apply(_clean_seq)
    adata.obs['cdr3_TRB_clean'] = adata.obs['cdr3_TRB'].fillna('').apply(_clean_seq)

    # K-mer encoding
    print(f"  K-mer encoding (k={kmer_k})...")
    _encode_tcr_kmers(adata, k=kmer_k, n_svd_components=n_svd_components)

    # One-hot encoding
    print(f"  One-hot encoding (max_length={max_onehot_length})...")
    _encode_tcr_onehot(adata, max_length=max_onehot_length)

    # Physicochemical features
    print(f"  Physicochemical features...")
    _encode_tcr_physicochemical(adata)

    print("TCR encoding complete")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "feature_engineering", format="h5ad")

    return adata


def _encode_tcr_kmers(adata: AnnData, k: int = 3, n_svd_components: int = 200) -> None:
    """Encode TCR sequences using k-mers with SVD reduction."""
    # Extract k-mers from all sequences
    tra_sequences = adata.obs['cdr3_TRA_clean'].tolist()
    trb_sequences = adata.obs['cdr3_TRB_clean'].tolist()

    # Convert sequences to k-mer strings
    tra_kmer_strings = [' '.join(kmer_encode_sequence(seq, k=k)) for seq in tra_sequences]
    trb_kmer_strings = [' '.join(kmer_encode_sequence(seq, k=k)) for seq in trb_sequences]

    # Vectorize k-mers using CountVectorizer
    vec_tra = CountVectorizer(analyzer='char', ngram_range=(k, k))
    vec_trb = CountVectorizer(analyzer='char', ngram_range=(k, k))

    try:
        X_kmer_tra = vec_tra.fit_transform(tra_kmer_strings)
        X_kmer_trb = vec_trb.fit_transform(trb_kmer_strings)

        # Reduce dimensionality using SVD
        n_comp_tra = min(n_svd_components, X_kmer_tra.shape[1])
        n_comp_trb = min(n_svd_components, X_kmer_trb.shape[1])

        if n_comp_tra > 0:
            svd_tra = TruncatedSVD(n_components=n_comp_tra, random_state=42)
            X_kmer_tra_reduced = svd_tra.fit_transform(X_kmer_tra)
            adata.obsm['X_tcr_tra_kmer'] = X_kmer_tra_reduced
            print(f"    TRA k-mers: {X_kmer_tra.shape[1]} → {n_comp_tra} dims")

        if n_comp_trb > 0:
            svd_trb = TruncatedSVD(n_components=n_comp_trb, random_state=42)
            X_kmer_trb_reduced = svd_trb.fit_transform(X_kmer_trb)
            adata.obsm['X_tcr_trb_kmer'] = X_kmer_trb_reduced
            print(f"    TRB k-mers: {X_kmer_trb.shape[1]} → {n_comp_trb} dims")

    except Exception as e:
        print(f"    Warning: K-mer encoding failed: {e}")
        # Add zeros as fallback
        adata.obsm['X_tcr_tra_kmer'] = np.zeros((adata.shape[0], n_svd_components))
        adata.obsm['X_tcr_trb_kmer'] = np.zeros((adata.shape[0], n_svd_components))


def _encode_tcr_onehot(adata: AnnData, max_length: int = 50) -> None:
    """Encode TCR sequences using one-hot encoding."""
    tra_sequences = adata.obs['cdr3_TRA_clean'].tolist()
    trb_sequences = adata.obs['cdr3_TRB_clean'].tolist()

    # One-hot encode all sequences
    X_onehot_tra = np.array([one_hot_encode_sequence(seq, max_length=max_length) for seq in tra_sequences])
    X_onehot_trb = np.array([one_hot_encode_sequence(seq, max_length=max_length) for seq in trb_sequences])

    adata.obsm['X_tcr_tra_onehot'] = X_onehot_tra
    adata.obsm['X_tcr_trb_onehot'] = X_onehot_trb

    print(f"    One-hot shapes: TRA {X_onehot_tra.shape}, TRB {X_onehot_trb.shape}")


def _encode_tcr_physicochemical(adata: AnnData) -> None:
    """Compute physicochemical features for TCR sequences."""
    tra_sequences = adata.obs['cdr3_TRA_clean'].tolist()
    trb_sequences = adata.obs['cdr3_TRB_clean'].tolist()

    # Compute features for TRA
    tra_features = [physicochemical_features(seq) for seq in tra_sequences]
    tra_df = pd.DataFrame(tra_features)
    tra_df.columns = ['tra_' + col for col in tra_df.columns]

    # Compute features for TRB
    trb_features = [physicochemical_features(seq) for seq in trb_sequences]
    trb_df = pd.DataFrame(trb_features)
    trb_df.columns = ['trb_' + col for col in trb_df.columns]

    # Add to obs
    for col in tra_df.columns:
        adata.obs[col] = tra_df[col].values

    for col in trb_df.columns:
        adata.obs[col] = trb_df[col].values

    # Also add combined array to obsm
    X_physico = np.column_stack([
        tra_df[['tra_molecular_weight', 'tra_hydrophobicity']].values,
        trb_df[['trb_molecular_weight', 'trb_hydrophobicity']].values
    ])
    adata.obsm['X_tcr_physico'] = X_physico

    print(f"    Physicochemical features: {X_physico.shape[1]} features")


def _get_obsm_or_zeros(adata: AnnData, key: str, mask: np.ndarray, n_cols: int) -> np.ndarray:
    """
    Get data from adata.obsm, or return zeros if not present.

    Args:
        adata: AnnData object
        key: Key in obsm
        mask: Boolean mask for rows
        n_cols: Number of columns to return

    Returns:
        Array from obsm or zeros
    """
    if key in adata.obsm:
        data = adata.obsm[key][mask]
        # Handle dimension mismatch
        if data.shape[1] < n_cols:
            # Pad with zeros
            padding = np.zeros((data.shape[0], n_cols - data.shape[1]))
            return np.column_stack([data, padding])
        else:
            return data[:, :n_cols]
    else:
        return np.zeros((np.sum(mask), n_cols))


def create_feature_sets(
    adata: AnnData,
    supervised_mask: np.ndarray,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> Dict[str, np.ndarray]:
    """
    Create multiple feature sets for modeling.

    Args:
        adata: AnnData object with all encodings
        supervised_mask: Boolean mask for supervised learning samples
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        Dictionary of feature sets
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "create_feature_sets",
            {"n_supervised": supervised_mask.sum(), "shape": adata.shape},
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "feature_engineering", format="pt")
        if cached_data is not None:
            print(f"Loaded feature sets from cache")
            return cached_data

    print(f"Creating feature sets (n_samples={supervised_mask.sum()})...")

    feature_sets = {}

    # Get gene features
    gene_pca = adata.obsm['X_gene_pca'][supervised_mask] if 'X_gene_pca' in adata.obsm else np.zeros((supervised_mask.sum(), 50))
    gene_svd = adata.obsm['X_gene_svd'][supervised_mask] if 'X_gene_svd' in adata.obsm else np.zeros((supervised_mask.sum(), 50))

    # Get TCR features
    tra_kmer = adata.obsm['X_tcr_tra_kmer'][supervised_mask] if 'X_tcr_tra_kmer' in adata.obsm else np.zeros((supervised_mask.sum(), 200))
    trb_kmer = adata.obsm['X_tcr_trb_kmer'][supervised_mask] if 'X_tcr_trb_kmer' in adata.obsm else np.zeros((supervised_mask.sum(), 200))

    # Get physicochemical features
    tcr_physico = np.column_stack([
        adata.obs[['tra_molecular_weight', 'tra_hydrophobicity']].fillna(0)[supervised_mask].values,
        adata.obs[['trb_molecular_weight', 'trb_hydrophobicity']].fillna(0)[supervised_mask].values
    ])

    # Get QC features
    qc_features = adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']].fillna(0)[supervised_mask].values

    # Feature Set 1: Basic (29 features)
    feature_sets['basic'] = np.column_stack([
        gene_pca[:, :20],  # Top 20 gene PCA
        tcr_physico,       # 4 physicochemical features
        qc_features        # 3 QC metrics
    ])
    print(f"  basic: {feature_sets['basic'].shape}")

    # Feature Set 2: Gene Enhanced (~100 features)
    feature_sets['gene_enhanced'] = np.column_stack([
        gene_pca,                                    # 50 PCA components
        gene_svd[:, :30],                            # 30 SVD components
        _get_obsm_or_zeros(adata, 'X_gene_umap', supervised_mask, 20),  # 20 UMAP components
    ])
    print(f"  gene_enhanced: {feature_sets['gene_enhanced'].shape}")

    # Feature Set 3: TCR Enhanced (~400 features)
    feature_sets['tcr_enhanced'] = np.column_stack([
        gene_pca[:, :15],  # 15 gene PCA
        tra_kmer,          # 200 TRA k-mers
        trb_kmer,          # 200 TRB k-mers
        tcr_physico,       # 4 physicochemical features
        qc_features        # 3 QC metrics
    ])
    print(f"  tcr_enhanced: {feature_sets['tcr_enhanced'].shape}")

    # Feature Set 4: Comprehensive (~450+ features)
    feature_sets['comprehensive'] = np.column_stack([
        gene_pca[:, :15],  # 15 gene PCA
        gene_svd[:, :25],  # 25 SVD components
        tra_kmer,          # 200 TRA k-mers
        trb_kmer,          # 200 TRB k-mers
        tcr_physico,       # 4 physicochemical features
        qc_features        # 3 QC metrics
    ])
    print(f"  comprehensive: {feature_sets['comprehensive'].shape}")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(feature_sets, cache_key, "feature_engineering", format="pt")

    return feature_sets
