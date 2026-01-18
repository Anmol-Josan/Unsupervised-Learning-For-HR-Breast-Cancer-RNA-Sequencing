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
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,  # H is ~10% protonated at pH 7
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

# Volume (Å³) - Zamyatnin, 1972
VOLUME = {
    'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
    'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
    'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
    'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
}

# Flexibility Index (Bhaskaran-Ponnuswamy, 1988)
FLEXIBILITY = {
    'A': 0.360, 'R': 0.530, 'N': 0.460, 'D': 0.510, 'C': 0.350,
    'Q': 0.490, 'E': 0.500, 'G': 0.540, 'H': 0.320, 'I': 0.460,
    'L': 0.370, 'K': 0.470, 'M': 0.300, 'F': 0.310, 'P': 0.510,
    'S': 0.510, 'T': 0.440, 'W': 0.310, 'Y': 0.420, 'V': 0.390
}

# Beta-sheet propensity (Chou-Fasman)
BETA_SHEET = {
    'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
    'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
    'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
    'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
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
    Compute comprehensive physicochemical properties of a TCR CDR3 sequence.
    Returns 26 features matching the notebook implementation.

    Args:
        sequence: TCR CDR3 amino acid sequence

    Returns:
        Dictionary of 26 physicochemical features
    """
    sequence = _clean_seq(sequence)

    if len(sequence) == 0:
        return {
            'hydro_mean': 0.0, 'hydro_sum': 0.0, 'hydro_min': 0.0, 'hydro_max': 0.0,
            'hydro_range': 0.0, 'hydro_std': 0.0,
            'net_charge': 0.0, 'positive_aa_count': 0.0, 'negative_aa_count': 0.0, 'charge_ratio': 0.0,
            'polarity_mean': 0.0, 'polarity_std': 0.0,
            'length': 0, 'total_mw': 0.0, 'mean_mw': 0.0, 'mean_volume': 0.0, 'total_volume': 0.0,
            'flexibility_mean': 0.0, 'flexibility_max': 0.0,
            'beta_propensity_mean': 0.0,
            'nterm_hydro': 0.0, 'cterm_hydro': 0.0, 'middle_hydro': 0.0,
            'nterm_charge': 0.0, 'cterm_charge': 0.0, 'middle_charge': 0.0
        }

    # === Hydrophobicity Features ===
    hydro_values = [HYDROPHOBICITY_KD.get(aa, 0) for aa in sequence]
    
    # === Charge Features ===
    charge_values = [CHARGE.get(aa, 0) for aa in sequence]
    positive_count = sum(1 for c in charge_values if c > 0)
    negative_count = sum(1 for c in charge_values if c < 0)
    
    # === Polarity Features ===
    polarity_values = [POLARITY.get(aa, 0) for aa in sequence]
    
    # === Size Features ===
    mw_values = [MOLECULAR_WEIGHT.get(aa, 0) for aa in sequence]
    volume_values = [VOLUME.get(aa, 0) for aa in sequence]
    
    # === Flexibility Features ===
    flex_values = [FLEXIBILITY.get(aa, 0) for aa in sequence]
    
    # === Beta-sheet Propensity ===
    beta_values = [BETA_SHEET.get(aa, 0) for aa in sequence]
    
    # === Positional Features (N-term, C-term, Middle) ===
    n_term = sequence[:3] if len(sequence) >= 3 else sequence
    c_term = sequence[-3:] if len(sequence) >= 3 else sequence
    middle = sequence[3:-3] if len(sequence) > 6 else sequence

    return {
        # Hydrophobicity (6 features)
        'hydro_mean': np.mean(hydro_values),
        'hydro_sum': np.sum(hydro_values),
        'hydro_min': np.min(hydro_values),
        'hydro_max': np.max(hydro_values),
        'hydro_range': np.max(hydro_values) - np.min(hydro_values),
        'hydro_std': np.std(hydro_values) if len(hydro_values) > 1 else 0.0,
        
        # Charge (4 features)
        'net_charge': np.sum(charge_values),
        'positive_aa_count': float(positive_count),
        'negative_aa_count': float(negative_count),
        'charge_ratio': positive_count / (negative_count + 1) if negative_count >= 0 else 0.0,
        
        # Polarity (2 features)
        'polarity_mean': np.mean(polarity_values),
        'polarity_std': np.std(polarity_values) if len(polarity_values) > 1 else 0.0,
        
        # Size (5 features)
        'length': len(sequence),
        'total_mw': np.sum(mw_values),
        'mean_mw': np.mean(mw_values),
        'mean_volume': np.mean(volume_values),
        'total_volume': np.sum(volume_values),
        
        # Flexibility (2 features)
        'flexibility_mean': np.mean(flex_values),
        'flexibility_max': np.max(flex_values),
        
        # Beta-sheet (1 feature)
        'beta_propensity_mean': np.mean(beta_values),
        
        # Positional features (6 features)
        'nterm_hydro': np.mean([HYDROPHOBICITY_KD.get(aa, 0) for aa in n_term]),
        'cterm_hydro': np.mean([HYDROPHOBICITY_KD.get(aa, 0) for aa in c_term]),
        'middle_hydro': np.mean([HYDROPHOBICITY_KD.get(aa, 0) for aa in middle]) if middle else 0.0,
        'nterm_charge': np.sum([CHARGE.get(aa, 0) for aa in n_term]),
        'cterm_charge': np.sum([CHARGE.get(aa, 0) for aa in c_term]),
        'middle_charge': np.sum([CHARGE.get(aa, 0) for aa in middle]) if middle else 0.0
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
    max_onehot_length: int = 20,  # Reduced from 50 to match notebook
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
    """Encode TCR sequences using k-mers with CountVectorizer and SVD reduction (matching notebook)."""
    # Extract and clean sequences
    tra_sequences = adata.obs['cdr3_TRA_clean'].tolist()
    trb_sequences = adata.obs['cdr3_TRB_clean'].tolist()

    # Vectorized k-mer encoding using CountVectorizer (sparse) - matching notebook cell 33
    vec_tra = CountVectorizer(analyzer='char', ngram_range=(k, k))
    vec_trb = CountVectorizer(analyzer='char', ngram_range=(k, k))
    
    try:
        tra_kmer_sparse = vec_tra.fit_transform(tra_sequences)
        trb_kmer_sparse = vec_trb.fit_transform(trb_sequences)
        
        print(f"    TRA k-mer sparse shape: {tra_kmer_sparse.shape}")
        print(f"    TRB k-mer sparse shape: {trb_kmer_sparse.shape}")

        # Reduce k-mer sparse matrices with TruncatedSVD to dense representation
        def _reduce_sparse(sparse_mat, n_components):
            n_comp = min(n_components, max(1, sparse_mat.shape[1] - 1))
            try:
                svd = TruncatedSVD(n_components=n_comp, random_state=42)
                return svd.fit_transform(sparse_mat)
            except Exception:
                # Fallback to dense (small datasets)
                return sparse_mat.toarray() if hasattr(sparse_mat, 'toarray') else np.asarray(sparse_mat)

        tra_kmer_matrix = _reduce_sparse(tra_kmer_sparse, n_components=n_svd_components)
        trb_kmer_matrix = _reduce_sparse(trb_kmer_sparse, n_components=n_svd_components)
        
        print(f"    TRA k-mer reduced shape: {tra_kmer_matrix.shape}")
        print(f"    TRB k-mer reduced shape: {trb_kmer_matrix.shape}")

        adata.obsm['X_tcr_tra_kmer'] = tra_kmer_matrix
        adata.obsm['X_tcr_trb_kmer'] = trb_kmer_matrix

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

    # Also add combined array to obsm (using key features matching notebook)
    # Notebook uses: tra_length, tra_total_mw, tra_hydro_mean, trb_length, trb_total_mw, trb_hydro_mean
    physico_cols_tra = ['tra_length', 'tra_total_mw', 'tra_hydro_mean']
    physico_cols_trb = ['trb_length', 'trb_total_mw', 'trb_hydro_mean']
    
    # Select available columns
    available_tra = [col for col in physico_cols_tra if col in tra_df.columns]
    available_trb = [col for col in physico_cols_trb if col in trb_df.columns]
    
    if available_tra and available_trb:
        X_physico = np.column_stack([
            tra_df[available_tra].values,
            trb_df[available_trb].values
        ])
    else:
        # Fallback: use length if other columns don't exist
        X_physico = np.column_stack([
            tra_df[['tra_length']].values if 'tra_length' in tra_df.columns else np.zeros((len(tra_df), 1)),
            trb_df[['trb_length']].values if 'trb_length' in trb_df.columns else np.zeros((len(trb_df), 1))
        ])
    
    adata.obsm['X_tcr_physico'] = X_physico

    print(f"    Physicochemical features: {len(tra_df.columns) + len(trb_df.columns)} total features ({len(tra_df.columns)} per chain)")


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

    # Get TCR k-mer features (full)
    tra_kmer_supervised = adata.obsm['X_tcr_tra_kmer'][supervised_mask] if 'X_tcr_tra_kmer' in adata.obsm else np.zeros((supervised_mask.sum(), 200))
    trb_kmer_supervised = adata.obsm['X_tcr_trb_kmer'][supervised_mask] if 'X_tcr_trb_kmer' in adata.obsm else np.zeros((supervised_mask.sum(), 200))

    # Reduce k-mer features by variance selection (matching notebook cell 43)
    def select_top_variance_features(X, n_features=200):
        """Select features with highest variance"""
        if X.shape[1] <= n_features:
            return X, np.arange(X.shape[1])
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-n_features:]
        return X[:, top_indices], top_indices

    print("  Reducing k-mer features by variance selection...")
    tra_kmer_reduced, tra_top_idx = select_top_variance_features(tra_kmer_supervised, n_features=200)
    trb_kmer_reduced, trb_top_idx = select_top_variance_features(trb_kmer_supervised, n_features=200)
    print(f"    TRA k-mers: {tra_kmer_supervised.shape[1]} → {tra_kmer_reduced.shape[1]}")
    print(f"    TRB k-mers: {trb_kmer_supervised.shape[1]} → {trb_kmer_reduced.shape[1]}")

    # Get physicochemical features (matching notebook: tra_length, tra_molecular_weight, tra_hydrophobicity, etc.)
    # Notebook uses: tra_length, tra_total_mw, tra_hydro_mean, trb_length, trb_total_mw, trb_hydro_mean
    physico_cols_tra = ['tra_length', 'tra_total_mw', 'tra_hydro_mean']
    physico_cols_trb = ['trb_length', 'trb_total_mw', 'trb_hydro_mean']
    
    # Use available columns, fallback to defaults if new columns don't exist
    available_tra = [col for col in physico_cols_tra if col in adata.obs.columns]
    available_trb = [col for col in physico_cols_trb if col in adata.obs.columns]
    
    if available_tra and available_trb:
        tcr_physico = np.column_stack([
            adata.obs[available_tra].fillna(0)[supervised_mask].values,
            adata.obs[available_trb].fillna(0)[supervised_mask].values
        ])
    else:
        # Fallback to original columns
        fallback_tra = ['tra_length', 'tra_total_mw'] if 'tra_total_mw' in adata.obs.columns else ['tra_length']
        fallback_trb = ['trb_length', 'trb_total_mw'] if 'trb_total_mw' in adata.obs.columns else ['trb_length']
        tcr_physico = np.column_stack([
            adata.obs[fallback_tra].fillna(0)[supervised_mask].values if fallback_tra else np.zeros((supervised_mask.sum(), 1)),
            adata.obs[fallback_trb].fillna(0)[supervised_mask].values if fallback_trb else np.zeros((supervised_mask.sum(), 1))
        ])

    # Get QC features
    qc_features = adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']].fillna(0)[supervised_mask].values

    # Feature Set 1: Basic (matching notebook)
    feature_sets['basic'] = np.column_stack([
        gene_pca[:, :20],  # Top 20 gene PCA
        tcr_physico,       # 6 physicochemical features (or fallback)
        qc_features        # 3 QC metrics
    ])
    print(f"  basic: {feature_sets['basic'].shape}")

    # Feature Set 2: Gene Enhanced (matching notebook)
    feature_sets['gene_enhanced'] = np.column_stack([
        gene_pca,                                    # 50 PCA components
        gene_svd[:, :30],                            # 30 SVD components
        _get_obsm_or_zeros(adata, 'X_gene_umap', supervised_mask, 20),  # 20 UMAP components
        tcr_physico,                                 # 6 physicochemical features
        qc_features                                   # 3 QC metrics
    ])
    print(f"  gene_enhanced: {feature_sets['gene_enhanced'].shape}")

    # Feature Set 3: TCR Enhanced (matching notebook)
    feature_sets['tcr_enhanced'] = np.column_stack([
        gene_pca[:, :15],  # 15 gene PCA
        tra_kmer_reduced,  # Top 200 TRA k-mers (variance-selected)
        trb_kmer_reduced,  # Top 200 TRB k-mers (variance-selected)
        tcr_physico,       # 6 physicochemical features
        qc_features        # 3 QC metrics
    ])
    print(f"  tcr_enhanced: {feature_sets['tcr_enhanced'].shape}")

    # Feature Set 4: Comprehensive (matching notebook cell 43 - reduced version)
    # Notebook: 15 PCA + 50 TRA k-mers + 50 TRB k-mers + 26 physico + 3 QC = 144 features
    # But actually uses: 15 + 50 + 50 + 6 + 3 = 124 features
    feature_sets['comprehensive'] = np.column_stack([
        gene_pca[:, :15],           # Top 15 gene PCA (vs 50) - explains 80-85% variance
        tra_kmer_reduced[:, :50],   # Top 50 TRA k-mers (vs 200) - sufficient for TCR diversity
        trb_kmer_reduced[:, :50],   # Top 50 TRB k-mers (vs 200)
        tcr_physico,                # 6 physicochemical features
        qc_features                  # 3 QC metrics
    ])
    print(f"  comprehensive: {feature_sets['comprehensive'].shape}")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(feature_sets, cache_key, "feature_engineering", format="pt")

    return feature_sets
