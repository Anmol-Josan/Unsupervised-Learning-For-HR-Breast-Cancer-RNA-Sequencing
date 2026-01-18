"""
Data loading module for downloading, loading, and merging GEX and TCR data.
"""

import glob
import gzip
import os
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import scanpy as sc
from anndata import AnnData

from pipeline.utils import CacheManager, compute_data_hash


def download_file(url: str, filename: str, destination_folder: Path, timeout: int = 300) -> Optional[Path]:
    """
    Download a file from a given URL to a specified destination folder.

    Args:
        url: URL to download from
        filename: Name of file to save
        destination_folder: Destination directory
        timeout: Download timeout in seconds

    Returns:
        Path to downloaded file or None if failed
    """
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    filepath = destination_folder / filename

    print(f"Attempting to download {filename} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Successfully downloaded {filename} to {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {filename}: {e}")
        return None


def decompress_gz_file(gz_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Decompress a .gz file.

    Args:
        gz_path: Path to .gz file
        output_path: Output path (default: remove .gz extension)

    Returns:
        Path to decompressed file
    """
    gz_path = Path(gz_path)
    if output_path is None:
        output_path = gz_path.with_suffix('')

    print(f"Decompressing {gz_path} → {output_path}")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    return output_path


def extract_tar(tar_path: Path, extract_dir: Path) -> Path:
    """
    Extract a tar archive.

    Args:
        tar_path: Path to tar file
        extract_dir: Directory to extract to

    Returns:
        Path to extraction directory
    """
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=extract_dir)

    print(f"Extraction complete to {extract_dir}")
    return extract_dir


def download_geo_data(
    data_dir: Path,
    geo_accession: str = "GSE300475",
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> Path:
    """
    Download GEO dataset.

    Args:
        data_dir: Directory to download data to
        geo_accession: GEO accession number
        use_cache: Whether to use cached downloads
        cache_manager: Cache manager instance

    Returns:
        Path to extracted data directory
    """
    data_dir = Path(data_dir)
    download_dir = data_dir / "downloads"
    extract_dir = data_dir / f"{geo_accession}_RAW"

    # Check if already extracted
    if extract_dir.exists() and len(list(extract_dir.glob("*"))) > 0:
        print(f"Data already extracted at {extract_dir}")
        return extract_dir

    # Download tar file
    url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={geo_accession}&format=file"
    tar_filename = f"{geo_accession}_RAW.tar"
    tar_path = download_file(url, tar_filename, download_dir)

    if tar_path is None:
        raise RuntimeError(f"Failed to download {geo_accession}")

    # Extract tar file
    extract_tar(tar_path, extract_dir)

    # Decompress all .gz files
    gz_files = list(extract_dir.glob("*.gz"))
    for gz_file in gz_files:
        decompress_gz_file(gz_file)

    return extract_dir


def load_10x_sample(
    sample_dir: Path,
    sample_id: str
) -> AnnData:
    """
    Load a single 10x sample from MTX files.

    Args:
        sample_dir: Directory containing matrix.mtx, genes.tsv, barcodes.tsv
        sample_id: Sample identifier

    Returns:
        AnnData object for the sample
    """
    print(f"Loading 10x data for sample {sample_id} from {sample_dir}")

    # Read 10x MTX format
    adata = sc.read_10x_mtx(sample_dir, var_names='gene_symbols', cache=False)

    # Add sample ID to observations
    adata.obs['sample_id'] = sample_id

    return adata


def find_sample_dirs(data_dir: Path) -> Dict[str, Path]:
    """
    Find all sample directories containing 10x data.

    Args:
        data_dir: Root data directory

    Returns:
        Dictionary mapping sample_id -> sample_dir
    """
    sample_dirs = {}

    # Look for matrix.mtx files (can be .mtx or .mtx.gz)
    matrix_files = list(data_dir.glob("*matrix.mtx*"))

    for matrix_file in matrix_files:
        # Extract sample ID from filename (e.g., GSM9061687_S1_matrix.mtx.gz -> S1)
        filename = matrix_file.name
        parts = filename.split('_')

        if len(parts) >= 2:
            sample_id = parts[1]  # e.g., S1, S2, etc.
            sample_dir = matrix_file.parent
            sample_dirs[sample_id] = sample_dir

    return sample_dirs


def load_gex_data(
    data_dir: Path,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Load gene expression data from all samples.

    Args:
        data_dir: Directory containing 10x data
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        Concatenated AnnData object with all samples
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "load_gex_data",
            {"data_dir": str(data_dir)},
            data_hash=compute_data_hash(data_dir)
        )
        cached_data = cache_manager.load_cache(cache_key, "data_loading", format="h5ad")
        if cached_data is not None:
            print(f"Loaded GEX data from cache")
            return cached_data

    # Find all sample directories
    sample_dirs = find_sample_dirs(data_dir)

    if not sample_dirs:
        raise ValueError(f"No 10x samples found in {data_dir}")

    print(f"Found {len(sample_dirs)} samples: {list(sample_dirs.keys())}")

    # Load each sample
    adata_list = []
    for sample_id, sample_dir in sorted(sample_dirs.items()):
        adata = load_10x_sample(sample_dir, sample_id)
        adata_list.append(adata)

    # Concatenate all samples
    print(f"Concatenating {len(adata_list)} samples...")
    adata = sc.concat(adata_list, label="sample_id", keys=[a.obs['sample_id'][0] for a in adata_list])

    print(f"Loaded GEX data: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "data_loading", format="h5ad")

    return adata


def load_tcr_data(
    data_dir: Path,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> pd.DataFrame:
    """
    Load TCR contig annotations from all samples.

    Args:
        data_dir: Directory containing TCR annotation files
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        DataFrame with TCR annotations
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "load_tcr_data",
            {"data_dir": str(data_dir)},
            data_hash=compute_data_hash(data_dir)
        )
        cached_data = cache_manager.load_cache(cache_key, "data_loading", format="pt")
        if cached_data is not None:
            print(f"Loaded TCR data from cache")
            return cached_data

    # Find all TCR contig annotation files
    contig_files = list(data_dir.glob("*_all_contig_annotations.csv"))

    if not contig_files:
        print("Warning: No TCR contig annotation files found")
        return pd.DataFrame()

    print(f"Found {len(contig_files)} TCR annotation files")

    # Load and concatenate all TCR data
    tcr_data_list = []
    for contig_file in contig_files:
        # Extract sample ID from filename
        filename = contig_file.name
        sample_id = filename.split('_')[1]  # e.g., S1, S2, etc.

        df = pd.read_csv(contig_file)
        df['sample_id'] = sample_id
        tcr_data_list.append(df)

    full_tcr_df = pd.concat(tcr_data_list, ignore_index=True)

    print(f"Loaded TCR data: {len(full_tcr_df)} contigs")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(full_tcr_df, cache_key, "data_loading", format="pt")

    return full_tcr_df


def merge_gex_tcr(
    adata: AnnData,
    tcr_df: pd.DataFrame,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> AnnData:
    """
    Merge GEX and TCR data into a single AnnData object.

    Args:
        adata: AnnData object with gene expression data
        tcr_df: DataFrame with TCR annotations
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        Merged AnnData object with TCR info in obs
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "merge_gex_tcr",
            {"adata_shape": adata.shape, "tcr_shape": tcr_df.shape},
            data_hash=compute_data_hash(adata)
        )
        cached_data = cache_manager.load_cache(cache_key, "data_loading", format="h5ad")
        if cached_data is not None:
            print(f"Loaded merged data from cache")
            return cached_data

    print("Merging GEX and TCR data...")

    # Filter for high-confidence, productive TRA/TRB chains (matching notebook)
    # Handle case where high_confidence column might not exist
    if 'high_confidence' in tcr_df.columns:
        tcr_to_agg = tcr_df[
            (tcr_df['high_confidence'] == True) &
            (tcr_df['productive'] == 'True') &
            (tcr_df['chain'].isin(['TRA', 'TRB']))
        ].copy()
    else:
        # If high_confidence column doesn't exist, just filter by productive and chain
        tcr_to_agg = tcr_df[
            (tcr_df['productive'] == 'True') &
            (tcr_df['chain'].isin(['TRA', 'TRB']))
        ].copy()

    # Pivot the data to create one row per barcode with TRA and TRB columns
    # This ensures each cell (barcode) has its TRA and TRB info in separate columns
    tcr_aggregated = tcr_to_agg.pivot_table(
        index=['sample_id', 'barcode'],
        columns='chain',
        values=['v_gene', 'j_gene', 'cdr3'],
        aggfunc='first'  # 'first' is safe as we expect at most one productive TRA/TRB per cell
    )

    # Flatten the multi-level column index (e.g., from ('v_gene', 'TRA') to 'v_gene_TRA')
    tcr_aggregated.columns = ['_'.join(col).strip() for col in tcr_aggregated.columns.values]
    tcr_aggregated.reset_index(inplace=True)

    # Prepare adata.obs for the merge by creating a matching barcode column
    # The index in adata.obs is like 'AGCCATGCAGCTGTTA-1-0' (barcode-batch_id)
    # The barcode in TCR data is like 'AGCCATGCAGCTGTTA-1'
    adata.obs['barcode'] = adata.obs.index
    adata.obs['barcode_for_merge'] = adata.obs.index.str.rsplit('-', n=1).str[0]

    # Perform a left merge. This keeps all cells from adata and adds TCR info where available
    # The number of rows will not change because tcr_aggregated has unique barcodes
    original_obs = adata.obs.copy()
    merged_obs = original_obs.merge(
        tcr_aggregated,
        left_on=['sample_id', 'barcode_for_merge'],
        right_on=['sample_id', 'barcode'],
        how='left'
    )
    
    # Restore the original index to the merged dataframe
    merged_obs.index = original_obs.index
    adata.obs = merged_obs

    # Add TCR length information
    adata.obs['tra_length'] = adata.obs['cdr3_TRA'].fillna('').str.len()
    adata.obs['trb_length'] = adata.obs['cdr3_TRB'].fillna('').str.len()

    # Mark cells with TCR data
    adata.obs['has_tcr'] = (~adata.obs['cdr3_TRA'].isna()) | (~adata.obs['cdr3_TRB'].isna())

    print(f"Merged data: {adata.shape[0]} cells, {adata.obs['has_tcr'].sum()} cells with TCR")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(adata, cache_key, "data_loading", format="h5ad")

    return adata


def load_metadata(
    metadata_path: Path,
    use_cache: bool = True,
    cache_manager: Optional[CacheManager] = None
) -> pd.DataFrame:
    """
    Load metadata file (Excel or CSV).

    Args:
        metadata_path: Path to metadata file
        use_cache: Whether to use cached data
        cache_manager: Cache manager instance

    Returns:
        DataFrame with metadata
    """
    # Check cache
    if use_cache and cache_manager:
        cache_key = cache_manager.generate_cache_key(
            "load_metadata",
            {"metadata_path": str(metadata_path)},
            data_hash=compute_data_hash(metadata_path)
        )
        cached_data = cache_manager.load_cache(cache_key, "data_loading", format="pt")
        if cached_data is not None:
            print(f"Loaded metadata from cache")
            return cached_data

    print(f"Loading metadata from {metadata_path}")

    # Load based on file extension
    if metadata_path.suffix in ['.xlsx', '.xls']:
        metadata_df = pd.read_excel(metadata_path)
    elif metadata_path.suffix == '.csv':
        metadata_df = pd.read_csv(metadata_path)
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path.suffix}")

    print(f"Loaded metadata: {len(metadata_df)} rows")

    # Save to cache
    if use_cache and cache_manager:
        cache_manager.save_cache(metadata_df, cache_key, "data_loading", format="pt")

    return metadata_df


def add_metadata_to_adata(
    adata: AnnData,
    metadata_df: pd.DataFrame,
    sample_id_col: str = 'sample_id',
    response_col: str = 'response',
    patient_col: str = 'patient_id'
) -> AnnData:
    """
    Add metadata columns to AnnData object.

    Args:
        adata: AnnData object
        metadata_df: Metadata DataFrame
        sample_id_col: Column name for sample IDs in metadata
        response_col: Column name for response labels
        patient_col: Column name for patient IDs

    Returns:
        AnnData with added metadata in obs
    """
    print("Adding metadata to AnnData...")

    # Merge metadata based on sample_id
    adata.obs = adata.obs.merge(
        metadata_df[[sample_id_col, response_col, patient_col]],
        on=sample_id_col,
        how='left'
    )

    print(f"Added metadata columns: {response_col}, {patient_col}")

    return adata
