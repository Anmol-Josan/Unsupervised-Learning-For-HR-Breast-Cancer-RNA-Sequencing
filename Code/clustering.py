#!/usr/bin/env python3
"""
Clustering Analysis for HR+ Breast Cancer RNA-seq Data

This script performs unsupervised clustering on processed single-cell RNA sequencing data
to identify approximately 7 clusters using best practices for biological data analysis.

Best practices implemented:
- Uses Leiden clustering algorithm (standard in scRNA-seq)
- Dimensionality reduction with PCA (if not present)
- UMAP for visualization (non-linear dimensionality reduction)
- Proper data preprocessing checks
- Reproducible random seeds
- Comprehensive logging
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy import stats
try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from pathlib import Path
from scipy.sparse import issparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
sc.settings.set_figure_params(dpi=100, facecolor='white')

def load_data(data_path):
    """Load AnnData object from h5ad file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")
    adata = sc.read_h5ad(data_path)
    logger.info(f"Data loaded: {adata.shape[0]} cells, {adata.shape[1]} genes")
    return adata

def preprocess_data(adata):
    """Perform preprocessing if not already done."""
    logger.info("Checking data preprocessing status...")

    # Check if data is normalized (sum of counts per cell should be similar)
    if 'log1p' not in adata.uns or not adata.uns.get('log1p', {}).get('base', None):
        logger.info("Data appears not log-normalized. Performing normalization...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata.uns['log1p'] = {'base': None}  # Mark as log-normalized

    # Check for highly variable genes
    if 'highly_variable' not in adata.var.columns:
        logger.info("Finding highly variable genes...")
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

    # Filter to highly variable genes if not already done
    if adata.shape[1] > 2000:  # Assuming more than 2000 genes means not filtered
        logger.info("Filtering to highly variable genes...")
        adata = adata[:, adata.var.highly_variable]

    # Scale data if not done
    if 'mean' not in adata.var.columns or 'std' not in adata.var.columns:
        logger.info("Scaling data...")
        sc.pp.scale(adata, max_value=10)

    return adata

def perform_clustering(adata, n_clusters_target=7):
    """Perform clustering to achieve approximately n_clusters_target clusters."""

    # PCA if not present
    if 'X_pca' not in adata.obsm:
        logger.info("Performing PCA...")
        sc.pp.pca(adata, n_comps=50, random_state=42)

    # Compute neighbors
    logger.info("Computing neighborhood graph...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, random_state=42)

    # Try different resolutions to get close to target clusters
    resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    best_resolution = None
    best_n_clusters = float('inf')

    for res in resolutions:
        logger.info(f"Trying resolution {res}...")
        sc.tl.leiden(adata, resolution=res, random_state=42, key_added=f'leiden_{res}')

        n_clusters = len(adata.obs[f'leiden_{res}'].unique())
        logger.info(f"Resolution {res}: {n_clusters} clusters")

        if abs(n_clusters - n_clusters_target) < abs(best_n_clusters - n_clusters_target):
            best_resolution = res
            best_n_clusters = n_clusters

    # Set the best clustering as the main one
    adata.obs['leiden'] = adata.obs[f'leiden_{best_resolution}']
    logger.info(f"Selected resolution {best_resolution} with {best_n_clusters} clusters")

    return adata

def visualize_clusters(adata, output_dir):
    """Create visualizations of the clustering results."""
    logger.info("Creating visualizations...")

    # UMAP if not present
    if 'X_umap' not in adata.obsm:
        logger.info("Computing UMAP...")
        sc.tl.umap(adata, random_state=42)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # UMAP plot colored by clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(adata, color='leiden', ax=ax, show=False, legend_loc='on data')
    plt.title(f'UMAP - Leiden Clustering ({len(adata.obs["leiden"].unique())} clusters)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Cluster sizes bar plot
    cluster_counts = adata.obs['leiden'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_counts.plot(kind='bar', ax=ax)
    plt.title('Cluster Sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Cells')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_sizes.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")

def save_results(adata, output_path):
    """Save the clustered data."""
    logger.info(f"Saving results to {output_path}")
    adata.write_h5ad(output_path)

def main():
    """Main function to run the clustering analysis."""

    # File paths
    data_paths = [
        'Archive/Processed_Data/processed_s_rna_seq_data_integrated.h5ad',
        'Archive/Processed_Data/processed_s_rna_seq_data.h5ad'
    ]

    output_dir = 'Output/Clustering_Results'
    output_file = os.path.join(output_dir, 'clustered_data.h5ad')

    # Try to load data
    adata = None
    for path in data_paths:
        try:
            adata = load_data(path)
            break
        except FileNotFoundError:
            continue

    if adata is None:
        raise FileNotFoundError("Could not find processed data file")

    # Preprocess if needed
    adata = preprocess_data(adata)

    # Perform clustering
    adata = perform_clustering(adata, n_clusters_target=7)

    # Visualize
    visualize_clusters(adata, output_dir)

    # Save results
    save_results(adata, output_file)

    logger.info("Clustering analysis completed successfully!")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Visualizations saved to {output_dir}")

# -------------------------
# Additional Kaggle Run 2-style analyses
# -------------------------

def safe_multipletests(pvals, alpha=0.05, method='fdr_bh'):
    """Adjust p-values using statsmodels if available, otherwise use BH fallback."""
    pvals = np.asarray(pvals, dtype=float)
    if multipletests is not None:
        try:
            _, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method=method)
            return pvals_corrected
        except Exception:
            pass

    # Benjamini-Hochberg fallback
    n = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(n, int)
    ranks[order] = np.arange(1, n + 1)
    qvals = pvals * n / ranks
    qvals[qvals > 1] = 1.0
    # enforce monotonicity
    qvals_corrected = np.empty(n, float)
    qvals_sorted = qvals[order]
    qvals_corrected_sorted = np.minimum.accumulate(qvals_sorted[::-1])[::-1]
    qvals_corrected[order] = qvals_corrected_sorted
    return qvals_corrected


def pseudobulk_counts(adata, sample_key='sample_id'):
    """Aggregate raw counts per sample (pseudobulk). Returns (counts_df, sample_meta_df)."""
    # Prefer .raw if present (usually holds untransformed counts)
    if hasattr(adata, 'raw') and adata.raw is not None and getattr(adata.raw, 'X', None) is not None:
        mat = adata.raw.X
        gene_names = list(map(str, adata.raw.var_names))
    else:
        mat = adata.X
        gene_names = list(map(str, adata.var_names))

    if issparse(mat):
        mat = mat.tocsr()

    samples = adata.obs[sample_key].astype(str).values
    unique_samples = np.unique(samples)
    agg = []
    for samp in unique_samples:
        idx = np.where(samples == samp)[0]
        if len(idx) == 0:
            agg.append(np.zeros(len(gene_names)))
            continue
        if issparse(mat):
            s_sum = np.array(mat[idx, :].sum(axis=0)).ravel()
        else:
            s_sum = mat[idx, :].sum(axis=0)
        agg.append(s_sum)

    counts_df = pd.DataFrame(np.vstack(agg), index=unique_samples, columns=gene_names)

    # Build sample-level metadata by taking the first cell's obs for each sample
    sample_meta = adata.obs[[sample_key]].copy()
    sample_meta = sample_meta.groupby(sample_key).first()

    return counts_df, sample_meta


def run_pseudobulk_DE(adata, sample_key='sample_id', group_key='response', groupA='Responder', groupB='Non-Responder', output_dir='Output/Kaggle_Run2_Analysis'):
    """Perform simple pseudobulk differential expression (t-test) between two groups of samples.

    Note: This is a lightweight pseudobulk implementation (t-test on aggregated counts).
    For publication-grade DE use DESeq2/edgeR in R on raw counts.
    """
    logger.info('Starting pseudobulk DE: %s vs %s', groupA, groupB)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    counts_df, sample_meta = pseudobulk_counts(adata, sample_key=sample_key)

    # attach group labels to sample_meta
    if group_key in adata.obs.columns:
        mapping = adata.obs[[sample_key, group_key]].groupby(sample_key).first()
        sample_meta[group_key] = mapping[group_key]
    else:
        sample_meta[group_key] = 'Unknown'

    groupA_samples = sample_meta[sample_meta[group_key] == groupA].index.tolist()
    groupB_samples = sample_meta[sample_meta[group_key] == groupB].index.tolist()

    if len(groupA_samples) < 2 or len(groupB_samples) < 2:
        logger.warning('Not enough samples for pseudobulk DE (%d vs %d). Skipping.', len(groupA_samples), len(groupB_samples))
        return None

    countsA = counts_df.loc[groupA_samples]
    countsB = counts_df.loc[groupB_samples]

    meanA = countsA.mean(axis=0)
    meanB = countsB.mean(axis=0)
    log2fc = np.log2((meanA + 1) / (meanB + 1))

    pvals = []
    for gene in counts_df.columns:
        a = countsA[gene].values
        b = countsB[gene].values
        try:
            _, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        except Exception:
            p = np.nan
        pvals.append(p)

    pvals = np.array(pvals)
    p_adj = safe_multipletests(np.nan_to_num(pvals, nan=1.0))

    de_df = pd.DataFrame({
        'gene': counts_df.columns,
        'log2FC': log2fc.values,
        'pval': pvals,
        'p_adj': p_adj
    })
    de_df = de_df.sort_values('p_adj')
    out_csv = os.path.join(output_dir, f'pseudobulk_DE_{groupA}_vs_{groupB}.csv')
    de_df.to_csv(out_csv, index=False)
    logger.info('Pseudobulk DE results written to %s', out_csv)
    return de_df


def compute_cluster_markers(adata, groupby='leiden', n_genes=50, output_dir='Output/Kaggle_Run2_Analysis'):
    """Compute marker genes per cluster using Scanpy's rank_genes_groups and save tables/plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info('Computing cluster markers (rank_genes_groups)')
    sc.tl.rank_genes_groups(adata, groupby, method='wilcoxon', n_genes=n_genes)

    try:
        df_all = sc.get.rank_genes_groups_df(adata, group=None)
        df_all.to_csv(os.path.join(output_dir, 'rank_genes_groups_all.csv'), index=False)
    except Exception:
        # Fallback: save raw uns object for inspection
        import json
        serializable = {}
        for k, v in adata.uns['rank_genes_groups'].items():
            try:
                serializable[k] = np.array(v).tolist()
            except Exception:
                serializable[k] = str(type(v))
        with open(os.path.join(output_dir, 'rank_genes_groups_raw.json'), 'w') as fh:
            json.dump(serializable, fh)

    # heatmap of top markers
    try:
        sc.pl.rank_genes_groups_heatmap(adata, n_genes=10, groupby=groupby, show=False, save=os.path.join(output_dir, 'rank_genes_groups_heatmap.png'))
    except Exception:
        # fallback: use matrixplot
        top_genes = []
        r = adata.uns.get('rank_genes_groups', {})
        names = r.get('names', None)
        if names is not None:
            # names usually shape (n_groups, n_genes)
            try:
                for i in range(min(len(names), 7)):
                    top_genes.extend([g for g in names[i][:5]])
            except Exception:
                pass
        top_genes = list(dict.fromkeys([g for g in top_genes if g is not None]))
        if top_genes:
            sc.pl.matrixplot(adata, var_names=top_genes, groupby=groupby, cmap='viridis', standard_scale='var', show=False)


def cluster_composition(adata, groupby='leiden', by='response', output_dir='Output/Kaggle_Run2_Analysis'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if groupby not in adata.obs.columns or by not in adata.obs.columns:
        logger.warning('Missing columns for composition (%s or %s). Skipping.', groupby, by)
        return
    comp = pd.crosstab(adata.obs[groupby], adata.obs[by], normalize='index') * 100
    comp.to_csv(os.path.join(output_dir, f'cluster_composition_by_{by}.csv'))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(comp, annot=True, fmt='.1f', cmap='viridis', ax=ax)
    plt.title(f'Cluster composition by {by} (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cluster_composition_by_{by}.png'), dpi=300, bbox_inches='tight')
    plt.close()


def annotate_clusters_by_markers(adata, markers_dict=None, groupby='leiden', output_dir='Output/Kaggle_Run2_Analysis'):
    """Annotate clusters with broad cell type labels using marker genes.

    `markers_dict` should map cell type name -> list of marker genes.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if markers_dict is None:
        markers_dict = {
            'T cells': ['CD3D', 'CD3E'],
            'CD8 T': ['CD8A', 'CD8B'],
            'CD4 T': ['CD4'],
            'B cells': ['MS4A1', 'CD79A'],
            'Monocytes': ['LYZ', 'S100A8', 'S100A9'],
            'NK': ['GNLY', 'NKG7'],
            'Platelets': ['PPBP']
        }

    # Use log-normalized expression for annotation
    try:
        expr = adata.to_df()
    except Exception:
        logger.warning('Could not convert adata to DataFrame for annotation. Skipping.')
        return

    cluster_means = expr.groupby(adata.obs[groupby]).mean()
    annotations = {}
    for cluster in cluster_means.index:
        scores = {}
        for ct, genes in markers_dict.items():
            genes_present = [g for g in genes if g in cluster_means.columns]
            if not genes_present:
                scores[ct] = -np.inf
                continue
            scores[ct] = cluster_means.loc[cluster, genes_present].mean()
        best_ct = max(scores, key=scores.get)
        annotations[cluster] = best_ct

    # map to adata.obs
    adata.obs[f'{groupby}_annotation'] = adata.obs[groupby].map(annotations).astype('category')
    pd.Series(annotations).to_csv(os.path.join(output_dir, 'cluster_annotations.csv'))
    logger.info('Cluster annotations saved to %s', output_dir)


def run_kaggle_run2_analysis(adata, output_dir='Output/Kaggle_Run2_Analysis'):
    """Run a set of analyses inspired by the Kaggle Run 2 notebook and save outputs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Ensure UMAP exists
    if 'X_umap' not in adata.obsm:
        logger.info('Computing UMAP for downstream plots...')
        sc.tl.umap(adata, random_state=42)

    # Save cell-level metadata + UMAP coordinates
    obs_df = adata.obs.copy()
    if 'X_umap' in adata.obsm:
        umap_df = pd.DataFrame(adata.obsm['X_umap'], index=adata.obs_names, columns=['UMAP1', 'UMAP2'])
        obs_df = pd.concat([obs_df, umap_df], axis=1)
    obs_df.to_csv(os.path.join(output_dir, 'cell_metadata_with_umap.csv'))

    # Compute cluster markers and visualizations
    compute_cluster_markers(adata, groupby='leiden', n_genes=50, output_dir=output_dir)

    # Cluster composition by response and timepoint
    cluster_composition(adata, groupby='leiden', by='response', output_dir=output_dir)
    if 'timepoint' in adata.obs.columns:
        cluster_composition(adata, groupby='leiden', by='timepoint', output_dir=output_dir)

    # Pseudobulk DE: Responder vs Non-Responder if available
    try:
        de_df = run_pseudobulk_DE(adata, sample_key='sample_id', group_key='response', groupA='Responder', groupB='Non-Responder', output_dir=output_dir)
    except Exception as e:
        logger.exception('Pseudobulk DE failed: %s', e)

    # Annotate clusters by marker genes
    annotate_clusters_by_markers(adata, groupby='leiden', output_dir=output_dir)

    # Save annotated AnnData
    annotated_file = os.path.join(output_dir, 'clustered_data_annotated.h5ad')
    try:
        adata.write_h5ad(annotated_file)
        logger.info('Annotated AnnData saved to %s', annotated_file)
    except Exception:
        logger.exception('Failed to save annotated AnnData')


def compute_and_save_silhouette(adata, cluster_key='leiden', use_rep='X_pca', output_dir='Output/Clustering_Results', metric='euclidean'):
    """Compute silhouette score (overall and per-cluster means) and save outputs.

    - `adata`: AnnData with clustering in `adata.obs[cluster_key]`.
    - `use_rep`: one of the keys in `adata.obsm`, e.g. 'X_pca' or 'X_umap'.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if cluster_key not in adata.obs.columns:
        logger.warning('Cluster key %s not found in adata.obs — skipping silhouette.', cluster_key)
        return None

    # Ensure representation is available
    if use_rep not in adata.obsm:
        if use_rep == 'X_pca':
            logger.info('PCA not found; computing PCA for silhouette...')
            sc.pp.pca(adata, n_comps=50, random_state=42)
        elif use_rep == 'X_umap':
            logger.info('UMAP not found; computing UMAP for silhouette...')
            sc.tl.umap(adata, random_state=42)
        else:
            logger.warning('%s not present in adata.obsm and not auto-computed.', use_rep)
            return None

    X = adata.obsm[use_rep]
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X)

    labels = adata.obs[cluster_key].astype(str).values
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning('Need at least 2 clusters for silhouette (found %d).', len(unique_labels))
        return None

    try:
        from sklearn.metrics import silhouette_score, silhouette_samples
    except Exception:
        logger.warning('scikit-learn not available; cannot compute silhouette score.')
        return None

    logger.info('Computing silhouette score using %s representation...', use_rep)
    overall = float(silhouette_score(X, labels, metric=metric))
    sample_vals = silhouette_samples(X, labels, metric=metric)

    # Attach per-cell silhouette values to adata.obs
    sil_col = f'silhouette_{cluster_key}'
    adata.obs[sil_col] = sample_vals

    # Per-cluster mean silhouette
    sil_df = pd.DataFrame({ 'cluster': labels, 'silhouette': sample_vals }, index=adata.obs_names)
    cluster_means = sil_df.groupby('cluster').silhouette.mean().sort_index()

    # Save overall and per-cluster results
    out_txt = os.path.join(output_dir, f'silhouette_overall_{cluster_key}_{use_rep}.txt')
    with open(out_txt, 'w') as fh:
        fh.write(f'Overall silhouette ({use_rep}): {overall:.6f}\n')
        fh.write(f'Number of clusters: {len(unique_labels)}\n')
        fh.write('\nPer-cluster mean silhouette:\n')
        for cl, val in cluster_means.items():
            fh.write(f'{cl}\t{val:.6f}\n')

    cluster_means.to_csv(os.path.join(output_dir, f'silhouette_cluster_means_{cluster_key}_{use_rep}.csv'))

    # Barplot of per-cluster mean silhouette
    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_means.plot(kind='bar', ax=ax, color='C0')
    ax.set_ylabel('Mean silhouette score')
    ax.set_xlabel('Cluster')
    plt.title(f'Mean silhouette per cluster ({use_rep}) — overall {overall:.3f}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'silhouette_cluster_means_{cluster_key}_{use_rep}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info('Silhouette computation complete — overall: %.4f', overall)
    return overall, cluster_means


if __name__ == "__main__":
    main()
    # Run the extended Kaggle Run 2-style analysis on the saved output
    output_dir_main = 'Output/Clustering_Results'
    output_file_main = os.path.join(output_dir_main, 'clustered_data.h5ad')
    try:
        adata_saved = load_data(output_file_main)
        run_kaggle_run2_analysis(adata_saved, output_dir=os.path.join(output_dir_main, 'Kaggle_Run2_Analysis'))
        # Compute silhouette score on the saved annotated data (if available)
        try:
            compute_and_save_silhouette(adata_saved, cluster_key='leiden', use_rep='X_pca', output_dir=os.path.join(output_dir_main, 'Kaggle_Run2_Analysis'))
        except Exception as e:
            logger.exception('Silhouette computation failed: %s', e)
    except Exception as e:
        logger.exception('Extended Kaggle Run 2 analysis failed: %s', e)