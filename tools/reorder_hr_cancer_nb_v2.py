"""
Reorder notebook script (v2) — dependency-aware prioritization.
Reads `Notebooks/hr-cancer.ipynb` and writes `Notebooks/hr-cancer_reordered_v2.ipynb`.

Ordering priorities (small -> early):
  0 - package installation cells
  1 - imports
  2 - data download & extraction
  3 - decompression/preview helpers
  4 - metadata/sample mapping
  5 - read/load AnnData (read_10x_mtx, concatenate)
  6 - TCR integration
  7 - QC and save processed .h5ad
  8 - heavy library installs (xgboost, tf, umap, etc.)
  9 - encoding function definitions
 10 - apply TCR/gene encoding to adata
 11 - combined multi-modal encodings
 12 - clustering and unsupervised analyses
 13 - cluster summaries / rank_genes_groups
 14 - dendrogram visualization
 15 - feature engineering for supervised
 16 - supervised training / GridSearchCV
 17 - multiclass cluster prediction cell
 18 - model evaluation and plots
 19 - experiments (sequence length, etc.)
 20 - pattern discovery & statistical tests
 21 - save results & summary
 22 - enhanced summary & LASSO
 23 - interpretation/next steps
 99 - uncategorized

This script is conservative: it does not modify cell sources, only reorders cells based on token matching.
"""
from pathlib import Path
import nbformat
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'Notebooks' / 'hr-cancer.ipynb'
OUTPUT = ROOT / 'Notebooks' / 'hr-cancer_reordered_v2.ipynb'

if not INPUT.exists():
    print(f"Input not found: {INPUT}")
    sys.exit(1)

nb = nbformat.read(str(INPUT), as_version=4)
orig_cells = nb.cells

# token -> priority mapping (more tokens increase chance of match)
priority_tokens = [
    (0, [r"%pip install", r"pip install ", r"install required packages", r"!pip install"]),
    (1, [r"^import ", r"from [a-zA-Z] import ", r"import scanpy", r"import pandas", r"import numpy"]),
    (2, [r"files_to_fetch", r"download_file", r"download_url", r"GSE300475", r"tarfile", r"requests.get", r"downloaded_filepath"]),
    (3, [r"decompress_gz", r"gzip.open", r"mmread\(|scipy.io.mmread", r"decompress", r"preview_file"]),
    (4, [r"Load Sample Metadata", r"metadata_list", r"GEX_Sample_ID", r"metadata_df", r"sample metadata"]),
    (5, [r"read_10x_mtx", r"AnnData.concatenate", r"sc.read_10x_mtx", r"adata_list", r"Processing GEX sample", r"sc.AnnData.concatenate"]),
    (6, [r"TCR data", r"all_contig_annotations", r"tcr_df", r"tcr_aggregated", r"high_confidence", r"merge into AnnData", r"v_gene_TRA"]),
    (7, [r"filter_cells\(|sc.pp.filter_cells", r"calculate_qc_metrics", r"pct_counts_mt|mt\)", r"write_h5ad", r"processed_s_rna_seq_data.h5ad"]),
    (8, [r"xgboost", r"tensorflow", r"umap-learn", r"hdbscan", r"from Bio", r"biopython"]),
    (9, [r"one_hot_encode_sequence", r"kmer_encode_sequence", r"physicochemical_features", r"encode_gene_expression_patterns", r"Genetic Sequence Encoding"]),
    (10, [r"X_tcr_tra_onehot", r"One-hot encoding of CDR3", r"Computing one-hot encodings", r"tra_onehot", r"trb_onehot"]),
    (11, [r"combined_gene_tcr", r"X_combined_gene_tcr", r"X_umap_combined", r"combined_gene_tcr_kmer"]),
    (12, [r"K-Means clustering", r"HDBSCAN", r"AgglomerativeClustering|Agglomerative clustering", r"DBSCAN", r"GaussianMixture", r"Hierarchical clustering", r"silhouette_score"]),
    (13, [r"rank_genes_groups", r"rank_genes_groups\(|sc.tl.rank_genes_groups", r"markers", r"marker genes"]),
    (14, [r"dendrogram", r"linkage\(|dendrogram\("]),
    (15, [r"feature_sets", r"Feature Engineering", r"comprehensive feature engineering", r"select_top_variance_features"]),
    (16, [r"Supervised Learning", r"GridSearchCV", r"param_grids", r"models = ", r"RandomForestClassifier", r"LogisticRegression", r"XGBoost"]),
    (17, [r"Multiclass classification", r"cluster_multiclass", r"predict cluster labels", r"multiclass"]),
    (18, [r"Model Performance", r"performance_report", r"ConfusionMatrixDisplay", r"AUC Comparison", r"performance_df"]),
    (19, [r"length_cutoffs", r"sequence length cutoffs", r"max sequence length", r"sequence length cutoff"]),
    (20, [r"sequence pattern discovery", r"TCR SEQUENCE PATTERNS", r"K-MER DIFFERENTIAL ANALYSIS", r"k-mer differential analysis"]),
    (21, [r"comprehensive_analysis_summary.json", r"comprehensive analysis summary", r"saved to CSV", r"output_path_enriched"]),
    (22, [r"Enhanced Summary Tables", r"LASSO|Logistic Regression L1", r"LASSO quick-check"]),
    (23, [r"Interpretation", r"Next steps", r"notable abnormalities", r"Interpretation, notable abnormalities"]),
]

# Precompile regex patterns for efficiency
compiled = []
for pri, toks in priority_tokens:
    compiled.append((pri, [re.compile(t, flags=re.IGNORECASE) for t in toks]))

# Evaluate each cell and give it a priority
cell_infos = []
for idx, cell in enumerate(orig_cells):
    text = cell.get('source', '')
    if isinstance(text, list):
        text = '\n'.join(text)
    score = 99  # default uncategorized priority
    # search tokens and pick smallest matching priority
    for pri, regexes in compiled:
        for rx in regexes:
            if rx.search(text):
                if pri < score:
                    score = pri
    cell_infos.append((idx, score, cell))

# Sort by priority then original index to keep stable ordering within same priority
cell_infos_sorted = sorted(cell_infos, key=lambda x: (x[1], x[0]))

# Build new cells list, preserving metadata.id when present and ensuring language field
new_cells = []
for orig_idx, score, cell in cell_infos_sorted:
    c = cell.copy()
    # preserve metadata.id if present
    meta = c.get('metadata', {}).copy()
    if c.get('cell_type') == 'code':
        meta['language'] = meta.get('language', 'python')
        c['metadata'] = meta
        # ensure outputs/execution_count exist
        if 'outputs' not in c:
            c['outputs'] = []
        c['execution_count'] = None
    elif c.get('cell_type') == 'markdown':
        meta['language'] = meta.get('language', 'markdown')
        c['metadata'] = meta
    new_cells.append(c)

# Create new notebook and write
new_nb = nbformat.v4.new_notebook()
new_nb['cells'] = new_cells
# copy metadata except kernelspec/execution if present
new_nb['metadata'] = nb.get('metadata', {})

nbformat.write(new_nb, str(OUTPUT))
print(f"Wrote reordered notebook to: {OUTPUT}")
