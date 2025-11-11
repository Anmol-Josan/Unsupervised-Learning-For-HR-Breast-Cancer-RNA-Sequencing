"""
Reorder notebook script (v3) — dependency-aware with specificity preference.
This version selects the most specific matching category per cell (highest priority number matched),
so cells that contain both general tokens like 'import' and specific tokens like 'K-Means clustering'
are classified as clustering, not as an import.
"""
from pathlib import Path
import nbformat
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'Notebooks' / 'hr-cancer.ipynb'
OUTPUT = ROOT / 'Notebooks' / 'hr-cancer_reordered_v3.ipynb'

if not INPUT.exists():
    print(f"Input not found: {INPUT}")
    sys.exit(1)

nb = nbformat.read(str(INPUT), as_version=4)
orig_cells = nb.cells

priority_tokens = [
    (0, [r"%pip install", r"pip install ", r"!pip install", r"install required packages"]),
    (1, [r"^import ", r"from [a-zA-Z] import ", r"import scanpy", r"import pandas", r"import numpy"]),
    (2, [r"files_to_fetch", r"download_file\(|download_url|GSE300475|tarfile|requests.get|downloaded_filepath"]),
    (3, [r"decompress_gz|gzip.open|mmread\(|scipy.io.mmread|decompress_gz_file|preview_file"]),
    (4, [r"Load Sample Metadata|metadata_list|GEX_Sample_ID|metadata_df|sample metadata"]),
    (5, [r"read_10x_mtx|AnnData.concatenate|sc.read_10x_mtx|adata_list|Processing GEX sample"]),
    (6, [r"TCR data|all_contig_annotations|tcr_df|tcr_aggregated|high_confidence|v_gene_TRA|v_gene_TRB"]),
    (7, [r"filter_cells\(|sc.pp.filter_cells|calculate_qc_metrics|pct_counts_mt|write_h5ad|processed_s_rna_seq_data.h5ad"]),
    (8, [r"xgboost|tensorflow|umap-learn|hdbscan|from Bio|biopython"]),
    (9, [r"one_hot_encode_sequence|kmer_encode_sequence|physicochemical_features|encode_gene_expression_patterns"]),
    (10, [r"X_tcr_tra_onehot|Computing one-hot encodings|tra_onehot|trb_onehot"]),
    (11, [r"combined_gene_tcr|X_combined_gene_tcr|X_umap_combined|combined_gene_tcr_kmer"]),
    (12, [r"K-Means clustering|HDBSCAN|AgglomerativeClustering|Agglomerative clustering|DBSCAN|GaussianMixture|Hierarchical clustering|silhouette_score"]),
    (13, [r"rank_genes_groups|sc.tl.rank_genes_groups|marker genes|markers"]),
    (14, [r"dendrogram|linkage\(|dendrogram\("]),
    (15, [r"feature_sets|Feature Engineering|comprehensive feature engineering|select_top_variance_features"]),
    (16, [r"Supervised Learning|GridSearchCV|param_grids|models = |RandomForestClassifier|LogisticRegression|XGBoost"]),
    (17, [r"Multiclass classification|cluster_multiclass|predict cluster labels|multiclass"]),
    (18, [r"Model Performance|performance_report|ConfusionMatrixDisplay|AUC Comparison|performance_df"]),
    (19, [r"length_cutoffs|sequence length cutoffs|max sequence length|sequence length cutoff"]),
    (20, [r"sequence pattern discovery|TCR SEQUENCE PATTERNS|K-MER DIFFERENTIAL ANALYSIS|k-mer differential analysis"]),
    (21, [r"comprehensive_analysis_summary.json|comprehensive analysis summary|saved to CSV|output_path_enriched"]),
    (22, [r"Enhanced Summary Tables|LASSO|Logistic Regression L1|LASSO quick-check"]),
    (23, [r"Interpretation|Next steps|notable abnormalities|Interpretation, notable abnormalities"]),
]

compiled = [(pri, [re.compile(t, flags=re.IGNORECASE) for t in toks]) for pri, toks in priority_tokens]

cell_infos = []
for idx, cell in enumerate(orig_cells):
    text = cell.get('source', '')
    if isinstance(text, list):
        text = '\n'.join(text)
    matched_pris = []
    for pri, regexes in compiled:
        for rx in regexes:
            if rx.search(text):
                matched_pris.append(pri)
                break
    if matched_pris:
        # pick the most specific (max priority number)
        score = max(matched_pris)
    else:
        score = 99
    cell_infos.append((idx, score, cell))

cell_infos_sorted = sorted(cell_infos, key=lambda x: (x[1], x[0]))

new_cells = []
for orig_idx, score, cell in cell_infos_sorted:
    c = cell.copy()
    meta = c.get('metadata', {}).copy()
    if c.get('cell_type') == 'code':
        meta['language'] = meta.get('language', 'python')
        c['metadata'] = meta
        if 'outputs' not in c:
            c['outputs'] = []
        c['execution_count'] = None
    elif c.get('cell_type') == 'markdown':
        meta['language'] = meta.get('language', 'markdown')
        c['metadata'] = meta
    new_cells.append(c)

new_nb = nbformat.v4.new_notebook()
new_nb['cells'] = new_cells
new_nb['metadata'] = nb.get('metadata', {})
nbformat.write(new_nb, str(OUTPUT))
print(f"Wrote reordered notebook to: {OUTPUT}")
