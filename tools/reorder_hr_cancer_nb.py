"""
Script: reorder_hr_cancer_nb.py

Reads Notebooks/hr-cancer.ipynb, heuristically reorders cells into a logical structure,
and writes a new notebook Notebooks/hr-cancer_reordered.ipynb preserving each cell's source.

Run: python tools\reorder_hr_cancer_nb.py
"""
from pathlib import Path
import nbformat
import re

INPUT = Path('Notebooks') / 'hr-cancer.ipynb'
OUTPUT = Path('Notebooks') / 'hr-cancer_reordered.ipynb'

if not INPUT.exists():
    print(f"Input notebook not found: {INPUT.resolve()}")
    raise SystemExit(1)

nb = nbformat.read(str(INPUT), as_version=4)
orig_cells = nb.cells

# Helper to get text of a cell (join list or use str)
def cell_text(cell):
    src = cell.get('source', '')
    if isinstance(src, list):
        return '\n'.join(src)
    return src

# Lowercase text for matching

categories = {
    'install': [],
    'imports': [],
    'download': [],
    'decompress': [],
    'metadata': [],
    'load_adata': [],
    'integrate_tcr': [],
    'save_processed': [],
    'additional_installs': [],
    'seq_encoding': [],
    'apply_seq_encoding': [],
    'encode_genes': [],
    'combined_encodings': [],
    'clustering': [],
    'dendrogram': [],
    'cluster_summary': [],
    'feature_engineering': [],
    'supervised': [],
    'multiclass': [],
    'model_evaluation': [],
    'sequence_length': [],
    'pattern_discovery': [],
    'save_results': [],
    'enhanced_summary': [],
    'interpretation': [],
    'uncategorized': []
}

# Matching heuristics (order matters)
patterns = [
    ('install', [r"%pip install", r"pip install "]),
    ('imports', [r"import scanpy", r"import pandas", r"import numpy", r"from sklearn", r"import scanpy as sc"]),
    ('download', [r"download_file", r"download_dir", r"GSE300475", r"download_url"]),
    ('decompress', [r"decompress_gz", r"gzip", r"mmread", r"decompress"]),
    ('metadata', [r"Load Sample Metadata", r"metadata_list", r"GEX_Sample_ID"]),
    ('load_adata', [r"read_10x_mtx", r"AnnData.concatenate", r"adata_list", r"Processing GEX sample"]),
    ('integrate_tcr', [r"TCR data", r"all_contig_annotations", r"high_confidence", r"tcr_aggregated", r"merge into AnnData"]),
    ('save_processed', [r"write_h5ad", r"Processed_Data", r"processed_s_rna_seq_data.h5ad"]),
    ('additional_installs', [r"xgboost", r"tensorflow", r"umap-learn", r"hdbscan"]),
    ('seq_encoding', [r"one_hot_encode_sequence", r"kmer_encode_sequence", r"physicochemical_features", r"Genetic Sequence Encoding"]),
    ('apply_seq_encoding', [r"X_tcr_tra_onehot", r"Computing one-hot encodings", r"TCR CDR3 sequences"]),
    ('encode_genes', [r"encode_gene_expression_patterns", r"Encoding gene expression patterns", r"highly_variable"]),
    ('combined_encodings', [r"combined_gene_tcr", r"X_combined_gene_tcr", r"X_umap_combined"]),
    ('clustering', [r"K-Means clustering", r"HDBSCAN", r"Agglomerative", r"DBSCAN", r"GaussianMixture", r"Hierarchical clustering", r"silhouette_score"]),
    ('dendrogram', [r"Dendrogram|dendrogram"]),
    ('cluster_summary', [r"cluster summary", r"centroid", r"rank_genes_groups", r"clustering_summary"]),
    ('feature_engineering', [r"Feature Engineering", r"feature_sets", r"kmer_reduced", r"comprehensive feature engineering"]),
    ('supervised', [r"Supervised Learning", r"GridSearchCV", r"param_grids", r"models = "]),
    ('multiclass', [r"Multiclass classification|multiclass", r"cluster_multiclass", r"predict cluster labels"]),
    ('model_evaluation', [r"Model Performance", r"performance_report", r"confusion_matrix", r"AUC Comparison"]),
    ('sequence_length', [r"sequence length", r"length_cutoffs", r"max sequence length"]),
    ('pattern_discovery', [r"sequence pattern discovery", r"TCR SEQUENCE PATTERNS"]),
    ('save_results', [r"comprehensive_analysis_summary", r"model_performance_results.csv", r"saved to"]),
    ('enhanced_summary', [r"Enhanced Summary Tables", r"LASSO (Logistic Regression L1)"]),
    ('interpretation', [r"Interpretation", r"Next steps", r"notable abnormalities"]) 
]

# Classify cells
for cell in orig_cells:
    text = cell_text(cell)
    low = text.lower()
    placed = False
    for cat, pats in patterns:
        for p in pats:
            if re.search(p.lower(), low):
                categories[cat].append(cell)
                placed = True
                break
        if placed:
            break
    if not placed:
        # fallback checks: use headings starting with '##'
        if cell.get('cell_type') == 'markdown' and '##' in text:
            t = text.splitlines()[0].lower()
            # heuristic mapping
            if any(k in t for k in ['load sample metadata', 'metadata']):
                categories['metadata'].append(cell); placed = True
        if not placed:
            categories['uncategorized'].append(cell)

# Build ordered list of cells
order = ['install','imports','download','decompress','metadata','load_adata','integrate_tcr','save_processed',
         'additional_installs','seq_encoding','apply_seq_encoding','encode_genes','combined_encodings',
         'clustering','dendrogram','cluster_summary','feature_engineering','supervised','multiclass',
         'model_evaluation','sequence_length','pattern_discovery','save_results','enhanced_summary','interpretation','uncategorized']

new_cells = []
for k in order:
    new_cells.extend(categories.get(k, []))

# Ensure each cell has metadata.language and code cells have outputs/execution_count
for c in new_cells:
    if c.get('cell_type') == 'code':
        meta = c.get('metadata', {})
        meta['language'] = 'python'
        c['metadata'] = meta
        # normalize outputs and execution count
        if 'outputs' not in c:
            c['outputs'] = []
        c['execution_count'] = None
    elif c.get('cell_type') == 'markdown':
        meta = c.get('metadata', {})
        meta['language'] = 'markdown'
        c['metadata'] = meta

# Assemble new notebook
new_nb = nbformat.v4.new_notebook()
new_nb['cells'] = new_cells
# copy some metadata from original if present
new_nb['metadata'] = nb.get('metadata', {})

# Write output
nbformat.write(new_nb, str(OUTPUT))
print(f"Wrote reordered notebook to: {OUTPUT.resolve()}")
