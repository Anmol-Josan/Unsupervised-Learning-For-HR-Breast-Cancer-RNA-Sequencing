# unsupervised-learning-for-hr-breast-cancer-rna-sequencing

How can unsupervised learning techniques, applied to single-cell RNA sequencing data of HR+ breast cancer patients undergoing nab-paclitaxel and pembrolizumab treatment, identify cell clusters predictive of treatment response, and what biomarkers cause these clusters?

This project explores the use of unsupervised machine learning (ML) to analyze single-cell RNA sequencing (scRNA-seq) data from HR+ breast cancer patients treated with the combination of nab-paclitaxel and pembrolizumab, with the goal of identifying immune cell clusters and transcriptomic biomarkers predictive of immunotherapy response.
Immunotherapy has shown benefits in HR+ breast cancer, however its clinical utility is limited by the absence of reliable biomarkers and the risk of immune-related toxicity. Building on recent findings that responders exhibit expansion of GZMB+ cytotoxic CD8 T cells, dynamic TCR clonality, and interferon-driven monocyte and B cell signatures while non-responders display exhausted, static immune states, this project uses clustering techniques to stratify patients according to their likelihood of treatment benefit.

1. Cluster patients based on gene sequence profiles
2. Cluster patients into most likely to respond to least likely to respond
    1. Find what are the biomarkers or identifications to figure out if people will respond
3. Cluster cells and their behavior based on their expression profile
4. Profiling gene sequences
5. Finding what clusters of genes will perform similarly -> May be useful for treatment

# Notebook Execution Time

The total wall time for all cells with timing measurements in the main analysis notebook (`hr-cancer.ipynb`) is approximately **4 hours, 50 minutes, and 33 seconds** using the free resources provided by Kaggle without any GPU acceleration. This includes data loading, processing, clustering algorithms, and visualization steps. Individual cell execution times vary significantly, with the unsupervised machine learning analysis cell taking up to 3 hours in recent runs.