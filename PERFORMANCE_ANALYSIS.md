# Performance Analysis: Why "Kaggle Run 2.ipynb" Takes 12 Hours to Run

## Executive Summary

This notebook performs comprehensive single-cell RNA sequencing (scRNA-seq) analysis with TCR (T-cell receptor) sequence data for breast cancer response prediction. The 12-hour runtime is primarily due to **nested cross-validation with multiple models, feature sets, and hyperparameter tuning** across a large single-cell dataset (~9,000 cells, ~36,000 genes).

---

## Dataset Scale

- **Cells**: ~9,000 cells across multiple patients
- **Genes**: ~36,000 features
- **Patients**: ~7 patients (for Leave-One-Patient-Out cross-validation)
- **TCR Sequences**: ~162,000 contig annotations across 10 samples
- **Data Size**: 565.5 MB compressed raw data

---

## Primary Performance Bottlenecks

### 1. **Leave-One-Patient-Out (LOPO) Cross-Validation** ⏱️ **Major Contributor**

**What it does:**
- For each patient (7 patients), trains on all other patients and tests on the held-out patient
- This creates **7 outer CV folds**

**Why it's slow:**
- Each fold requires training multiple models on ~6/7 of the data
- With nested hyperparameter tuning, this multiplies the computational cost

**Code location:** Lines ~4578-4832, ~4937-5451

---

### 2. **Nested Cross-Validation for Hyperparameter Tuning** ⏱️ **Major Contributor**

**What it does:**
- Within each LOPO fold, performs **GroupKFold cross-validation** (3 splits) to select hyperparameters
- This means for each LOPO fold, it trains models multiple times on different train/validation splits

**Computational impact:**
- **7 LOPO folds × 3 inner CV splits = 21 training iterations per model per feature set**
- Each iteration involves hyperparameter search

**Code location:** Lines ~4722-4750

---

### 3. **Multiple Feature Sets** ⏱️ **Moderate Contributor**

**What it does:**
- Iterates over multiple feature representations:
  - Gene expression features (PCA, SVD, UMAP)
  - TCR sequence features (k-mer, one-hot, physicochemical)
  - Combined/comprehensive feature sets
  - Multiple encoding schemes

**Why it's slow:**
- Each feature set requires separate model training
- Feature engineering itself is computationally expensive (PCA, UMAP, sequence encoding)

**Code location:** Lines ~4705-4832

---

### 4. **Multiple Machine Learning Models** ⏱️ **Moderate Contributor**

**What it does:**
- Trains 4 different models for each feature set:
  - Logistic Regression
  - Decision Tree
  - Random Forest (100 trees)
  - XGBoost (100 estimators)

**Computational impact:**
- **7 LOPO folds × 4 models × multiple feature sets × nested CV = hundreds of model training iterations**

**Code location:** Lines ~4660-4724

---

### 5. **Hyperparameter Grid Search** ⏱️ **Major Contributor**

**What it does:**
- Uses `RandomizedSearchCV` with **15 iterations** per fold (or `GridSearchCV` for smaller grids)
- Tests multiple hyperparameter combinations:
  - XGBoost: `max_depth` (2 values) × `learning_rate` (2 values) × `subsample` (2 values) × `colsample_bytree` (2 values) = 16 combinations
  - Random Forest: Multiple `max_depth` and `min_samples_split` combinations
  - Other models: Various regularization parameters

**Computational impact:**
- Even with RandomizedSearchCV limiting to 15 iterations, this adds significant overhead
- Each hyperparameter combination requires training and evaluating the model

**Code location:** Lines ~4731-4750

---

### 6. **Deep Learning Models with Extensive Hyperparameter Search** ⏱️ **Major Contributor**

**What it does:**
- Trains 4 deep learning architectures:
  - MLP (Multi-Layer Perceptron)
  - CNN (Convolutional Neural Network)
  - BiLSTM (Bidirectional LSTM)
  - Transformer

**Hyperparameter grid:**
- Architecture: 4 options
- Hidden units: 2 options (64, 128)
- Dropout: 2 options (0.2, 0.3)
- Learning rate: 2 options (1e-3, 1e-4)
- Batch size: 1 option (32)
- Epochs: 1 option (30)
- **Total combinations: 4 × 2 × 2 × 2 = 32 hyperparameter combinations**

**Why it's extremely slow:**
- For each LOPO fold (7 folds):
  - For each feature set (3 feature sets):
    - For each architecture (4 architectures):
      - For each hyperparameter combination (32 combinations):
        - Trains model with nested CV (3 inner folds)
        - Each training can take minutes for deep learning models
- **Estimated: 7 × 3 × 4 × 32 × 3 = 8,064 deep learning training iterations**
- Each iteration trains for up to 30 epochs with early stopping

**Code location:** Lines ~4937-5451, ~5126-5342

---

### 7. **Leiden Clustering at Multiple Resolutions** ⏱️ **Moderate Contributor**

**What it does:**
- Tests **26 different resolution parameters** to find optimal clustering:
  ```
  resolutions = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 
                 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 
                 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]
  ```

**Why it's slow:**
- Each resolution requires building a neighbor graph and running the Leiden algorithm
- With ~9,000 cells, neighbor graph construction is expensive
- 26 iterations multiply this cost

**Code location:** Lines ~4133-4157

---

### 8. **TCR Sequence Encoding** ⏱️ **Moderate Contributor**

**What it does:**
- Computes multiple encoding schemes for TCR CDR3 sequences:
  - **K-mer encoding**: 3-mer extraction and vectorization (sparse matrices)
  - **One-hot encoding**: Positional amino acid encoding
  - **Physicochemical features**: Computed per sequence
  - **TruncatedSVD reduction**: Dimensionality reduction on sparse k-mer matrices

**Why it's slow:**
- Processing ~162,000 TCR sequences
- K-mer encoding creates very high-dimensional sparse matrices
- SVD reduction on large sparse matrices is computationally intensive
- Multiple encoding schemes multiply the cost

**Code location:** Lines ~3682-3760

---

### 9. **Dimensionality Reduction Operations** ⏱️ **Moderate Contributor**

**What it does:**
- Performs multiple dimensionality reduction techniques:
  - **PCA**: On gene expression data (50 components)
  - **TruncatedSVD**: On TCR k-mer sparse matrices (200 components)
  - **UMAP**: For visualization and feature engineering
  - **t-SNE**: For visualization (on subsets)

**Why it's slow:**
- PCA/SVD on ~9,000 × ~36,000 gene expression matrix
- UMAP is particularly slow (non-linear, iterative optimization)
- Multiple reductions for different feature sets

**Code location:** Lines ~3604-3937

---

### 10. **Memory-Intensive Operations** ⏱️ **Moderate Contributor**

**What it does:**
- Stores multiple representations of the data:
  - Raw gene expression
  - Normalized/scaled data
  - Multiple PCA/SVD/UMAP embeddings
  - TCR sequence encodings (one-hot, k-mer, physicochemical)
  - Neighbor graphs for clustering

**Why it's slow:**
- Memory pressure can cause swapping to disk
- Large sparse matrices require efficient handling
- Multiple copies of transformed data increase memory footprint

**Code location:** Throughout, with explicit cleanup attempts at lines ~4285-4320

---

## Computational Complexity Breakdown

### Traditional ML Pipeline (Lines ~4578-4832)

```
Time ≈ 7 LOPO folds × 4 feature sets × 4 models × 
       (15 hyperparameter iterations × 3 inner CV splits) × 
       training_time_per_model
```

**Estimated:** ~1,680 model training iterations for traditional ML

### Deep Learning Pipeline (Lines ~4937-5451)

```
Time ≈ 7 LOPO folds × 3 feature sets × 4 architectures × 
       32 hyperparameter combinations × 3 inner CV splits × 
       (30 epochs × training_time_per_epoch)
```

**Estimated:** ~8,064 deep learning training iterations

### Clustering (Lines ~4078-4157)

```
Time ≈ 26 resolutions × (neighbor_graph_construction + leiden_algorithm)
```

---

## Optimization Attempts Already Made

The code includes several optimizations:

1. **RandomizedSearchCV**: Limits hyperparameter search to 15 iterations instead of exhaustive grid search
2. **Reduced hyperparameter grids**: XGBoost grid reduced from 162 to 16 combinations
3. **Early stopping**: Deep learning models use early stopping (patience=5)
4. **Memory cleanup**: Explicit cleanup of large matrices between sections
5. **CPU optimization**: XGBoost uses `tree_method='hist'` for faster CPU training
6. **Parallelization**: `n_jobs=-1` for CV parallelization (though `n_jobs=1` for models to avoid contention)

---

## Why 12 Hours is Expected

Given the computational complexity:

1. **Nested CV structure**: 7 outer × 3 inner = 21× multiplier
2. **Multiple models**: 4 traditional + 4 deep learning = 8× multiplier  
3. **Multiple feature sets**: 3-4 feature sets = 3-4× multiplier
4. **Hyperparameter search**: 15-32 combinations per fold = 15-32× multiplier
5. **Deep learning training**: Each model trains for multiple epochs

**Rough estimate:**
- Traditional ML: ~1,680 iterations × ~30 seconds/iteration ≈ 14 hours
- Deep Learning: ~8,064 iterations × ~5 seconds/iteration ≈ 11 hours
- Clustering + preprocessing: ~2 hours
- **Total: ~27 hours theoretical maximum**

With optimizations and early stopping, **12 hours is reasonable** for this comprehensive analysis.

---

## Recommendations for Faster Execution

1. **Reduce LOPO folds**: Use GroupKFold(3) instead of LeaveOneGroupOut (if statistically acceptable)
2. **Reduce hyperparameter search**: Further limit RandomizedSearchCV iterations (e.g., 5-10 instead of 15)
3. **Skip some architectures**: Focus on best-performing DL architectures only
4. **Reduce clustering resolutions**: Test fewer resolution values (e.g., 10 instead of 26)
5. **Use GPU acceleration**: Enable GPU for XGBoost and deep learning (code has GPU detection)
6. **Parallelize more aggressively**: Use more parallel workers where possible
7. **Reduce epochs**: Lower max epochs for deep learning (e.g., 20 instead of 30)
8. **Subsample cells**: For initial exploration, use a subset of cells

---

## Conclusion

The 12-hour runtime is **expected and justified** given:
- The comprehensive nested cross-validation strategy (necessary for patient-level generalization)
- Multiple models and feature sets being evaluated
- Extensive hyperparameter tuning
- Deep learning model training
- Large-scale single-cell RNA-seq data processing

This represents a thorough, methodologically sound analysis that prioritizes **statistical rigor** (avoiding data leakage) and **comprehensive evaluation** over speed.
