# Notebook vs Pipeline: Will Results Match?

## Answer: **NO, they will NOT give identical results**

## Critical Differences Found:

### 1. **PREPROCESSING ORDER OF OPERATIONS** ⚠️ CRITICAL

**Notebook (Cell 18):**
```
1. Filter TCR cells (keep only cells with v_gene_TRA)
2. Filter cells (min_genes=200)
3. Filter genes (min_cells=3)
4. Calculate QC metrics
```

**Pipeline (`preprocess_pipeline`):**
```
1. Calculate QC metrics FIRST
2. Filter cells (min_genes=200, min_counts=500, max_pct_mt=20.0)
3. Filter genes (min_cells=3)
4. Filter TCR cells (keep only cells with v_gene_TRA)
```

**Impact:** QC metrics are calculated on different data states, which affects downstream filtering decisions.

### 2. **MISSING FILTERS IN NOTEBOOK** ⚠️ CRITICAL

**Notebook:**
- ✅ Filters by `min_genes=200`
- ❌ **Does NOT filter by `min_counts=500`**
- ❌ **Does NOT filter by `max_pct_mt=20.0`**

**Pipeline:**
- ✅ Filters by `min_genes=200`
- ✅ Filters by `min_counts=500`
- ✅ Filters by `max_pct_mt=20.0`

**Impact:** The notebook will keep more low-quality cells (low UMI counts, high mitochondrial percentage) that the pipeline would remove.

### 3. **HYPERPARAMETER GRIDS** ⚠️ MODERATE

**Notebook (Cell 22, 48):**
```python
'Logistic Regression': {'C': [0.01, 0.1, 1, 10, 100], ...}
'Decision Tree': {'max_depth': [5, 10, 20, None], ...}
'Random Forest': {'n_estimators': [50, 100, 200], ...}
'XGBoost': {'max_depth': [3, 6, 9], 'learning_rate': [0.01, 0.1, 0.3], ...}
```

**Pipeline (`get_default_param_grids`):**
```python
'LogisticRegression': {'C': [0.1, 1, 10], ...}  # Reduced from 5 to 3
'DecisionTree': {'max_depth': [5, 10], ...}  # Removed 20 and None
'RandomForest': {'n_estimators': [100], ...}  # Fixed value
'XGBoost': {'max_depth': [3, 5], 'learning_rate': [0.05, 0.1], ...}  # Reduced
```

**Impact:** Different hyperparameter search spaces will lead to different best models selected.

### 4. **FEATURE ENGINEERING IMPLEMENTATIONS** ⚠️ MODERATE

**Notebook:** Custom implementations in Cell 28, 33
**Pipeline:** Standardized implementations in `pipeline/feature_engineering.py`

While the logic appears similar, there may be subtle differences in:
- Sequence cleaning
- K-mer extraction
- Physicochemical feature calculations
- PCA/SVD implementations

### 5. **RANDOM SEEDS** ✅ SAME

Both use `random_state=42`, so this is consistent.

### 6. **CACHING** ⚠️ MINOR

**Notebook:** No caching system
**Pipeline:** Has caching system (won't affect results, but affects performance)

## Summary Table

| Aspect | Notebook | Pipeline | Impact |
|--------|----------|----------|--------|
| Filter order | TCR → Cells → Genes → QC | QC → Cells → Genes → TCR | **HIGH** |
| min_counts filter | ❌ Missing | ✅ 500 | **HIGH** |
| max_pct_mt filter | ❌ Missing | ✅ 20.0 | **HIGH** |
| Hyperparameter grids | Larger | Smaller (reduced) | **MODERATE** |
| Feature engineering | Custom | Standardized | **MODERATE** |
| Random seed | 42 | 42 | ✅ Same |
| Caching | None | Yes | **MINOR** |

## Expected Differences in Results:

1. **Different number of cells after preprocessing** (notebook will have more cells due to missing filters)
2. **Different QC metrics** (calculated at different stages)
3. **Different model performance** (due to different data + different hyperparameter grids)
4. **Potentially different feature importance rankings** (due to different input data)

## Recommendation:

To make the notebook match the pipeline results, you would need to:

1. **Change preprocessing order** to match pipeline
2. **Add missing filters** (`min_counts=500`, `max_pct_mt=20.0`)
3. **Use pipeline hyperparameter grids** (or update config to match notebook)
4. **Use pipeline feature engineering functions** instead of custom implementations

Alternatively, update the pipeline config to match the notebook's approach, but this is **not recommended** as the pipeline's approach is more standard and includes important QC filters.
