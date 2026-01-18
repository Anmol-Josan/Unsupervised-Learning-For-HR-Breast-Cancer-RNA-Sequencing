# Product Requirements Document (PRD): Notebook Performance Optimization

## Document Information
- **Version**: 1.0
- **Date**: 2026-01-17
- **Status**: Draft
- **Target Runtime**: < 2 hours (from current 12 hours)
- **Target Speedup**: 6× improvement

## ⚠️ CRITICAL DIRECTIVES FOR AI WORKERS

### ⚠️ DIRECTIVE 1: DO NOT COMMIT CODE
**DO NOT COMMIT CODE UNDER ANY CIRCUMSTANCES**
- This code is for local development only
- Do not use `git commit`, `git add`, or any version control commands
- Do not create pull requests or push to repositories
- The user will handle all version control manually
- Focus on creating and modifying code files only

### ⚠️ DIRECTIVE 2: YOU MUST PARSE THE NOTEBOOK YOURSELF
**YOU ARE REQUIRED TO READ AND PARSE THE ENTIRE 7,400-LINE NOTEBOOK**
- **MANDATORY**: You MUST read `Code/Kaggle Run 2.ipynb` in its entirety (all 7,400+ lines)
- **MANDATORY**: You MUST extract code from the notebook cells yourself - cell references in this PRD are approximate guides only
- **MANDATORY**: You MUST identify all functions, global variables, dependencies, and data structures by parsing the notebook
- **MANDATORY**: You MUST understand the complete data flow by reading through all notebook cells sequentially
- **DO NOT** ask the user for cell numbers or code locations - find them yourself by parsing the notebook
- **DO NOT** assume cell references in this PRD are exact - they are approximate starting points for your search
- The notebook is the single source of truth - parse it completely before starting implementation

---

## Executive Summary

This PRD outlines a comprehensive optimization strategy to reduce the runtime of the single-cell RNA-seq analysis pipeline from **12 hours to < 2 hours** through modularization, intelligent caching, and distributed parallelization.

**⚠️ IMPORTANT FOR AI WORKERS**: This PRD assumes you will parse the entire `Code/Kaggle Run 2.ipynb` notebook (7,400+ lines) yourself. Cell references in this document are approximate guides only. You must read the notebook completely to extract all code, functions, and logic.

---

## 1. Goals and Objectives

### Primary Goals
1. **Reduce total runtime by 6×**: From 12 hours to < 2 hours
2. **Enable incremental development**: Run only changed portions without full re-execution
3. **Support distributed execution**: Parallelize across multiple VM instances
4. **Maintain reproducibility**: Ensure results are identical to original notebook

### Success Metrics
- **Runtime**: < 2 hours end-to-end (6× speedup)
- **Cache hit rate**: > 80% for repeated runs
- **Parallelization efficiency**: > 70% (utilize 70%+ of available compute)
- **Code maintainability**: Modular functions with clear interfaces
- **Reproducibility**: Identical results to original notebook (within floating-point precision)

---

## 2. Current State Analysis

### Current Architecture
- **Format**: Single Jupyter notebook (~7,400 lines)
- **Execution**: Sequential, monolithic
- **Caching**: None
- **Parallelization**: Limited (only within sklearn CV, n_jobs=-1)

### Identified Bottlenecks (from PERFORMANCE_ANALYSIS.md)
1. **Nested CV**: 7 LOPO folds × 3 inner CV splits = 21× overhead
2. **Deep Learning**: ~8,064 training iterations
3. **Traditional ML**: ~1,680 training iterations
4. **Clustering**: 26 resolution tests
5. **Feature Engineering**: Expensive operations (PCA, UMAP, SVD, sequence encoding)

---

## 3. Proposed Solution Architecture

### 3.1 Modularization Strategy

#### Module Structure (Simplified for Easier Editing)
```
pipeline/
├── __init__.py
├── data_loading.py          # Download, load GEX, load TCR, merge data
├── preprocessing.py         # Normalization, QC filtering, feature selection
├── feature_engineering.py   # Gene encoding (PCA/SVD/UMAP), TCR encoding (k-mer/one-hot/physico), feature combinations
├── clustering.py            # Leiden clustering, TCR clustering
├── modeling.py              # Traditional ML, Deep Learning, CV strategies
├── evaluation.py            # Metrics computation, patient-level aggregation
└── utils.py                 # Caching, parallelization, config management

scripts/
├── run_full_pipeline.py     # Main execution script
├── run_step.py              # Run individual pipeline step
└── parallel_runner.py       # Distributed execution coordinator

config/
└── pipeline_config.yaml     # Configuration file

cache/                        # Auto-generated cache directory
└── [cache files]

tests/                        # Unit tests
└── test_*.py
```

**Note**: This simplified structure groups related functionality into single files rather than splitting into many small modules. This makes it easier for end users to understand and edit the codebase. Each file contains multiple related functions organized by comments/sections.

#### Function Signatures (Key Examples - there are more functions than this that will require reading the IPYNB)

```python
# pipeline/data_loading.py
def load_gex_data(
    data_dir: Path,
    sample_ids: List[str],
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> Dict[str, AnnData]:
    """Load gene expression data for multiple samples."""
    pass

def load_tcr_data(
    data_dir: Path,
    sample_ids: List[str],
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """Load TCR contig annotations."""
    pass

def merge_gex_tcr(
    adata: AnnData,
    tcr_df: pd.DataFrame,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> AnnData:
    """Merge GEX and TCR data into AnnData object."""
    pass

# pipeline/feature_engineering.py
def encode_gene_expression(
    adata: AnnData,
    n_pca_components: int = 50,
    n_svd_components: int = 50,
    compute_umap: bool = False,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """Compute gene expression encodings."""
    pass

def encode_tcr_sequences(
    adata: AnnData,
    kmer_k: int = 3,
    n_svd_components: int = 200,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> Dict[str, np.ndarray]:
    """Compute TCR sequence encodings (k-mer, one-hot, physicochemical)."""
    pass

# pipeline/modeling.py
def train_traditional_ml_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    models: Dict[str, Any],
    param_grids: Dict[str, Dict],
    cv_strategy: str = "lopo",
    n_jobs: int = -1,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """Train traditional ML models with cross-validation."""
    pass

def train_deep_learning_models(
    X_gene: Optional[np.ndarray],
    X_seq: Optional[np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    config: Dict[str, Any],
    cache_dir: Optional[Path] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """Train deep learning models (MLP, CNN, BiLSTM, Transformer)."""
    pass
```

---

### 3.2 Caching Strategy

#### Cache Format
- **Primary**: PyTorch `.pt` files (efficient for NumPy arrays and AnnData objects)
- **Fallback**: `pickle` for complex objects, `h5ad` for AnnData
- **Metadata**: JSON files for cache validation and dependency tracking

#### Cache Directory Structure
```
cache/                        # Auto-generated, organized by module
├── data_loading/
│   ├── gex_data.pt              # Gene expression matrices
│   ├── tcr_data.pt              # TCR contig annotations
│   └── merged_adata.h5ad        # Merged AnnData object
├── preprocessing/
│   ├── normalized_adata.h5ad
│   └── hvg_mask.pt
├── feature_engineering/
│   ├── gene_pca.pt              # PCA components
│   ├── gene_svd.pt              # SVD components
│   ├── tcr_kmer_tra.pt          # TRA k-mer encodings
│   ├── tcr_kmer_trb.pt          # TRB k-mer encodings
│   └── tcr_physico.pt           # Physicochemical features
├── clustering/
│   ├── leiden_resolutions.pt    # All resolution results
│   └── neighbor_graph.pt       # Neighbor graph for clustering
└── modeling/
    ├── traditional_ml_results.pt
    └── deep_learning_results.pt
```

#### Cache Key Generation
```python
def generate_cache_key(
    function_name: str,
    inputs: Dict[str, Any],
    params: Dict[str, Any],
    data_hash: Optional[str] = None
) -> str:
    """
    Generate cache key from function name, inputs, and parameters.
    Includes data hash for invalidation when input data changes.
    """
    # Hash inputs and params
    # Include function name and version
    # Return deterministic cache key
    pass
```

#### Cache Invalidation Strategy
1. **Data hash**: Compute hash of input data files (MD5/SHA256)
2. **Parameter changes**: Invalidate if function parameters change
3. **Code versioning**: Include code version in cache key
4. **Manual invalidation**: CLI flag to clear specific cache sections

#### Cache Configuration
```python
# config/cache_config.yaml
cache:
  enabled: true
  base_dir: "./cache"
  formats:
    arrays: "pt"      # PyTorch for NumPy arrays
    adata: "h5ad"     # AnnData native format
    models: "pt"      # PyTorch for sklearn models
    results: "pt"     # PyTorch for DataFrames
  invalidation:
    check_data_hash: true
    check_code_version: true
  compression:
    enabled: true
    level: 6          # zlib compression level
```

---

### 3.3 Parallelization Strategy

#### Level 1: Intra-Process Parallelization (Current)
- **sklearn CV**: `n_jobs=-1` (already implemented)
- **XGBoost**: `n_jobs=1` per model (to avoid contention)
- **Deep Learning**: Batch processing with GPU if available

#### Level 2: Multi-Process Parallelization (New)
- **LOPO folds**: Distribute 7 folds across processes/cores
- **Feature sets**: Parallelize feature engineering for different sets
- **Model types**: Train traditional ML and DL in parallel

#### Level 3: Distributed Parallelization (New)
- **VM-level distribution**: Split LOPO folds across VM instances using Ray
- **Ray Object Store**: Efficient shared memory for large data structures
- **Auto-scaling**: Ray automatically distributes tasks across available VMs
- **Fault tolerance**: Automatic retry and recovery on worker failures
- **Result aggregation**: Ray handles result collection automatically

#### Parallelization Architecture
```
Local Execution (Multiprocessing):
Main Process
└── Process Pool (n_workers=8)
    ├── Worker 1: Processes fold_1
    ├── Worker 2: Processes fold_2
    ├── Worker 3: Processes fold_3
    └── ... (all 7 folds in parallel)

Distributed Execution (Ray):
Ray Head Node (VM 1)
├── Ray Object Store (shared memory)
├── Ray Scheduler (task distribution)
└── Remote Workers:
    ├── VM 1: Processes folds [1, 2]
    ├── VM 2: Processes folds [3, 4]
    ├── VM 3: Processes folds [5, 6]
    └── VM 4: Processes fold [7]
    
    (Ray automatically handles load balancing, fault tolerance, and result aggregation)
```

#### Implementation Strategy (Performance-Optimized)

**Decision: Hybrid Approach - Multiprocessing for Local, Ray for Distributed**

**Local Parallelization (Single Machine): Use Multiprocessing**
- **Rationale**: Lowest overhead, fastest for single-machine multi-core execution
- **Performance**: ~3-4× speedup on 8-core machine
- **Implementation**:
```python
from multiprocessing import Pool
from functools import partial

def process_lopo_fold(fold_idx, train_idx, test_idx, config):
    """Process a single LOPO fold."""
    # Load cached features if available
    # Train models
    # Return results
    pass

# Parallelize LOPO folds
with Pool(processes=n_workers) as pool:
    results = pool.starmap(
        process_lopo_fold,
        [(i, train_idx, test_idx, config) for i, (train_idx, test_idx) in enumerate(logo.split(...))]
    )
```

**Distributed Parallelization (Multiple VMs): Use Ray**
- **Rationale**: Superior performance vs Celery (object store, auto-scaling, better fault tolerance)
- **Performance**: Near-linear scaling across VMs (6-7× speedup with 7 VMs)
- **Implementation**:
```python
import ray

# Initialize Ray cluster (auto-detects if running on cluster)
ray.init(address='auto' if ray.is_initialized() else None)

@ray.remote(num_cpus=4)  # Allocate resources per task
def process_lopo_fold_ray(fold_idx, train_idx, test_idx, config):
    """Ray remote function for distributed execution."""
    # Load cached features if available
    # Train models
    # Return results
    pass

# Distribute across cluster (7 folds across available VMs)
results = [
    process_lopo_fold_ray.remote(i, train_idx, test_idx, config)
    for i, (train_idx, test_idx) in enumerate(logo.split(...))
]
final_results = ray.get(results)  # Blocks until all complete
```

**Auto-Detection Logic**:
```python
def get_parallel_strategy(n_vms: int = 1, n_local_cores: int = 8):
    """Auto-select best parallelization strategy."""
    if n_vms > 1:
        return "ray"  # Use Ray for multi-VM distribution
    elif n_local_cores >= 4:
        return "multiprocessing"  # Use multiprocessing for local
    else:
        return "sequential"  # Fallback to sequential
```

---

## 4. Implementation Plan

### Phase 1: Modularization (Week 1-2)
**Goal**: Refactor notebook into modular Python files

1. **Extract data loading** (Days 1-2)
   - Create `data_loading/` module
   - Extract download, GEX loading, TCR loading functions
   - Add unit tests

2. **Extract preprocessing** (Days 3-4)
   - Create `preprocessing/` module
   - Extract normalization, QC, feature selection
   - Add unit tests

3. **Extract feature engineering** (Days 5-7)
   - Create `feature_engineering/` module
   - Extract gene encoding, TCR encoding functions
   - Add unit tests

4. **Extract modeling** (Days 8-10)
   - Create `modeling/` module
   - Extract traditional ML and DL training functions
   - Add unit tests

5. **Create main script** (Days 11-14)
   - Create `run_full_pipeline.py`
   - Integrate all modules
   - Validate against original notebook results

**Deliverables**:
- Modular codebase with unit tests
- `run_full_pipeline.py` producing identical results
- Documentation for each module

---

### Phase 2: Caching Implementation (Week 3)
**Goal**: Add intelligent caching system

1. **Implement cache utilities** (Days 1-3)
   - Create `utils/caching.py`
   - Implement cache key generation
   - Implement save/load functions for different formats

2. **Add caching to data loading** (Day 4)
   - Cache raw data loads
   - Cache merged AnnData objects

3. **Add caching to feature engineering** (Days 5-6)
   - Cache PCA, SVD, UMAP results
   - Cache TCR encodings

4. **Add caching to modeling** (Days 7-8)
   - Cache trained models (if memory permits)
   - Cache CV results

5. **Add cache management CLI** (Days 9-10)
   - CLI to clear specific cache sections
   - CLI to inspect cache status
   - Cache validation utilities

**Deliverables**:
- Caching system with 80%+ cache hit rate on repeated runs
- Cache management CLI
- Documentation

---

### Phase 3: Parallelization (Week 4)
**Goal**: Enable parallel execution

1. **Intra-process parallelization** (Days 1-2)
   - Optimize existing `n_jobs` usage
   - Profile and fix bottlenecks

2. **Multi-process parallelization** (Days 3-5)
   - Implement LOPO fold parallelization
   - Implement feature set parallelization
   - Add process pool management

3. **Distributed execution with Ray** (Days 6-8)
   - Implement Ray-based distributed execution
   - Create worker scripts for VM execution
   - Add result aggregation and fault tolerance
   - Implement auto-scaling and resource management

**Deliverables**:
- Parallelized pipeline with 4-6× speedup (local) or 6-7× speedup (distributed)
- Ray-based distributed execution system
- Performance benchmarks and scaling analysis

---

### Phase 4: Integration & Testing (Week 5)
**Goal**: Validate optimization and ensure reproducibility

1. **End-to-end testing** (Days 1-3)
   - Run full pipeline with caching enabled
   - Run full pipeline with parallelization enabled
   - Compare results to original notebook

2. **Performance benchmarking** (Days 4-5)
   - Measure runtime improvements
   - Profile remaining bottlenecks
   - Optimize further if needed

3. **Documentation** (Days 6-7)
   - Update README with usage instructions
   - Document caching strategy
   - Document parallelization setup

**Deliverables**:
- Validated optimized pipeline
- Performance benchmarks showing < 2 hour runtime
- Complete documentation

---

## 5. Technical Specifications

### 5.1 Cache Format Details

#### PyTorch (.pt) Format
```python
import torch

# Save
torch.save({
    'data': numpy_array,
    'metadata': {'shape': shape, 'dtype': str(dtype)},
    'hash': data_hash,
    'version': code_version
}, cache_path)

# Load
cache = torch.load(cache_path)
data = cache['data']
```

#### AnnData (.h5ad) Format
```python
import anndata as ad

# Save (native format, already efficient)
adata.write_h5ad(cache_path)

# Load
adata = ad.read_h5ad(cache_path)
```

#### Metadata JSON
```json
{
    "function": "encode_gene_expression",
    "version": "1.0.0",
    "data_hash": "abc123...",
    "params": {"n_components": 50},
    "created": "2026-01-17T10:00:00Z",
    "size_mb": 125.5
}
```

### 5.2 Configuration Management

#### YAML Configuration
```yaml
# config/pipeline_config.yaml
pipeline:
  data_dir: "./Data"
  cache_dir: "./cache"
  output_dir: "./Output"
  
  steps:
    data_loading:
      enabled: true
      use_cache: true
    preprocessing:
      enabled: true
      use_cache: true
    feature_engineering:
      enabled: true
      use_cache: true
      compute_umap: false  # Skip expensive UMAP
    clustering:
      enabled: true
      use_cache: true
      resolutions: [0.1, 0.2, 0.3, 0.5, 0.8]  # Reduced from 26
    modeling:
      enabled: true
      use_cache: true
      traditional_ml:
        enabled: true
        n_iter_search: 10  # Reduced from 15
      deep_learning:
        enabled: true
        max_epochs: 20  # Reduced from 30
        hyperparam_combinations: 16  # Reduced from 32

parallelization:
  enabled: true
  strategy: "auto"  # "auto", "multiprocessing", or "ray"
  n_workers: 4  # For local multiprocessing
  distributed:
    enabled: false  # Set to true for multi-VM execution
    ray_address: null  # Auto-detect if null, or specify "head-node-ip:6379"
    n_vms: 3  # Number of VM instances available
```

### 5.3 Main Execution Script

```python
# scripts/run_full_pipeline.py
#!/usr/bin/env python3
"""
Main pipeline execution script with caching and parallelization support.
"""

import argparse
from pathlib import Path
from src.utils.config import load_config
from src.utils.caching import CacheManager
from src.data_loading import load_all_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.clustering import perform_clustering
from src.modeling import train_models
from src.evaluation import evaluate_models

def main():
    parser = argparse.ArgumentParser(description="Run scRNA-seq analysis pipeline")
    parser.add_argument("--config", type=Path, default="config/pipeline_config.yaml")
    parser.add_argument("--steps", nargs="+", choices=[
        "data_loading", "preprocessing", "feature_engineering", 
        "clustering", "modeling", "evaluation"
    ], default="all")
    parser.add_argument("--use-cache", action="store_true", default=True)
    parser.add_argument("--clear-cache", nargs="+", help="Clear specific cache sections")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--n-workers", type=int, default=4)
    
    args = parser.parse_args()
    config = load_config(args.config)
    cache_manager = CacheManager(config['cache']['base_dir'])
    
    # Clear cache if requested
    if args.clear_cache:
        cache_manager.clear_sections(args.clear_cache)
    
    # Execute pipeline steps
    if args.steps == "all" or "data_loading" in args.steps:
        adata = load_all_data(
            config['pipeline']['data_dir'],
            use_cache=args.use_cache,
            cache_manager=cache_manager
        )
    
    if args.steps == "all" or "preprocessing" in args.steps:
        adata = preprocess_data(
            adata,
            use_cache=args.use_cache,
            cache_manager=cache_manager
        )
    
    if args.steps == "feature_engineering" in args.steps:
        features = engineer_features(
            adata,
            config['pipeline']['steps']['feature_engineering'],
            use_cache=args.use_cache,
            cache_manager=cache_manager
        )
    
    if args.steps == "all" or "clustering" in args.steps:
        adata = perform_clustering(
            adata,
            config['pipeline']['steps']['clustering'],
            use_cache=args.use_cache,
            cache_manager=cache_manager
        )
    
    if args.steps == "all" or "modeling" in args.steps:
        results = train_models(
            adata,
            features,
            config['pipeline']['steps']['modeling'],
            parallel=args.parallel,
            n_workers=args.n_workers,
            use_cache=args.use_cache,
            cache_manager=cache_manager
        )
    
    if args.steps == "all" or "evaluation" in args.steps:
        metrics = evaluate_models(results)
        metrics.to_csv(config['pipeline']['output_dir'] / "results.csv")

if __name__ == "__main__":
    main()
```

### 5.4 Distributed Execution Script (Ray-Based)

```python
# scripts/parallel_runner.py
#!/usr/bin/env python3
"""
Ray-based distributed execution coordinator for VM-level parallelization.
"""

import argparse
import ray
from pathlib import Path
from pipeline.modeling import process_lopo_fold

@ray.remote(num_cpus=4, num_gpus=0)  # Allocate 4 CPUs per task
def process_fold_remote(fold_idx, train_idx, test_idx, config, cache_dir):
    """Ray remote function for distributed LOPO fold processing."""
    return process_lopo_fold(fold_idx, train_idx, test_idx, config, cache_dir)

def main():
    parser = argparse.ArgumentParser(description="Distributed pipeline execution with Ray")
    parser.add_argument("--ray-address", type=str, default=None, 
                       help="Ray cluster address (auto-detects if None)")
    parser.add_argument("--config", type=Path, default="config/pipeline_config.yaml")
    parser.add_argument("--n-workers", type=int, default=None,
                       help="Number of workers (auto-detects if None)")
    
    args = parser.parse_args()
    
    # Initialize Ray (connects to cluster if address provided, else starts local)
    if not ray.is_initialized():
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    
    print(f"Ray cluster resources: {ray.cluster_resources()}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Generate LOPO folds
    from pipeline.modeling import lopo_cv_split
    X, y, groups = load_data_for_modeling(config)
    folds = list(lopo_cv_split(X, y, groups))
    
    # Distribute folds across cluster
    print(f"Distributing {len(folds)} LOPO folds across Ray cluster...")
    results_refs = [
        process_fold_remote.remote(
            fold_idx=i,
            train_idx=train_idx,
            test_idx=test_idx,
            config=config,
            cache_dir=Path(config['cache']['base_dir'])
        )
        for i, (train_idx, test_idx) in enumerate(folds)
    ]
    
    # Collect results (Ray handles fault tolerance automatically)
    print("Collecting results from distributed workers...")
    results = ray.get(results_refs)  # Blocks until all complete
    
    # Aggregate results
    final_results = aggregate_results(results)
    save_results(final_results, config['pipeline']['output_dir'])
    
    print(f"Completed {len(folds)} folds. Results saved.")
    
    # Shutdown Ray (optional - keeps cluster alive for reuse)
    # ray.shutdown()

if __name__ == "__main__":
    main()
```

**Ray Cluster Setup** (for multi-VM execution):
```bash
# On head node (VM 1)
ray start --head --port=6379

# On worker nodes (VM 2-N)
ray start --address=<head-node-ip>:6379

# Run pipeline (connects to cluster automatically)
python scripts/parallel_runner.py --ray-address=<head-node-ip>:6379
```

---

## 6. Expected Performance Improvements

### Baseline (Current)
- **Total Runtime**: 12 hours
- **Breakdown**:
  - Data loading: 10 min
  - Preprocessing: 15 min
  - Feature engineering: 45 min
  - Clustering: 30 min
  - Traditional ML: 4 hours
  - Deep Learning: 6 hours
  - Evaluation: 20 min

### Optimized (Target)
- **Total Runtime**: < 2 hours
- **Breakdown**:
  - Data loading: 2 min (cached)
  - Preprocessing: 3 min (cached)
  - Feature engineering: 5 min (cached, parallelized)
  - Clustering: 5 min (reduced resolutions, cached)
  - Traditional ML: 30 min (parallelized LOPO folds, reduced hyperparams)
  - Deep Learning: 60 min (parallelized, reduced epochs/combinations)
  - Evaluation: 5 min

### Speedup Breakdown
1. **Caching**: 2× speedup (skip recomputation)
2. **Parallelization**: 3× speedup (distribute LOPO folds)
3. **Reduced hyperparameters**: 1.5× speedup (fewer combinations)
4. **Reduced resolutions**: 1.2× speedup (fewer clustering tests)
5. **Total**: ~6× speedup (2 × 3 × 1.5 × 1.2 ≈ 10.8× theoretical, ~6× practical)

---

## 7. Risks and Mitigations

### Risk 1: Cache Invalidation Issues
**Impact**: High - Wrong results if cache is stale  
**Mitigation**:
- Include data hash in cache keys
- Version cache format
- Provide `--clear-cache` flag
- Add cache validation checks

### Risk 2: Parallelization Overhead
**Impact**: Medium - May not achieve expected speedup  
**Mitigation**:
- Profile to identify optimal worker count
- Use process pools instead of threads for CPU-bound tasks
- Implement work-stealing for load balancing

### Risk 3: Memory Constraints
**Impact**: High - May cause OOM errors  
**Mitigation**:
- Implement memory-efficient data loading
- Use generators for large datasets
- Add memory profiling and warnings
- Support chunked processing

### Risk 4: Reproducibility Issues
**Impact**: High - Results must match original  
**Mitigation**:
- Set random seeds consistently
- Use deterministic algorithms where possible
- Add result validation tests
- Document any expected differences

### Risk 5: Distributed Execution Complexity
**Impact**: Medium - Adds operational complexity  
**Mitigation**:
- Use Ray's built-in fault tolerance and auto-recovery
- Provide clear documentation and examples
- Implement robust error handling and retries
- Auto-detect execution environment (local vs distributed)
- Fallback to multiprocessing if Ray unavailable

---

## 8. Success Criteria

### Must Have (MVP)
- [ ] Modular codebase with clear separation of concerns
- [ ] Caching system with >80% hit rate on repeated runs
- [ ] Parallelization achieving 4×+ speedup
- [ ] Results identical to original notebook (within FP precision)
- [ ] Runtime < 3 hours (2.5× improvement)

### Should Have (Full Implementation)
- [ ] Runtime < 2 hours (6× improvement)
- [ ] Ray-based distributed execution (VM-level) with auto-detection
- [ ] Comprehensive unit tests (>80% coverage)
- [ ] Complete documentation
- [ ] CLI tools for cache management
- [ ] Performance profiling and bottleneck identification

### Nice to Have (Future Enhancements)
- [ ] GPU acceleration for deep learning (Ray supports GPU allocation)
- [ ] Automatic hyperparameter optimization (Optuna/Bayesian with Ray Tune)
- [ ] Web UI for monitoring pipeline execution (Ray Dashboard)
- [ ] Incremental learning support
- [ ] Cloud deployment scripts (AWS/GCP/Azure) with Ray cluster setup

---

## 9. Dependencies and Requirements

### New Dependencies
```txt
# Core
torch>=2.0.0          # For .pt cache format
pyyaml>=6.0           # Configuration files
tqdm>=4.65.0          # Progress bars

# Distributed execution (required for VM-level parallelization)
ray>=2.5.0            # Distributed computing framework (superior to Celery for performance)

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### System Requirements
- **CPU**: 8+ cores recommended for parallelization
- **RAM**: 32GB+ recommended (16GB minimum)
- **Storage**: 50GB+ for cache and outputs
- **Python**: 3.9+

---

## 10. Timeline and Milestones

| Phase | Duration | Key Milestones |
|-------|----------|----------------|
| Phase 1: Modularization | 2 weeks | Modular codebase, unit tests |
| Phase 2: Caching | 1 week | Caching system, 80%+ hit rate |
| Phase 3: Parallelization | 1 week | 4×+ speedup achieved |
| Phase 4: Integration | 1 week | < 2 hour runtime, validated results |
| **Total** | **5 weeks** | **Production-ready optimized pipeline** |

---

## 11. Next Steps

1. **Review and approve PRD** (Stakeholders)
2. **Set up development environment** (DevOps)
3. **Create project structure** (Developer)
4. **Begin Phase 1: Modularization** (Developer)
5. **Weekly progress reviews** (Team)

---

## 12. Implementation Instructions for AI Workers (Claude Code, GitHub Copilot, etc.)

### ⚠️ CRITICAL REMINDER: YOU MUST PARSE THE NOTEBOOK YOURSELF
**Before reading any further, understand this:**
- The notebook `Code/Kaggle Run 2.ipynb` contains 7,400+ lines of code
- **YOU MUST** read and parse this entire notebook yourself
- **YOU MUST** extract all code, functions, and logic from the notebook
- **YOU MUST** identify all global variables, helper functions, and dependencies
- Cell references in this PRD are approximate guides - find exact locations yourself
- The notebook is the single source of truth - parse it completely

### ⚠️ CRITICAL: DO NOT COMMIT CODE
**IMPORTANT DIRECTIVE FOR ALL AI WORKERS:**
- **DO NOT** use `git commit`, `git add`, or any version control commands
- **DO NOT** create pull requests or push to repositories
- **DO NOT** modify `.gitignore` or git-related files
- **ONLY** create and modify code files as specified in this PRD
- The user will handle all version control manually

### Implementation Approach

#### ⚠️ STEP 0: MANDATORY NOTEBOOK PARSING (DO THIS FIRST)
**YOU MUST COMPLETE THIS STEP BEFORE ANY IMPLEMENTATION**

**CRITICAL REQUIREMENT**: You MUST read and parse the entire `Code/Kaggle Run 2.ipynb` notebook (all 7,400+ lines) before starting any implementation. This is not optional - it is mandatory.

1. **READ THE ENTIRE NOTEBOOK**: `Code/Kaggle Run 2.ipynb` (all 7,400+ lines)
   - Parse every cell sequentially from start to finish
   - Extract all code blocks, functions, and logic
   - Document all global variables (e.g., `IS_KAGGLE`, `adata`, `supervised_mask`, `metadata_df`, `extract_dir`, `raw_data_dir`)
   - Identify all helper functions (e.g., `encode_gene_expression_patterns`, `physicochemical_features`, `cleanup_after_clustering`, `_clean_seq`, `_reduce_sparse`, `_onehot_flat_list`)
   - Map all imports and their sources
   - Understand the complete execution flow
   - Note all data structure modifications (what gets added to `adata.obs`, `adata.obsm`, `adata.var`, etc.)

2. **CREATE A CODE MAP**: Document your findings
   - Map notebook cells to functionality (data loading, preprocessing, feature engineering, clustering, modeling, evaluation)
   - List all global variables and their purposes, including where they're set
   - List all helper functions and their complete implementations (extract the full function code)
   - Document data structure assumptions:
     - AnnData structure: what columns exist in `adata.obs` (e.g., `patient_id`, `response`, `sample_id`, `timepoint`)
     - What gets stored in `adata.obsm` (e.g., `X_pca`, `X_tcr_tra_kmer`, `X_gene_pca`)
     - What gets stored in `adata.var` (e.g., `highly_variable`)
   - Note environment-specific code (Kaggle vs local path handling)
   - Document all feature set names and how they're created

3. **IDENTIFY DEPENDENCIES**: Track what depends on what
   - Which cells must run before others (execution order)
   - Which variables are set in which cells and used where
   - Which functions are defined where and called where
   - Which imports are needed for each module
   - How `supervised_mask` is created and used throughout

4. **EXTRACT HELPER FUNCTIONS**: These are defined in the notebook and MUST be extracted
   - `encode_gene_expression_patterns()` - Find this function definition in the notebook
   - `physicochemical_features()` - Find this function definition in the notebook  
   - `cleanup_after_clustering()` - Find this function definition in the notebook
   - Any other helper functions you find - extract them all

**REMINDER**: Cell references in this PRD (e.g., "cells 3-6", "~2350+", "~3824+") are APPROXIMATE GUIDES ONLY. They are starting points for your search, not exact locations. You must find the exact code locations by parsing the notebook yourself. The notebook is the single source of truth.

#### ⚠️ CRITICAL: What You Must Extract from the Notebook

**Global Variables** (find where these are set and used):
- `IS_KAGGLE` - Environment detection flag
- `adata` - Main AnnData object (created, modified throughout)
- `supervised_mask` - Boolean mask for supervised learning (CRITICAL - find how this is created)
- `metadata_df` - Patient/sample metadata DataFrame
- `extract_dir`, `raw_data_dir` - Data directory paths
- `feature_sets` - Dictionary of feature sets (find how this is created)
- `y_encoded`, `y_all` - Encoded target variables
- `groups_all`, `groups_all_local` - Grouping variables for CV
- `unique_patients` - List of unique patient IDs
- Any other global variables you encounter

**Helper Functions** (extract complete implementations):
- `encode_gene_expression_patterns()` - Gene expression encoding function
- `physicochemical_features()` - TCR physicochemical property computation
- `cleanup_after_clustering()` - Memory cleanup function
- `_clean_seq()` - Sequence cleaning helper
- `_reduce_sparse()` - SVD reduction helper
- `_onehot_flat_list()` - One-hot encoding helper
- `_apply_gpu_patches()` - GPU configuration helper
- Any other helper functions defined in the notebook

**Data Structures** (document what gets stored where):
- `adata.obs` columns: `patient_id`, `response`, `sample_id`, `timepoint`, `cdr3_TRA`, `cdr3_TRB`, clustering results, etc.
- `adata.obsm` keys: `X_pca`, `X_tcr_tra_kmer`, `X_tcr_trb_kmer`, `X_gene_pca`, `X_umap`, etc.
- `adata.var` columns: `highly_variable`, gene names, etc.
- Feature set dictionary structure and keys

**Search Patterns** (use these to find code in the notebook):
- Data loading: Search for `sc.read_10x_mtx`, `download_file`, `tarfile.open`, `all_contig_annotations`
- Preprocessing: Search for `sc.pp.normalize_total`, `sc.pp.log1p`, `sc.pp.highly_variable_genes`, `sc.pp.scale`
- Feature engineering: Search for `encode_gene_expression_patterns`, `CountVectorizer`, `TruncatedSVD`, `physicochemical_features`
- Clustering: Search for `sc.tl.leiden`, `sc.pp.neighbors`, `KMeans`, `resolutions`
- Modeling: Search for `LeaveOneGroupOut`, `GridSearchCV`, `RandomizedSearchCV`, MLP/CNN/BiLSTM/Transformer
- Evaluation: Search for `accuracy_score`, `roc_auc_score`, patient-level aggregation

#### Step 1: Understand the Source Material
1. **You have already parsed the notebook** (from Step 0 above)
2. **Understand the data flow**: 
   - Data loading → Preprocessing → Feature Engineering → Clustering → Modeling → Evaluation
3. **Identify key functions**: Extract reusable code blocks from notebook cells (you found these in Step 0)
4. **Note dependencies**: Track imports and data structures used (you documented these in Step 0)

#### Step 2: Create Directory Structure
```bash
# Create the simplified directory structure
mkdir -p pipeline scripts config tests cache
touch pipeline/__init__.py
touch scripts/run_full_pipeline.py
touch scripts/run_step.py
touch scripts/parallel_runner.py
touch config/pipeline_config.yaml
```

#### Step 3: Implement Modules (In Order)

**3.1 Start with `pipeline/utils.py`** (Foundation)
- Implement `CacheManager` class with:
  - `generate_cache_key()` - Create deterministic cache keys
  - `save_cache()` - Save data to .pt or .h5ad files
  - `load_cache()` - Load cached data
  - `clear_cache()` - Clear specific cache sections
  - `validate_cache()` - Check cache validity
- Implement `ConfigManager` class for YAML config loading
- Implement basic parallelization helpers

**3.2 Implement `pipeline/data_loading.py`**
- **PARSE THE NOTEBOOK** to find exact locations for:
  - Data download logic (search for `download_file`, `files_to_fetch`, tar extraction)
  - GEX loading (search for `sc.read_10x_mtx`, `AnnData`, gene expression loading)
  - TCR loading (search for `all_contig_annotations.csv`, TCR data loading)
  - Merge logic (search for `merge_gex_tcr`, `concatenate`, combining GEX and TCR)
- Extract all code from the notebook cells you identified
- Handle global variables like `IS_KAGGLE`, `metadata_df` appropriately
- Add caching support using `CacheManager`
- Function signatures:
  ```python
  def download_data(data_dir: Path, urls: List[Dict]) -> Path
  def load_gex_data(data_dir: Path, sample_ids: List[str], cache_dir: Path, use_cache: bool) -> Dict[str, AnnData]
  def load_tcr_data(data_dir: Path, sample_ids: List[str], cache_dir: Path, use_cache: bool) -> pd.DataFrame
  def merge_gex_tcr(adata: AnnData, tcr_df: pd.DataFrame, cache_dir: Path, use_cache: bool) -> AnnData
  ```

**3.3 Implement `pipeline/preprocessing.py`**
- **PARSE THE NOTEBOOK** to find exact locations for:
  - Normalization (search for `sc.pp.normalize_total`, `sc.pp.log1p`, normalization logic)
  - QC filtering (search for `filter_cells`, `min_genes`, `min_cells`, quality control)
  - Highly variable gene selection (search for `sc.pp.highly_variable_genes`, `hvg`)
- Extract all code from the notebook cells you identified
- Handle any global state dependencies
- Add caching support
- Function signatures:
  ```python
  def normalize_data(adata: AnnData, cache_dir: Path, use_cache: bool) -> AnnData
  def filter_cells(adata: AnnData, min_genes: int, min_cells: int) -> AnnData
  def select_hvg(adata: AnnData, n_top_genes: int, cache_dir: Path, use_cache: bool) -> AnnData
  ```

**3.4 Implement `pipeline/feature_engineering.py`**
- **PARSE THE NOTEBOOK** to find exact locations for:
  - Gene encoding (search for `encode_gene_expression_patterns`, PCA, SVD, UMAP on gene expression)
  - TCR encoding (search for `CountVectorizer`, k-mer encoding, `physicochemical_features`, one-hot encoding)
  - Feature combination (search for feature set creation, combining gene + TCR features)
- **CRITICAL**: Extract helper functions like `encode_gene_expression_patterns()` and `physicochemical_features()` - these are defined in the notebook
- Extract all code from the notebook cells you identified
- Add caching for expensive operations (PCA, SVD, UMAP, k-mer encoding)
- Function signatures:
  ```python
  def encode_gene_expression(adata: AnnData, n_pca: int, n_svd: int, compute_umap: bool, cache_dir: Path, use_cache: bool) -> Dict[str, np.ndarray]
  def encode_tcr_sequences(adata: AnnData, kmer_k: int, n_svd: int, cache_dir: Path, use_cache: bool) -> Dict[str, np.ndarray]
  def create_feature_sets(gene_encodings: Dict, tcr_encodings: Dict, supervised_mask: np.ndarray) -> Dict[str, np.ndarray]
  ```

**3.5 Implement `pipeline/clustering.py`**
- **PARSE THE NOTEBOOK** to find exact locations for:
  - Leiden clustering (search for `sc.tl.leiden`, `resolutions`, clustering logic)
  - TCR-specific clustering (search for KMeans on TCR features, `tra_kmer_clusters`, `trb_kmer_clusters`)
  - Neighbor graph computation (search for `sc.pp.neighbors`)
- Extract all code from the notebook cells you identified
- Cache neighbor graphs and clustering results
- Function signatures:
  ```python
  def perform_leiden_clustering(adata: AnnData, resolutions: List[float], cache_dir: Path, use_cache: bool) -> AnnData
  def cluster_tcr_sequences(adata: AnnData, n_clusters: int, cache_dir: Path, use_cache: bool) -> AnnData
  ```

**3.6 Implement `pipeline/modeling.py`**
- **PARSE THE NOTEBOOK** to find exact locations for:
  - Traditional ML (search for `LeaveOneGroupOut`, `GridSearchCV`, `RandomizedSearchCV`, sklearn models)
  - Deep learning (search for MLP, CNN, BiLSTM, Transformer, PyTorch models)
  - LOPO CV implementation (search for `logo.split`, patient-level cross-validation)
- Extract all code from the notebook cells you identified
- **CRITICAL**: Understand how `supervised_mask` is created and used - find this in the notebook
- Implement LOPO CV strategy exactly as in the notebook
- Add parallelization for LOPO folds
- Add caching for trained models (if memory permits)
- Function signatures:
  ```python
  def train_traditional_ml_models(X: np.ndarray, y: np.ndarray, groups: np.ndarray, models: Dict, param_grids: Dict, cv_strategy: str, n_jobs: int, cache_dir: Path, use_cache: bool) -> pd.DataFrame
  def train_deep_learning_models(X_gene: np.ndarray, X_seq: np.ndarray, y: np.ndarray, groups: np.ndarray, config: Dict, cache_dir: Path, use_cache: bool) -> pd.DataFrame
  def lopo_cv_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
  ```

**3.7 Implement `pipeline/evaluation.py`**
- **PARSE THE NOTEBOOK** to find exact locations for:
  - Metrics computation (search for `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`)
  - Patient-level aggregation (search for patient-level result aggregation, groupby operations)
- Extract all code from the notebook cells you identified
- Function signatures:
  ```python
  def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]
  def aggregate_patient_level(predictions: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame
  ```

**3.8 Implement `scripts/run_full_pipeline.py`**
- Create main execution script with argparse
- Integrate all pipeline modules
- Add step-by-step execution with progress tracking
- Support for `--steps`, `--use-cache`, `--clear-cache`, `--parallel` flags
- Reference the example in section 5.3

**3.9 Implement `scripts/parallel_runner.py`** (Required)
- Create Ray-based distributed execution coordinator
- Create worker script for VM execution
- Implement auto-detection of local vs distributed mode
- Add fault tolerance and result aggregation
- Reference the example in section 5.4

**3.10 Create `config/pipeline_config.yaml`**
- Create YAML configuration file
- Include all configurable parameters
- Reference the example in section 5.2

#### Step 4: Testing Strategy
1. **Unit Tests**: Create `tests/test_*.py` files for each module
2. **Integration Test**: Run full pipeline and compare results to original notebook
   - **Validation Criteria**: Results must match within floating-point precision (rtol=1e-5, atol=1e-8)
   - **Metrics to Compare**: Accuracy, precision, recall, F1, ROC-AUC for each model and feature set
   - **Outputs to Validate**: All prediction arrays, probability arrays, aggregated patient-level results
3. **Cache Test**: Verify caching works correctly
   - Test cache hit/miss behavior
   - Test cache invalidation on parameter changes
   - Test cache loading produces identical results
4. **Parallelization Test**: Verify parallel execution produces same results
   - Compare sequential vs parallel results (must be identical)
   - Verify LOPO fold results match notebook exactly

#### Step 5: Validation Checklist
- [ ] **Notebook fully parsed** - All 7,400+ lines read and understood
- [ ] **All helper functions extracted** - Functions like `encode_gene_expression_patterns`, `physicochemical_features`, etc. extracted from notebook
- [ ] **All global variables handled** - Variables like `IS_KAGGLE`, `adata`, `supervised_mask` properly managed
- [ ] All functions from notebook extracted and modularized
- [ ] Caching implemented for expensive operations
- [ ] Parallelization implemented for LOPO folds
- [ ] Results match original notebook (within floating-point precision: rtol=1e-5, atol=1e-8)
- [ ] Runtime reduced to < 2 hours
- [ ] Code is well-documented with docstrings
- [ ] Configuration file is complete and tested
- [ ] Environment detection (Kaggle vs local) works correctly
- [ ] Memory cleanup functions included where needed

### Implementation Tips for AI Workers

1. **PARSE THE NOTEBOOK FIRST** (MANDATORY):
   - **YOU MUST** read the entire 7,400-line notebook before starting
   - **YOU MUST** extract all code, functions, and logic yourself
   - **YOU MUST** identify all global variables and their sources
   - **YOU MUST** find all helper functions defined in the notebook
   - Cell references in this PRD are approximate - find exact locations yourself

2. **Extract Code Systematically**: 
   - Go through notebook cells sequentially (you've already parsed it)
   - Identify reusable code blocks (you've already mapped them)
   - Note dependencies between cells (you've already documented them)

3. **Preserve Functionality**:
   - Keep all random seeds (random_state=42)
   - Maintain exact parameter values
   - Preserve data transformations exactly

4. **Add Caching Strategically**:
   - Cache expensive operations: PCA, SVD, UMAP, k-mer encoding, model training
   - Use cache keys that include function name, parameters, and data hash
   - Validate cache before use

5. **Implement Parallelization Carefully**:
   - Parallelize LOPO folds (7 folds can run in parallel)
   - Use multiprocessing.Pool for local single-machine execution (lowest overhead)
   - Use Ray for distributed multi-VM execution (best performance, auto-scaling)
   - Implement auto-detection: use multiprocessing if single machine, Ray if cluster detected
   - Ensure thread-safety for shared resources (Ray handles this automatically)

6. **Error Handling**:
   - Add try-except blocks for file operations
   - Validate inputs at function boundaries
   - Provide clear error messages

7. **Documentation**:
   - Add docstrings to all functions
   - Document parameters and return values
   - Add comments explaining complex logic

### Common Pitfalls to Avoid

1. **Don't change algorithm parameters** - Keep exact values from notebook
2. **Don't skip data validation** - Ensure data shapes match expectations
3. **Don't forget random seeds** - Reproducibility is critical (use `random_state=42` everywhere)
4. **Don't commit code** - User handles version control
5. **Don't optimize prematurely** - Focus on correctness first, then speed
6. **Don't assume cell references are exact** - Parse the notebook to find actual code locations
7. **Don't miss helper functions** - Functions like `encode_gene_expression_patterns`, `physicochemical_features`, `cleanup_after_clustering` are defined in the notebook - extract them
8. **Don't ignore global variables** - Variables like `IS_KAGGLE`, `adata`, `supervised_mask`, `metadata_df` are set in the notebook - handle them appropriately
9. **Don't skip environment detection** - The notebook has Kaggle vs local logic - preserve this functionality
10. **Don't forget memory cleanup** - The notebook has `cleanup_after_clustering` and garbage collection - include similar logic

### Expected Deliverables

After implementation, you should have:
- `pipeline/` directory with 7 Python files
- `scripts/` directory with 3 execution scripts
- `config/` directory with YAML configuration
- `tests/` directory with unit tests
- `cache/` directory (auto-generated)
- Updated `requirements.txt` with new dependencies
- `README.md` with usage instructions

---

## Appendix A: Example Usage

**⚠️ REMINDER FOR AI WORKERS**: Before implementing any of these examples, you MUST have already parsed the entire `Code/Kaggle Run 2.ipynb` notebook (all 7,400+ lines) to understand the codebase structure and extract all necessary functions and logic.

### Basic Usage (Full Pipeline)
```bash
python scripts/run_full_pipeline.py \
    --config config/pipeline_config.yaml \
    --use-cache \
    --parallel \
    --n-workers 8
```

### Run Specific Steps Only
```bash
python scripts/run_full_pipeline.py \
    --steps modeling evaluation \
    --use-cache \
    --parallel
```

### Clear Cache and Recompute
```bash
python scripts/run_full_pipeline.py \
    --clear-cache feature_engineering modeling \
    --steps feature_engineering modeling
```

### Distributed Execution with Ray (Multi-VM)

**Setup Ray Cluster:**
```bash
# On head node (VM 1)
ray start --head --port=6379 --dashboard-host=0.0.0.0

# On worker nodes (VM 2-N)
ray start --address=<head-node-ip>:6379
```

**Run Pipeline:**
```bash
# Connects to Ray cluster automatically
python scripts/parallel_runner.py \
    --ray-address=<head-node-ip>:6379 \
    --config config/pipeline_config.yaml
```

**Monitor Execution:**
```bash
# Access Ray Dashboard
open http://<head-node-ip>:8265
```

---

## Appendix B: Cache Key Examples

**⚠️ REMINDER FOR AI WORKERS**: These cache key examples assume you have already parsed the notebook and understand the function signatures and data structures used throughout the pipeline.

```python
# Example cache keys
"data_loading.load_gex_data.v1.0.abc123def456.pt"
"feature_engineering.encode_gene_expression.v1.0.xyz789.n_components=50.pt"
"modeling.train_traditional_ml_models.v1.0.data_hash.lopo_fold=0.model=xgboost.pt"
```

---

## Document Approval

- **Author**: [Your Name]
- **Reviewers**: [Reviewer Names]
- **Approved**: [ ] Yes [ ] No
- **Date**: ___________

---

**End of PRD**
