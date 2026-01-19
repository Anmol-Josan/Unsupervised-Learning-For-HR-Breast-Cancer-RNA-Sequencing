#!/usr/bin/env python3
"""
Profiling script to identify performance bottlenecks.
"""

import argparse
import cProfile
import pstats
import sys
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_function(func, *args, **kwargs):
    """Profile a function and return stats."""
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()

    return result, profiler


def print_profiling_stats(profiler, top_n: int = 20):
    """Print profiling statistics."""
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)

    print("\n" + "=" * 80)
    print(f"TOP {top_n} FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 80)
    print(s.getvalue())


def profile_data_loading():
    """Profile data loading step."""
    from pipeline.data_loading import load_gex_data, load_tcr_data
    from pipeline.utils import CacheManager

    data_dir = Path("./Data/GSE300475_RAW")
    cache_manager = CacheManager("./cache", enabled=False)

    print("\nProfiling: Data Loading")
    print("-" * 80)

    _, profiler = profile_function(
        load_gex_data,
        data_dir,
        use_cache=False,
        cache_manager=None
    )

    print_profiling_stats(profiler, top_n=15)


def profile_preprocessing(adata):
    """Profile preprocessing step."""
    from pipeline.preprocessing import preprocess_pipeline

    print("\nProfiling: Preprocessing")
    print("-" * 80)

    _, profiler = profile_function(
        preprocess_pipeline,
        adata.copy(),
        min_genes=200,
        use_cache=False,
        cache_manager=None
    )

    print_profiling_stats(profiler, top_n=15)


def profile_feature_engineering(adata):
    """Profile feature engineering step."""
    from pipeline.feature_engineering import (
        encode_gene_expression_patterns,
        encode_tcr_sequences
    )

    print("\nProfiling: Gene Expression Encoding")
    print("-" * 80)

    _, profiler = profile_function(
        encode_gene_expression_patterns,
        adata.copy(),
        n_pca_components=50,
        compute_umap=False,
        use_cache=False,
        cache_manager=None
    )

    print_profiling_stats(profiler, top_n=15)


def profile_modeling(X, y, groups):
    """Profile modeling step."""
    from pipeline.modeling import train_traditional_ml
    from sklearn.linear_model import LogisticRegression

    models = {"LogisticRegression": LogisticRegression(random_state=42)}
    param_grids = {"LogisticRegression": {"C": [1.0]}}

    print("\nProfiling: Modeling")
    print("-" * 80)

    _, profiler = profile_function(
        train_traditional_ml,
        X, y, groups, "test",
        models=models,
        param_grids=param_grids,
        use_random_search=False,
        use_cache=False,
        cache_manager=None
    )

    print_profiling_stats(profiler, top_n=15)


def main():
    parser = argparse.ArgumentParser(description="Profile pipeline performance")
    parser.add_argument(
        "--step",
        choices=["all", "data_loading", "preprocessing", "feature_engineering", "modeling"],
        default="all",
        help="Pipeline step to profile"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to preprocessed AnnData file"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PIPELINE PROFILING")
    print("=" * 80)

    if args.step in ["all", "data_loading"]:
        try:
            profile_data_loading()
        except Exception as e:
            print(f"Data loading profiling failed: {e}")

    if args.step in ["all", "preprocessing", "feature_engineering", "modeling"]:
        # Need AnnData for these steps
        if args.data_path and args.data_path.exists():
            import scanpy as sc
            adata = sc.read_h5ad(args.data_path)

            if args.step in ["all", "preprocessing"]:
                profile_preprocessing(adata)

            if args.step in ["all", "feature_engineering"]:
                profile_feature_engineering(adata)

            if args.step in ["all", "modeling"]:
                # Need processed data for modeling
                from pipeline.preprocessing import preprocess_pipeline
                from pipeline.feature_engineering import encode_gene_expression_patterns, create_feature_sets
                from pipeline.modeling import create_supervised_mask, prepare_ml_data

                adata_processed = preprocess_pipeline(
                    adata.copy(),
                    min_genes=200,
                    n_top_genes=500,
                    use_cache=False
                )
                adata_processed = encode_gene_expression_patterns(
                    adata_processed,
                    n_pca_components=20,
                    use_cache=False
                )

                supervised_mask = create_supervised_mask(adata_processed)
                feature_sets = create_feature_sets(adata_processed, supervised_mask, use_cache=False)
                feature_sets, y, groups = prepare_ml_data(adata_processed, feature_sets, supervised_mask)

                X = feature_sets["basic"]
                profile_modeling(X, y, groups)
        else:
            print("Data path required for profiling preprocessing/feature_engineering/modeling")
            print("Provide --data-path argument")

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
