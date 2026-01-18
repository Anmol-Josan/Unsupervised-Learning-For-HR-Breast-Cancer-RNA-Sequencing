#!/usr/bin/env python3
"""
Main pipeline execution script with caching and parallelization support.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.clustering import cluster_pipeline
from pipeline.data_loading import (
    add_metadata_to_adata,
    download_geo_data,
    load_gex_data,
    load_metadata,
    load_tcr_data,
    merge_gex_tcr,
)
from pipeline.evaluation import evaluate_results, print_evaluation_summary
from pipeline.feature_engineering import (
    create_feature_sets,
    encode_gene_expression_patterns,
    encode_tcr_sequences,
)
from pipeline.modeling import (
    create_supervised_mask,
    prepare_ml_data,
    train_all_feature_sets,
)
from pipeline.preprocessing import preprocess_pipeline
from pipeline.utils import CacheManager, ConfigManager, detect_environment, setup_random_seeds


def main():
    parser = argparse.ArgumentParser(description="Run scRNA-seq + TCR analysis pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default="config/pipeline_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["all", "data_loading", "preprocessing", "feature_engineering", "clustering", "modeling", "evaluation"],
        default=["all"],
        help="Pipeline steps to run"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use cached results"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--clear-cache",
        nargs="+",
        choices=["all", "data_loading", "preprocessing", "feature_engineering", "clustering", "modeling"],
        help="Clear specific cache sections before running"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    print("=" * 80)
    print("SINGLE-CELL RNA-SEQ + TCR ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Configuration: {args.config}")

    config = ConfigManager(args.config)

    # Detect environment
    env = detect_environment()
    print(f"Environment: {env['platform']}")

    # Setup random seeds
    seed = config.get('pipeline.random_seed', 42)
    setup_random_seeds(seed)
    print(f"Random seed: {seed}")

    # Setup cache manager
    use_cache = args.use_cache and not args.no_cache and config.get('cache.enabled', True)
    cache_dir = Path(config.get('cache.base_dir', './cache'))
    cache_manager = CacheManager(cache_dir, enabled=use_cache)
    print(f"Caching: {'enabled' if use_cache else 'disabled'}")

    # Clear cache if requested
    if args.clear_cache:
        if 'all' in args.clear_cache:
            cache_manager.clear_all()
        else:
            for section in args.clear_cache:
                cache_manager.clear_section(section)

    # Determine steps to run
    steps_to_run = args.steps
    if "all" in steps_to_run:
        steps_to_run = ["data_loading", "preprocessing", "feature_engineering", "clustering", "modeling", "evaluation"]

    # Setup directories
    data_dir = Path(env['data_dir'] if 'data_dir' in env else config.get('pipeline.data_dir', './Data'))
    output_dir = args.output_dir if args.output_dir else Path(env.get('output_dir', config.get('pipeline.output_dir', './Output')))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    print("\n" + "=" * 80)
    print(f"PIPELINE STEPS: {', '.join(steps_to_run)}")
    print("=" * 80 + "\n")

    # ========================================
    # STEP 1: DATA LOADING
    # ========================================
    if "data_loading" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 1: DATA LOADING")
        print("=" * 80)

        # Download data if needed
        geo_accession = config.get('pipeline.steps.data_loading.geo_accession', 'GSE300475')
        extract_dir = download_geo_data(data_dir, geo_accession, use_cache=use_cache, cache_manager=cache_manager)

        # Load GEX data
        adata = load_gex_data(extract_dir, use_cache=use_cache, cache_manager=cache_manager)

        # Load TCR data
        tcr_df = load_tcr_data(extract_dir, use_cache=use_cache, cache_manager=cache_manager)

        # Merge GEX and TCR
        adata = merge_gex_tcr(adata, tcr_df, use_cache=use_cache, cache_manager=cache_manager)

        # Load metadata if available
        metadata_path = data_dir / "metadata.xlsx"
        if metadata_path.exists():
            metadata_df = load_metadata(metadata_path, use_cache=use_cache, cache_manager=cache_manager)
            adata = add_metadata_to_adata(adata, metadata_df)

        # Save checkpoint
        checkpoint_path = output_dir / "adata_raw.h5ad"
        adata.write_h5ad(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # ========================================
    # STEP 2: PREPROCESSING
    # ========================================
    if "preprocessing" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 2: PREPROCESSING")
        print("=" * 80)

        # Load from checkpoint if not already in memory
        if 'adata' not in locals():
            checkpoint_path = output_dir / "adata_raw.h5ad"
            if checkpoint_path.exists():
                import scanpy as sc
                adata = sc.read_h5ad(checkpoint_path)
            else:
                raise RuntimeError("No data found. Run data_loading step first.")

        # Preprocess
        adata = preprocess_pipeline(
            adata,
            min_genes=config.get('pipeline.steps.preprocessing.min_genes', 200),
            min_counts=config.get('pipeline.steps.preprocessing.min_counts', 500),
            max_pct_mt=config.get('pipeline.steps.preprocessing.max_pct_mt', 20.0),
            min_cells=config.get('pipeline.steps.preprocessing.min_cells', 3),
            target_sum=config.get('pipeline.steps.preprocessing.target_sum', 10000),
            n_top_genes=config.get('pipeline.steps.preprocessing.n_top_genes', 2000),
            hvg_flavor=config.get('pipeline.steps.preprocessing.hvg_flavor', 'seurat'),
            scale_max_value=config.get('pipeline.steps.preprocessing.scale_max_value', 10.0),
            filter_tcr_cells=config.get('pipeline.steps.preprocessing.filter_tcr_cells', True),
            use_cache=use_cache,
            cache_manager=cache_manager
        )

        # Save checkpoint
        checkpoint_path = output_dir / "adata_preprocessed.h5ad"
        adata.write_h5ad(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # ========================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================
    if "feature_engineering" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE ENGINEERING")
        print("=" * 80)

        # Load from checkpoint if not already in memory
        if 'adata' not in locals():
            checkpoint_path = output_dir / "adata_preprocessed.h5ad"
            if checkpoint_path.exists():
                import scanpy as sc
                adata = sc.read_h5ad(checkpoint_path)
            else:
                raise RuntimeError("No preprocessed data found. Run preprocessing step first.")

        # Encode gene expression
        adata = encode_gene_expression_patterns(
            adata,
            n_pca_components=config.get('pipeline.steps.feature_engineering.gene_encoding.n_pca_components', 50),
            n_svd_components=config.get('pipeline.steps.feature_engineering.gene_encoding.n_svd_components', 50),
            compute_umap=config.get('pipeline.steps.feature_engineering.gene_encoding.compute_umap', False),
            n_umap_components=config.get('pipeline.steps.feature_engineering.gene_encoding.n_umap_components', 20),
            use_cache=use_cache,
            cache_manager=cache_manager
        )

        # Encode TCR sequences
        adata = encode_tcr_sequences(
            adata,
            kmer_k=config.get('pipeline.steps.feature_engineering.tcr_encoding.kmer_k', 3),
            n_svd_components=config.get('pipeline.steps.feature_engineering.tcr_encoding.n_svd_components', 200),
            max_onehot_length=config.get('pipeline.steps.feature_engineering.tcr_encoding.max_onehot_length', 50),
            use_cache=use_cache,
            cache_manager=cache_manager
        )

        # Save checkpoint
        checkpoint_path = output_dir / "adata_features.h5ad"
        adata.write_h5ad(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # ========================================
    # STEP 4: CLUSTERING
    # ========================================
    if "clustering" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 4: CLUSTERING")
        print("=" * 80)

        # Load from checkpoint if not already in memory
        if 'adata' not in locals():
            checkpoint_path = output_dir / "adata_features.h5ad"
            if checkpoint_path.exists():
                import scanpy as sc
                adata = sc.read_h5ad(checkpoint_path)
            else:
                raise RuntimeError("No feature data found. Run feature_engineering step first.")

        # Cluster
        adata = cluster_pipeline(
            adata,
            n_neighbors=config.get('pipeline.steps.clustering.n_neighbors', 15),
            n_pcs=config.get('pipeline.steps.clustering.n_pcs', 50),
            leiden_resolutions=config.get('pipeline.steps.clustering.leiden_resolutions', [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5]),
            target_clusters=config.get('pipeline.steps.clustering.target_clusters', 7),
            kmeans_n_clusters=config.get('pipeline.steps.clustering.kmeans_n_clusters', 6),
            use_cache=use_cache,
            cache_manager=cache_manager
        )

        # Save checkpoint
        checkpoint_path = output_dir / "adata_clustered.h5ad"
        adata.write_h5ad(checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # ========================================
    # STEP 5: MODELING
    # ========================================
    if "modeling" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 5: MODELING")
        print("=" * 80)

        # Load from checkpoint if not already in memory
        if 'adata' not in locals():
            checkpoint_path = output_dir / "adata_features.h5ad"
            if checkpoint_path.exists():
                import scanpy as sc
                adata = sc.read_h5ad(checkpoint_path)
            else:
                raise RuntimeError("No feature data found. Run feature_engineering step first.")

        # Create supervised mask
        supervised_mask = create_supervised_mask(adata, response_col=config.get('pipeline.steps.modeling.response_col', 'response'))

        # Create feature sets
        feature_sets = create_feature_sets(adata, supervised_mask, use_cache=use_cache, cache_manager=cache_manager)

        # Prepare ML data
        feature_sets, y, groups = prepare_ml_data(
            adata,
            feature_sets,
            supervised_mask,
            response_col=config.get('pipeline.steps.modeling.response_col', 'response'),
            patient_col=config.get('pipeline.steps.modeling.patient_col', 'patient_id')
        )

        # Train models
        results_df = train_all_feature_sets(
            feature_sets,
            y,
            groups,
            use_random_search=config.get('pipeline.steps.modeling.traditional_ml.use_random_search', True),
            n_iter=config.get('pipeline.steps.modeling.traditional_ml.n_iter_search', 15),
            use_cache=use_cache,
            cache_manager=cache_manager
        )

        # Save results
        results_path = output_dir / "modeling_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"Saved results: {results_path}")

    # ========================================
    # STEP 6: EVALUATION
    # ========================================
    if "evaluation" in steps_to_run:
        print("\n" + "=" * 80)
        print("STEP 6: EVALUATION")
        print("=" * 80)

        # Load results if not already in memory
        if 'results_df' not in locals():
            results_path = output_dir / "modeling_results.csv"
            if results_path.exists():
                import pandas as pd
                results_df = pd.read_csv(results_path)
            else:
                raise RuntimeError("No modeling results found. Run modeling step first.")

        # Evaluate
        eval_df = evaluate_results(results_df)

        # Print summary
        print_evaluation_summary(eval_df, top_n=10)

        # Save evaluation
        eval_path = output_dir / "evaluation_results.csv"
        eval_df.to_csv(eval_path, index=False)
        print(f"\nSaved evaluation: {eval_path}")

    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")

    # Show cache info
    if use_cache:
        cache_info = cache_manager.get_cache_info()
        print(f"\nCache usage: {cache_info['total_size_mb']:.1f} MB")
        for section, info in cache_info['sections'].items():
            print(f"  {section}: {info['size_mb']:.1f} MB ({info['num_files']} files)")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
