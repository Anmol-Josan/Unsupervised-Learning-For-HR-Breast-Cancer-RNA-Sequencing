#!/usr/bin/env python3
"""
Benchmark script to measure pipeline performance and compare against baseline.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from pipeline.utils import CacheManager, ConfigManager, setup_random_seeds


class PipelineBenchmark:
    """Benchmark pipeline performance."""

    def __init__(self, config_path: Path):
        """Initialize benchmark."""
        self.config = ConfigManager(config_path)
        self.results = []
        setup_random_seeds(42)

    def time_step(self, step_name: str, func, *args, **kwargs):
        """Time a pipeline step."""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {step_name}")
        print(f"{'='*60}")

        start_time = time.time()
        start_memory = self._get_memory_usage()

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = self._get_memory_usage()

        elapsed = end_time - start_time
        memory_delta = end_memory - start_memory

        self.results.append({
            "step": step_name,
            "duration_seconds": elapsed,
            "duration_minutes": elapsed / 60,
            "memory_mb": memory_delta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

        print(f"Duration: {elapsed:.2f}s ({elapsed/60:.2f}min)")
        print(f"Memory: {memory_delta:.2f} MB")

        return result

    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0

    def run_full_benchmark(self, use_cache: bool = False):
        """Run full pipeline benchmark."""
        print("\n" + "="*80)
        print("PIPELINE PERFORMANCE BENCHMARK")
        print("="*80)
        print(f"Cache enabled: {use_cache}")
        print(f"Configuration: {self.config.config_path}")

        cache_dir = Path(self.config.get("cache.base_dir", "./cache"))
        cache_manager = CacheManager(cache_dir, enabled=use_cache)

        data_dir = Path(self.config.get("pipeline.data_dir", "./Data"))
        output_dir = Path("./Benchmark_Output")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Data Loading
            from pipeline.data_loading import load_gex_data, load_tcr_data, merge_gex_tcr

            # Assume data already downloaded for benchmarking
            extract_dir = data_dir / "GSE300475_RAW"
            if not extract_dir.exists():
                print(f"Warning: Data directory {extract_dir} not found. Skipping data loading.")
                adata = None
            else:
                adata = self.time_step(
                    "1. Load GEX Data",
                    load_gex_data,
                    extract_dir,
                    use_cache=use_cache,
                    cache_manager=cache_manager
                )

                tcr_df = self.time_step(
                    "2. Load TCR Data",
                    load_tcr_data,
                    extract_dir,
                    use_cache=use_cache,
                    cache_manager=cache_manager
                )

                adata = self.time_step(
                    "3. Merge GEX + TCR",
                    merge_gex_tcr,
                    adata,
                    tcr_df,
                    use_cache=use_cache,
                    cache_manager=cache_manager
                )

            if adata is None:
                print("Skipping remaining steps due to missing data.")
                return self.save_results(output_dir)

            # Step 2: Preprocessing
            from pipeline.preprocessing import preprocess_pipeline

            adata = self.time_step(
                "4. Preprocessing",
                preprocess_pipeline,
                adata,
                min_genes=200,
                min_counts=500,
                n_top_genes=2000,
                use_cache=use_cache,
                cache_manager=cache_manager
            )

            # Step 3: Feature Engineering
            from pipeline.feature_engineering import (
                encode_gene_expression_patterns,
                encode_tcr_sequences,
                create_feature_sets
            )

            adata = self.time_step(
                "5. Gene Expression Encoding",
                encode_gene_expression_patterns,
                adata,
                n_pca_components=50,
                n_svd_components=50,
                compute_umap=False,
                use_cache=use_cache,
                cache_manager=cache_manager
            )

            adata = self.time_step(
                "6. TCR Sequence Encoding",
                encode_tcr_sequences,
                adata,
                kmer_k=3,
                n_svd_components=200,
                use_cache=use_cache,
                cache_manager=cache_manager
            )

            from pipeline.modeling import create_supervised_mask
            supervised_mask = create_supervised_mask(adata)

            feature_sets = self.time_step(
                "7. Create Feature Sets",
                create_feature_sets,
                adata,
                supervised_mask,
                use_cache=use_cache,
                cache_manager=cache_manager
            )

            # Step 4: Clustering (optional for benchmarking)
            from pipeline.clustering import cluster_pipeline

            adata = self.time_step(
                "8. Clustering",
                cluster_pipeline,
                adata,
                leiden_resolutions=[0.5],
                use_cache=use_cache,
                cache_manager=cache_manager
            )

            # Step 5: Modeling (limited for benchmarking)
            from pipeline.modeling import prepare_ml_data, train_traditional_ml
            from sklearn.linear_model import LogisticRegression

            feature_sets_ml, y, groups = prepare_ml_data(adata, feature_sets, supervised_mask)

            # Use only basic feature set and LogisticRegression for speed
            models = {"LogisticRegression": LogisticRegression(random_state=42)}
            param_grids = {"LogisticRegression": {"C": [1.0]}}

            results_df = self.time_step(
                "9. Modeling (Basic + LogReg)",
                train_traditional_ml,
                feature_sets_ml["basic"],
                y,
                groups,
                "basic",
                models=models,
                param_grids=param_grids,
                use_random_search=False,
                use_cache=use_cache,
                cache_manager=cache_manager
            )

            print("\n" + "="*80)
            print("BENCHMARK COMPLETE")
            print("="*80)

        except Exception as e:
            print(f"\nBenchmark failed with error: {e}")
            import traceback
            traceback.print_exc()

        return self.save_results(output_dir)

    def save_results(self, output_dir: Path):
        """Save benchmark results."""
        if not self.results:
            print("No benchmark results to save.")
            return None

        results_df = pd.DataFrame(self.results)

        # Calculate total time
        total_time = results_df["duration_seconds"].sum()
        total_memory = results_df["memory_mb"].sum()

        # Add summary row
        summary = {
            "step": "TOTAL",
            "duration_seconds": total_time,
            "duration_minutes": total_time / 60,
            "memory_mb": total_memory,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

        # Save to CSV
        output_path = output_dir / f"benchmark_results_{int(time.time())}.csv"
        results_df.to_csv(output_path, index=False)

        # Print summary
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("\n" + "="*80)
        print(f"Total Duration: {total_time:.2f}s ({total_time/60:.2f}min / {total_time/3600:.2f}hr)")
        print(f"Peak Memory: {total_memory:.2f} MB")
        print(f"Results saved to: {output_path}")
        print("="*80)

        return results_df

    def compare_with_baseline(self, baseline_path: Path):
        """Compare current benchmark with baseline."""
        if not baseline_path.exists():
            print(f"Baseline file not found: {baseline_path}")
            return

        baseline_df = pd.read_csv(baseline_path)
        current_df = pd.DataFrame(self.results)

        print("\n" + "="*80)
        print("COMPARISON WITH BASELINE")
        print("="*80)

        for _, row in current_df.iterrows():
            step = row["step"]
            baseline_row = baseline_df[baseline_df["step"] == step]

            if not baseline_row.empty:
                baseline_time = baseline_row.iloc[0]["duration_seconds"]
                current_time = row["duration_seconds"]
                speedup = baseline_time / current_time if current_time > 0 else 0

                print(f"\n{step}:")
                print(f"  Baseline: {baseline_time:.2f}s")
                print(f"  Current:  {current_time:.2f}s")
                print(f"  Speedup:  {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline performance")
    parser.add_argument(
        "--config",
        type=Path,
        default="config/pipeline_config.yaml",
        help="Configuration file"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable caching during benchmark"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching during benchmark"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline benchmark results for comparison"
    )

    args = parser.parse_args()

    use_cache = args.use_cache and not args.no_cache

    # Run benchmark
    benchmark = PipelineBenchmark(args.config)
    results = benchmark.run_full_benchmark(use_cache=use_cache)

    # Compare with baseline if provided
    if args.baseline and results is not None:
        benchmark.compare_with_baseline(args.baseline)


if __name__ == "__main__":
    main()
