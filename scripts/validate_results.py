#!/usr/bin/env python3
"""
Validation script to compare pipeline results against notebook baseline.
Ensures reproducibility within floating-point precision.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from anndata import AnnData


class ResultValidator:
    """Validate pipeline results against baseline."""

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        """
        Initialize validator.

        Args:
            rtol: Relative tolerance for floating-point comparison
            atol: Absolute tolerance for floating-point comparison
        """
        self.rtol = rtol
        self.atol = atol
        self.validation_results = []

    def validate_adata(
        self,
        pipeline_adata: AnnData,
        notebook_adata: AnnData,
        check_obs: bool = True,
        check_var: bool = True,
        check_obsm: bool = True
    ) -> bool:
        """
        Validate AnnData objects match.

        Args:
            pipeline_adata: AnnData from pipeline
            notebook_adata: AnnData from notebook baseline
            check_obs: Whether to check obs DataFrame
            check_var: Whether to check var DataFrame
            check_obsm: Whether to check obsm matrices

        Returns:
            True if validation passes
        """
        print("\n" + "=" * 60)
        print("Validating AnnData Objects")
        print("=" * 60)

        all_valid = True

        # Check shapes
        if pipeline_adata.shape != notebook_adata.shape:
            print(f"❌ Shape mismatch: {pipeline_adata.shape} != {notebook_adata.shape}")
            all_valid = False
        else:
            print(f"✓ Shapes match: {pipeline_adata.shape}")

        # Check X matrix
        try:
            if hasattr(pipeline_adata.X, 'toarray'):
                X_pipeline = pipeline_adata.X.toarray()
            else:
                X_pipeline = pipeline_adata.X

            if hasattr(notebook_adata.X, 'toarray'):
                X_notebook = notebook_adata.X.toarray()
            else:
                X_notebook = notebook_adata.X

            if np.allclose(X_pipeline, X_notebook, rtol=self.rtol, atol=self.atol):
                print(f"✓ X matrix matches (rtol={self.rtol}, atol={self.atol})")
            else:
                max_diff = np.abs(X_pipeline - X_notebook).max()
                print(f"❌ X matrix mismatch (max diff: {max_diff})")
                all_valid = False
        except Exception as e:
            print(f"❌ X matrix comparison failed: {e}")
            all_valid = False

        # Check obs
        if check_obs:
            if set(pipeline_adata.obs.columns) == set(notebook_adata.obs.columns):
                print(f"✓ obs columns match: {len(pipeline_adata.obs.columns)} columns")
            else:
                missing = set(notebook_adata.obs.columns) - set(pipeline_adata.obs.columns)
                extra = set(pipeline_adata.obs.columns) - set(notebook_adata.obs.columns)
                print(f"❌ obs columns mismatch")
                if missing:
                    print(f"  Missing: {missing}")
                if extra:
                    print(f"  Extra: {extra}")
                all_valid = False

        # Check var
        if check_var:
            if set(pipeline_adata.var.columns) == set(notebook_adata.var.columns):
                print(f"✓ var columns match: {len(pipeline_adata.var.columns)} columns")
            else:
                print(f"❌ var columns mismatch")
                all_valid = False

        # Check obsm
        if check_obsm:
            pipeline_keys = set(pipeline_adata.obsm.keys())
            notebook_keys = set(notebook_adata.obsm.keys())

            if pipeline_keys == notebook_keys:
                print(f"✓ obsm keys match: {len(pipeline_keys)} keys")

                # Check each obsm matrix
                for key in pipeline_keys:
                    if np.allclose(
                        pipeline_adata.obsm[key],
                        notebook_adata.obsm[key],
                        rtol=self.rtol,
                        atol=self.atol
                    ):
                        print(f"  ✓ obsm['{key}'] matches")
                    else:
                        max_diff = np.abs(
                            pipeline_adata.obsm[key] - notebook_adata.obsm[key]
                        ).max()
                        print(f"  ❌ obsm['{key}'] mismatch (max diff: {max_diff})")
                        all_valid = False
            else:
                print(f"❌ obsm keys mismatch")
                print(f"  Missing: {notebook_keys - pipeline_keys}")
                print(f"  Extra: {pipeline_keys - notebook_keys}")
                all_valid = False

        return all_valid

    def validate_model_results(
        self,
        pipeline_results: pd.DataFrame,
        notebook_results: pd.DataFrame
    ) -> bool:
        """
        Validate model results match.

        Args:
            pipeline_results: Results DataFrame from pipeline
            notebook_results: Results DataFrame from notebook baseline

        Returns:
            True if validation passes
        """
        print("\n" + "=" * 60)
        print("Validating Model Results")
        print("=" * 60)

        all_valid = True

        # Check shapes
        if pipeline_results.shape != notebook_results.shape:
            print(f"❌ Shape mismatch: {pipeline_results.shape} != {notebook_results.shape}")
            all_valid = False
        else:
            print(f"✓ Shapes match: {pipeline_results.shape}")

        # Check key metrics
        metric_cols = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        for col in metric_cols:
            if col in pipeline_results.columns and col in notebook_results.columns:
                pipeline_vals = pipeline_results[col].values
                notebook_vals = notebook_results[col].values

                if np.allclose(pipeline_vals, notebook_vals, rtol=self.rtol, atol=self.atol, equal_nan=True):
                    print(f"✓ {col} matches")
                else:
                    max_diff = np.abs(pipeline_vals - notebook_vals).max()
                    print(f"❌ {col} mismatch (max diff: {max_diff})")
                    all_valid = False

        return all_valid

    def validate_predictions(
        self,
        pipeline_predictions: np.ndarray,
        notebook_predictions: np.ndarray,
        prediction_type: str = "labels"
    ) -> bool:
        """
        Validate predictions match.

        Args:
            pipeline_predictions: Predictions from pipeline
            notebook_predictions: Predictions from notebook baseline
            prediction_type: Type of predictions ("labels" or "probabilities")

        Returns:
            True if validation passes
        """
        print("\n" + "=" * 60)
        print(f"Validating {prediction_type.title()}")
        print("=" * 60)

        all_valid = True

        if pipeline_predictions.shape != notebook_predictions.shape:
            print(f"❌ Shape mismatch: {pipeline_predictions.shape} != {notebook_predictions.shape}")
            return False

        if prediction_type == "labels":
            # Exact match for labels
            if np.array_equal(pipeline_predictions, notebook_predictions):
                print(f"✓ {prediction_type} match exactly")
            else:
                n_diff = np.sum(pipeline_predictions != notebook_predictions)
                print(f"❌ {prediction_type} mismatch: {n_diff}/{len(pipeline_predictions)} different")
                all_valid = False
        else:
            # Approximate match for probabilities
            if np.allclose(pipeline_predictions, notebook_predictions, rtol=self.rtol, atol=self.atol):
                print(f"✓ {prediction_type} match (rtol={self.rtol}, atol={self.atol})")
            else:
                max_diff = np.abs(pipeline_predictions - notebook_predictions).max()
                mean_diff = np.abs(pipeline_predictions - notebook_predictions).mean()
                print(f"❌ {prediction_type} mismatch (max: {max_diff}, mean: {mean_diff})")
                all_valid = False

        return all_valid

    def generate_report(self, output_path: Path):
        """Generate validation report."""
        if not self.validation_results:
            print("No validation results to report.")
            return

        report = []
        report.append("=" * 80)
        report.append("VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Tolerance: rtol={self.rtol}, atol={self.atol}")
        report.append("")

        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r['passed'])

        report.append(f"Tests: {passed_tests}/{total_tests} passed")
        report.append("")

        for result in self.validation_results:
            status = "✓ PASS" if result['passed'] else "❌ FAIL"
            report.append(f"{status}: {result['test_name']}")
            if 'details' in result:
                report.append(f"  {result['details']}")

        report.append("=" * 80)

        report_text = "\n".join(report)
        print("\n" + report_text)

        # Save to file
        with open(output_path, 'w') as f:
            f.write(report_text)

        print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate pipeline results against notebook baseline")
    parser.add_argument(
        "--pipeline-adata",
        type=Path,
        help="Path to pipeline AnnData file"
    )
    parser.add_argument(
        "--notebook-adata",
        type=Path,
        help="Path to notebook baseline AnnData file"
    )
    parser.add_argument(
        "--pipeline-results",
        type=Path,
        help="Path to pipeline results CSV"
    )
    parser.add_argument(
        "--notebook-results",
        type=Path,
        help="Path to notebook results CSV"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for comparison (default: 1e-5)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for comparison (default: 1e-8)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="./validation_report.txt",
        help="Output path for validation report"
    )

    args = parser.parse_args()

    validator = ResultValidator(rtol=args.rtol, atol=args.atol)

    # Validate AnnData if provided
    if args.pipeline_adata and args.notebook_adata:
        if args.pipeline_adata.exists() and args.notebook_adata.exists():
            print("Loading AnnData objects...")
            import scanpy as sc
            pipeline_adata = sc.read_h5ad(args.pipeline_adata)
            notebook_adata = sc.read_h5ad(args.notebook_adata)

            adata_valid = validator.validate_adata(pipeline_adata, notebook_adata)
            validator.validation_results.append({
                'test_name': 'AnnData Validation',
                'passed': adata_valid
            })

    # Validate results if provided
    if args.pipeline_results and args.notebook_results:
        if args.pipeline_results.exists() and args.notebook_results.exists():
            print("Loading results...")
            pipeline_results = pd.read_csv(args.pipeline_results)
            notebook_results = pd.read_csv(args.notebook_results)

            results_valid = validator.validate_model_results(pipeline_results, notebook_results)
            validator.validation_results.append({
                'test_name': 'Model Results Validation',
                'passed': results_valid
            })

    # Generate report
    validator.generate_report(args.output)


if __name__ == "__main__":
    main()
