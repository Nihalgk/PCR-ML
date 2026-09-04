"""
Command-line interface (CLI) for PCR-ML pipeline operations.
"""

import argparse
from pathlib import Path
import sys
import pandas as pd

from .parser import load_all_runs
from .features import engineer_features
from .evaluation import evaluate_leave_one_out, evaluate_model_pipeline
from .c_code_gen import generate_arduino_header


def main():
    parser = argparse.ArgumentParser(
        description="PCR-ML: Embedded Machine Learning Thermal Lag Compensation & Validation Suite"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Run Leave-One-Run-Out Cross Validation across datasets")
    eval_parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")

    # Command: generate-c
    gen_parser = subparsers.add_parser("generate-c", help="Export trained model weights as C++ header for Arduino")
    gen_parser.add_argument("--output", "-o", type=str, default="firmware/pcr_sensorless_stage1/ml_model_weights.h", help="Output .h path")
    gen_parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")

    # Command: summary
    sum_parser = subparsers.add_parser("summary", help="Print detailed dataset and phase accuracy summary")
    sum_parser.add_argument("--data-dir", type=str, default=None, help="Path to data directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    repo_dir = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir) if args.data_dir else repo_dir

    print(f"[PCR-ML] Loading experimental PCR datasets from: {data_dir}")
    runs = load_all_runs(data_dir)
    if not runs:
        print(f"[ERROR] No PCR dataset files (*.txt) found in {data_dir}")
        sys.exit(1)

    print(f"[INFO] Loaded {len(runs)} experimental runs: {list(runs.keys())}\n")

    if args.command == "evaluate":
        print("=" * 70)
        print("LEAVE-ONE-RUN-OUT CROSS-VALIDATION BENCHMARK")
        print("=" * 70)
        df_cv = evaluate_leave_one_out(runs)
        print(df_cv.to_string(index=False))
        print("-" * 70)
        print(f"Mean Global Linear MAE   : {df_cv['Global Linear MAE (°C)'].mean():.2f} deg C")
        print(f"Mean Phase-Specific MAE  : {df_cv['Phase-Specific MAE (°C)'].mean():.2f} deg C")
        print(f"Mean Random Forest MAE   : {df_cv['Random Forest MAE (°C)'].mean():.2f} deg C")
        print("=" * 70)

    elif args.command == "generate-c":
        all_dfs = [engineer_features(df) for df in runs.values()]
        df_all = pd.concat(all_dfs, ignore_index=True)
        out_path = Path(args.output)
        generate_arduino_header(df_all, output_path=out_path)
        print(f"[SUCCESS] Exported Arduino C++ header to: {out_path.resolve()}")

    elif args.command == "summary":
        all_dfs = [engineer_features(df) for df in runs.values()]
        df_all = pd.concat(all_dfs, ignore_index=True)
        report = evaluate_model_pipeline(df_all)
        print("=" * 70)
        print("DETAILED PHASE & HOLD ACCURACY BREAKDOWN (Trained on All Data)")
        print("=" * 70)
        for section, metrics in report.items():
            print(f"--- {section.upper()} ---")
            print(f"  MAE       : {metrics['mae']:.2f} deg C")
            print(f"  RMSE      : {metrics['rmse']:.2f} deg C")
            print(f"  Max Error : {metrics['max_error']:.2f} deg C")
            print(f"  R^2 Score : {metrics['r2']:.4f}")
        print("=" * 70)


if __name__ == "__main__":
    main()
