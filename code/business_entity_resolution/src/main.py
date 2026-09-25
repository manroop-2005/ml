"""
Business Entity Resolution - Unified CLI Entrypoint.
Commands:
  train     - Train GBDT matching model and optimize F_0.5 threshold
  predict   - Run candidate generation and match inference on test data
  validate  - Validate output TSVs using challenge submission validator
  run-all   - Run complete end-to-end pipeline (train -> predict -> validate)
"""

import sys
import argparse
import subprocess
from pathlib import Path

from .config import (
    TRAIN_DIR,
    TEST_DIR,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH
)
from .train import train_pipeline
from .predict import predict_pipeline


def validate_outputs(output_dir: str = DEFAULT_OUTPUT_DIR, test_dir: str = TEST_DIR):
    """Run utils/validate_submission.py on produced output files."""
    matching_path = Path(output_dir) / "matching_results.tsv"
    candidate_path = Path(output_dir) / "candidate_pairs.tsv"
    validator_path = Path(__file__).resolve().parent.parent.parent.parent / "utils" / "validate_submission.py"

    print("=" * 60)
    print("Running Submission Validator...")
    print(f"Validator: {validator_path}")
    print(f"Matching:  {matching_path}")
    print(f"Candidate: {candidate_path}")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir)
    ]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n>>> VALIDATION SUCCESSFUL: Submission files meet all challenge criteria! <<<")
    else:
        print("\n>>> VALIDATION FAILED: Please inspect errors above. <<<")
    return res.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # train command
    p_train = subparsers.add_parser("train", help="Train matching model")
    p_train.add_argument("--data-dir", default=TRAIN_DIR, help="Path to training data directory")
    p_train.add_argument("--sample-size", type=int, default=30000, help="Number of S1 entities to sample (0 = all)")
    p_train.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    p_train.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Output path for model artifact")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")

    # predict command
    p_pred = subparsers.add_parser("predict", help="Generate predictions on test set")
    p_pred.add_argument("--test-dir", default=TEST_DIR, help="Path to test directory")
    p_pred.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to model artifact")
    p_pred.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    p_pred.add_argument("--country", default=None, help="Filter to specific country")
    p_pred.add_argument("--quick-limit", type=int, default=0, help="Limit number of test S1 records (0 = all)")
    p_pred.add_argument("--batch-size", type=int, default=2000, help="Batch size")

    # validate command
    p_val = subparsers.add_parser("validate", help="Validate output submission files")
    p_val.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory containing TSVs")
    p_val.add_argument("--test-dir", default=TEST_DIR, help="Path to test dataset directory")

    # run-all command
    p_all = subparsers.add_parser("run-all", help="Train, predict, and validate end-to-end")
    p_all.add_argument("--sample-size", type=int, default=30000, help="Training sample size (0 = all)")
    p_all.add_argument("--quick-limit", type=int, default=0, help="Test set limit (0 = all)")

    args = parser.parse_args()

    if args.command == "train":
        train_pipeline(
            train_dir=args.data_dir,
            sample_size=args.sample_size,
            val_ratio=args.val_ratio,
            model_out=args.model_out,
            seed=args.seed
        )
    elif args.command == "predict":
        predict_pipeline(
            test_dir=args.test_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            country_filter=args.country,
            quick_limit=args.quick_limit,
            batch_size=args.batch_size
        )
    elif args.command == "validate":
        validate_outputs(args.output_dir, args.test_dir)
    elif args.command == "run-all":
        print("=== Step 1: Training Model ===")
        train_pipeline(sample_size=args.sample_size)
        print("\n=== Step 2: Running Inference on Test Set ===")
        predict_pipeline(quick_limit=args.quick_limit)
        print("\n=== Step 3: Validating Outputs ===")
        validate_outputs()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
