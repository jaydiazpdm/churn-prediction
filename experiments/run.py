#!/usr/bin/env python3
"""
CLI entry point for experiment pipeline.

Usage:
    # Run single experiment
    python -m experiments.run configs/v3_features.yaml

    # Run multiple experiments
    python -m experiments.run configs/*.yaml --batch

    # List all experiments
    python -m experiments.run --list

    # Compare experiments
    python -m experiments.run --compare
"""

import argparse
import sys
from pathlib import Path

from .runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Churn prediction experiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m experiments.run configs/v3_features.yaml
  python -m experiments.run configs/baseline.yaml configs/v3_features.yaml
  python -m experiments.run --list
  python -m experiments.run --compare
        """,
    )

    parser.add_argument(
        "configs",
        nargs="*",
        help="Path(s) to YAML config file(s)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all past experiments",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all experiments (sorted by accuracy)",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop batch run if any experiment errors",
    )

    args = parser.parse_args()

    runner = ExperimentRunner()

    # List experiments
    if args.list:
        df = runner.list_experiments()
        if df.empty:
            print("No experiments found.")
        else:
            print(df.to_string(index=False))
        return 0

    # Compare experiments
    if args.compare:
        df = runner.list_experiments()
        if df.empty:
            print("No experiments found.")
        else:
            # Sort by accuracy descending
            if "accuracy" in df.columns:
                df = df.sort_values("accuracy", ascending=False)
            print("\nExperiment Comparison (sorted by accuracy):\n")
            print(df.to_string(index=False))
        return 0

    # Run experiments
    if not args.configs:
        parser.print_help()
        return 1

    # Resolve config paths relative to experiments/ directory
    experiments_dir = Path(__file__).parent

    results = []
    for config_path in args.configs:
        path = Path(config_path)

        # Try resolving path relative to experiments directory first
        if not path.is_absolute():
            experiments_relative = experiments_dir / path
            if experiments_relative.exists():
                path = experiments_relative
            elif not path.exists():
                print(f"Config not found: {config_path}")
                if args.stop_on_failure:
                    return 1
                continue

        if not path.exists():
            print(f"Config not found: {config_path}")
            if args.stop_on_failure:
                return 1
            continue

        try:
            print(f"\n{'=' * 60}")
            print(f"Running: {path.name}")
            print("=" * 60)

            result = runner.run_from_yaml(path)
            results.append(result)

            print(result.summary())

            if result.passed:
                print(f"\nArtifacts saved to: artifacts/{result.experiment_id}/")

        except Exception as e:
            print(f"ERROR: {e}")
            if args.stop_on_failure:
                return 1

    # Summary
    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("BATCH SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in results if r.passed)
        print(f"Total: {len(results)}, Passed: {passed}, Failed: {len(results) - passed}")

        print("\nResults:")
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            acc = r.metrics.get("test_accuracy", 0)
            print(f"  [{status}] {r.config.name}: {acc:.1%} accuracy")

    return 0


if __name__ == "__main__":
    sys.exit(main())
