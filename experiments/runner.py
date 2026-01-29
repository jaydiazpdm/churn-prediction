"""
Experiment runner for churn prediction pipeline.

Single entry point for running experiments.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid

import pandas as pd

from .config import ExperimentConfig
from .evaluator import Evaluator
from .logger import ExperimentLogger
from .artifacts import ArtifactManager


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    experiment_id: str
    config: ExperimentConfig
    metrics: dict
    threshold: float
    passed: bool
    timestamp: datetime
    duration_seconds: float

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        test_acc = self.metrics.get("test_accuracy", 0)
        test_rec = self.metrics.get("test_recall", 0)
        test_f1 = self.metrics.get("test_f1", 0)

        return (
            f"[{self.experiment_id}] {self.config.name} - {status}\n"
            f"  Accuracy: {test_acc:.1%}\n"
            f"  Recall:   {test_rec:.1%}\n"
            f"  F1:       {test_f1:.3f}\n"
            f"  Threshold: {self.threshold:.0f}"
        )


class ExperimentRunner:
    """
    Single entry point for running experiments.

    Usage:
        runner = ExperimentRunner()

        # From YAML config
        result = runner.run_from_yaml("configs/v3_features.yaml")

        # From ExperimentConfig object
        config = ExperimentConfig(name="custom", ...)
        result = runner.run(config)

        # Batch run
        results = runner.run_batch(["configs/v1.yaml", "configs/v2.yaml"])
    """

    def __init__(
        self,
        base_path: Optional[Path] = None,
        logs_dir: str = "logs",
        artifacts_dir: str = "artifacts",
    ):
        """
        Initialize runner.

        Args:
            base_path: Base path for experiments (default: this file's parent)
            logs_dir: Subdirectory for logs
            artifacts_dir: Subdirectory for artifacts
        """
        self.base_path = base_path or Path(__file__).parent
        self.logs_dir = self.base_path / logs_dir
        self.artifacts_dir = self.base_path / artifacts_dir

        self.logger = ExperimentLogger(self.logs_dir)
        self.artifact_manager = ArtifactManager(self.artifacts_dir)
        self.evaluator = Evaluator()

    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID: exp_YYYYMMDD_XXXX"""
        date_str = datetime.now().strftime("%Y%m%d")
        short_uuid = uuid.uuid4().hex[:4]
        return f"exp_{date_str}_{short_uuid}"

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            config: ExperimentConfig to run

        Returns:
            ExperimentResult with metrics and pass/fail status
        """
        experiment_id = self.generate_experiment_id()
        start_time = datetime.now()

        try:
            # Load data
            train_df = pd.read_csv(self.base_path / config.train_path)
            val_df = pd.read_csv(self.base_path / config.val_path)
            test_df = pd.read_csv(self.base_path / config.test_path)

            # Score datasets
            train_scored = self.evaluator.score_dataset(train_df, config)
            val_scored = self.evaluator.score_dataset(val_df, config)
            test_scored = self.evaluator.score_dataset(test_df, config)

            # Find optimal threshold on validation set
            threshold, val_sweep = self.evaluator.find_optimal_threshold(
                val_scored, config
            )

            # Calculate final metrics on all splits
            train_metrics = self.evaluator.calculate_metrics(train_scored, threshold)
            val_metrics = self.evaluator.calculate_metrics(val_scored, threshold)
            test_metrics = self.evaluator.calculate_metrics(test_scored, threshold)

            # Flatten metrics with split prefixes
            metrics = {}
            for split, split_metrics in [
                ("train", train_metrics),
                ("val", val_metrics),
                ("test", test_metrics),
            ]:
                for metric_name, value in split_metrics.items():
                    metrics[f"{split}_{metric_name}"] = value

            # Determine pass/fail
            passed = self._evaluate_pass_fail(metrics, config)

            duration = (datetime.now() - start_time).total_seconds()

            result = ExperimentResult(
                experiment_id=experiment_id,
                config=config,
                metrics=metrics,
                threshold=threshold,
                passed=passed,
                timestamp=start_time,
                duration_seconds=duration,
            )

            # Always log
            self.logger.log_experiment(result)

            # Save full artifacts only if passed
            if passed:
                self.artifact_manager.save_artifacts(
                    result,
                    threshold_sweep=val_sweep,
                    test_predictions=test_scored,
                )

            return result

        except Exception as e:
            # Log failure
            self.logger.log_failure(experiment_id, config, str(e))
            raise

    def run_from_yaml(self, config_path: str | Path) -> ExperimentResult:
        """
        Load config from YAML and run.

        Args:
            config_path: Path to YAML config (relative to base_path or absolute)

        Returns:
            ExperimentResult
        """
        path = Path(config_path)
        if not path.is_absolute():
            path = self.base_path / path
        config = ExperimentConfig.from_yaml(path)
        return self.run(config)

    def run_batch(
        self,
        config_paths: list[str | Path],
        stop_on_failure: bool = False,
    ) -> list[ExperimentResult]:
        """
        Run multiple experiments in sequence.

        Args:
            config_paths: List of paths to YAML configs
            stop_on_failure: Whether to stop if an experiment errors

        Returns:
            List of ExperimentResults
        """
        results = []
        for path in config_paths:
            try:
                result = self.run_from_yaml(path)
                results.append(result)
                print(result.summary())
                print()
            except Exception as e:
                print(f"ERROR: {path} - {e}")
                if stop_on_failure:
                    raise
        return results

    def list_experiments(self) -> pd.DataFrame:
        """
        Get summary of all past experiments.

        Returns:
            DataFrame with experiment history
        """
        return self.logger.get_summary_dataframe()

    def _evaluate_pass_fail(
        self,
        metrics: dict,
        config: ExperimentConfig,
    ) -> bool:
        """Evaluate if experiment passes kill criteria."""
        # Check minimum accuracy (kill criteria)
        if metrics["test_accuracy"] < config.min_accuracy:
            return False

        # Check minimum F1 if specified
        if config.min_f1 and metrics["test_f1"] < config.min_f1:
            return False

        return True
