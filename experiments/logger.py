"""
Experiment logging for churn prediction pipeline.

Writes JSON logs for all experiments (pass or fail).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .runner import ExperimentResult
    from .config import ExperimentConfig


class ExperimentLogger:
    """Structured JSON logging for experiments."""

    def __init__(self, logs_dir: Path):
        """
        Initialize logger.

        Args:
            logs_dir: Directory to write log files
        """
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def log_experiment(self, result: "ExperimentResult") -> Path:
        """
        Log experiment result to JSON file.

        Args:
            result: ExperimentResult from runner

        Returns:
            Path to log file
        """
        log_entry = {
            "experiment_id": result.experiment_id,
            "timestamp": result.timestamp.isoformat(),
            "duration_seconds": result.duration_seconds,
            "config": {
                "name": result.config.name,
                "description": result.config.description,
                "engineered_features": result.config.engineered_features,
                "weights": result.config.weights,
                "optimize_metric": result.config.optimize_metric,
                "min_recall": result.config.min_recall,
                "min_accuracy": result.config.min_accuracy,
            },
            "results": {
                "threshold": result.threshold,
                "metrics": result.metrics,
            },
            "passed": result.passed,
            "status": "PASS" if result.passed else "FAIL",
        }

        log_path = self.logs_dir / f"{result.experiment_id}.json"
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)

        return log_path

    def log_failure(
        self,
        experiment_id: str,
        config: "ExperimentConfig",
        error: str,
    ) -> Path:
        """
        Log failed/errored experiment.

        Args:
            experiment_id: Unique experiment ID
            config: ExperimentConfig used
            error: Error message

        Returns:
            Path to log file
        """
        log_entry = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "name": config.name,
                "description": config.description,
            },
            "status": "ERROR",
            "error": error,
        }

        log_path = self.logs_dir / f"{experiment_id}.json"
        with open(log_path, "w") as f:
            json.dump(log_entry, f, indent=2)

        return log_path

    def get_all_logs(self) -> list[dict]:
        """
        Load all experiment logs.

        Returns:
            List of log dictionaries, sorted by timestamp
        """
        logs = []
        for log_file in sorted(self.logs_dir.glob("exp_*.json")):
            with open(log_file) as f:
                logs.append(json.load(f))
        return logs

    def get_passing_experiments(self) -> list[dict]:
        """Get only passing experiments."""
        return [log for log in self.get_all_logs() if log.get("passed")]

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get summary of all experiments as DataFrame.

        Returns:
            DataFrame with experiment summaries
        """
        logs = self.get_all_logs()
        if not logs:
            return pd.DataFrame()

        summary = []
        for log in logs:
            entry = {
                "experiment_id": log["experiment_id"],
                "name": log["config"]["name"],
                "timestamp": log["timestamp"],
                "status": log["status"],
            }

            # Add metrics if available
            if "results" in log:
                entry["threshold"] = log["results"].get("threshold")
                metrics = log["results"].get("metrics", {})
                for key in ["accuracy", "precision", "recall", "f1"]:
                    entry[key] = metrics.get(f"test_{key}")

            summary.append(entry)

        df = pd.DataFrame(summary)
        return df.sort_values("timestamp", ascending=False)
