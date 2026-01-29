"""
Artifact management for churn prediction experiments.

Saves plots and CSVs for PASSING experiments only.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

if TYPE_CHECKING:
    from .runner import ExperimentResult


class ArtifactManager:
    """Manages saving experiment artifacts."""

    def __init__(self, artifacts_dir: Path):
        """
        Initialize artifact manager.

        Args:
            artifacts_dir: Base directory for artifacts
        """
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_artifacts(
        self,
        result: "ExperimentResult",
        threshold_sweep: pd.DataFrame,
        test_predictions: pd.DataFrame,
    ) -> Path:
        """
        Save full artifacts for a passing experiment.

        Args:
            result: ExperimentResult from runner
            threshold_sweep: DataFrame from threshold optimization
            test_predictions: Test set with predictions

        Returns:
            Path to experiment artifacts directory
        """
        exp_dir = self.artifacts_dir / result.experiment_id
        exp_dir.mkdir(exist_ok=True)

        # Save config
        result.config.to_yaml(exp_dir / "config.yaml")

        # Save metrics
        metrics_df = pd.DataFrame([{
            "split": "test",
            "threshold": result.threshold,
            **{k: v for k, v in result.metrics.items() if k.startswith("test_")}
        }])
        metrics_df.to_csv(exp_dir / "metrics.csv", index=False)

        # Save threshold sweep
        threshold_sweep.to_csv(exp_dir / "threshold_sweep.csv", index=False)

        # Save predictions
        test_predictions.to_csv(exp_dir / "predictions.csv", index=False)

        # Generate plots
        self._plot_confusion_matrix(test_predictions, result.threshold, exp_dir)
        self._plot_threshold_sweep(threshold_sweep, result.threshold, exp_dir)

        return exp_dir

    def _plot_confusion_matrix(
        self,
        df: pd.DataFrame,
        threshold: float,
        output_dir: Path,
    ) -> None:
        """Generate confusion matrix plot."""
        y_true = df["IS_CHURN"]
        y_pred = (df["RISK_SCORE"] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Predicted Active", "Predicted Churn"],
            yticklabels=["Actual Active", "Actual Churn"],
        )
        ax.set_title(f"Confusion Matrix (Threshold = {threshold:.0f})")
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
        plt.close()

    def _plot_threshold_sweep(
        self,
        results_df: pd.DataFrame,
        best_threshold: float,
        output_dir: Path,
    ) -> None:
        """Generate threshold sweep plot."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric in ["accuracy", "precision", "recall", "f1"]:
            ax.plot(
                results_df["threshold"],
                results_df[metric],
                label=metric.capitalize(),
                linewidth=2,
            )

        ax.axvline(
            x=best_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best ({best_threshold:.0f})",
        )
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Score")
        ax.set_title("Metrics vs. Threshold")
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "threshold_sweep.png", dpi=150)
        plt.close()

    def load_experiment(self, experiment_id: str) -> dict | None:
        """
        Load artifacts for a specific experiment.

        Args:
            experiment_id: Experiment ID to load

        Returns:
            Dictionary with loaded artifacts, or None if not found
        """
        exp_dir = self.artifacts_dir / experiment_id
        if not exp_dir.exists():
            return None

        from .config import ExperimentConfig

        return {
            "config": ExperimentConfig.from_yaml(exp_dir / "config.yaml"),
            "metrics": pd.read_csv(exp_dir / "metrics.csv"),
            "threshold_sweep": pd.read_csv(exp_dir / "threshold_sweep.csv"),
            "predictions": pd.read_csv(exp_dir / "predictions.csv"),
        }
