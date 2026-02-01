"""
Regression tests for model performance and behavior.

Tests that model maintains historical performance and produces
deterministic, reproducible results.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from model import ChurnScorer, generate_sample_data


class TestPerformanceRegression:
    """Regression tests against historical benchmarks."""

    @pytest.fixture
    def performance_baseline(self):
        """Load performance baseline from fixture."""
        baseline_path = Path("tests/fixtures/baseline_metrics.json")

        if not baseline_path.exists():
            pytest.skip("No baseline metrics file found")

        with open(baseline_path) as f:
            return json.load(f)

    def test_accuracy_no_regression(self, performance_baseline):
        """Accuracy should not degrade >5% from baseline."""
        baseline_acc = performance_baseline["metrics"]["test_accuracy"]

        # Run v3_features experiment (the baseline config)
        config = ExperimentConfig.from_yaml("experiments/configs/v3_features.yaml")
        runner = ExperimentRunner()
        result = runner.run(config)

        current_acc = result.metrics["test_accuracy"]
        degradation = baseline_acc - current_acc

        assert degradation <= 0.05, \
            f"Accuracy degraded: {baseline_acc:.1%} -> {current_acc:.1%} (degradation: {degradation:.1%})"

    def test_recall_no_regression(self, performance_baseline):
        """Recall is critical - should not degrade significantly."""
        baseline_recall = performance_baseline["metrics"]["test_recall"]

        config = ExperimentConfig.from_yaml("experiments/configs/v3_features.yaml")
        runner = ExperimentRunner()
        result = runner.run(config)

        current_recall = result.metrics["test_recall"]
        degradation = baseline_recall - current_recall

        # Allow up to 3% degradation in recall
        assert degradation <= 0.03, \
            f"Recall degraded: {baseline_recall:.1%} -> {current_recall:.1%} (degradation: {degradation:.1%})"

    def test_f1_no_regression(self, performance_baseline):
        """F1 score should not degrade >5% from baseline."""
        baseline_f1 = performance_baseline["metrics"]["test_f1"]

        config = ExperimentConfig.from_yaml("experiments/configs/v3_features.yaml")
        runner = ExperimentRunner()
        result = runner.run(config)

        current_f1 = result.metrics["test_f1"]
        degradation = baseline_f1 - current_f1

        assert degradation <= 0.05, \
            f"F1 score degraded: {baseline_f1:.3f} -> {current_f1:.3f} (degradation: {degradation:.3f})"


class TestComponentDeterminism:
    """Test deterministic behavior of scoring components."""

    def test_scorer_produces_same_results(self):
        """Same input should always produce same scores."""
        df = generate_sample_data(n_clients=100, seed=42)

        scorer1 = ChurnScorer()
        scorer2 = ChurnScorer()

        result1 = scorer1.score(df)
        result2 = scorer2.score(df)

        # Compare risk scores
        pd.testing.assert_series_equal(
            result1.df["RISK_SCORE"],
            result2.df["RISK_SCORE"],
            check_names=False
        )

    def test_component_scores_deterministic(self):
        """Each component should produce identical scores on re-run."""
        df = generate_sample_data(n_clients=100, seed=42)
        scorer = ChurnScorer()

        result1 = scorer.score(df)
        result2 = scorer.score(df)

        # Get component columns (they start with specific prefixes)
        component_cols = [col for col in result1.df.columns
                         if any(prefix in col for prefix in
                               ["tier_", "blueprint_", "contract_", "urgency_", "financial_", "tenure_"])]

        for col in component_cols:
            if col in result1.df.columns and col in result2.df.columns:
                pd.testing.assert_series_equal(
                    result1.df[col],
                    result2.df[col],
                    check_names=False,
                    check_exact=True
                )

    def test_score_single_deterministic(self):
        """Single client scoring should be deterministic."""
        client_data = {
            "CLIENT_ID": "TEST001",
            "TIER_NAME": "Core",
            "TOTAL_BLUEPRINTS": 1,
            "CONTRACT_DURATION": 6,
            "MONTHS_UNTIL_END": 2,
            "FIRST_LENGTH": 3,
            "DIFF_RETAINER": -0.2,
        }

        scorer = ChurnScorer()

        result1 = scorer.score_single(client_data)
        result2 = scorer.score_single(client_data)

        assert result1["RISK_SCORE"] == result2["RISK_SCORE"]
        assert result1["RISK_LEVEL"] == result2["RISK_LEVEL"]


class TestConfigBackwardCompatibility:
    """Ensure old configs remain runnable."""

    def test_baseline_config_loads(self):
        """Baseline config from project start should still load."""
        config = ExperimentConfig.from_yaml("experiments/configs/baseline.yaml")
        assert config.name == "baseline"
        assert config.min_accuracy == 0.50

    def test_v3_config_loads(self):
        """V3 feature engineering config should load."""
        config = ExperimentConfig.from_yaml("experiments/configs/v3_features.yaml")
        assert config.name == "v3_features"
        assert len(config.engineered_features) > 0

    def test_baseline_config_runs(self):
        """Baseline config should execute without errors."""
        config = ExperimentConfig.from_yaml("experiments/configs/baseline.yaml")
        runner = ExperimentRunner()

        result = runner.run(config)
        assert result.experiment_id is not None
        assert "test_accuracy" in result.metrics

    def test_v3_config_runs_and_passes(self):
        """V3 config should execute and pass criteria."""
        config = ExperimentConfig.from_yaml("experiments/configs/v3_features.yaml")
        runner = ExperimentRunner()

        result = runner.run(config)
        assert result.experiment_id is not None
        assert result.passed, \
            f"V3 experiment no longer passes (accuracy: {result.metrics.get('test_accuracy', 0):.1%})"
