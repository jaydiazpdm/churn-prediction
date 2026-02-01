"""
End-to-end integration tests.

Tests complete workflows from data loading through scoring to output.
"""

from pathlib import Path

import pandas as pd
import pytest

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from model import ChurnScorer


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_full_experiment_pipeline(self):
        """Complete experiment run should succeed."""
        config = ExperimentConfig.from_yaml("experiments/configs/baseline.yaml")
        runner = ExperimentRunner()

        result = runner.run(config)

        # Should complete without errors
        assert result.experiment_id is not None
        assert result.passed is not None
        assert "test_accuracy" in result.metrics
        assert result.metrics["test_accuracy"] > 0

    def test_batch_scoring_pipeline(self):
        """Production batch scoring workflow."""
        # 1. Load data
        df = pd.read_csv("experiments/data/test.csv")
        assert len(df) > 0

        # 2. Score
        scorer = ChurnScorer()
        result = scorer.score(df)
        assert len(result.df) == len(df)

        # 3. Filter high-risk
        high_risk = result.get_high_risk("High")

        # 4. Verify output
        assert len(high_risk) >= 0, "Should return empty or non-empty DataFrame"
        assert "RISK_SCORE" in high_risk.columns
        assert "RISK_LEVEL" in high_risk.columns

        # 5. Verify high-risk clients actually have high scores
        if len(high_risk) > 0:
            assert (high_risk["RISK_LEVEL"].isin(["High", "Critical"])).all()

    def test_scoring_with_filtering_workflow(self):
        """Score data and apply various filters."""
        df = pd.read_csv("experiments/data/test.csv")
        scorer = ChurnScorer()

        # Score all clients
        result = scorer.score(df)

        # Get summary statistics
        summary = result.summary()
        assert len(summary) > 0

        # Get component breakdown
        breakdown = result.component_breakdown()
        assert len(breakdown) > 0

        # Filter by different risk levels
        critical = result.get_high_risk("Critical")
        high = result.get_high_risk("High")
        medium = result.get_high_risk("Medium")

        # Critical should be subset of high
        assert len(critical) <= len(high)
        # High should be subset of medium
        assert len(high) <= len(medium)

    def test_single_client_scoring_workflow(self):
        """Score a single client end-to-end."""
        client_data = {
            "CLIENT_ID": "TEST_CLIENT",
            "TIER_NAME": "Core",
            "TOTAL_BLUEPRINTS": 1,
            "CONTRACT_DURATION": 3,
            "MONTHS_UNTIL_END": 1,
            "FIRST_LENGTH": 3,
            "DIFF_RETAINER": -0.3,
        }

        scorer = ChurnScorer()
        result = scorer.score_single(client_data)

        # Verify output structure
        assert "RISK_SCORE" in result
        assert "RISK_LEVEL" in result
        assert isinstance(result["RISK_SCORE"], (int, float))
        assert isinstance(result["RISK_LEVEL"], str)

        # Score should be reasonable for high-risk profile
        # (Core tier, single blueprint, short contract, urgent, declining revenue)
        assert result["RISK_SCORE"] > 30, \
            f"Expected high risk score for this profile, got {result['RISK_SCORE']}"


class TestExperimentConfigs:
    """Validate all experiment configs."""

    @pytest.fixture
    def all_config_files(self):
        """List all YAML configs."""
        config_dir = Path("experiments/configs")
        return list(config_dir.glob("*.yaml"))

    def test_all_configs_load(self, all_config_files):
        """All YAML files should load without errors."""
        for config_file in all_config_files:
            config = ExperimentConfig.from_yaml(config_file)
            assert config.name is not None, f"Config {config_file} has no name"
            assert config.name != "", f"Config {config_file} has empty name"

    def test_all_configs_have_required_fields(self, all_config_files):
        """All configs should have name and min_accuracy."""
        for config_file in all_config_files:
            config = ExperimentConfig.from_yaml(config_file)

            assert config.name != "", f"Config {config_file} has empty name"
            assert config.min_accuracy >= 0.50, \
                f"Config {config_file} has min_accuracy below kill criteria"

    def test_all_configs_have_valid_data_paths(self, all_config_files):
        """All configs should reference valid data files."""
        for config_file in all_config_files:
            config = ExperimentConfig.from_yaml(config_file)

            # Check if data paths exist
            if hasattr(config, "train_path") and config.train_path:
                train_path = Path(config.train_path)
                assert train_path.exists(), \
                    f"Config {config_file} references missing train data: {config.train_path}"

    @pytest.mark.slow
    def test_all_configs_runnable(self, all_config_files):
        """All configs should execute successfully (slow test)."""
        runner = ExperimentRunner()

        for config_file in all_config_files:
            config = ExperimentConfig.from_yaml(config_file)

            result = runner.run(config)
            assert result.experiment_id is not None, \
                f"Config {config_file} failed to run"


class TestDataPipeline:
    """Test data loading and preprocessing pipeline."""

    def test_train_val_test_split_integrity(self):
        """Train, validation, and test sets should not overlap."""
        train_df = pd.read_csv("experiments/data/train.csv")
        val_df = pd.read_csv("experiments/data/val.csv")
        test_df = pd.read_csv("experiments/data/test.csv")

        # Check no overlap in CLIENT_IDs
        train_ids = set(train_df["CLIENT_ID"])
        val_ids = set(val_df["CLIENT_ID"])
        test_ids = set(test_df["CLIENT_ID"])

        assert len(train_ids & val_ids) == 0, "Train and validation sets overlap"
        assert len(train_ids & test_ids) == 0, "Train and test sets overlap"
        assert len(val_ids & test_ids) == 0, "Validation and test sets overlap"

    def test_data_split_sizes_reasonable(self):
        """Data splits should have reasonable proportions."""
        train_df = pd.read_csv("experiments/data/train.csv")
        val_df = pd.read_csv("experiments/data/val.csv")
        test_df = pd.read_csv("experiments/data/test.csv")

        total = len(train_df) + len(val_df) + len(test_df)

        train_pct = len(train_df) / total
        val_pct = len(val_df) / total
        test_pct = len(test_df) / total

        # Should be roughly 60/20/20
        assert 0.50 <= train_pct <= 0.70, f"Train split unusual: {train_pct:.1%}"
        assert 0.10 <= val_pct <= 0.30, f"Val split unusual: {val_pct:.1%}"
        assert 0.10 <= test_pct <= 0.30, f"Test split unusual: {test_pct:.1%}"

    def test_target_variable_present_in_all_splits(self):
        """IS_CHURN should be present in all data splits."""
        for split_name in ["train", "val", "test"]:
            df = pd.read_csv(f"experiments/data/{split_name}.csv")
            assert "IS_CHURN" in df.columns, \
                f"{split_name} split missing IS_CHURN column"
            assert df["IS_CHURN"].isin([0, 1]).all(), \
                f"{split_name} split has invalid IS_CHURN values"
