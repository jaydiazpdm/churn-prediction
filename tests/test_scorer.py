"""
Integration tests for ChurnScorer.
"""

import time

import pandas as pd
import pytest

from model import ChurnScorer, ScoringConfig
from model.scorer import generate_sample_data


class TestChurnScorer:
    """Integration tests for the main scorer class."""

    def test_score_returns_all_components(self, scorer, sample_data):
        """Result should include all component scores."""
        result = scorer.score(sample_data)

        assert "RISK_SCORE" in result.df.columns
        assert "RISK_LEVEL" in result.df.columns
        assert "tier_score" in result.df.columns
        assert "blueprint_score" in result.df.columns
        assert "contract_score" in result.df.columns
        assert "urgency_score" in result.df.columns
        assert "financial_score" in result.df.columns
        assert "tenure_score" in result.df.columns

    def test_risk_score_is_sum_of_components(self, scorer, sample_data):
        """RISK_SCORE should equal sum of all component scores."""
        result = scorer.score(sample_data)

        component_sum = (
            result.df["tier_score"]
            + result.df["blueprint_score"]
            + result.df["contract_score"]
            + result.df["urgency_score"]
            + result.df["financial_score"]
            + result.df["tenure_score"]
        )

        assert (result.df["RISK_SCORE"] == component_sum).all()

    def test_high_risk_client_flagged_critical(self, scorer, edge_cases):
        """Highest risk client should be Critical."""
        result = scorer.score(edge_cases)

        high_risk = result.df[result.df["CLIENT_ID"] == "EDGE_HIGH_RISK"]
        assert high_risk["RISK_LEVEL"].values[0] == "Critical"

    def test_low_risk_client_flagged_low(self, scorer, edge_cases):
        """Lowest risk client should be Low."""
        result = scorer.score(edge_cases)

        low_risk = result.df[result.df["CLIENT_ID"] == "EDGE_LOW_RISK"]
        assert low_risk["RISK_LEVEL"].values[0] == "Low"

    def test_scores_within_expected_range(self, scorer, sample_data):
        """All scores should be within 0 to max_score."""
        result = scorer.score(sample_data)

        assert result.df["RISK_SCORE"].min() >= 0
        assert result.df["RISK_SCORE"].max() <= scorer.config.max_score

    def test_all_risk_levels_present(self, scorer, sample_data):
        """Sample data should produce variety of risk levels."""
        result = scorer.score(sample_data)
        levels = result.df["RISK_LEVEL"].unique()

        # Should have at least 3 different risk levels
        assert len(levels) >= 3

    def test_vectorized_performance(self, scorer):
        """Scoring 10k clients should complete in <1 second."""
        large_data = generate_sample_data(n_clients=10000, seed=123)

        start = time.time()
        result = scorer.score(large_data)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Scoring took {elapsed:.2f}s, expected <1s"
        assert len(result.df) == 10000

    def test_missing_column_raises_error(self, scorer):
        """Missing required columns should raise ValueError."""
        bad_data = pd.DataFrame({"CLIENT_ID": ["TEST"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            scorer.score(bad_data)

    def test_get_high_risk_filters_correctly(self, scorer, edge_cases):
        """get_high_risk should filter by risk level."""
        result = scorer.score(edge_cases)

        high_risk = result.get_high_risk("High")
        assert all(high_risk["RISK_LEVEL"].isin(["High", "Critical"]))

        critical = result.get_high_risk("Critical")
        assert all(critical["RISK_LEVEL"] == "Critical")

    def test_summary_returns_dataframe(self, scorer, sample_data):
        """summary() should return aggregated stats."""
        result = scorer.score(sample_data)
        summary = result.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "count" in summary.columns
        assert "avg_score" in summary.columns

    def test_component_breakdown_returns_stats(self, scorer, sample_data):
        """component_breakdown() should show each component's contribution."""
        result = scorer.score(sample_data)
        breakdown = result.component_breakdown()

        assert isinstance(breakdown, pd.DataFrame)
        assert "tier" in breakdown.index
        assert "blueprint" in breakdown.index
        assert "mean" in breakdown.columns


class TestScoreSingle:
    """Tests for scoring individual clients."""

    def test_score_single_returns_dict(self, scorer):
        """score_single should return a dict with score and components."""
        client = {
            "CLIENT_ID": "SINGLE_001",
            "TIER_NAME": "Growth",
            "TOTAL_BLUEPRINTS": 2,
            "CONTRACT_DURATION": 6,
            "MONTHS_UNTIL_END": 3,
            "FIRST_LENGTH": 6,
            "DIFF_RETAINER": -0.10,
        }
        result = scorer.score_single(client)

        assert "RISK_SCORE" in result
        assert "RISK_LEVEL" in result
        assert "components" in result
        assert isinstance(result["RISK_SCORE"], int)

    def test_score_single_components_sum_to_total(self, scorer):
        """Component scores should sum to RISK_SCORE."""
        client = {
            "CLIENT_ID": "SINGLE_002",
            "TIER_NAME": "Core",
            "TOTAL_BLUEPRINTS": 1,
            "CONTRACT_DURATION": 3,
            "MONTHS_UNTIL_END": 1,
            "FIRST_LENGTH": 3,
            "DIFF_RETAINER": -0.50,
        }
        result = scorer.score_single(client)

        component_sum = sum(result["components"].values())
        assert result["RISK_SCORE"] == component_sum


class TestCustomConfig:
    """Tests for custom configuration."""

    def test_custom_tier_points(self):
        """Custom tier points should be used."""
        custom_config = ScoringConfig(
            tier_points={"Core": 50, "Growth": 30, "Enterprise": 10}
        )
        scorer = ChurnScorer(custom_config)

        df = pd.DataFrame([{
            "CLIENT_ID": "CUSTOM",
            "TIER_NAME": "Core",
            "TOTAL_BLUEPRINTS": 3,
            "CONTRACT_DURATION": 10,
            "MONTHS_UNTIL_END": 8,
            "FIRST_LENGTH": 10,
            "DIFF_RETAINER": 0.10,
        }])
        result = scorer.score(df)

        assert result.df["tier_score"].iloc[0] == 50

    def test_custom_risk_levels(self):
        """Custom risk level thresholds should work."""
        custom_config = ScoringConfig(
            risk_levels={
                "Safe": (0, 20),
                "Watch": (21, 40),
                "Danger": (41, 120),
            }
        )
        scorer = ChurnScorer(custom_config)

        df = pd.DataFrame([{
            "CLIENT_ID": "CUSTOM",
            "TIER_NAME": "Enterprise",
            "TOTAL_BLUEPRINTS": 5,
            "CONTRACT_DURATION": 12,
            "MONTHS_UNTIL_END": 10,
            "FIRST_LENGTH": 12,
            "DIFF_RETAINER": 0.20,
        }])
        result = scorer.score(df)

        # Low score should be "Safe" with custom config
        assert result.df["RISK_LEVEL"].iloc[0] == "Safe"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_null_values_handled(self, scorer):
        """Null values in optional fields should not crash."""
        df = pd.DataFrame([{
            "CLIENT_ID": "NULL_TEST",
            "TIER_NAME": "Growth",
            "TOTAL_BLUEPRINTS": 2,
            "CONTRACT_DURATION": 6,
            "MONTHS_UNTIL_END": 3,
            "FIRST_LENGTH": 6,
            "DIFF_RETAINER": None,  # Null
        }])

        # Should handle null (may produce NaN score)
        result = scorer.score(df)
        assert len(result.df) == 1

    def test_extreme_values(self, scorer):
        """Extreme values should be handled gracefully."""
        df = pd.DataFrame([{
            "CLIENT_ID": "EXTREME",
            "TIER_NAME": "Core",
            "TOTAL_BLUEPRINTS": 100,  # Very high
            "CONTRACT_DURATION": 0,   # Zero
            "MONTHS_UNTIL_END": -1,   # Negative (overdue)
            "FIRST_LENGTH": 100,      # Very long
            "DIFF_RETAINER": -2.0,    # Beyond -100%
        }])
        result = scorer.score(df)

        # Should not crash and produce a valid score
        assert result.df["RISK_SCORE"].iloc[0] >= 0

    def test_empty_dataframe(self, scorer):
        """Empty DataFrame should return empty result."""
        df = pd.DataFrame(columns=scorer.REQUIRED_COLUMNS)
        result = scorer.score(df)

        assert len(result.df) == 0
