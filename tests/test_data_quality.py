"""
Data quality and schema validation tests.

Tests input data validation, range checks, and data quality assurance.
"""

import warnings

import pandas as pd
import pandera as pa
import pytest

from model import ChurnScorer
from model.schemas import SCORING_INPUT_SCHEMA


class TestDataQuality:
    """Data quality and schema validation tests."""

    def test_scoring_input_schema_valid(self):
        """Production scoring input must match schema."""
        df = pd.read_csv("experiments/data/test.csv")

        # This will raise if schema is violated
        validated_df = SCORING_INPUT_SCHEMA.validate(df)
        assert len(validated_df) > 0

    def test_scorer_validates_schema(self):
        """ChurnScorer should validate input schema."""
        bad_df = pd.DataFrame({
            "CLIENT_ID": ["TEST"],
            "TIER_NAME": ["InvalidTier"],  # Bad tier
            "TOTAL_BLUEPRINTS": [1],
            "CONTRACT_DURATION": [6],
            "MONTHS_UNTIL_END": [3],
            "FIRST_LENGTH": [6],
            "DIFF_RETAINER": [0.0],
        })

        scorer = ChurnScorer()

        # Should raise validation error
        with pytest.raises(pa.errors.SchemaError):
            scorer.score(bad_df)

    def test_null_handling_documented(self):
        """DIFF_RETAINER can be null, others cannot."""
        # Valid: null DIFF_RETAINER
        df1 = pd.DataFrame({
            "CLIENT_ID": ["TEST"],
            "TIER_NAME": ["Core"],
            "TOTAL_BLUEPRINTS": [1],
            "CONTRACT_DURATION": [6],
            "MONTHS_UNTIL_END": [3],
            "FIRST_LENGTH": [6],
            "DIFF_RETAINER": [None],
        })
        validated = SCORING_INPUT_SCHEMA.validate(df1)
        assert len(validated) == 1  # Should pass

        # Invalid: null CLIENT_ID
        df2 = pd.DataFrame({
            "CLIENT_ID": [None],
            "TIER_NAME": ["Core"],
            "TOTAL_BLUEPRINTS": [1],
            "CONTRACT_DURATION": [6],
            "MONTHS_UNTIL_END": [3],
            "FIRST_LENGTH": [6],
            "DIFF_RETAINER": [0.0],
        })

        with pytest.raises(pa.errors.SchemaError):
            SCORING_INPUT_SCHEMA.validate(df2)


class TestDataRanges:
    """Tests for reasonable data ranges and outlier detection."""

    def test_no_negative_counts(self):
        """Count fields should never be negative."""
        df = pd.read_csv("experiments/data/test.csv")

        assert (df["TOTAL_BLUEPRINTS"] >= 0).all(), \
            "Negative blueprint counts found"
        assert (df["CONTRACT_DURATION"] >= 0).all(), \
            "Negative contract durations found"
        assert (df["FIRST_LENGTH"] >= 0).all(), \
            "Negative first length values found"

    def test_contract_duration_reasonable(self):
        """Contracts should be <120 months (10 years)."""
        df = pd.read_csv("experiments/data/test.csv")

        max_duration = df["CONTRACT_DURATION"].max()
        assert max_duration <= 120, \
            f"Unreasonable contract duration: {max_duration} months"

    def test_retainer_change_within_bounds(self):
        """DIFF_RETAINER should be between -100% and +500%."""
        df = pd.read_csv("experiments/data/test.csv")

        valid_range = (
            df["DIFF_RETAINER"].between(-1.0, 5.0, inclusive="both") |
            df["DIFF_RETAINER"].isna()
        )

        invalid_values = df[~valid_range]["DIFF_RETAINER"]
        assert valid_range.all(), \
            f"DIFF_RETAINER out of bounds: {invalid_values.values}"

    @pytest.mark.parametrize("column,expected_max,threshold", [
        ("TOTAL_BLUEPRINTS", 20, 0.10),  # Warn if >10% exceed 20
        ("CONTRACT_DURATION", 36, 0.10),  # Warn if >10% exceed 3 years
        ("FIRST_LENGTH", 36, 0.10),
    ])
    def test_outlier_warning(self, column, expected_max, threshold):
        """Warn if >threshold of values exceed expected maxima."""
        df = pd.read_csv("experiments/data/test.csv")

        outlier_pct = (df[column] > expected_max).mean()

        if outlier_pct > threshold:
            warnings.warn(
                f"{column}: {outlier_pct:.1%} of values >{expected_max}. "
                f"Check data pipeline."
            )

    def test_tier_distribution_reasonable(self):
        """Tier distribution should match expected proportions."""
        df = pd.read_csv("experiments/data/test.csv")

        tier_counts = df["TIER_NAME"].value_counts(normalize=True)

        # Based on EDA: Core (43%), Growth (32%), Enterprise (25%)
        # Allow 20% deviation from expected
        if "Core" in tier_counts:
            assert 0.30 <= tier_counts["Core"] <= 0.60, \
                f"Core tier proportion unusual: {tier_counts['Core']:.1%}"

    def test_churn_rate_reasonable(self):
        """Churn rate should be between 20-60% (balanced dataset)."""
        df = pd.read_csv("experiments/data/test.csv")

        if "IS_CHURN" in df.columns:
            churn_rate = df["IS_CHURN"].mean()
            assert 0.20 <= churn_rate <= 0.60, \
                f"Unusual churn rate: {churn_rate:.1%} (expected 20-60%)"


class TestDataConsistency:
    """Tests for logical data consistency."""

    def test_contract_duration_vs_first_length(self):
        """CONTRACT_DURATION should generally be >= FIRST_LENGTH."""
        df = pd.read_csv("experiments/data/test.csv")

        # Allow some exceptions (10%) for edge cases
        inconsistent = df["CONTRACT_DURATION"] < df["FIRST_LENGTH"]
        inconsistent_pct = inconsistent.mean()

        assert inconsistent_pct <= 0.15, \
            f"Too many inconsistent duration vs first_length: {inconsistent_pct:.1%}"

    def test_no_duplicate_client_ids(self):
        """Each CLIENT_ID should appear exactly once."""
        df = pd.read_csv("experiments/data/test.csv")

        duplicates = df["CLIENT_ID"].duplicated().sum()
        assert duplicates == 0, \
            f"Found {duplicates} duplicate CLIENT_IDs"

    def test_required_columns_present(self):
        """All required columns must be present."""
        df = pd.read_csv("experiments/data/test.csv")

        required = [
            "CLIENT_ID", "TIER_NAME", "TOTAL_BLUEPRINTS",
            "CONTRACT_DURATION", "MONTHS_UNTIL_END", "FIRST_LENGTH", "DIFF_RETAINER"
        ]

        missing = set(required) - set(df.columns)
        assert len(missing) == 0, \
            f"Missing required columns: {missing}"
