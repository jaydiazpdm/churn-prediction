"""
Unit tests for individual scoring components.
"""

import pandas as pd
import pytest

from model.config import ScoringConfig
from model.components.tier import TierScorer
from model.components.blueprint import BlueprintScorer
from model.components.contract import ContractScorer
from model.components.urgency import UrgencyScorer
from model.components.financial import FinancialScorer
from model.components.tenure import TenureScorer


class TestTierScorer:
    """Tests for tier-based risk scoring."""

    def test_tier_points_correct(self, default_config):
        """Core=25, Growth=15, Enterprise=5."""
        scorer = TierScorer(default_config)
        df = pd.DataFrame({"TIER_NAME": ["Core", "Growth", "Enterprise"]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 25  # Core
        assert scores.iloc[1] == 15  # Growth
        assert scores.iloc[2] == 5   # Enterprise

    def test_unknown_tier_uses_default(self, default_config):
        """Unknown tiers should get default points."""
        scorer = TierScorer(default_config)
        df = pd.DataFrame({"TIER_NAME": ["Unknown", "Premium", None]})
        scores = scorer.score(df)

        assert scores.iloc[0] == default_config.tier_default
        assert scores.iloc[1] == default_config.tier_default

    def test_missing_column_raises_error(self, default_config):
        """Should raise ValueError if TIER_NAME missing."""
        scorer = TierScorer(default_config)
        df = pd.DataFrame({"OTHER_COLUMN": ["Core"]})

        with pytest.raises(ValueError, match="TIER_NAME"):
            scorer.score(df)


class TestBlueprintScorer:
    """Tests for engagement depth scoring."""

    def test_single_blueprint_highest_risk(self, default_config):
        """Single blueprint should score highest (20 points)."""
        scorer = BlueprintScorer(default_config)
        df = pd.DataFrame({"TOTAL_BLUEPRINTS": [1, 2, 3, 5]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 20  # Single: high risk
        assert scores.iloc[1] == 10  # Two: moderate
        assert scores.iloc[2] == 0   # Three+: low risk
        assert scores.iloc[3] == 0   # Five: low risk

    def test_zero_blueprints_uses_default(self, default_config):
        """Edge case: 0 blueprints."""
        scorer = BlueprintScorer(default_config)
        df = pd.DataFrame({"TOTAL_BLUEPRINTS": [0]})
        scores = scorer.score(df)

        assert scores.iloc[0] == default_config.blueprint_default


class TestContractScorer:
    """Tests for contract stability scoring."""

    def test_short_contract_high_risk(self, default_config):
        """Contracts <=3 months should score 20."""
        scorer = ContractScorer(default_config)
        df = pd.DataFrame({"CONTRACT_DURATION": [1, 3, 4, 6, 9, 10, 12]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 20  # 1 month: very short
        assert scores.iloc[1] == 20  # 3 months: very short
        assert scores.iloc[2] == 15  # 4 months: short-term
        assert scores.iloc[3] == 15  # 6 months: short-term
        assert scores.iloc[4] == 10  # 9 months: approaching
        assert scores.iloc[5] == 0   # 10 months: past pivot
        assert scores.iloc[6] == 0   # 12 months: stable

    def test_nine_month_pivot(self, default_config):
        """9 months is the stability threshold."""
        scorer = ContractScorer(default_config)
        df = pd.DataFrame({"CONTRACT_DURATION": [6, 8, 9, 10]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 15  # 6: short-term (4-6 months)
        assert scores.iloc[1] == 10  # 8: approaching (7-9 months)
        assert scores.iloc[2] == 10  # 9: at threshold
        assert scores.iloc[3] == 0   # 10: past threshold


class TestUrgencyScorer:
    """Tests for renewal urgency scoring."""

    def test_urgent_renewal_high_score(self, default_config):
        """<=1 month should score 25 (critical)."""
        scorer = UrgencyScorer(default_config)
        df = pd.DataFrame({"MONTHS_UNTIL_END": [0, 1, 2, 3, 6, 7, 12]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 25  # 0: critical
        assert scores.iloc[1] == 25  # 1: critical
        assert scores.iloc[2] == 15  # 2: urgent
        assert scores.iloc[3] == 15  # 3: urgent
        assert scores.iloc[4] == 5   # 6: monitor
        assert scores.iloc[5] == 0   # 7: safe
        assert scores.iloc[6] == 0   # 12: very safe


class TestFinancialScorer:
    """Tests for financial health scoring."""

    def test_severe_budget_cut_highest(self, default_config):
        """<=-50% should score 15."""
        scorer = FinancialScorer(default_config)
        df = pd.DataFrame({"DIFF_RETAINER": [-0.70, -0.50, -0.30, -0.10, 0.0, 0.10]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 15  # -70%: severe
        assert scores.iloc[1] == 15  # -50%: severe (threshold)
        assert scores.iloc[2] == 10  # -30%: significant
        assert scores.iloc[3] == 5   # -10%: slight decline
        assert scores.iloc[4] == 5   # 0%: at zero threshold
        assert scores.iloc[5] == 0   # +10%: growing

    def test_positive_growth_zero_risk(self, default_config):
        """Positive growth should score 0."""
        scorer = FinancialScorer(default_config)
        df = pd.DataFrame({"DIFF_RETAINER": [0.01, 0.50, 1.0]})
        scores = scorer.score(df)

        assert all(scores == 0)


class TestTenureScorer:
    """Tests for initial commitment scoring."""

    def test_short_initial_high_risk(self, default_config):
        """<=3 months initial should score 15."""
        scorer = TenureScorer(default_config)
        df = pd.DataFrame({"FIRST_LENGTH": [1, 3, 4, 8, 9, 12]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 15  # 1: short trial
        assert scores.iloc[1] == 15  # 3: short trial
        assert scores.iloc[2] == 10  # 4: moderate
        assert scores.iloc[3] == 10  # 8: moderate
        assert scores.iloc[4] == 0   # 9: committed
        assert scores.iloc[5] == 0   # 12: very committed

    def test_nine_month_pivot_commitment(self, default_config):
        """9+ month initial = committed from start."""
        scorer = TenureScorer(default_config)
        df = pd.DataFrame({"FIRST_LENGTH": [8, 9, 10]})
        scores = scorer.score(df)

        assert scores.iloc[0] == 10  # 8: moderate
        assert scores.iloc[1] == 0   # 9: committed
        assert scores.iloc[2] == 0   # 10: committed
