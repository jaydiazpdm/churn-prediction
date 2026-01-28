"""
Main ChurnScorer class - orchestrates scoring components.

Usage:
    from model import ChurnScorer, ScoringConfig

    # With default config
    scorer = ChurnScorer()
    result = scorer.score(df)

    # With custom config
    config = ScoringConfig(tier_points={"Core": 30, ...})
    scorer = ChurnScorer(config)
    result = scorer.score(df)

    # Access results
    print(result.df[["CLIENT_ID", "RISK_SCORE", "RISK_LEVEL"]])
    print(result.summary())
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import ScoringConfig, DEFAULT_CONFIG
from .components import (
    TierScorer,
    BlueprintScorer,
    ContractScorer,
    UrgencyScorer,
    FinancialScorer,
    TenureScorer,
)


@dataclass
class ScoringResult:
    """
    Container for scoring results with component breakdown.

    Attributes:
        df: Original DataFrame with scores added
        component_columns: List of component score column names
    """

    df: pd.DataFrame
    component_columns: list[str]

    def get_high_risk(self, min_level: str = "High") -> pd.DataFrame:
        """
        Get clients at or above a risk level.

        Args:
            min_level: Minimum risk level ("Low", "Medium", "High", "Critical")

        Returns:
            DataFrame filtered to clients at or above the specified level
        """
        level_order = ["Low", "Medium", "High", "Critical"]
        min_idx = level_order.index(min_level)
        valid_levels = level_order[min_idx:]
        return self.df[self.df["RISK_LEVEL"].isin(valid_levels)]

    def summary(self) -> pd.DataFrame:
        """
        Generate summary statistics by tier and risk level.

        Returns:
            DataFrame with counts and percentages
        """
        return (
            self.df.groupby(["TIER_NAME", "RISK_LEVEL"])
            .agg(
                count=("CLIENT_ID", "count"),
                avg_score=("RISK_SCORE", "mean"),
            )
            .round(1)
        )

    def component_breakdown(self) -> pd.DataFrame:
        """
        Show average contribution of each component.

        Returns:
            DataFrame with component statistics
        """
        stats = {}
        for col in self.component_columns:
            component_name = col.replace("_score", "")
            stats[component_name] = {
                "mean": self.df[col].mean(),
                "max": self.df[col].max(),
                "min": self.df[col].min(),
            }
        return pd.DataFrame(stats).T.round(1)


class ChurnScorer:
    """
    Vectorized churn risk scoring engine.

    Calculates component scores independently using pandas operations,
    then combines them into a total risk score.

    Components:
    - Tier Risk (0-25): Based on client tier
    - Engagement Depth (0-20): Based on blueprint count
    - Contract Stability (0-20): Based on contract duration
    - Renewal Urgency (0-25): Based on months until contract end
    - Financial Health (0-15): Based on retainer change
    - Initial Commitment (0-15): Based on first contract length
    """

    REQUIRED_COLUMNS = [
        "CLIENT_ID",
        "TIER_NAME",
        "TOTAL_BLUEPRINTS",
        "CONTRACT_DURATION",
        "MONTHS_UNTIL_END",
        "FIRST_LENGTH",
        "DIFF_RETAINER",
    ]

    def __init__(self, config: Optional[ScoringConfig] = None):
        """
        Initialize scorer with configuration.

        Args:
            config: ScoringConfig instance. Uses DEFAULT_CONFIG if None.
        """
        self.config = config or DEFAULT_CONFIG
        self._init_components()

    def _init_components(self) -> None:
        """Initialize all scoring components."""
        self.components = {
            "tier": TierScorer(self.config),
            "blueprint": BlueprintScorer(self.config),
            "contract": ContractScorer(self.config),
            "urgency": UrgencyScorer(self.config),
            "financial": FinancialScorer(self.config),
            "tenure": TenureScorer(self.config),
        }

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate required columns exist.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If required columns are missing
        """
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def score(self, df: pd.DataFrame) -> ScoringResult:
        """
        Calculate churn risk scores for all clients.

        Args:
            df: DataFrame with required columns

        Returns:
            ScoringResult with scores and component breakdown

        Example:
            >>> scorer = ChurnScorer()
            >>> result = scorer.score(client_df)
            >>> high_risk = result.get_high_risk("High")
        """
        self.validate_input(df)
        result = df.copy()

        # Calculate all component scores (vectorized)
        component_cols = []
        for name, component in self.components.items():
            col_name = f"{name}_score"
            result[col_name] = component.score(result)
            component_cols.append(col_name)

        # Sum all components for total score
        result["RISK_SCORE"] = result[component_cols].sum(axis=1)

        # Categorize risk level
        result["RISK_LEVEL"] = result["RISK_SCORE"].apply(
            self.config.get_risk_level
        )

        return ScoringResult(df=result, component_columns=component_cols)

    def score_single(self, client_data: dict) -> dict:
        """
        Score a single client (convenience method).

        Args:
            client_data: Dictionary with required fields

        Returns:
            Dictionary with scores and risk level
        """
        df = pd.DataFrame([client_data])
        result = self.score(df)
        row = result.df.iloc[0]
        return {
            "RISK_SCORE": int(row["RISK_SCORE"]),
            "RISK_LEVEL": row["RISK_LEVEL"],
            "components": {
                col.replace("_score", ""): int(row[col])
                for col in result.component_columns
            },
        }


def generate_sample_data(n_clients: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic sample data for testing.

    Distributions based on EDA findings:
    - Tier: Core 43%, Growth 32%, Enterprise 25%
    - ~50% have single blueprint
    - Contract durations centered around 6 months
    """
    np.random.seed(seed)

    # Tier distribution (from EDA)
    tiers = np.random.choice(
        ["Core", "Growth", "Enterprise"],
        size=n_clients,
        p=[0.43, 0.32, 0.25],
    )

    # Blueprint count (50% have only 1)
    blueprints = np.where(
        np.random.random(n_clients) < 0.50,
        1,
        np.random.choice([2, 3, 4, 5], size=n_clients, p=[0.5, 0.25, 0.15, 0.1]),
    )

    # Contract duration (months) - center around 6, range 1-12
    contract_duration = np.clip(
        np.random.normal(loc=6, scale=3, size=n_clients).astype(int),
        1,
        12,
    )

    # Months until contract end
    months_until_end = np.random.randint(0, 13, size=n_clients)

    # First length (initial contract) - 9 month pivot
    first_length = np.clip(
        np.random.normal(loc=7, scale=3, size=n_clients).astype(int),
        1,
        12,
    )

    # DIFF_RETAINER (% change, slightly negative on average)
    diff_retainer = np.clip(
        np.random.normal(loc=-0.05, scale=0.25, size=n_clients),
        -1.0,
        1.0,
    )

    return pd.DataFrame(
        {
            "CLIENT_ID": [f"CLIENT_{i:04d}" for i in range(n_clients)],
            "TIER_NAME": tiers,
            "TOTAL_BLUEPRINTS": blueprints,
            "CONTRACT_DURATION": contract_duration,
            "MONTHS_UNTIL_END": months_until_end,
            "FIRST_LENGTH": first_length,
            "DIFF_RETAINER": diff_retainer.round(2),
        }
    )
