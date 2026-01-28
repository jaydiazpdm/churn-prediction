"""Tier-based risk scoring component."""

import pandas as pd

from .base import BaseScorer


class TierScorer(BaseScorer):
    """
    Score based on client tier (Core/Growth/Enterprise).

    Core tier has highest churn volatility (25% spike observed in EDA).
    Enterprise clients tend to have longer, more stable relationships.

    Points:
    - Core: 25 (highest risk)
    - Growth: 15 (moderate)
    - Enterprise: 5 (lowest risk)
    """

    name = "tier"

    @property
    def required_columns(self) -> list[str]:
        return ["TIER_NAME"]

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Map tier names to risk points."""
        self.validate(df)
        return (
            df["TIER_NAME"]
            .map(self.config.tier_points)
            .fillna(self.config.tier_default)
            .astype(int)
        )
