"""Renewal urgency scoring component."""

import numpy as np
import pandas as pd

from .base import BaseScorer


class UrgencyScorer(BaseScorer):
    """
    Score based on months until contract end.

    Closer to contract renewal = higher immediate churn risk.
    This is an "action trigger" - when urgency is high,
    intervention is needed NOW.

    Points:
    - <=1 month: 25 (critical - immediate action needed)
    - 2-3 months: 15 (urgent - schedule intervention)
    - 4-6 months: 5 (monitor)
    - >6 months: 0 (safe for now)
    """

    name = "urgency"

    @property
    def required_columns(self) -> list[str]:
        return ["MONTHS_UNTIL_END"]

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate renewal urgency score."""
        self.validate(df)
        months_left = df["MONTHS_UNTIL_END"]

        # Build conditions from thresholds (first match wins)
        conditions = []
        choices = []

        for max_months, points in self.config.urgency_thresholds:
            conditions.append(months_left <= max_months)
            choices.append(points)

        return pd.Series(
            np.select(conditions, choices, default=self.config.urgency_default),
            index=df.index,
            dtype=int,
        )
