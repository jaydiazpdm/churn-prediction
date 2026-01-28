"""Contract duration stability scoring component."""

import numpy as np
import pandas as pd

from .base import BaseScorer


class ContractScorer(BaseScorer):
    """
    Score based on current contract duration.

    Based on the 9-month pivot point from EDA:
    Clients with contracts >9 months have passed the
    "stability threshold" and show better retention.

    Points:
    - <=3 months: 20 (very short, no commitment)
    - 4-6 months: 15 (short-term)
    - 7-9 months: 10 (approaching stability)
    - >9 months: 0 (past pivot, stable)
    """

    name = "contract"

    @property
    def required_columns(self) -> list[str]:
        return ["CONTRACT_DURATION"]

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate contract stability score."""
        self.validate(df)
        duration = df["CONTRACT_DURATION"]

        # Build conditions from thresholds (first match wins)
        conditions = []
        choices = []

        for max_months, points in self.config.contract_thresholds:
            conditions.append(duration <= max_months)
            choices.append(points)

        return pd.Series(
            np.select(conditions, choices, default=self.config.contract_default),
            index=df.index,
            dtype=int,
        )
