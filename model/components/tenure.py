"""Initial commitment (tenure) scoring component."""

import numpy as np
import pandas as pd

from .base import BaseScorer


class TenureScorer(BaseScorer):
    """
    Score based on FIRST_LENGTH (initial contract length).

    The 9-month pivot point is critical from EDA:
    Clients with 9+ month initial contracts stay ~60% longer
    than those with short initial contracts.

    Short initial contracts suggest the client was testing,
    not committing from the start.

    Points:
    - <=3 months: 15 (short trial - higher risk)
    - 4-8 months: 10 (moderate commitment)
    - >=9 months: 0 (committed from start)
    """

    name = "tenure"

    @property
    def required_columns(self) -> list[str]:
        return ["FIRST_LENGTH"]

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate initial commitment score."""
        self.validate(df)
        first_length = df["FIRST_LENGTH"]

        # Build conditions from thresholds (first match wins)
        conditions = []
        choices = []

        for max_months, points in self.config.tenure_thresholds:
            conditions.append(first_length <= max_months)
            choices.append(points)

        return pd.Series(
            np.select(conditions, choices, default=self.config.tenure_default),
            index=df.index,
            dtype=int,
        )
