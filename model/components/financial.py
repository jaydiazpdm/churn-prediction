"""Financial health scoring component."""

import numpy as np
import pandas as pd

from .base import BaseScorer


class FinancialScorer(BaseScorer):
    """
    Score based on DIFF_RETAINER (% change in retainer fee).

    Negative retainer change signals budget pressure,
    which is one of the top churn drivers identified in EDA.

    DIFF_RETAINER is a decimal: -0.5 means -50% change.

    Points:
    - <= -50%: 15 (severe budget cut)
    - -50% to -20%: 10 (significant reduction)
    - -20% to 0%: 5 (slight decline)
    - > 0%: 0 (growing - positive signal)
    """

    name = "financial"

    @property
    def required_columns(self) -> list[str]:
        return ["DIFF_RETAINER"]

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate financial health score."""
        self.validate(df)
        diff_retainer = df["DIFF_RETAINER"]

        # Build conditions from thresholds (first match wins)
        conditions = []
        choices = []

        for threshold, points in self.config.financial_thresholds:
            conditions.append(diff_retainer <= threshold)
            choices.append(points)

        return pd.Series(
            np.select(conditions, choices, default=self.config.financial_default),
            index=df.index,
            dtype=int,
        )
