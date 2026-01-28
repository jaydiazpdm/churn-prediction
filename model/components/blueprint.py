"""Blueprint engagement depth scoring component."""

import numpy as np
import pandas as pd

from .base import BaseScorer


class BlueprintScorer(BaseScorer):
    """
    Score based on number of blueprints (engagement depth).

    Single blueprint clients have a single point of failure -
    if that project ends, the relationship ends.

    Multiple blueprints indicate deeper investment and
    multiple touchpoints with the client.

    Points:
    - 1 blueprint: 20 (high risk - single point of failure)
    - 2 blueprints: 10 (moderate)
    - 3+ blueprints: 0 (well invested)
    """

    name = "blueprint"

    @property
    def required_columns(self) -> list[str]:
        return ["TOTAL_BLUEPRINTS"]

    def score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate blueprint engagement score."""
        self.validate(df)
        blueprints = df["TOTAL_BLUEPRINTS"]

        # Build conditions from thresholds
        conditions = []
        choices = []

        for threshold, points in self.config.blueprint_thresholds:
            conditions.append(blueprints == threshold)
            choices.append(points)

        return pd.Series(
            np.select(conditions, choices, default=self.config.blueprint_default),
            index=df.index,
            dtype=int,
        )
