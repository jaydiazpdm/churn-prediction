"""Base class for scoring components."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..config import ScoringConfig


class BaseScorer(ABC):
    """
    Abstract base class for scoring components.

    Each component calculates a single aspect of churn risk
    using vectorized pandas operations.
    """

    name: str = "base"

    def __init__(self, config: "ScoringConfig"):
        """
        Initialize scorer with configuration.

        Args:
            config: ScoringConfig instance with thresholds and weights
        """
        self.config = config

    @abstractmethod
    def score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate component score for all rows.

        Must be implemented by subclasses using vectorized operations.

        Args:
            df: DataFrame with required columns

        Returns:
            Series of integer scores
        """
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """List of columns required by this scorer."""
        pass

    def validate(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(
                f"{self.__class__.__name__} requires columns: {missing}"
            )
