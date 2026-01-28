"""
Churn Prediction Model Package

A rule-based scoring system for predicting client churn risk.
"""

from .scorer import ChurnScorer
from .config import ScoringConfig

__all__ = ["ChurnScorer", "ScoringConfig"]
__version__ = "1.0.0"
