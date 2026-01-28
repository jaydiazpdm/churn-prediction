"""
Scoring configuration for churn prediction model.

All scoring thresholds and weights are defined here for easy tuning.
Based on EDA findings:
- 9-month pivot point for retention
- Core tier has highest churn volatility
- 3.7% 12-month churn rate
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class ScoringConfig:
    """
    Configuration for all scoring components.

    Total max score: 120 points
    - Tier Risk: 0-25
    - Engagement Depth: 0-20
    - Contract Stability: 0-20
    - Renewal Urgency: 0-25
    - Financial Health: 0-15
    - Initial Commitment: 0-15
    """

    # === Tier Risk (0-25 points) ===
    # Core has highest volatility (25% spike in Jan 2025)
    tier_points: Dict[str, int] = field(default_factory=lambda: {
        "Core": 25,       # 43% of clients, highest volatility
        "Growth": 15,     # 32% of clients, moderate risk
        "Enterprise": 5,  # 25% of clients, most stable
    })
    tier_default: int = 15  # Default for unknown tiers

    # === Engagement Depth (0-20 points) ===
    # Single blueprint = single point of failure
    blueprint_thresholds: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 20),   # Single blueprint: high risk
        (2, 10),   # Two blueprints: moderate
        # 3+: 0 points (well invested)
    ])
    blueprint_default: int = 0

    # === Contract Stability (0-20 points) ===
    # Based on 9-month pivot point
    contract_thresholds: List[Tuple[int, int]] = field(default_factory=lambda: [
        (3, 20),   # <=3 months: very short, no commitment
        (6, 15),   # 4-6 months: short-term
        (9, 10),   # 7-9 months: approaching stability
        # >9 months: 0 points (past pivot, stable)
    ])
    contract_default: int = 0

    # === Renewal Urgency (0-25 points) ===
    # Closer to contract end = higher immediate risk
    urgency_thresholds: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 25),   # <=1 month: critical
        (3, 15),   # 2-3 months: urgent
        (6, 5),    # 4-6 months: monitor
        # >6 months: 0 points (safe)
    ])
    urgency_default: int = 0

    # === Financial Health (0-15 points) ===
    # Negative retainer change signals budget pressure
    # DIFF_RETAINER is a percentage change (e.g., -0.5 = -50%)
    financial_thresholds: List[Tuple[float, int]] = field(default_factory=lambda: [
        (-0.50, 15),  # <= -50%: severe budget cut
        (-0.20, 10),  # -50% to -20%: significant reduction
        (0.0, 5),     # -20% to 0%: slight decline
        # > 0%: 0 points (growing)
    ])
    financial_default: int = 0

    # === Initial Commitment (0-15 points) ===
    # First contract length predicts long-term retention
    # 9-month pivot: clients with 9+ month initial stay ~60% longer
    tenure_thresholds: List[Tuple[int, int]] = field(default_factory=lambda: [
        (3, 15),   # <=3 months: short trial, higher risk
        (8, 10),   # 4-8 months: moderate commitment
        # >=9 months: 0 points (committed from start)
    ])
    tenure_default: int = 0

    # === Risk Level Categorization ===
    risk_levels: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "Low": (0, 30),
        "Medium": (31, 50),
        "High": (51, 70),
        "Critical": (71, 120),
    })

    # === Metadata ===
    max_score: int = 120
    version: str = "1.0.0"

    def get_risk_level(self, score: int) -> str:
        """Map numeric score to risk level."""
        for level, (low, high) in self.risk_levels.items():
            if low <= score <= high:
                return level
        return "Unknown"


# Default configuration instance
DEFAULT_CONFIG = ScoringConfig()
