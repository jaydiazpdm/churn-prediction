"""
Experiment configuration for churn prediction pipeline.

Defines the ExperimentConfig dataclass for YAML-driven experimentation.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment.

    Load from YAML:
        config = ExperimentConfig.from_yaml("configs/v3_features.yaml")

    Create programmatically:
        config = ExperimentConfig(
            name="test_heavy_tier",
            description="Test with increased tier weight",
            weights={"tier": 2.0}
        )
    """

    # Metadata
    name: str
    description: str = ""

    # Feature engineering toggles
    # Available: IS_FIRST_CONTRACT, TENURE_BUCKET, CORE_SINGLE_EXPOSURE,
    #            NEW_SHORT_CONTRACT_RISK, CONTRACTS_COMPLETED
    engineered_features: list[str] = field(default_factory=list)

    # Weight configuration
    # Base component multipliers (1.0 = default from model/config.py)
    # And point values for engineered features
    weights: dict[str, float] = field(default_factory=lambda: {
        # Base component multipliers
        "tier": 1.0,
        "blueprint": 1.0,
        "contract": 1.0,
        "urgency": 1.0,
        "financial": 1.0,
        "tenure": 1.0,
        # Engineered feature points
        "first_contract": 15,
        "very_new": 20,
        "establishing": 10,
        "mature_bonus": 5,
        "core_single": 20,
        "new_short": 15,
        "multi_renewal": 10,
    })

    # Threshold optimization
    optimize_metric: Literal["f1", "accuracy", "precision", "recall"] = "f1"
    min_recall: Optional[float] = None
    min_precision: Optional[float] = None
    threshold_step: int = 5

    # Pass/fail criteria (kill criteria from CLAUDE.md: <50% = kill)
    min_accuracy: float = 0.50
    min_f1: Optional[float] = None

    # Data paths (relative to experiments/)
    train_path: str = "data/train.csv"
    val_path: str = "data/validation.csv"
    test_path: str = "data/test.csv"

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def get_weight(self, key: str, default: float = 1.0) -> float:
        """Get a weight value with fallback to default."""
        return self.weights.get(key, default)
