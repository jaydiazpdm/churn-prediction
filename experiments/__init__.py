"""
Experiment pipeline for churn prediction.

Usage:
    from experiments import ExperimentRunner, ExperimentConfig

    # Run from YAML
    runner = ExperimentRunner()
    result = runner.run_from_yaml("configs/v3_features.yaml")
    print(result.summary())

    # Run programmatically
    config = ExperimentConfig(
        name="custom",
        description="Test custom weights",
        weights={"tier": 2.0}
    )
    result = runner.run(config)

CLI:
    python -m experiments.run configs/v3_features.yaml
    python -m experiments.run --list
"""

from .config import ExperimentConfig
from .runner import ExperimentRunner, ExperimentResult
from .evaluator import Evaluator
from .features import engineer_features, AVAILABLE_FEATURES
from .logger import ExperimentLogger
from .artifacts import ArtifactManager

__all__ = [
    "ExperimentConfig",
    "ExperimentRunner",
    "ExperimentResult",
    "Evaluator",
    "ExperimentLogger",
    "ArtifactManager",
    "engineer_features",
    "AVAILABLE_FEATURES",
]
