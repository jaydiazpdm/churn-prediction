"""
Pytest fixtures for churn prediction model tests.
"""

import pandas as pd
import pytest

# Add model to path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import ScoringConfig
from model.scorer import ChurnScorer, generate_sample_data


@pytest.fixture
def default_config():
    """Default scoring configuration."""
    return ScoringConfig()


@pytest.fixture
def scorer(default_config):
    """ChurnScorer with default config."""
    return ChurnScorer(default_config)


@pytest.fixture
def sample_data():
    """100 sample clients with realistic distributions."""
    return generate_sample_data(n_clients=100, seed=42)


@pytest.fixture
def edge_cases():
    """Specific edge cases for testing boundary conditions."""
    return pd.DataFrame([
        # Highest risk: Core, single blueprint, short contract, urgent, declining revenue
        {
            "CLIENT_ID": "EDGE_HIGH_RISK",
            "TIER_NAME": "Core",
            "TOTAL_BLUEPRINTS": 1,
            "CONTRACT_DURATION": 2,
            "MONTHS_UNTIL_END": 1,
            "FIRST_LENGTH": 2,
            "DIFF_RETAINER": -0.60,
        },
        # Lowest risk: Enterprise, many blueprints, long contract, not urgent, growing
        {
            "CLIENT_ID": "EDGE_LOW_RISK",
            "TIER_NAME": "Enterprise",
            "TOTAL_BLUEPRINTS": 5,
            "CONTRACT_DURATION": 12,
            "MONTHS_UNTIL_END": 10,
            "FIRST_LENGTH": 12,
            "DIFF_RETAINER": 0.20,
        },
        # 9-month pivot: At the stability threshold
        {
            "CLIENT_ID": "EDGE_PIVOT",
            "TIER_NAME": "Growth",
            "TOTAL_BLUEPRINTS": 2,
            "CONTRACT_DURATION": 9,
            "MONTHS_UNTIL_END": 4,
            "FIRST_LENGTH": 9,
            "DIFF_RETAINER": -0.10,
        },
        # Financial distress: Large negative retainer change
        {
            "CLIENT_ID": "EDGE_FINANCIAL",
            "TIER_NAME": "Growth",
            "TOTAL_BLUEPRINTS": 3,
            "CONTRACT_DURATION": 8,
            "MONTHS_UNTIL_END": 5,
            "FIRST_LENGTH": 6,
            "DIFF_RETAINER": -0.70,
        },
        # Urgent renewal: Contract ending soon but otherwise healthy
        {
            "CLIENT_ID": "EDGE_URGENT",
            "TIER_NAME": "Enterprise",
            "TOTAL_BLUEPRINTS": 4,
            "CONTRACT_DURATION": 10,
            "MONTHS_UNTIL_END": 0,
            "FIRST_LENGTH": 10,
            "DIFF_RETAINER": 0.05,
        },
    ])


@pytest.fixture
def single_client():
    """Single client for simple tests."""
    return pd.DataFrame([{
        "CLIENT_ID": "TEST_001",
        "TIER_NAME": "Growth",
        "TOTAL_BLUEPRINTS": 2,
        "CONTRACT_DURATION": 6,
        "MONTHS_UNTIL_END": 3,
        "FIRST_LENGTH": 6,
        "DIFF_RETAINER": -0.10,
    }])


@pytest.fixture
def tier_test_data():
    """Data for testing tier scoring specifically."""
    return pd.DataFrame([
        {"CLIENT_ID": "CORE", "TIER_NAME": "Core", "TOTAL_BLUEPRINTS": 2,
         "CONTRACT_DURATION": 6, "MONTHS_UNTIL_END": 3, "FIRST_LENGTH": 6, "DIFF_RETAINER": 0},
        {"CLIENT_ID": "GROWTH", "TIER_NAME": "Growth", "TOTAL_BLUEPRINTS": 2,
         "CONTRACT_DURATION": 6, "MONTHS_UNTIL_END": 3, "FIRST_LENGTH": 6, "DIFF_RETAINER": 0},
        {"CLIENT_ID": "ENTERPRISE", "TIER_NAME": "Enterprise", "TOTAL_BLUEPRINTS": 2,
         "CONTRACT_DURATION": 6, "MONTHS_UNTIL_END": 3, "FIRST_LENGTH": 6, "DIFF_RETAINER": 0},
        {"CLIENT_ID": "UNKNOWN", "TIER_NAME": "Unknown", "TOTAL_BLUEPRINTS": 2,
         "CONTRACT_DURATION": 6, "MONTHS_UNTIL_END": 3, "FIRST_LENGTH": 6, "DIFF_RETAINER": 0},
    ])
