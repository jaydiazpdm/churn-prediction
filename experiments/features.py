"""
Feature engineering for churn prediction experiments.

Extracted from evaluate_rule_model_v3.py for reuse across experiments.
"""

import numpy as np
import pandas as pd

from .config import ExperimentConfig


# Registry of available engineered features
AVAILABLE_FEATURES = [
    "IS_FIRST_CONTRACT",
    "TENURE_BUCKET",
    "CORE_SINGLE_EXPOSURE",
    "NEW_SHORT_CONTRACT_RISK",
    "CONTRACTS_COMPLETED",
]


def get_tenure_bucket(months: float) -> str:
    """
    Categorize client tenure into risk buckets.

    Based on 9-month pivot point from EDA.
    """
    if pd.isna(months):
        return "Unknown"
    if months <= 3:
        return "Very_New"  # Highest risk
    elif months <= 9:
        return "Establishing"  # Before pivot
    elif months <= 18:
        return "Stable"  # Past pivot
    return "Mature"  # Lowest risk


def engineer_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """
    Apply feature engineering based on experiment config.

    Args:
        df: DataFrame with raw features
        config: ExperimentConfig specifying which features to create

    Returns:
        DataFrame with engineered features added
    """
    df = df.copy()
    features_to_create = config.engineered_features

    # IS_FIRST_CONTRACT: Are they still on their first contract?
    if "IS_FIRST_CONTRACT" in features_to_create:
        if "ACTIVE_TIME" in df.columns and "FIRST_LENGTH" in df.columns:
            df["IS_FIRST_CONTRACT"] = (
                df["ACTIVE_TIME"] <= df["FIRST_LENGTH"]
            ).astype(int)
        else:
            df["IS_FIRST_CONTRACT"] = 0

    # TENURE_BUCKET: Non-linear risk based on 9-month pivot
    if "TENURE_BUCKET" in features_to_create:
        if "ACTIVE_TIME" in df.columns:
            df["TENURE_BUCKET"] = df["ACTIVE_TIME"].apply(get_tenure_bucket)
        else:
            df["TENURE_BUCKET"] = "Unknown"

    # CORE_SINGLE_EXPOSURE: Interaction feature (highest risk combo)
    if "CORE_SINGLE_EXPOSURE" in features_to_create:
        df["CORE_SINGLE_EXPOSURE"] = (
            (df["TIER_NAME"] == "Core") & (df["TOTAL_BLUEPRINTS"] == 1)
        ).astype(int)

    # NEW_SHORT_CONTRACT_RISK: New client with short initial contract
    if "NEW_SHORT_CONTRACT_RISK" in features_to_create:
        if "ACTIVE_TIME" in df.columns and "FIRST_LENGTH" in df.columns:
            df["NEW_SHORT_CONTRACT_RISK"] = (
                (df["ACTIVE_TIME"] <= 6) & (df["FIRST_LENGTH"] <= 6)
            ).astype(int)
        else:
            df["NEW_SHORT_CONTRACT_RISK"] = 0

    # CONTRACTS_COMPLETED: Approximate number of renewals
    if "CONTRACTS_COMPLETED" in features_to_create:
        if "ACTIVE_TIME" in df.columns and "FIRST_LENGTH" in df.columns:
            df["CONTRACTS_COMPLETED"] = np.where(
                df["FIRST_LENGTH"] > 0,
                df["ACTIVE_TIME"] / df["FIRST_LENGTH"],
                0,
            ).round(1)
        else:
            df["CONTRACTS_COMPLETED"] = 0.0

    return df
