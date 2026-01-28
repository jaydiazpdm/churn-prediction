"""
Churn Prediction: Create Prospective Evaluation Dataset

This script creates a properly constructed dataset for model evaluation by:
1. Simulating features as they would appear at a 90-day prediction horizon
2. Removing data leakage from time-sensitive features
3. Creating train/validation/test splits

LIMITATION: The source dataset lacks date columns, so we cannot perform true
temporal validation. We use stratified random splits instead and document this
as a key limitation of the current evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = Path(__file__).parent.parent / "dataset.csv"
OUTPUT_DIR = Path(__file__).parent / "data"
PREDICTION_HORIZON_MONTHS = 3  # 90 days
RANDOM_STATE = 42

# Feature transformation documentation
FEATURE_TRANSFORMATIONS = """
Feature Transformations for 90-Day Prediction Horizon
======================================================

Original → Prospective Feature Mapping:

1. MONTHS_UNTIL_END
   - Original: Snapshot at data collection (often 0 or negative for churned clients)
   - Prospective: Fixed at 3.0 months (by definition of 90-day horizon)
   - Rationale: At decision point, we always have 3 months until contract end

2. DIFF_RETAINER
   - Original: (LAST_RETAINER - FIRST_RETAINER) / FIRST_RETAINER
   - Prospective: Set to 0.0 (no change observable at 90-day horizon)
   - Rationale: LAST_RETAINER is post-outcome for churned clients; without
     monthly retainer history, we assume no change is known yet
   - LIMITATION: This removes signal. Proper fix requires monthly retainer data.

3. CONTRACT_DURATION
   - Original: Duration of current contract (post-hoc for completed contracts)
   - Prospective: Use FIRST_LENGTH as proxy for expected contract duration
   - Rationale: At 90-day horizon, we can only use historical contract patterns

4. TIER_NAME, TOTAL_BLUEPRINTS, FIRST_LENGTH
   - No change needed: These are stable/historical features without leakage

5. LAST_TEAM_SENTIMENT_SCORE
   - Kept as-is for now, but noted as potentially leaky (timing unclear)
"""


def create_prospective_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform features to simulate what would be known at 90-day horizon.

    This removes data leakage by replacing time-sensitive features with
    values that would be available 90 days before contract end.
    """
    df_prospective = df.copy()

    # 1. MONTHS_UNTIL_END: Fixed at prediction horizon
    df_prospective['MONTHS_UNTIL_END_ORIGINAL'] = df_prospective['MONTHS_UNTIL_END']
    df_prospective['MONTHS_UNTIL_END'] = PREDICTION_HORIZON_MONTHS

    # 2. DIFF_RETAINER: Set to 0 (no change known at decision point)
    df_prospective['DIFF_RETAINER_ORIGINAL'] = df_prospective['DIFF_RETAINER']
    df_prospective['DIFF_RETAINER'] = 0.0

    # 3. CONTRACT_DURATION: Use FIRST_LENGTH as proxy
    df_prospective['CONTRACT_DURATION_ORIGINAL'] = df_prospective['CONTRACT_DURATION']
    df_prospective['CONTRACT_DURATION'] = df_prospective['FIRST_LENGTH']

    return df_prospective


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.

    LIMITATION: Without date columns, we cannot perform temporal validation.
    This stratified random split is a fallback that still validates generalization
    but may overestimate performance on truly future data.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

    # Create target variable
    df['IS_CHURN'] = (df['CLIENT_STATUS'] == 'Churn').astype(int)

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df['IS_CHURN'],
        random_state=RANDOM_STATE
    )

    # Second split: train vs val
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        stratify=train_val_df['IS_CHURN'],
        random_state=RANDOM_STATE
    )

    return train_df, val_df, test_df


def print_split_summary(train_df, val_df, test_df):
    """Print summary of data splits."""
    print("\n" + "=" * 60)
    print("DATASET SPLIT SUMMARY")
    print("=" * 60)

    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        n_total = len(df)
        n_churn = df['IS_CHURN'].sum()
        churn_rate = n_churn / n_total
        print(f"\n{name}:")
        print(f"  Total: {n_total}")
        print(f"  Churn: {n_churn} ({churn_rate:.1%})")
        print(f"  Active: {n_total - n_churn} ({1 - churn_rate:.1%})")


def print_feature_comparison(df_original, df_prospective):
    """Compare original vs prospective feature distributions."""
    print("\n" + "=" * 60)
    print("FEATURE TRANSFORMATION COMPARISON")
    print("=" * 60)

    features = ['MONTHS_UNTIL_END', 'DIFF_RETAINER', 'CONTRACT_DURATION']

    for feat in features:
        orig_col = f"{feat}_ORIGINAL"
        print(f"\n{feat}:")
        print(f"  Original:    mean={df_prospective[orig_col].mean():.2f}, "
              f"std={df_prospective[orig_col].std():.2f}")
        print(f"  Prospective: mean={df_prospective[feat].mean():.2f}, "
              f"std={df_prospective[feat].std():.2f}")


def main():
    print("Loading raw dataset...")
    df_original = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_original)} rows")

    print("\nApplying prospective feature transformations...")
    df_prospective = create_prospective_features(df_original)

    print_feature_comparison(df_original, df_prospective)

    print("\nCreating train/validation/test splits...")
    train_df, val_df, test_df = create_train_val_test_split(df_prospective)

    print_split_summary(train_df, val_df, test_df)

    # Save datasets
    OUTPUT_DIR.mkdir(exist_ok=True)

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)
    df_prospective.to_csv(OUTPUT_DIR / "full_prospective.csv", index=False)

    print(f"\nSaved datasets to {OUTPUT_DIR}/")
    print("  - train.csv")
    print("  - validation.csv")
    print("  - test.csv")
    print("  - full_prospective.csv")

    # Save transformation documentation
    with open(OUTPUT_DIR / "TRANSFORMATIONS.md", "w") as f:
        f.write(FEATURE_TRANSFORMATIONS)
    print("  - TRANSFORMATIONS.md")

    print("\n" + "=" * 60)
    print("IMPORTANT LIMITATIONS")
    print("=" * 60)
    print("""
1. NO TEMPORAL SPLIT: Dataset lacks date columns. Using stratified random
   split instead. This may overestimate performance on truly future data.

2. DIFF_RETAINER ZEROED: Without monthly retainer history, we cannot
   compute prospective retainer changes. Signal is lost.

3. CONTRACT_DURATION PROXIED: Using FIRST_LENGTH as proxy for expected
   duration. This assumes contract patterns repeat.

RECOMMENDATION: For production deployment, collect:
- Churn/contract end dates (for temporal validation)
- Monthly retainer history (for prospective DIFF_RETAINER)
""")

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = main()
