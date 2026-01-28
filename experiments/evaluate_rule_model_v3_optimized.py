"""
Churn Model V3 - Optimized for Accuracy with Recall Constraint

This version optimizes for ACCURACY (not F1) while maintaining recall >= 70%.

Key insight from threshold sweep:
- Threshold 40: Accuracy 60.3%, Recall 74.5% - ACHIEVES TARGETS
- Lower thresholds have higher recall but more false positives
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from itertools import product

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results_v3_optimized"
RESULTS_DIR.mkdir(exist_ok=True)

TARGET_ACCURACY = 0.60
TARGET_RECALL = 0.70


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing data."""
    df = df.copy()

    # IS_FIRST_CONTRACT
    df['IS_FIRST_CONTRACT'] = (df['ACTIVE_TIME'] <= df['FIRST_LENGTH']).astype(int)

    # TENURE_BUCKET
    def get_tenure_bucket(months):
        if pd.isna(months):
            return 'Unknown'
        if months <= 3:
            return 'Very_New'
        elif months <= 9:
            return 'Establishing'
        elif months <= 18:
            return 'Stable'
        else:
            return 'Mature'

    df['TENURE_BUCKET'] = df['ACTIVE_TIME'].apply(get_tenure_bucket)

    # CORE_SINGLE_EXPOSURE
    df['CORE_SINGLE_EXPOSURE'] = (
        (df['TIER_NAME'] == 'Core') & (df['TOTAL_BLUEPRINTS'] == 1)
    ).astype(int)

    # NEW_SHORT_CONTRACT_RISK
    df['NEW_SHORT_CONTRACT_RISK'] = (
        (df['ACTIVE_TIME'] <= 6) & (df['FIRST_LENGTH'] <= 6)
    ).astype(int)

    # CONTRACTS_COMPLETED
    df['CONTRACTS_COMPLETED'] = np.where(
        df['FIRST_LENGTH'] > 0,
        df['ACTIVE_TIME'] / df['FIRST_LENGTH'],
        0
    ).round(1)

    return df


def calculate_risk_score_parameterized(row: pd.Series, weights: dict) -> int:
    """
    Parameterized scoring for grid search optimization.

    Args:
        row: Data row
        weights: Dictionary of weight parameters
    """
    points = 0

    # NEW FEATURES
    if row.get('IS_FIRST_CONTRACT', 0) == 1:
        points += weights.get('first_contract', 15)

    tenure_bucket = row.get('TENURE_BUCKET', 'Unknown')
    if tenure_bucket == 'Very_New':
        points += weights.get('very_new', 20)
    elif tenure_bucket == 'Establishing':
        points += weights.get('establishing', 10)
    elif tenure_bucket == 'Mature':
        points -= weights.get('mature_bonus', 5)

    if row.get('CORE_SINGLE_EXPOSURE', 0) == 1:
        points += weights.get('core_single', 20)

    if row.get('NEW_SHORT_CONTRACT_RISK', 0) == 1:
        points += weights.get('new_short', 15)

    contracts_completed = row.get('CONTRACTS_COMPLETED', 0)
    if contracts_completed >= 2:
        points -= weights.get('multi_renewal_bonus', 10)

    # ORIGINAL FEATURES
    if row['TIER_NAME'] == 'Core':
        points += weights.get('tier_core', 25)
    elif row['TIER_NAME'] == 'Growth':
        points += weights.get('tier_growth', 15)
    elif row['TIER_NAME'] == 'Enterprise':
        points += weights.get('tier_enterprise', 5)

    if row['TOTAL_BLUEPRINTS'] == 1:
        points += weights.get('single_bp', 20)
    elif row['TOTAL_BLUEPRINTS'] == 2:
        points += weights.get('two_bp', 10)

    if row['CONTRACT_DURATION'] <= 3:
        points += weights.get('contract_short', 20)
    elif row['CONTRACT_DURATION'] <= 6:
        points += weights.get('contract_medium', 15)
    elif row['CONTRACT_DURATION'] <= 9:
        points += weights.get('contract_long', 10)

    if row['FIRST_LENGTH'] <= 3:
        points += weights.get('first_short', 15)
    elif row['FIRST_LENGTH'] <= 8:
        points += weights.get('first_medium', 10)

    return max(0, points)


# Default weights from V3
DEFAULT_WEIGHTS = {
    'first_contract': 15,
    'very_new': 20,
    'establishing': 10,
    'mature_bonus': 5,
    'core_single': 20,
    'new_short': 15,
    'multi_renewal_bonus': 10,
    'tier_core': 25,
    'tier_growth': 15,
    'tier_enterprise': 5,
    'single_bp': 20,
    'two_bp': 10,
    'contract_short': 20,
    'contract_medium': 15,
    'contract_long': 10,
    'first_short': 15,
    'first_medium': 10,
}


def score_dataset(df: pd.DataFrame, weights: dict = None) -> pd.DataFrame:
    """Score dataset with given weights."""
    df = df.copy()
    df = engineer_features(df)
    weights = weights or DEFAULT_WEIGHTS
    df['RISK_SCORE'] = df.apply(lambda row: calculate_risk_score_parameterized(row, weights), axis=1)
    return df


def find_optimal_threshold_for_accuracy(df: pd.DataFrame, min_recall: float = 0.70) -> tuple:
    """Find threshold that maximizes accuracy while maintaining recall >= min_recall."""
    y_true = df['IS_CHURN']
    scores = df['RISK_SCORE']

    thresholds = np.arange(scores.min(), scores.max() + 1, 5)
    best_threshold = None
    best_accuracy = 0

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        if recall >= min_recall and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh

    return best_threshold, best_accuracy


def calculate_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Calculate metrics at given threshold."""
    y_true = df['IS_CHURN']
    y_pred = (df['RISK_SCORE'] >= threshold).astype(int)
    max_score = df['RISK_SCORE'].max()
    y_prob = df['RISK_SCORE'] / max_score if max_score > 0 else df['RISK_SCORE']

    return {
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
        'True Positives': ((y_pred == 1) & (y_true == 1)).sum(),
        'True Negatives': ((y_pred == 0) & (y_true == 0)).sum(),
        'False Positives': ((y_pred == 1) & (y_true == 0)).sum(),
        'False Negatives': ((y_pred == 0) & (y_true == 1)).sum(),
    }


def grid_search_weights(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        min_recall: float = 0.70) -> tuple:
    """
    Grid search over weight multipliers to find best configuration.

    Searches over multipliers for key weight groups.
    """
    print("\n[GRID SEARCH - Weight Optimization]")

    # Define multiplier ranges for each weight group
    multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    # Group weights for more efficient search
    weight_groups = {
        'new_features': ['first_contract', 'very_new', 'establishing', 'core_single', 'new_short'],
        'bonuses': ['mature_bonus', 'multi_renewal_bonus'],
        'tier': ['tier_core', 'tier_growth', 'tier_enterprise'],
        'blueprint': ['single_bp', 'two_bp'],
    }

    best_accuracy = 0
    best_config = None
    best_threshold = None
    results = []

    # Reduced grid: only search over 4 multiplier groups
    total_combinations = len(multipliers) ** 4
    print(f"  Searching {total_combinations} weight combinations...")

    count = 0
    for new_mult, bonus_mult, tier_mult, bp_mult in product(multipliers, repeat=4):
        count += 1
        if count % 100 == 0:
            print(f"  Progress: {count}/{total_combinations}")

        # Build weights with multipliers
        weights = DEFAULT_WEIGHTS.copy()
        for key in weight_groups['new_features']:
            weights[key] = int(DEFAULT_WEIGHTS[key] * new_mult)
        for key in weight_groups['bonuses']:
            weights[key] = int(DEFAULT_WEIGHTS[key] * bonus_mult)
        for key in weight_groups['tier']:
            weights[key] = int(DEFAULT_WEIGHTS[key] * tier_mult)
        for key in weight_groups['blueprint']:
            weights[key] = int(DEFAULT_WEIGHTS[key] * bp_mult)

        # Score validation set
        val_scored = score_dataset(val_df, weights)

        # Find best threshold for accuracy
        threshold, accuracy = find_optimal_threshold_for_accuracy(val_scored, min_recall)

        if threshold is not None and accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = {
                'new_features_mult': new_mult,
                'bonuses_mult': bonus_mult,
                'tier_mult': tier_mult,
                'blueprint_mult': bp_mult,
                'weights': weights.copy(),
            }
            best_threshold = threshold
            results.append({
                'new_mult': new_mult,
                'bonus_mult': bonus_mult,
                'tier_mult': tier_mult,
                'bp_mult': bp_mult,
                'threshold': threshold,
                'accuracy': accuracy,
            })

    print(f"\n  Best validation accuracy: {best_accuracy:.1%}")
    print(f"  Best threshold: {best_threshold}")
    print(f"  Multipliers: new={best_config['new_features_mult']}, "
          f"bonus={best_config['bonuses_mult']}, "
          f"tier={best_config['tier_mult']}, "
          f"bp={best_config['blueprint_mult']}")

    return best_config, best_threshold, pd.DataFrame(results)


def main():
    print("=" * 70)
    print("CHURN MODEL V3 - OPTIMIZED FOR ACCURACY")
    print("=" * 70)
    print(f"\nTargets: Accuracy >= {TARGET_ACCURACY:.0%}, Recall >= {TARGET_RECALL:.0%}")

    # Load data
    print("\n[1] Loading datasets...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    # First: Evaluate with default weights but optimize for accuracy
    print("\n[2] Evaluating with default weights (optimizing for accuracy)...")
    val_scored = score_dataset(val_df, DEFAULT_WEIGHTS)

    threshold_default, accuracy_default = find_optimal_threshold_for_accuracy(val_scored, TARGET_RECALL)
    print(f"  Default weights - Best threshold: {threshold_default}, Accuracy: {accuracy_default:.1%}")

    # Grid search for better weights
    print("\n[3] Running grid search for optimal weights...")
    best_config, best_threshold, search_results = grid_search_weights(train_df, val_df, TARGET_RECALL)

    # Evaluate on test set with best configuration
    print("\n[4] Evaluating on TEST set with best configuration...")
    test_scored = score_dataset(test_df, best_config['weights'])

    # Use threshold optimized on validation
    test_metrics = calculate_metrics(test_scored, best_threshold)

    # Also test with default weights
    test_default = score_dataset(test_df, DEFAULT_WEIGHTS)
    test_default_metrics = calculate_metrics(test_default, threshold_default)

    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS - TEST SET")
    print("=" * 70)

    print("\n[DEFAULT WEIGHTS]")
    print(f"  Threshold: {threshold_default}")
    print(f"  Accuracy:  {test_default_metrics['Accuracy']:.1%} {'PASS' if test_default_metrics['Accuracy'] >= TARGET_ACCURACY else 'FAIL'}")
    print(f"  Recall:    {test_default_metrics['Recall']:.1%} {'PASS' if test_default_metrics['Recall'] >= TARGET_RECALL else 'FAIL'}")
    print(f"  Precision: {test_default_metrics['Precision']:.1%}")
    print(f"  F1 Score:  {test_default_metrics['F1 Score']:.3f}")
    print(f"  AUC-ROC:   {test_default_metrics['AUC-ROC']:.3f}")

    print("\n[OPTIMIZED WEIGHTS]")
    print(f"  Threshold: {best_threshold}")
    print(f"  Accuracy:  {test_metrics['Accuracy']:.1%} {'PASS' if test_metrics['Accuracy'] >= TARGET_ACCURACY else 'FAIL'}")
    print(f"  Recall:    {test_metrics['Recall']:.1%} {'PASS' if test_metrics['Recall'] >= TARGET_RECALL else 'FAIL'}")
    print(f"  Precision: {test_metrics['Precision']:.1%}")
    print(f"  F1 Score:  {test_metrics['F1 Score']:.3f}")
    print(f"  AUC-ROC:   {test_metrics['AUC-ROC']:.3f}")

    print("\n[IMPROVEMENT]")
    print(f"  Accuracy: {test_default_metrics['Accuracy']:.1%} -> {test_metrics['Accuracy']:.1%} "
          f"({test_metrics['Accuracy'] - test_default_metrics['Accuracy']:+.1%})")
    print(f"  Recall:   {test_default_metrics['Recall']:.1%} -> {test_metrics['Recall']:.1%}")

    print("\n[CONFUSION MATRIX]")
    print(f"  True Positives:  {test_metrics['True Positives']}")
    print(f"  True Negatives:  {test_metrics['True Negatives']}")
    print(f"  False Positives: {test_metrics['False Positives']}")
    print(f"  False Negatives: {test_metrics['False Negatives']}")

    print("\n[BEST WEIGHT MULTIPLIERS]")
    print(f"  New features: {best_config['new_features_mult']}x")
    print(f"  Bonuses:      {best_config['bonuses_mult']}x")
    print(f"  Tier:         {best_config['tier_mult']}x")
    print(f"  Blueprint:    {best_config['blueprint_mult']}x")

    # Save results
    search_results.to_csv(RESULTS_DIR / 'grid_search_results.csv', index=False)

    final_results = pd.DataFrame([
        {**test_default_metrics, 'Version': 'Default Weights'},
        {**test_metrics, 'Version': 'Optimized Weights'},
    ])
    final_results.to_csv(RESULTS_DIR / 'final_results.csv', index=False)

    # Save best weights
    weights_df = pd.DataFrame([
        {'parameter': k, 'value': v}
        for k, v in best_config['weights'].items()
    ])
    weights_df.to_csv(RESULTS_DIR / 'optimized_weights.csv', index=False)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return test_metrics


if __name__ == "__main__":
    metrics = main()
