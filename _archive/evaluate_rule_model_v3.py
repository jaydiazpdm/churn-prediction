"""
Churn Prediction: Rule-Based Model Evaluation (V3 - Feature Engineering)

Improvements over V2:
1. NEW FEATURES from existing but unused data:
   - IS_FIRST_CONTRACT: Using ACTIVE_TIME (was completely unused!)
   - TENURE_BUCKET: Captures 9-month pivot point
   - CORE_SINGLE_EXPOSURE: Interaction feature for highest risk combo
   - RETAINER_SIZE_BUCKET: Uses FIRST_RETAINER_VALUE

2. WEIGHT ADJUSTMENTS:
   - Reduced weights for fixed components (urgency, financial)
   - Added weights for new engineered features

3. GOAL: >60% accuracy while maintaining >70% recall
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# Setup
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results_v3"
RESULTS_DIR.mkdir(exist_ok=True)

# Kill criteria (updated target)
KILL_CRITERIA_F1 = 0.4
TARGET_ACCURACY = 0.60
TARGET_RECALL = 0.70

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing but underutilized columns.

    Key insight: ACTIVE_TIME exists in the data but wasn't being used!
    """
    df = df.copy()

    # 1. IS_FIRST_CONTRACT: Are they still on their first contract?
    # First-contract clients churn at much higher rates
    df['IS_FIRST_CONTRACT'] = (df['ACTIVE_TIME'] <= df['FIRST_LENGTH']).astype(int)

    # 2. TENURE_BUCKET: Non-linear risk based on 9-month pivot point from EDA
    def get_tenure_bucket(months):
        if pd.isna(months):
            return 'Unknown'
        if months <= 3:
            return 'Very_New'      # Highest risk
        elif months <= 9:
            return 'Establishing'   # Medium risk (before pivot)
        elif months <= 18:
            return 'Stable'         # Lower risk (past pivot)
        else:
            return 'Mature'         # Lowest risk

    df['TENURE_BUCKET'] = df['ACTIVE_TIME'].apply(get_tenure_bucket)

    # 3. CORE_SINGLE_EXPOSURE: Interaction feature (highest risk combination)
    df['CORE_SINGLE_EXPOSURE'] = (
        (df['TIER_NAME'] == 'Core') & (df['TOTAL_BLUEPRINTS'] == 1)
    ).astype(int)

    # 4. RETAINER_SIZE_BUCKET: Using FIRST_RETAINER_VALUE
    def get_retainer_bucket(value):
        if pd.isna(value):
            return 'Unknown'
        if value <= 5000:
            return 'Small'
        elif value <= 10000:
            return 'Medium'
        elif value <= 20000:
            return 'Large'
        else:
            return 'Enterprise'

    df['RETAINER_SIZE_BUCKET'] = df['FIRST_RETAINER_VALUE'].apply(get_retainer_bucket)

    # 5. NEW_SHORT_CONTRACT_RISK: New client with short initial contract
    df['NEW_SHORT_CONTRACT_RISK'] = (
        (df['ACTIVE_TIME'] <= 6) & (df['FIRST_LENGTH'] <= 6)
    ).astype(int)

    # 6. CONTRACTS_COMPLETED: Approximate number of renewals
    df['CONTRACTS_COMPLETED'] = np.where(
        df['FIRST_LENGTH'] > 0,
        df['ACTIVE_TIME'] / df['FIRST_LENGTH'],
        0
    ).round(1)

    return df


# =============================================================================
# SCORING LOGIC V3
# =============================================================================

def calculate_risk_score_v3(row: pd.Series) -> int:
    """
    Enhanced scoring with new engineered features.

    Changes from V2:
    - Added points for IS_FIRST_CONTRACT (+15 if first contract)
    - Added points for TENURE_BUCKET (Very_New: +20, Establishing: +10)
    - Added points for CORE_SINGLE_EXPOSURE (+20 interaction bonus)
    - Added points for NEW_SHORT_CONTRACT_RISK (+15)
    - Reduced base points for urgency/financial (since they don't vary)

    Max possible score ~150 (increased from 120 to accommodate new features)
    """
    points = 0

    # =========================================================================
    # NEW FEATURE SCORES (highest impact additions)
    # =========================================================================

    # IS_FIRST_CONTRACT: First-contract clients are highest risk
    if row.get('IS_FIRST_CONTRACT', 0) == 1:
        points += 15

    # TENURE_BUCKET: Captures client maturity
    tenure_bucket = row.get('TENURE_BUCKET', 'Unknown')
    if tenure_bucket == 'Very_New':
        points += 20
    elif tenure_bucket == 'Establishing':
        points += 10
    elif tenure_bucket == 'Stable':
        points += 0
    elif tenure_bucket == 'Mature':
        points -= 5  # Protective factor (reduce risk)

    # CORE_SINGLE_EXPOSURE: Interaction bonus (Core + single BP is worse than sum)
    if row.get('CORE_SINGLE_EXPOSURE', 0) == 1:
        points += 20

    # NEW_SHORT_CONTRACT_RISK: Double risk flag
    if row.get('NEW_SHORT_CONTRACT_RISK', 0) == 1:
        points += 15

    # CONTRACTS_COMPLETED: Multiple renewals = lower risk
    contracts_completed = row.get('CONTRACTS_COMPLETED', 0)
    if contracts_completed >= 2:
        points -= 10  # Significant protective factor

    # =========================================================================
    # ORIGINAL FEATURES (with adjusted weights)
    # =========================================================================

    # TIER RISK (0-25 points) - unchanged, this varies
    if row['TIER_NAME'] == 'Core':
        points += 25
    elif row['TIER_NAME'] == 'Growth':
        points += 15
    elif row['TIER_NAME'] == 'Enterprise':
        points += 5

    # BLUEPRINT COUNT (0-20 points) - unchanged, this varies
    if row['TOTAL_BLUEPRINTS'] == 1:
        points += 20
    elif row['TOTAL_BLUEPRINTS'] == 2:
        points += 10
    # 3+ blueprints: 0 points (well diversified)

    # CONTRACT DURATION (0-20 points) - unchanged, this varies
    # Note: CONTRACT_DURATION is proxied by FIRST_LENGTH in prospective data
    if row['CONTRACT_DURATION'] <= 3:
        points += 20
    elif row['CONTRACT_DURATION'] <= 6:
        points += 15
    elif row['CONTRACT_DURATION'] <= 9:
        points += 10
    # >9 months: 0 points (past pivot)

    # FIRST_LENGTH (0-15 points) - unchanged
    if row['FIRST_LENGTH'] <= 3:
        points += 15
    elif row['FIRST_LENGTH'] <= 8:
        points += 10
    # >=9 months: 0 points (committed from start)

    # URGENCY (0-5 points) - REDUCED from 0-25 since MONTHS_UNTIL_END is fixed at 3
    # At 90-day horizon, everyone has ~3 months until end
    months_until_end = row['MONTHS_UNTIL_END']
    if months_until_end <= 1:
        points += 5
    elif months_until_end <= 3:
        points += 3  # This is where most clients fall
    elif months_until_end <= 6:
        points += 1
    # >6 months: 0 points

    # FINANCIAL (0-3 points) - REDUCED from 0-15 since DIFF_RETAINER is 0
    # No discriminative power at prospective horizon
    if row['DIFF_RETAINER'] <= -0.5:
        points += 3
    elif row['DIFF_RETAINER'] <= -0.2:
        points += 2
    elif row['DIFF_RETAINER'] <= 0:
        points += 1
    # Growing: 0 points

    # Ensure non-negative score
    return max(0, points)


def calculate_risk_score_v2_baseline(row: pd.Series) -> int:
    """
    Original V2 scoring for baseline comparison.
    Simplified version that captures the key logic.
    """
    points = 0
    months_until_end = row['MONTHS_UNTIL_END']

    if row['TIER_NAME'] == 'Core':
        points += 40
        if row['TOTAL_BLUEPRINTS'] == 1:
            points += 40
            if row['CONTRACT_DURATION'] <= 3:
                points += 40
                if months_until_end < 3:
                    points += 40
                else:
                    points += 5
            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if months_until_end <= 3:
                    points += 40
                else:
                    points += 5
            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if months_until_end <= 3:
                    points += 40
                else:
                    points += 5
            else:
                points += 5
        else:
            points += 10
            if row['DIFF_RETAINER'] <= -0.3:
                points += 20
            elif row['DIFF_RETAINER'] <= 0:
                points += 10
            if row['CONTRACT_DURATION'] <= 6:
                points += 20
            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
            else:
                points += 5

    elif row['TIER_NAME'] == 'Growth':
        points += 10
        if row['TOTAL_BLUEPRINTS'] == 1:
            points += 40
            if row['CONTRACT_DURATION'] <= 3:
                points += 40
            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
            else:
                points += 5
        else:
            points += 10
            if row['CONTRACT_DURATION'] <= 6:
                points += 20
            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
            else:
                points += 5

    elif row['TIER_NAME'] == 'Enterprise':
        points += 5
        if row['TOTAL_BLUEPRINTS'] == 1:
            points += 40
            if row['CONTRACT_DURATION'] <= 3:
                points += 20
            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
            else:
                points += 5
        else:
            points += 10
            if row['CONTRACT_DURATION'] <= 6:
                points += 20
            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
            else:
                points += 5

    return points


# =============================================================================
# DATA LOADING AND SCORING
# =============================================================================

def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, val_df, test_df


def score_dataset(df: pd.DataFrame, version: str = 'v3') -> pd.DataFrame:
    """Apply risk scoring to a dataset."""
    df = df.copy()

    # Engineer features first
    df = engineer_features(df)

    # Apply scoring
    if version == 'v3':
        df['RISK_SCORE'] = df.apply(calculate_risk_score_v3, axis=1)
    else:
        df['RISK_SCORE'] = df.apply(calculate_risk_score_v2_baseline, axis=1)

    return df


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def find_optimal_threshold(df: pd.DataFrame, metric: str = 'f1',
                          min_recall: float = 0.0) -> tuple[float, pd.DataFrame]:
    """
    Find optimal threshold on validation set.

    Args:
        df: Scored DataFrame
        metric: Metric to optimize ('f1', 'accuracy', etc.)
        min_recall: Minimum recall constraint
    """
    y_true = df['IS_CHURN']
    scores = df['RISK_SCORE']

    thresholds = np.arange(scores.min(), scores.max() + 1, 5)
    results = []

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)

        results.append({
            'threshold': thresh,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall,
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })

    results_df = pd.DataFrame(results)

    # Filter by minimum recall constraint if specified
    if min_recall > 0:
        valid_results = results_df[results_df['recall'] >= min_recall]
        if len(valid_results) > 0:
            best_idx = valid_results[metric].idxmax()
        else:
            print(f"  Warning: No threshold achieves recall >= {min_recall}")
            best_idx = results_df[metric].idxmax()
    else:
        best_idx = results_df[metric].idxmax()

    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df


def calculate_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Calculate all metrics at a given threshold."""
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


def analyze_feature_impact(df: pd.DataFrame):
    """Analyze how new features correlate with churn."""
    print("\n[FEATURE IMPACT ANALYSIS]")

    # IS_FIRST_CONTRACT
    first_contract_churn = df[df['IS_FIRST_CONTRACT'] == 1]['IS_CHURN'].mean()
    multi_contract_churn = df[df['IS_FIRST_CONTRACT'] == 0]['IS_CHURN'].mean()
    print(f"\n  IS_FIRST_CONTRACT:")
    print(f"    First contract clients churn rate:  {first_contract_churn:.1%}")
    print(f"    Multi-contract clients churn rate:  {multi_contract_churn:.1%}")
    print(f"    Lift: {first_contract_churn / multi_contract_churn:.1f}x" if multi_contract_churn > 0 else "")

    # TENURE_BUCKET
    print(f"\n  TENURE_BUCKET churn rates:")
    for bucket in ['Very_New', 'Establishing', 'Stable', 'Mature']:
        subset = df[df['TENURE_BUCKET'] == bucket]
        if len(subset) > 0:
            churn_rate = subset['IS_CHURN'].mean()
            print(f"    {bucket}: {churn_rate:.1%} (n={len(subset)})")

    # CORE_SINGLE_EXPOSURE
    exposure_churn = df[df['CORE_SINGLE_EXPOSURE'] == 1]['IS_CHURN'].mean()
    no_exposure_churn = df[df['CORE_SINGLE_EXPOSURE'] == 0]['IS_CHURN'].mean()
    print(f"\n  CORE_SINGLE_EXPOSURE:")
    print(f"    Core + Single BP churn rate: {exposure_churn:.1%}")
    print(f"    Other combinations:          {no_exposure_churn:.1%}")

    # CONTRACTS_COMPLETED
    print(f"\n  CONTRACTS_COMPLETED:")
    for n in [0, 1, 2]:
        if n == 2:
            subset = df[df['CONTRACTS_COMPLETED'] >= 2]
            label = "2+"
        else:
            subset = df[(df['CONTRACTS_COMPLETED'] >= n) & (df['CONTRACTS_COMPLETED'] < n+1)]
            label = str(n)
        if len(subset) > 0:
            churn_rate = subset['IS_CHURN'].mean()
            print(f"    {label} contracts: {churn_rate:.1%} (n={len(subset)})")


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_comparison(v2_results: pd.DataFrame, v3_results: pd.DataFrame,
                   v2_threshold: float, v3_threshold: float):
    """Plot V2 vs V3 comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Threshold sweep comparison
    ax = axes[0]
    ax.plot(v2_results['threshold'], v2_results['f1'],
            label='V2 (baseline)', linewidth=2, linestyle='--', alpha=0.7)
    ax.plot(v3_results['threshold'], v3_results['f1'],
            label='V3 (feature engineering)', linewidth=2)
    ax.axvline(x=v2_threshold, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=v3_threshold, color='blue', linestyle=':', alpha=0.5)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score: V2 vs V3')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Metric comparison at optimal thresholds
    ax = axes[1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    v2_vals = [v2_results.loc[v2_results['threshold'] == v2_threshold, m.lower()].values[0]
               for m in ['accuracy', 'precision', 'recall', 'f1']]
    v3_vals = [v3_results.loc[v3_results['threshold'] == v3_threshold, m.lower()].values[0]
               for m in ['accuracy', 'precision', 'recall', 'f1']]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, v2_vals, width, label='V2 (baseline)', alpha=0.7)
    ax.bar(x + width/2, v3_vals, width, label='V3 (feature eng.)')
    ax.set_ylabel('Score')
    ax.set_title('Metrics Comparison (Validation Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Target (60%)')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'v2_v3_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(df: pd.DataFrame, threshold: float, split_name: str):
    """Plot confusion matrix."""
    y_true = df['IS_CHURN']
    y_pred = (df['RISK_SCORE'] >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Active', 'Predicted Churn'],
                yticklabels=['Actual Active', 'Actual Churn'])
    ax.set_title(f'Confusion Matrix - {split_name} (Threshold = {threshold:.0f})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'confusion_matrix_{split_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distribution(train_df, val_df, test_df):
    """Plot risk score distributions by churn status."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, df) in zip(axes, [("Train", train_df), ("Validation", val_df), ("Test", test_df)]):
        for status, label, color in [(0, 'Active', '#2ecc71'), (1, 'Churn', '#e74c3c')]:
            subset = df[df['IS_CHURN'] == status]['RISK_SCORE']
            ax.hist(subset, bins=20, alpha=0.5, label=label, color=color, density=True)
        ax.set_title(f'{name} Set', fontsize=12, fontweight='bold')
        ax.set_xlabel('Risk Score (V3)')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'score_distribution_v3.png', dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def print_summary(v2_metrics, v3_metrics, best_threshold):
    """Print comparison summary."""
    print("\n" + "=" * 70)
    print("CHURN MODEL EVALUATION: V2 (BASELINE) vs V3 (FEATURE ENGINEERING)")
    print("=" * 70)

    print("\n[NEW FEATURES IN V3]")
    print("  1. IS_FIRST_CONTRACT - Using ACTIVE_TIME (was unused!)")
    print("  2. TENURE_BUCKET - Captures 9-month pivot point")
    print("  3. CORE_SINGLE_EXPOSURE - Interaction: Core + single blueprint")
    print("  4. NEW_SHORT_CONTRACT_RISK - New client + short contract")
    print("  5. CONTRACTS_COMPLETED - Renewal history proxy")

    print("\n[WEIGHT ADJUSTMENTS IN V3]")
    print("  - Reduced urgency points: 25 -> 5 (MONTHS_UNTIL_END fixed at 3)")
    print("  - Reduced financial points: 15 -> 3 (DIFF_RETAINER fixed at 0)")
    print("  - Added points for new features")

    print(f"\n[OPTIMAL THRESHOLD: {best_threshold}]")

    print("\n[METRICS COMPARISON - TEST SET]")
    print(f"{'Metric':<15} {'V2 Baseline':>12} {'V3 Features':>12} {'Change':>10}")
    print("-" * 50)

    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']:
        v2_val = v2_metrics[metric]
        v3_val = v3_metrics[metric]
        change = v3_val - v2_val
        sign = "+" if change >= 0 else ""
        print(f"{metric:<15} {v2_val:>11.1%} {v3_val:>11.1%} {sign}{change:>9.1%}")

    print("\n[TARGET EVALUATION]")
    print(f"  Accuracy >= 60%: {'PASS' if v3_metrics['Accuracy'] >= 0.60 else 'FAIL'} ({v3_metrics['Accuracy']:.1%})")
    print(f"  Recall >= 70%:   {'PASS' if v3_metrics['Recall'] >= 0.70 else 'FAIL'} ({v3_metrics['Recall']:.1%})")
    print(f"  F1 >= 0.60:      {'PASS' if v3_metrics['F1 Score'] >= 0.60 else 'FAIL'} ({v3_metrics['F1 Score']:.3f})")

    print("\n[CONFUSION MATRIX - TEST SET]")
    print(f"  True Positives:  {v3_metrics['True Positives']} (correctly predicted churn)")
    print(f"  True Negatives:  {v3_metrics['True Negatives']} (correctly predicted active)")
    print(f"  False Positives: {v3_metrics['False Positives']} (false alarms)")
    print(f"  False Negatives: {v3_metrics['False Negatives']} (missed churns)")


def main():
    print("=" * 70)
    print("CHURN MODEL V3 EVALUATION - FEATURE ENGINEERING")
    print("=" * 70)

    print("\n[1] Loading datasets...")
    train_df, val_df, test_df = load_datasets()
    print(f"  Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    print("\n[2] Engineering features and scoring V3...")
    train_v3 = score_dataset(train_df, version='v3')
    val_v3 = score_dataset(val_df, version='v3')
    test_v3 = score_dataset(test_df, version='v3')

    print("\n[3] Scoring V2 baseline for comparison...")
    train_v2 = score_dataset(train_df, version='v2')
    val_v2 = score_dataset(val_df, version='v2')
    test_v2 = score_dataset(test_df, version='v2')

    print("\n[4] Analyzing feature impact...")
    analyze_feature_impact(train_v3)

    print("\n[5] Finding optimal thresholds (validation set)...")
    # V2 baseline
    v2_threshold, v2_results = find_optimal_threshold(val_v2, metric='f1')
    print(f"  V2 optimal threshold: {v2_threshold}")

    # V3 with recall constraint
    v3_threshold, v3_results = find_optimal_threshold(val_v3, metric='f1', min_recall=TARGET_RECALL)
    print(f"  V3 optimal threshold: {v3_threshold} (with recall >= {TARGET_RECALL})")

    print("\n[6] Calculating final metrics...")
    v2_test_metrics = calculate_metrics(test_v2, v2_threshold)
    v3_test_metrics = calculate_metrics(test_v3, v3_threshold)

    print("\n[7] Generating plots...")
    plot_comparison(v2_results, v3_results, v2_threshold, v3_threshold)
    plot_confusion_matrix(test_v3, v3_threshold, "Test_V3")
    plot_score_distribution(train_v3, val_v3, test_v3)
    print(f"  Plots saved to {RESULTS_DIR}/")

    # Print summary
    print_summary(v2_test_metrics, v3_test_metrics, v3_threshold)

    # Save results
    all_metrics = pd.DataFrame([
        {**calculate_metrics(train_v3, v3_threshold), 'Split': 'Train', 'Version': 'V3'},
        {**calculate_metrics(val_v3, v3_threshold), 'Split': 'Validation', 'Version': 'V3'},
        {**v3_test_metrics, 'Split': 'Test', 'Version': 'V3'},
        {**calculate_metrics(train_v2, v2_threshold), 'Split': 'Train', 'Version': 'V2'},
        {**calculate_metrics(val_v2, v2_threshold), 'Split': 'Validation', 'Version': 'V2'},
        {**v2_test_metrics, 'Split': 'Test', 'Version': 'V2'},
    ])
    all_metrics.to_csv(RESULTS_DIR / 'evaluation_report_v3.csv', index=False)

    v3_results.to_csv(RESULTS_DIR / 'threshold_sweep_v3.csv', index=False)

    print(f"\nSaved: {RESULTS_DIR / 'evaluation_report_v3.csv'}")
    print(f"Saved: {RESULTS_DIR / 'threshold_sweep_v3.csv'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return v3_test_metrics


if __name__ == "__main__":
    metrics = main()
