"""
Churn Prediction: Rule-Based Model Evaluation (V2 - Proper Methodology)

This version fixes critical methodological issues from V1:
1. Uses prospective features (90-day prediction horizon)
2. Proper train/validation/test split
3. Threshold optimization on validation set only
4. F1 > 0.4 as kill criteria (not accuracy)
5. Clear documentation of limitations

Key changes from V1:
- MONTHS_UNTIL_END is fixed at 3 (by definition of 90-day horizon)
- DIFF_RETAINER is 0 (no change known at decision point)
- CONTRACT_DURATION uses FIRST_LENGTH as proxy
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
RESULTS_DIR = Path(__file__).parent / "results_v2"
RESULTS_DIR.mkdir(exist_ok=True)

# Kill criteria
KILL_CRITERIA_F1 = 0.4

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_risk_score_prospective(row: pd.Series) -> int:
    """
    Apply rule-based scoring logic adapted for 90-day prediction horizon.

    Key differences from original:
    - MONTHS_UNTIL_END is always 3 (by definition)
    - Original checks like "MONTHS_UNTIL_END < 3" are always FALSE
    - Model relies more heavily on TIER_NAME, TOTAL_BLUEPRINTS, CONTRACT_DURATION (from FIRST_LENGTH)

    NOTE: This dramatically changes the scoring behavior. The original model
    heavily weighted imminent renewal timing, which is now removed.
    """
    points = 0

    # At 90-day horizon, MONTHS_UNTIL_END = 3 always
    # So any check for < 3, <= 2, <= 1 is FALSE
    months_until_end = row['MONTHS_UNTIL_END']  # Will be 3.0

    if row['TIER_NAME'] == 'Core':
        points += 40

        if row['TOTAL_BLUEPRINTS'] == 1:
            points += 40

            if row['CONTRACT_DURATION'] <= 3:
                points += 40
                # MONTHS_UNTIL_END < 3 is FALSE when horizon = 3
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

            # DIFF_RETAINER is 0 at prospective horizon
            if row['DIFF_RETAINER'] <= -0.3:
                points += 20
            elif row['DIFF_RETAINER'] <= 0:
                points += 10
            else:
                points += 0

            if row['CONTRACT_DURATION'] <= 3:
                points += 20
                if months_until_end < 3:
                    points += 40
                    if row['FIRST_LENGTH'] <= 3:
                        points += 40
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if months_until_end <= 3:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 40
                    elif row['FIRST_LENGTH'] <= 3:
                        points += 10
                    elif row['FIRST_LENGTH'] <= 9:
                        points += 20
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if months_until_end <= 3:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 20
                    elif row['FIRST_LENGTH'] <= 9:
                        points += 10
                    else:
                        points += 5
                else:
                    points += 5
            else:
                points += 5
                if months_until_end <= 3:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 20
                    elif row['FIRST_LENGTH'] <= 9:
                        points += 10
                    else:
                        points += 5
                else:
                    points += 5

    elif row['TIER_NAME'] == 'Growth':
        points += 10

        if row['TOTAL_BLUEPRINTS'] == 1:
            points += 40

            if row['CONTRACT_DURATION'] <= 3:
                points += 40
                if months_until_end < 3:
                    points += 5
                else:
                    points += 40

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if months_until_end <= 2:
                    points += 40
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if months_until_end <= 2:
                    points += 40
                else:
                    points += 5
            else:
                points += 5

        else:
            points += 10

            if row['DIFF_RETAINER'] <= -0.6:
                points += 40
            elif row['DIFF_RETAINER'] <= -0.3:
                points += 20
            elif row['DIFF_RETAINER'] <= 0:
                points += 10
            else:
                points += 5

            if row['CONTRACT_DURATION'] <= 3:
                points += 20
                if months_until_end < 2:
                    points += 40
                    if row['FIRST_LENGTH'] <= 3:
                        points += 40
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if months_until_end <= 2:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 40
                    elif row['FIRST_LENGTH'] <= 3:
                        points += 5
                    elif row['FIRST_LENGTH'] < 9:
                        points += 20
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if months_until_end <= 2:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 20
                    elif row['FIRST_LENGTH'] <= 9:
                        points += 5
                    else:
                        points += 5
                else:
                    points += 5
            else:
                points += 5
                if months_until_end <= 2:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 20
                    elif row['FIRST_LENGTH'] <= 9:
                        points += 10
                    else:
                        points += 5
                else:
                    points += 5

    elif row['TIER_NAME'] == 'Enterprise':
        points += 5

        if row['TOTAL_BLUEPRINTS'] == 1:
            points += 40

            if row['CONTRACT_DURATION'] <= 3:
                points += 20
                if months_until_end < 2:
                    points += 20
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if months_until_end <= 2:
                    points += 40
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if months_until_end <= 2:
                    points += 40
                else:
                    points += 5
            else:
                points += 5

        else:
            points += 10

            if row['DIFF_RETAINER'] <= -0.6:
                points += 40
            elif row['DIFF_RETAINER'] <= 0:
                points += 20
            else:
                points += 5

            if row['CONTRACT_DURATION'] <= 3:
                points += 20
                if months_until_end < 1:
                    points += 40
                    if row['FIRST_LENGTH'] <= 3:
                        points += 10
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if months_until_end <= 1:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 40
                    elif row['FIRST_LENGTH'] <= 3:
                        points += 5
                    elif row['FIRST_LENGTH'] < 9:
                        points += 20
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if months_until_end <= 1:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 20
                    elif row['FIRST_LENGTH'] <= 9:
                        points += 5
                    else:
                        points += 5
                else:
                    points += 5
            else:
                points += 5
                if months_until_end <= 1:
                    points += 40
                    if (row['FIRST_LENGTH'] >= 9) and (row['FIRST_LENGTH'] <= 10):
                        points += 20
                    elif row['FIRST_LENGTH'] <= 9:
                        points += 10
                    else:
                        points += 5
                else:
                    points += 5

    return points


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test datasets."""
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, val_df, test_df


def score_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply risk scoring to a dataset."""
    df = df.copy()
    df['RISK_SCORE'] = df.apply(calculate_risk_score_prospective, axis=1)
    return df


def find_optimal_threshold(df: pd.DataFrame, metric: str = 'f1') -> tuple[float, pd.DataFrame]:
    """
    Find optimal threshold on validation set.

    Returns the best threshold and a DataFrame with all threshold results.
    """
    y_true = df['IS_CHURN']
    scores = df['RISK_SCORE']

    thresholds = np.arange(scores.min(), scores.max() + 1, 5)
    results = []

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        results.append({
            'threshold': thresh,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df[metric].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    return best_threshold, results_df


def calculate_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Calculate all metrics at a given threshold."""
    y_true = df['IS_CHURN']
    y_pred = (df['RISK_SCORE'] >= threshold).astype(int)
    y_prob = df['RISK_SCORE'] / df['RISK_SCORE'].max() if df['RISK_SCORE'].max() > 0 else df['RISK_SCORE']

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


def plot_threshold_analysis(results_df: pd.DataFrame, best_threshold: float, split_name: str):
    """Plot metrics vs threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        ax.plot(results_df['threshold'], results_df[metric], label=metric.capitalize(), linewidth=2)

    ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7,
               label=f'Best threshold ({best_threshold:.0f})')
    ax.set_xlabel('Threshold (Risk Score)', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Model Performance vs. Threshold ({split_name})', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'threshold_analysis_{split_name.lower()}.png', dpi=150, bbox_inches='tight')
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


def plot_roc_curve(df: pd.DataFrame, split_name: str):
    """Plot ROC curve."""
    y_true = df['IS_CHURN']
    y_prob = df['RISK_SCORE'] / df['RISK_SCORE'].max() if df['RISK_SCORE'].max() > 0 else df['RISK_SCORE']

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {split_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'roc_curve_{split_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distribution(train_df, val_df, test_df):
    """Plot risk score distributions by status across splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, df) in zip(axes, [("Train", train_df), ("Validation", val_df), ("Test", test_df)]):
        for status, color in [('Active', '#2ecc71'), ('Churn', '#e74c3c')]:
            subset = df[df['CLIENT_STATUS'] == status]['RISK_SCORE']
            ax.hist(subset, bins=15, alpha=0.5, label=status, color=color, density=True)
        ax.set_title(f'{name} Set', fontsize=12, fontweight='bold')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Density')
        ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def print_summary(train_metrics, val_metrics, test_metrics, best_threshold):
    """Print final summary report."""
    print("\n" + "=" * 70)
    print("CHURN PREDICTION MODEL EVALUATION SUMMARY (V2 - Proper Methodology)")
    print("=" * 70)

    print("\n[METHODOLOGY]")
    print("  Prediction Horizon: 90 days before contract end")
    print("  Feature Treatment: Prospective (no data leakage)")
    print("  Split Strategy: Stratified random (60/20/20)")
    print("  Threshold Selection: Optimized on VALIDATION set")
    print("  Final Metrics: Reported on TEST set")

    print("\n[DATASET SPLITS]")
    for name, metrics in [("Train", train_metrics), ("Validation", val_metrics), ("Test", test_metrics)]:
        total = metrics['True Positives'] + metrics['True Negatives'] + \
                metrics['False Positives'] + metrics['False Negatives']
        print(f"  {name}: {total} samples")

    print(f"\n[OPTIMAL THRESHOLD]")
    print(f"  Selected: {best_threshold:.0f} (from validation set)")

    print("\n[VALIDATION SET METRICS]")
    print(f"  Accuracy:  {val_metrics['Accuracy']:.1%}")
    print(f"  Precision: {val_metrics['Precision']:.1%}")
    print(f"  Recall:    {val_metrics['Recall']:.1%}")
    print(f"  F1 Score:  {val_metrics['F1 Score']:.3f}")
    print(f"  AUC-ROC:   {val_metrics['AUC-ROC']:.3f}")

    print("\n[TEST SET METRICS - FINAL EVALUATION]")
    print(f"  Accuracy:  {test_metrics['Accuracy']:.1%}")
    print(f"  Precision: {test_metrics['Precision']:.1%}")
    print(f"  Recall:    {test_metrics['Recall']:.1%}")
    print(f"  F1 Score:  {test_metrics['F1 Score']:.3f}  ", end="")

    # Kill criteria evaluation
    if test_metrics['F1 Score'] >= KILL_CRITERIA_F1:
        print(f"PASS (kill criteria: F1 >= {KILL_CRITERIA_F1})")
    else:
        print(f"FAIL (kill criteria: F1 >= {KILL_CRITERIA_F1})")

    print(f"  AUC-ROC:   {test_metrics['AUC-ROC']:.3f}")

    print("\n[TEST SET CONFUSION MATRIX]")
    print(f"  True Positives (correctly predicted churn):  {test_metrics['True Positives']}")
    print(f"  True Negatives (correctly predicted active): {test_metrics['True Negatives']}")
    print(f"  False Positives (false alarms):              {test_metrics['False Positives']}")
    print(f"  False Negatives (missed churns):             {test_metrics['False Negatives']}")

    print("\n[IMPORTANT LIMITATIONS]")
    print("  1. No temporal split (dataset lacks date columns)")
    print("  2. DIFF_RETAINER set to 0 (no monthly retainer data)")
    print("  3. CONTRACT_DURATION proxied by FIRST_LENGTH")
    print("  4. Model logic designed for imminent renewal detection,")
    print("     but now evaluated at 90-day horizon where timing signals are lost")


def main():
    print("Loading datasets...")
    train_df, val_df, test_df = load_datasets()

    print("\nScoring datasets...")
    train_df = score_dataset(train_df)
    val_df = score_dataset(val_df)
    test_df = score_dataset(test_df)

    print("\nFinding optimal threshold on VALIDATION set...")
    best_threshold, val_threshold_results = find_optimal_threshold(val_df, metric='f1')
    print(f"  Optimal threshold: {best_threshold}")

    print("\nCalculating metrics...")
    train_metrics = calculate_metrics(train_df, best_threshold)
    val_metrics = calculate_metrics(val_df, best_threshold)
    test_metrics = calculate_metrics(test_df, best_threshold)

    print("\nGenerating plots...")
    plot_threshold_analysis(val_threshold_results, best_threshold, "Validation")
    plot_confusion_matrix(test_df, best_threshold, "Test")
    plot_roc_curve(test_df, "Test")
    plot_score_distribution(train_df, val_df, test_df)
    print(f"  Plots saved to {RESULTS_DIR}/")

    # Print summary
    print_summary(train_metrics, val_metrics, test_metrics, best_threshold)

    # Save metrics
    all_metrics = pd.DataFrame([
        {**train_metrics, 'Split': 'Train'},
        {**val_metrics, 'Split': 'Validation'},
        {**test_metrics, 'Split': 'Test'}
    ])
    all_metrics.to_csv(RESULTS_DIR / 'evaluation_report.csv', index=False)
    print(f"\nSaved: {RESULTS_DIR / 'evaluation_report.csv'}")

    val_threshold_results.to_csv(RESULTS_DIR / 'threshold_sweep_validation.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'threshold_sweep_validation.csv'}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return test_metrics


if __name__ == "__main__":
    test_metrics = main()
