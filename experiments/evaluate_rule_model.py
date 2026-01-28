"""
Churn Prediction: Rule-Based Model Evaluation

Evaluates the current rule-based scoring model to understand baseline performance.
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
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_PATH = Path(__file__).parent.parent / "dataset.csv"

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_risk_score(row: pd.Series) -> int:
    """Apply the rule-based scoring logic from model/model.py"""
    points = 0

    if row['TIER_NAME'] == 'Core':
        points += 40

        if row['TOTAL_BLUEPRINTS'] == 1:
            points += 40

            if row['CONTRACT_DURATION'] <= 3:
                points += 40
                if row['MONTHS_UNTIL_END'] < 3:
                    points += 40
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if row['MONTHS_UNTIL_END'] <= 3:
                    points += 40
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if row['MONTHS_UNTIL_END'] <= 3:
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
            else:
                points += 0

            if row['CONTRACT_DURATION'] <= 3:
                points += 20
                if row['MONTHS_UNTIL_END'] < 3:
                    points += 40
                    if row['FIRST_LENGTH'] <= 3:
                        points += 40
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if row['MONTHS_UNTIL_END'] <= 3:
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
                if row['MONTHS_UNTIL_END'] <= 3:
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
                if row['MONTHS_UNTIL_END'] <= 3:
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
                if row['MONTHS_UNTIL_END'] < 3:
                    points += 5
                else:
                    points += 40

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if row['MONTHS_UNTIL_END'] <= 2:
                    points += 40
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if row['MONTHS_UNTIL_END'] <= 2:
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
                if row['MONTHS_UNTIL_END'] < 2:
                    points += 40
                    if row['FIRST_LENGTH'] <= 3:
                        points += 40
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if row['MONTHS_UNTIL_END'] <= 2:
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
                if row['MONTHS_UNTIL_END'] <= 2:
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
                if row['MONTHS_UNTIL_END'] <= 2:
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
                if row['MONTHS_UNTIL_END'] < 2:
                    points += 20
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if row['MONTHS_UNTIL_END'] <= 2:
                    points += 40
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 9:
                points += 10
                if row['MONTHS_UNTIL_END'] <= 2:
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
                if row['MONTHS_UNTIL_END'] < 1:
                    points += 40
                    if row['FIRST_LENGTH'] <= 3:
                        points += 10
                    else:
                        points += 5
                else:
                    points += 5

            elif row['CONTRACT_DURATION'] <= 6:
                points += 20
                if row['MONTHS_UNTIL_END'] <= 1:
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
                if row['MONTHS_UNTIL_END'] <= 1:
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
                if row['MONTHS_UNTIL_END'] <= 1:
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


def load_and_score_data() -> pd.DataFrame:
    """Load dataset and apply risk scoring"""
    df = pd.read_csv(DATA_PATH)
    df['RISK_SCORE'] = df.apply(calculate_risk_score, axis=1)
    df['IS_CHURN'] = (df['CLIENT_STATUS'] == 'Churn').astype(int)
    return df


def plot_score_distribution(df: pd.DataFrame):
    """Create boxplot and histogram of risk scores by status"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    ax1 = axes[0]
    colors = {'Active': '#2ecc71', 'Churn': '#e74c3c'}
    df.boxplot(column='RISK_SCORE', by='CLIENT_STATUS', ax=ax1)
    ax1.set_title('Risk Score Distribution by Client Status', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Client Status')
    ax1.set_ylabel('Risk Score')
    plt.suptitle('')  # Remove automatic title

    # Add means as diamond markers
    means = df.groupby('CLIENT_STATUS')['RISK_SCORE'].mean()
    for i, (status, mean_val) in enumerate(means.items(), 1):
        ax1.scatter(i, mean_val, marker='D', color='red', s=50, zorder=5, label=f'Mean' if i == 1 else '')
    ax1.legend()

    # Histogram/KDE
    ax2 = axes[1]
    for status, color in colors.items():
        subset = df[df['CLIENT_STATUS'] == status]['RISK_SCORE']
        ax2.hist(subset, bins=20, alpha=0.5, label=status, color=color, density=True)
        subset.plot.kde(ax=ax2, color=color, linewidth=2)
    ax2.set_title('Risk Score Density by Client Status', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Risk Score')
    ax2.set_ylabel('Density')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'score_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'score_boxplot.png'}")


def analyze_thresholds(df: pd.DataFrame) -> dict:
    """Sweep thresholds and find optimal cutoff"""
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

    # Find optimal threshold (max F1)
    best_idx = results_df['f1'].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        ax.plot(results_df['threshold'], results_df[metric], label=metric.capitalize(), linewidth=2)

    ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, label=f'Best threshold ({best_threshold:.0f})')
    ax.set_xlabel('Threshold (Risk Score)', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Model Performance vs. Risk Score Threshold', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'threshold_analysis.png'}")

    return {
        'best_threshold': best_threshold,
        'results_df': results_df,
        'best_metrics': results_df.loc[best_idx].to_dict()
    }


def calculate_final_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Calculate all metrics at optimal threshold"""
    y_true = df['IS_CHURN']
    y_pred = (df['RISK_SCORE'] >= threshold).astype(int)
    y_prob = df['RISK_SCORE'] / df['RISK_SCORE'].max()  # Normalize for AUC

    metrics = {
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_prob),
        'True Positives': ((y_pred == 1) & (y_true == 1)).sum(),
        'True Negatives': ((y_pred == 0) & (y_true == 0)).sum(),
        'False Positives': ((y_pred == 1) & (y_true == 0)).sum(),
        'False Negatives': ((y_pred == 0) & (y_true == 1)).sum(),
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Active', 'Predicted Churn'],
                yticklabels=['Actual Active', 'Actual Churn'])
    ax.set_title(f'Confusion Matrix (Threshold = {threshold:.0f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'confusion_matrix.png'}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metrics["AUC-ROC"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'roc_curve.png'}")

    return metrics


def error_analysis(df: pd.DataFrame, threshold: float):
    """Analyze where the model fails"""
    y_true = df['IS_CHURN']
    y_pred = (df['RISK_SCORE'] >= threshold).astype(int)

    df['PREDICTION'] = y_pred
    df['CORRECT'] = (y_true == y_pred)

    # False negatives (churned but predicted active - missed churns)
    fn = df[(y_true == 1) & (y_pred == 0)]
    # False positives (active but predicted churn - false alarms)
    fp = df[(y_true == 0) & (y_pred == 1)]

    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    print(f"\nFalse Negatives (Missed Churns): {len(fn)}")
    if len(fn) > 0:
        print("  Tier breakdown:")
        print(fn['TIER_NAME'].value_counts().to_string().replace('\n', '\n    '))
        print(f"  Avg Risk Score: {fn['RISK_SCORE'].mean():.1f}")
        print(f"  Avg Contract Duration: {fn['CONTRACT_DURATION'].mean():.1f} months")

    print(f"\nFalse Positives (False Alarms): {len(fp)}")
    if len(fp) > 0:
        print("  Tier breakdown:")
        print(fp['TIER_NAME'].value_counts().to_string().replace('\n', '\n    '))
        print(f"  Avg Risk Score: {fp['RISK_SCORE'].mean():.1f}")
        print(f"  Avg Contract Duration: {fp['CONTRACT_DURATION'].mean():.1f} months")

    # Accuracy by tier
    print("\nAccuracy by Tier:")
    for tier in df['TIER_NAME'].unique():
        tier_df = df[df['TIER_NAME'] == tier]
        acc = accuracy_score(tier_df['IS_CHURN'], tier_df['PREDICTION'])
        print(f"  {tier}: {acc:.1%} (n={len(tier_df)})")

    return fn, fp


def print_summary(df: pd.DataFrame, metrics: dict, threshold_results: dict):
    """Print final summary report"""
    print("\n" + "="*60)
    print("CHURN PREDICTION MODEL EVALUATION SUMMARY")
    print("="*60)

    print("\n[DATASET]")
    print(f"  Total clients: {len(df)}")
    print(f"  Active: {(df['CLIENT_STATUS'] == 'Active').sum()} ({(df['CLIENT_STATUS'] == 'Active').mean():.1%})")
    print(f"  Churned: {(df['CLIENT_STATUS'] == 'Churn').sum()} ({(df['CLIENT_STATUS'] == 'Churn').mean():.1%})")

    print("\n[RISK SCORE DISTRIBUTION]")
    for status in ['Active', 'Churn']:
        subset = df[df['CLIENT_STATUS'] == status]['RISK_SCORE']
        print(f"  {status}:")
        print(f"    Mean: {subset.mean():.1f}, Median: {subset.median():.1f}, Std: {subset.std():.1f}")
        print(f"    Range: [{subset.min():.0f}, {subset.max():.0f}]")

    print("\n[MODEL PERFORMANCE]")
    print(f"  Optimal Threshold: {metrics['Threshold']:.0f}")
    print(f"  Accuracy:  {metrics['Accuracy']:.1%}  {'PASS' if metrics['Accuracy'] >= 0.5 else 'FAIL'} (kill criteria: >=50%)")
    print(f"  Precision: {metrics['Precision']:.1%}")
    print(f"  Recall:    {metrics['Recall']:.1%}")
    print(f"  F1 Score:  {metrics['F1 Score']:.3f}")
    print(f"  AUC-ROC:   {metrics['AUC-ROC']:.3f}")

    print("\n[CONFUSION MATRIX]")
    print(f"  True Positives (correctly predicted churn):  {metrics['True Positives']}")
    print(f"  True Negatives (correctly predicted active): {metrics['True Negatives']}")
    print(f"  False Positives (false alarms):              {metrics['False Positives']}")
    print(f"  False Negatives (missed churns):             {metrics['False Negatives']}")


def main():
    print("Loading and scoring data...")
    df = load_and_score_data()

    print("\nGenerating score distribution plots...")
    plot_score_distribution(df)

    print("\nAnalyzing thresholds...")
    threshold_results = analyze_thresholds(df)
    best_threshold = threshold_results['best_threshold']

    print(f"\nCalculating metrics at optimal threshold ({best_threshold})...")
    metrics = calculate_final_metrics(df, best_threshold)

    # Error analysis
    fn, fp = error_analysis(df, best_threshold)

    # Print summary
    print_summary(df, metrics, threshold_results)

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(RESULTS_DIR / 'evaluation_report.csv', index=False)
    print(f"\nSaved: {RESULTS_DIR / 'evaluation_report.csv'}")

    # Save threshold sweep results
    threshold_results['results_df'].to_csv(RESULTS_DIR / 'threshold_sweep.csv', index=False)
    print(f"Saved: {RESULTS_DIR / 'threshold_sweep.csv'}")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

    return df, metrics


if __name__ == "__main__":
    df, metrics = main()
