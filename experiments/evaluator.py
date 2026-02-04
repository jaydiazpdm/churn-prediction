"""
Evaluator for churn prediction experiments.

Consolidates scoring and metrics logic from V1-V3 evaluation scripts.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    confusion_matrix,
)

from .config import ExperimentConfig
from .features import engineer_features


class Evaluator:
    """Handles scoring and metric calculation for experiments."""

    def score_dataset(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> pd.DataFrame:
        """
        Apply feature engineering and scoring based on config.

        Args:
            df: DataFrame with raw features
            config: ExperimentConfig with weights and feature toggles

        Returns:
            DataFrame with RISK_SCORE column added
        """
        df = df.copy()

        # Apply feature engineering
        if config.engineered_features:
            df = engineer_features(df, config)

        # Calculate risk score
        df["RISK_SCORE"] = df.apply(
            lambda row: self._calculate_score(row, config), axis=1
        )

        return df

    def _calculate_score(self, row: pd.Series, config: ExperimentConfig) -> int:
        """
        Calculate risk score for a single row based on config weights.

        Combines base component scores with engineered feature scores.
        """
        points = 0
        w = config.weights

        # === BASE COMPONENT SCORES ===

        # Tier Risk (0-25 base points)
        tier_base = {"Core": 25, "Growth": 15, "Enterprise": 5}.get(
            row.get("TIER_NAME", ""), 15
        )
        points += int(tier_base * w.get("tier", 1.0))

        # Blueprint (0-20 base points)
        bp = row.get("TOTAL_BLUEPRINTS", 0)
        if bp == 1:
            bp_points = 20
        elif bp == 2:
            bp_points = 10
        else:
            bp_points = 0
        points += int(bp_points * w.get("blueprint", 1.0))

        # Contract Duration (0-20 base points)
        duration = row.get("CONTRACT_DURATION", 12)
        if duration <= 3:
            contract_points = 20
        elif duration <= 6:
            contract_points = 15
        elif duration <= 9:
            contract_points = 10
        else:
            contract_points = 0
        points += int(contract_points * w.get("contract", 1.0))

        # Urgency (0-25 base points)
        months_end = row.get("MONTHS_UNTIL_END", 12)
        if months_end <= 1:
            urgency_points = 25
        elif months_end <= 3:
            urgency_points = 15
        elif months_end <= 6:
            urgency_points = 5
        else:
            urgency_points = 0
        points += int(urgency_points * w.get("urgency", 1.0))

        # Financial (0-15 base points)
        diff = row.get("DIFF_RETAINER", 0)
        if diff <= -0.5:
            fin_points = 15
        elif diff <= -0.2:
            fin_points = 10
        elif diff <= 0:
            fin_points = 5
        else:
            fin_points = 0
        points += int(fin_points * w.get("financial", 1.0))

        # Tenure/First Length (0-15 base points)
        first_len = row.get("FIRST_LENGTH", 12)
        if first_len <= 3:
            tenure_points = 15
        elif first_len <= 8:
            tenure_points = 10
        else:
            tenure_points = 0
        points += int(tenure_points * w.get("tenure", 1.0))

        # === ENGINEERED FEATURE SCORES ===

        # IS_FIRST_CONTRACT
        if row.get("IS_FIRST_CONTRACT", 0) == 1:
            points += int(w.get("first_contract", 15))

        # TENURE_BUCKET
        bucket = row.get("TENURE_BUCKET", "Unknown")
        if bucket == "Very_New":
            points += int(w.get("very_new", 20))
        elif bucket == "Establishing":
            points += int(w.get("establishing", 10))
        elif bucket == "Mature":
            points -= int(w.get("mature_bonus", 5))

        # CORE_SINGLE_EXPOSURE
        if row.get("CORE_SINGLE_EXPOSURE", 0) == 1:
            points += int(w.get("core_single", 20))

        # NEW_SHORT_CONTRACT_RISK
        if row.get("NEW_SHORT_CONTRACT_RISK", 0) == 1:
            points += int(w.get("new_short", 15))

        # CONTRACTS_COMPLETED
        if row.get("CONTRACTS_COMPLETED", 0) >= 2:
            points -= int(w.get("multi_renewal", 10))

        return max(0, points)

    def find_optimal_threshold(
        self, df: pd.DataFrame, config: ExperimentConfig
    ) -> Tuple[float, pd.DataFrame]:
        """
        Find optimal threshold on validation set.

        Args:
            df: Scored DataFrame with RISK_SCORE and IS_CHURN
            config: ExperimentConfig with optimization settings

        Returns:
            Tuple of (best_threshold, sweep_results_df)
        """
        y_true = df["IS_CHURN"]
        scores = df["RISK_SCORE"]

        thresholds = np.arange(
            scores.min(), scores.max() + 1, config.threshold_step
        )
        results = []

        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            results.append({
                "threshold": thresh,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            })

        results_df = pd.DataFrame(results)

        # Apply constraints
        valid = results_df.copy()
        if config.min_recall:
            valid = valid[valid["recall"] >= config.min_recall]
        if config.min_precision:
            valid = valid[valid["precision"] >= config.min_precision]

        if len(valid) == 0:
            valid = results_df  # Fallback to unconstrained

        # Find best threshold for specified metric
        best_idx = valid[config.optimize_metric].idxmax()
        best_threshold = results_df.loc[best_idx, "threshold"]

        return best_threshold, results_df

    def calculate_metrics(
        self, df: pd.DataFrame, threshold: float
    ) -> dict:
        """
        Calculate all metrics at a given threshold.

        Args:
            df: Scored DataFrame with RISK_SCORE and IS_CHURN
            threshold: Score threshold for classification

        Returns:
            Dictionary with all metrics
        """
        y_true = df["IS_CHURN"]
        y_pred = (df["RISK_SCORE"] >= threshold).astype(int)

        # For AUC, normalize scores to [0, 1]
        max_score = df["RISK_SCORE"].max()
        y_prob = df["RISK_SCORE"] / max_score if max_score > 0 else df["RISK_SCORE"]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            "auc_roc": roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) > 1
            else 0.0,
            "true_positives": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
            "true_negatives": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
            "false_positives": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
            "false_negatives": int(cm[1, 0]) if cm.shape == (2, 2) else 0,
        }
