"""
Med-Nexus Evaluation Metrics
Computes classification metrics (Accuracy, F1, AUROC, AUPRC), probability calibration metrics
(Expected Calibration Error - ECE), and computational efficiency stats.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Any, List


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes classification accuracy."""
    return float(np.mean(y_true == y_pred))


def compute_expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) for probability calibration assessment.
    ECE = sum_{m=1}^M (|B_m|/N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_comprehensive_metrics(y_true: List[int], y_pred: List[int], probs: List[List[float]]) -> Dict[str, float]:
    """
    Computes complete diagnostic evaluation metrics package.
    """
    if len(y_true) == 0:
        return {}

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    p_arr = np.array(probs)

    acc = compute_accuracy(y_t, y_p)
    ece = compute_expected_calibration_error(y_t, p_arr)

    # Precision, Recall, F1 for positive class (1: Abnormal)
    tp = np.sum((y_t == 1) & (y_p == 1))
    fp = np.sum((y_t == 0) & (y_p == 1))
    fn = np.sum((y_t == 1) & (y_p == 0))
    tn = np.sum((y_t == 0) & (y_p == 0))

    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Calculate AUROC and AUPRC
    try:
        if len(np.unique(y_t)) > 1:
            auroc = roc_auc_score(y_t, p_arr[:, 1])
            auprc = average_precision_score(y_t, p_arr[:, 1])
        else:
            auroc = 0.0
            auprc = 0.0
    except Exception:
        auroc = 0.0
        auprc = 0.0

    return {
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "ece": round(ece, 4)
    }
