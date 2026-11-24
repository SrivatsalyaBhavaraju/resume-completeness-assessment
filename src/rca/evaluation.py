"""
Metrics shared by every method so the comparison is like-for-like.

Threshold policy
----------------
Probabilistic methods output P(deficiency). The decision threshold for
each label is chosen on the VALIDATION split (the value maximising F1)
and then applied unchanged to the TEST split. Tuning on test would
inflate the scores. Rules produce 0/1 directly and need no threshold.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from rca import config


def tune_thresholds(probabilities, Y, M):
    thresholds = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        rows = M[:, j] == 1
        if rows.sum() == 0 or Y[rows, j].sum() == 0:
            continue
        candidates = np.linspace(0.05, 0.95, 91)
        scores = [f1_score(Y[rows, j], probabilities[rows, j] >= t, zero_division=0) for t in candidates]
        thresholds[j] = candidates[int(np.argmax(scores))]
    return thresholds


def label_metrics(predictions, Y, M, probabilities=None):
    """Per-label precision / recall / F1 (+AUROC) and macro averages."""
    per_label = {}
    for j, name in enumerate(config.LABEL_NAMES):
        rows = M[:, j] == 1
        y_true, y_pred = Y[rows, j], predictions[rows, j]
        entry = {
            "support_units": int(rows.sum()),
            "positives": int(y_true.sum()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        if probabilities is not None and 0 < y_true.sum() < len(y_true):
            entry["auroc"] = roc_auc_score(y_true, probabilities[rows, j])
        per_label[name] = {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in entry.items()}

    macro_f1 = float(np.mean([v["f1"] for v in per_label.values()]))
    result = {"per_label": per_label, "macro_f1": round(macro_f1, 4)}
    if probabilities is not None:
        aucs = [v["auroc"] for v in per_label.values() if "auroc" in v]
        result["macro_auroc"] = round(float(np.mean(aucs)), 4)
    return result


def type_metrics(predicted_ids, true_ids):
    return {
        "accuracy": round(float(accuracy_score(true_ids, predicted_ids)), 4),
        "macro_f1": round(float(f1_score(true_ids, predicted_ids, average="macro")), 4),
    }
