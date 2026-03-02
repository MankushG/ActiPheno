"""
evaluate.py  —  ActiPheno
Metrics, confusion matrix helpers, pretty-printing.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict:
    y_pred = (y_prob >= thr).astype(int)

    # labels=[0,1] keeps the CM the right shape when a fold only predicts
    # one class — happens constantly in early training epochs
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc = float("nan")
        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1":          float(f1),
        "mcc":         float(matthews_corrcoef(y_true, y_pred)),
        "precision":   float(prec),
        "recall":      float(rec),
        "specificity": float(spec),
        "roc_auc":     roc,
        "cm":          cm,
        "thr":         thr,
        "n":           int(len(y_true)),
    }


def metrics_from_cm(cm: np.ndarray) -> Dict:
    """Scalar metrics from a 2×2 confusion matrix.

    Lets the global summary be computed from the accumulated CM at the end
    of LOPO without keeping raw prediction arrays in memory.
    """
    tn, fp, fn, tp = cm.ravel()
    total = int(cm.sum())
    acc  = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
    num  = float(tp * tn - fp * fn)
    den  = float(np.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    ))
    mcc  = num / den if den else 0.0
    return {
        "accuracy":    float(acc),
        "f1":          float(f1),
        "mcc":         float(mcc),
        "precision":   float(prec),
        "recall":      float(rec),
        "specificity": float(spec),
        "total":       total,
    }


def print_global(title: str, cm: np.ndarray) -> None:
    m = metrics_from_cm(cm)
    tn, fp, fn, tp = cm.ravel()
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print("-" * 72)
    print(f"  Total samples  : {m['total']}")
    print(f"  Accuracy       : {m['accuracy']:.4f}")
    print(f"  F1 Score       : {m['f1']:.4f}")
    print(f"  MCC            : {m['mcc']:.4f}")
    print(f"  Precision      : {m['precision']:.4f}")
    print(f"  Recall (Sens.) : {m['recall']:.4f}")
    print(f"  Specificity    : {m['specificity']:.4f}")
    print()
    print("  Confusion Matrix")
    print("-" * 72)
    print(f"  {'':>20}  {'Pred Healthy':>14}  {'Pred Depressed':>14}")
    print(f"  {'True Healthy':>20}  {tn:>14}  {fp:>14}")
    print(f"  {'True Depressed':>20}  {fn:>14}  {tp:>14}")
    print(f"{sep}\n")
