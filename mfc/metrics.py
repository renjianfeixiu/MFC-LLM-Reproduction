from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, r2_score


def compute_metrics(labels, probabilities, rul_true, rul_pred, rul_mask, num_classes):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    predictions = probabilities.argmax(1)
    all_labels = np.arange(num_classes)
    result = {"accuracy": float(accuracy_score(labels, predictions)),
              "f1_macro": float(f1_score(labels, predictions, labels=all_labels, average="macro", zero_division=0)),
              "f1_weighted": float(f1_score(labels, predictions, labels=all_labels, average="weighted", zero_division=0)),
              "precision_macro": float(precision_score(labels, predictions, labels=all_labels, average="macro", zero_division=0)),
              "recall_macro": float(recall_score(labels, predictions, labels=all_labels, average="macro", zero_division=0)),
              "confusion_matrix": confusion_matrix(labels, predictions, labels=all_labels).tolist(),
              "num_samples": len(labels), "num_rul_samples": int(np.asarray(rul_mask).sum())}
    if result["num_rul_samples"]:
        y = np.asarray(rul_true)[rul_mask]
        p = np.asarray(rul_pred)[rul_mask]
        result.update(rul_mae=float(np.mean(np.abs(p - y))), rul_rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                      rul_r2=float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else None)
    return result
