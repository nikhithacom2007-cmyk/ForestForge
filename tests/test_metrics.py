"""
Unit tests for custom evaluation metrics.
"""

import numpy as np

from forestforge.metrics.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def test_binary_metrics():
    # 6 samples: 3 positives, 3 negatives
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])

    # Accuracy: 4 out of 6 correct -> 2/3 ~ 0.6667
    acc = accuracy_score(y_true, y_pred)
    assert np.isclose(acc, 4/6), f"Expected 4/6, got {acc}"

    # Precision (binary, positive label = 1):
    # TP = 2 (indices 1, 4)
    # FP = 1 (index 5)
    # Precision = 2 / (2 + 1) = 2/3 ~ 0.6667
    prec = precision_score(y_true, y_pred, average="binary")
    assert np.isclose(prec, 2/3), f"Expected 2/3, got {prec}"

    # Recall (binary, positive label = 1):
    # TP = 2
    # FN = 1 (index 2)
    # Recall = 2 / (2 + 1) = 2/3 ~ 0.6667
    rec = recall_score(y_true, y_pred, average="binary")
    assert np.isclose(rec, 2/3), f"Expected 2/3, got {rec}"

    # F1 (binary):
    # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3 ~ 0.6667
    f1 = f1_score(y_true, y_pred, average="binary")
    assert np.isclose(f1, 2/3), f"Expected 2/3, got {f1}"

    # Confusion matrix:
    # True negatives (0,0): 2
    # False positives (0,1): 1
    # False negatives (1,0): 1
    # True positives (1,1): 2
    cm = confusion_matrix(y_true, y_pred)
    expected_cm = np.array([[2, 1], [1, 2]])
    assert np.array_equal(cm, expected_cm), f"Expected cm {expected_cm}, got {cm}"


def test_multiclass_metrics():
    # 3 classes: 0, 1, 2
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    # Accuracy: 4 out of 6 correct -> 2/3 ~ 0.6667
    acc = accuracy_score(y_true, y_pred)
    assert np.isclose(acc, 4/6), f"Expected 4/6, got {acc}"

    # Class 0:
    # TP = 1 (idx 0), FP = 1 (idx 5 is predicted 0 but true 2), FN = 1 (idx 1 is true 0 but pred 1)
    # prec = 1/2, rec = 1/2, f1 = 1/2

    # Class 1:
    # TP = 2 (idx 2, 3), FP = 1 (idx 1 is predicted 1 but true 0), FN = 0
    # prec = 2/3, rec = 2/2 = 1, f1 = 4/5 = 0.8

    # Class 2:
    # TP = 1 (idx 4), FP = 0, FN = 1 (idx 5 is true 2 but pred 0)
    # prec = 1/1 = 1, rec = 1/2, f1 = 2/3 ~ 0.6667

    # Macro Precision: (0.5 + 2/3 + 1.0) / 3 = 1.3333 / 2 ~ 0.7222
    prec = precision_score(y_true, y_pred, average="macro")
    assert np.isclose(prec, 13/18), f"Expected 13/18, got {prec}"

    # Macro Recall: (0.5 + 1.0 + 0.5) / 3 = 2.0 / 3 = 0.6667
    rec = recall_score(y_true, y_pred, average="macro")
    assert np.isclose(rec, 2/3), f"Expected 2/3, got {rec}"

    # Macro F1: (0.5 + 0.8 + 2/3) / 3 = (0.5 + 0.8 + 0.6667) / 3 ~ 0.6556
    f1 = f1_score(y_true, y_pred, average="macro")
    assert np.isclose(f1, 59/90), f"Expected 59/90, got {f1}"


if __name__ == "__main__":
    test_binary_metrics()
    test_multiclass_metrics()
    print("Metrics unit tests passed successfully!")
