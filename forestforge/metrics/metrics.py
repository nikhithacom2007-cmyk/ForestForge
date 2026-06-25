"""
ForestForge - Metrics
Custom evaluation metrics built from scratch using NumPy.

Author: Your Name
"""

import numpy as np


def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy classification score.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        return 0.0

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average="macro"):
    """
    Calculate precision score.
    Supports average: 'binary', 'macro', 'weighted'.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        return 0.0

    classes = np.unique(y_true)

    if average == "binary":
        # Usually positive label is 1
        pos_label = 1
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))

        if tp + fp == 0:
            return 0.0
        return float(tp / (tp + fp))

    elif average in ("macro", "weighted"):
        precisions = []
        weights = []

        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            weight = np.sum(y_true == c)

            if tp + fp == 0:
                prec = 0.0
            else:
                prec = float(tp / (tp + fp))

            precisions.append(prec)
            weights.append(weight)

        if average == "macro":
            return float(np.mean(precisions))
        else:  # weighted
            total_weight = np.sum(weights)
            if total_weight == 0:
                return 0.0
            return float(np.sum(np.array(precisions) * np.array(weights)) / total_weight)

    else:
        raise ValueError("average must be 'binary', 'macro', or 'weighted'")


def recall_score(y_true, y_pred, average="macro"):
    """
    Calculate recall score.
    Supports average: 'binary', 'macro', 'weighted'.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        return 0.0

    classes = np.unique(y_true)

    if average == "binary":
        pos_label = 1
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        if tp + fn == 0:
            return 0.0
        return float(tp / (tp + fn))

    elif average in ("macro", "weighted"):
        recalls = []
        weights = []

        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            weight = np.sum(y_true == c)

            if tp + fn == 0:
                rec = 0.0
            else:
                rec = float(tp / (tp + fn))

            recalls.append(rec)
            weights.append(weight)

        if average == "macro":
            return float(np.mean(recalls))
        else:  # weighted
            total_weight = np.sum(weights)
            if total_weight == 0:
                return 0.0
            return float(np.sum(np.array(recalls) * np.array(weights)) / total_weight)

    else:
        raise ValueError("average must be 'binary', 'macro', or 'weighted'")


def f1_score(y_true, y_pred, average="macro"):
    """
    Calculate F1 score.
    Supports average: 'binary', 'macro', 'weighted'.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0:
        return 0.0

    classes = np.unique(y_true)

    if average == "binary":
        prec = precision_score(y_true, y_pred, average="binary")
        rec = recall_score(y_true, y_pred, average="binary")

        if prec + rec == 0:
            return 0.0
        return float(2 * (prec * rec) / (prec + rec))

    elif average in ("macro", "weighted"):
        f1s = []
        weights = []

        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            weight = np.sum(y_true == c)

            if 2 * tp + fp + fn == 0:
                f1 = 0.0
            else:
                f1 = float(2 * tp / (2 * tp + fp + fn))

            f1s.append(f1)
            weights.append(weight)

        if average == "macro":
            return float(np.mean(f1s))
        else:  # weighted
            total_weight = np.sum(weights)
            if total_weight == 0:
                return 0.0
            return float(np.sum(np.array(f1s) * np.array(weights)) / total_weight)

    else:
        raise ValueError("average must be 'binary', 'macro', or 'weighted'")


def confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Find the unique sorted classes in both predictions and ground truth
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true], class_to_idx[pred]] += 1

    return cm
