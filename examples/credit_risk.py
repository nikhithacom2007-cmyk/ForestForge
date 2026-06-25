"""
ForestForge - Credit Risk Modeling Example
Demonstrates how to use the custom DecisionTree and RandomForest classifiers
on a synthetic Credit Risk Assessment dataset.

Author: Your Name
"""

import numpy as np
from sklearn.model_selection import train_test_split

from forestforge.tree import DecisionTreeClassifier
from forestforge.ensemble import RandomForestClassifier
from forestforge.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def generate_credit_data(n_samples=500, random_state=42):
    """
    Generate a synthetic credit risk dataset.
    Features:
        - Income (in thousands)
        - Debt-to-Income (DTI) ratio
        - Credit Score (300 to 850)
        - Employment Length (years)
    Target:
        - Default (1: Yes, 0: No)
    """
    rng = np.random.default_rng(random_state)

    # Features
    income = rng.uniform(20, 150, size=n_samples)
    dti = rng.uniform(0.05, 0.75, size=n_samples)
    credit_score = rng.uniform(400, 850, size=n_samples)
    emp_length = rng.uniform(0, 20, size=n_samples)

    X = np.column_stack([income, dti, credit_score, emp_length])

    # Target logic: higher probability of default for high DTI, low credit score, and low income
    default_prob = (
        (dti * 2.0)
        - (credit_score / 400.0)
        - (income / 100.0)
        - (emp_length / 10.0)
        + 1.5
    )
    # Sigmoid function to map to probabilities
    prob = 1 / (1 + np.exp(-default_prob))

    # Determine class based on probability
    y = (rng.uniform(0, 1, size=n_samples) < prob).astype(int)

    return X, y, ["Income", "DTI", "CreditScore", "EmpLength"]


def print_metrics(model_name, y_true, y_pred):
    """
    Helper function to print formatted classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="binary")
    rec = recall_score(y_true, y_pred, average="binary")
    f1 = f1_score(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 40)
    print(f" {model_name} Evaluation Summary".center(40))
    print("=" * 40)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f} (class 1)")
    print(f"Recall    : {rec:.4f} (class 1)")
    print(f"F1-Score  : {f1:.4f} (class 1)")
    print("\nConfusion Matrix:")
    print(f"   Predicted 0   Predicted 1")
    print(f"Actual 0: {cm[0, 0]:^13} {cm[0, 1]:^13}")
    print(f"Actual 1: {cm[1, 0]:^13} {cm[1, 1]:^13}")
    print("=" * 40)
    print()


def main():
    print("Generating synthetic Credit Risk dataset...")
    X, y, feature_names = generate_credit_data(n_samples=600, random_state=42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Dataset generated:")
    print(f"  Training samples : {X_train.shape[0]}")
    print(f"  Testing samples  : {X_test.shape[0]}")
    print(f"  Features         : {', '.join(feature_names)}")
    print(f"  Class 1 (Default) ratio in Train : {np.mean(y_train):.2%}")
    print(f"  Class 1 (Default) ratio in Test  : {np.mean(y_test):.2%}\n")

    # 1. Train Decision Tree Classifier
    print("Training DecisionTreeClassifier...")
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=5, criterion="gini", random_state=42)
    dt.fit(X_train, y_train)
    dt_preds = dt.predict(X_test)
    print_metrics("Decision Tree Classifier", y_test, dt_preds)

    # 2. Train Random Forest Classifier
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=6,
        min_samples_split=5,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    print_metrics("Random Forest Classifier", y_test, rf_preds)


if __name__ == "__main__":
    main()
