"""
Unit tests for RandomForestClassifier.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from forestforge import RandomForestClassifier


def test_random_forest():
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Initialize and fit RandomForest
    rf = RandomForestClassifier(
        n_estimators=15,
        max_depth=5,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=42
    )

    rf.fit(X_train, y_train)

    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)

    print(f"Random Forest - Train Accuracy: {train_acc:.4f}")
    print(f"Random Forest - Test Accuracy: {test_acc:.4f}")

    # Assert test accuracy is reasonable (Iris is a simple dataset)
    assert test_acc > 0.85, f"Expected test accuracy > 0.85, got {test_acc:.4f}"

    # Verify prediction shape
    preds = rf.predict(X_test)
    assert preds.shape == y_test.shape, "Prediction shape mismatch"


if __name__ == "__main__":
    test_random_forest()
    print("Random Forest Classifier unit tests passed successfully!")
