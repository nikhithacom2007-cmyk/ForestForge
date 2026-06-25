# ForestForge 🌳🛠️

`ForestForge` is a robust, modular, and high-performance Machine Learning library for Decision Trees and Random Forests built completely from scratch using Python and NumPy.

It is designed to be highly readable, standard-compliant (mirroring Scikit-Learn's API pattern), and fully self-contained.

---

## Features

- **DecisionTreeClassifier**:
  - Custom recursive tree building algorithm.
  - Supports **Gini Impurity** and **Information Entropy** split criteria.
  - Custom split stopping conditions: `max_depth`, `min_samples_split`.
  - Feature subspacing support (`max_features`) for Random Forest integration.
  - Fully deterministic execution via standard `random_state` controls.
  - Built-in tree visualization layout (`print_tree`).

- **RandomForestClassifier**:
  - Ensemble classifier implementing Bootstrap Aggregating (Bagging).
  - Multi-tree majority voting logic.
  - Supports feature subset selection (`max_features="sqrt"`, `"log2"`, float, int).
  - Reproducible building using tree-specific sub-seeding.

- **Custom Metrics**:
  - Accuracy, Precision, Recall, F1-Score (supporting Binary, Macro, and Weighted averaging).
  - Confusion Matrix generation.

---

## Installation

You can install the package locally in editable mode:

```bash
pip install -e .
```

---

## Project Structure

```text
ForestForge/
├── forestforge/
│   ├── __init__.py          # Main exports
│   ├── ensemble/
│   │   ├── __init__.py      # RandomForestClassifier export
│   │   └── random_forest.py # Bagging & majority vote logic
│   ├── tree/
│   │   ├── __init__.py      # DecisionTree exports
│   │   ├── decision_tree.py # Tree recursion, splitting, information gain
│   │   └── node.py          # Node structure representing splits/leaves
│   └── metrics/
│       ├── __init__.py      # Metric exports
│       └── metrics.py       # Custom evaluation algorithms
├── examples/
│   └── credit_risk.py       # Full pipeline example
├── tests/
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   └── test_metrics.py
├── setup.py                 # Package setup configuration
└── requirements.txt         # Dev dependencies
```

---

## Quick Start Example

Here is how you can load a dataset, train a Random Forest, and print evaluation metrics using `ForestForge`:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from forestforge.ensemble import RandomForestClassifier
from forestforge.metrics import accuracy_score, confusion_matrix

# 1. Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 2. Train Forest
rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    max_features="sqrt",
    random_state=42
)
rf.fit(X_train, y_train)

# 3. Predict and Evaluate
predictions = rf.predict(X_test)
acc = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)
```

---

## Verification and Testing

Verify the implementation by running the test suite:

```bash
# Run unit tests
PYTHONPATH=. python tests/test_decision_tree.py
PYTHONPATH=. python tests/test_random_forest.py
PYTHONPATH=. python tests/test_metrics.py

# Run credit risk modeling example
PYTHONPATH=. python examples/credit_risk.py
```
