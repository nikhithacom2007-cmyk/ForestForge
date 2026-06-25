"""
ForestForge - Random Forest Classifier
Built completely from scratch using NumPy.

Author: Your Name
"""

import numpy as np
from ..tree.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """
    Random Forest Classifier implemented from scratch.
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        criterion="gini",
        max_features="sqrt",
        bootstrap=True,
        random_state=None
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of decision trees in the forest.

        max_depth : int
            Maximum depth of each decision tree.

        min_samples_split : int
            Minimum number of samples required to split an internal node.

        criterion : str
            "gini" or "entropy".

        max_features : int, float, or str
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features.
            - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered.
            - If "sqrt", then `max_features=int(sqrt(n_features))`.
            - If "log2", then `max_features=int(log2(n_features))`.
            - If None, then `max_features=n_features`.

        bootstrap : bool
            Whether bootstrap samples are used when building trees.

        random_state : int or None
            Controls both the randomness of the bootstrapping of the samples
            used when building trees and the sampling of the features to consider
            when looking for the best split at each node.
        """

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees = []

    def fit(self, X, y):
        """
        Build a forest of trees from the training set (X, y).
        """

        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]

        # Initialize the random number generator
        if self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            # Generate deterministic seeds for each tree
            tree_seeds = rng.integers(0, 2**31 - 1, size=self.n_estimators)
        else:
            rng = np.random.default_rng()
            tree_seeds = [None] * self.n_estimators

        self.trees = []

        for i in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                # If random_state is set, we want the bootstrap selection to be deterministic
                # We can draw indices using our main rng
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_b, y_b = X[indices], y[indices]
            else:
                X_b, y_b = X, y

            # Initialize and fit the tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
                random_state=tree_seeds[i]
            )

            tree.fit(X_b, y_b)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Predict class for X.
        """

        X = np.array(X)

        if len(self.trees) == 0:
            raise ValueError("The forest has not been fitted yet.")

        # Get predictions from all trees
        # shape: (n_estimators, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote for each sample
        n_samples = X.shape[0]
        predictions = []

        for i in range(n_samples):
            sample_preds = tree_preds[:, i]
            classes, counts = np.unique(
                sample_preds,
                return_counts=True
            )
            predictions.append(
                classes[np.argmax(counts)]
            )

        return np.array(predictions)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """

        predictions = self.predict(X)

        accuracy = np.mean(
            predictions == y
        )

        return accuracy

    def __repr__(self):
        return (
            f"RandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"criterion='{self.criterion}', "
            f"max_features='{self.max_features}', "
            f"bootstrap={self.bootstrap}, "
            f"random_state={self.random_state})"
        )
