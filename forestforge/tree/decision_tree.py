"""
ForestForge - Decision Tree Classifier
Built completely from scratch using NumPy.

Author: Your Name
"""

import numpy as np

from .node import Node


class DecisionTreeClassifier:
    """
    Decision Tree Classifier implemented from scratch.
    """

    def __init__(
        self,
        max_depth=10,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        random_state=None
    ):
        """
        Parameters
        ----------
        max_depth : int
            Maximum depth of the tree.

        min_samples_split : int
            Minimum number of samples required to split.

        criterion : str
            "gini" or "entropy"

        max_features : int, float, or str
            The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features.
            - If float, then `max_features` is a fraction and `int(max_features * n_features)` features are considered.
            - If "sqrt", then `max_features=int(sqrt(n_features))`.
            - If "log2", then `max_features=int(log2(n_features))`.
            - If None, then `max_features=n_features`.

        random_state : int or None
            Controls the randomness of feature selection.
        """

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.root = None

    ####################################################
    # GINI IMPURITY
    ####################################################

    def _gini(self, y):
        """
        Calculate Gini Impurity.
        """

        classes, counts = np.unique(
            y,
            return_counts=True
        )

        probabilities = counts / len(y)

        gini = 1 - np.sum(
            probabilities ** 2
        )

        return gini

    ####################################################
    # ENTROPY
    ####################################################

    def _entropy(self, y):
        """
        Calculate Entropy.
        """

        classes, counts = np.unique(
            y,
            return_counts=True
        )

        probabilities = counts / len(y)

        probabilities = probabilities[
            probabilities > 0
        ]

        entropy = -np.sum(
            probabilities *
            np.log2(probabilities)
        )

        return entropy
            ####################################################
    # IMPURITY
    ####################################################

    def _impurity(self, y):
        """
        Return impurity based on the selected criterion.
        """

        if self.criterion == "gini":
            return self._gini(y)

        elif self.criterion == "entropy":
            return self._entropy(y)

        else:
            raise ValueError(
                "criterion must be 'gini' or 'entropy'"
            )

    ####################################################
    # SPLIT DATASET
    ####################################################

    def _split(
        self,
        X,
        y,
        feature,
        threshold
    ):
        """
        Split the dataset into left and right subsets.
        """

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        X_left = X[left_mask]
        y_left = y[left_mask]

        X_right = X[right_mask]
        y_right = y[right_mask]

        return (
            X_left,
            y_left,
            X_right,
            y_right
        )

    ####################################################
    # INFORMATION GAIN
    ####################################################

    def _information_gain(
        self,
        parent,
        left,
        right
    ):
        """
        Calculate Information Gain.
        """

        if len(left) == 0 or len(right) == 0:
            return 0

        parent_impurity = self._impurity(parent)

        left_impurity = self._impurity(left)

        right_impurity = self._impurity(right)

        n = len(parent)

        n_left = len(left)

        n_right = len(right)

        weighted_impurity = (
            (n_left / n) * left_impurity
            +
            (n_right / n) * right_impurity
        )

        information_gain = (
            parent_impurity - weighted_impurity
        )

        return information_gain
            ####################################################
    # BEST SPLIT
    ####################################################

    def _best_split(
        self,
        X,
        y
    ):
        """
        Find the best feature and threshold to split the data.
        """

        best_gain = 0.0

        best_feature = None

        best_threshold = None

        n_features = X.shape[1]
        features = np.arange(n_features)

        if not hasattr(self, "rng_"):
            if self.random_state is not None:
                self.rng_ = np.random.default_rng(self.random_state)
            else:
                self.rng_ = np.random.default_rng()

        if self.max_features is not None:
            if self.max_features == "sqrt":
                n_select = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_select = int(np.log2(n_features))
            elif isinstance(self.max_features, int):
                n_select = self.max_features
            elif isinstance(self.max_features, float):
                n_select = int(self.max_features * n_features)
            else:
                n_select = n_features

            n_select = max(1, min(n_select, n_features))
            features = self.rng_.choice(features, size=n_select, replace=False)

        for feature in features:

            thresholds = np.unique(
                X[:, feature]
            )

            for threshold in thresholds:

                (
                    X_left,
                    y_left,
                    X_right,
                    y_right
                ) = self._split(
                    X,
                    y,
                    feature,
                    threshold
                )

                gain = self._information_gain(
                    y,
                    y_left,
                    y_right
                )

                if gain > best_gain:

                    best_gain = gain

                    best_feature = feature

                    best_threshold = threshold

        return (
            best_feature,
            best_threshold,
            best_gain
        )

    ####################################################
    # MOST COMMON LABEL
    ####################################################

    def _most_common_label(
        self,
        y
    ):
        """
        Return the most frequent class label.
        """

        classes, counts = np.unique(
            y,
            return_counts=True
        )

        return classes[
            np.argmax(counts)
        ]
            ####################################################
    # BUILD TREE
    ####################################################

    def _build_tree(
        self,
        X,
        y,
        depth=0
    ):
        """
        Recursively build the Decision Tree.
        """

        n_samples = X.shape[0]

        n_classes = len(np.unique(y))

        ################################################
        # STOPPING CONDITIONS
        ################################################

        # Case 1: All samples belong to one class
        if n_classes == 1:

            return Node(
                value=y[0]
            )

        # Case 2: Maximum depth reached
        if depth >= self.max_depth:

            return Node(
                value=self._most_common_label(y)
            )

        # Case 3: Too few samples
        if n_samples < self.min_samples_split:

            return Node(
                value=self._most_common_label(y)
            )

        ################################################
        # FIND BEST SPLIT
        ################################################

        feature, threshold, gain = self._best_split(
            X,
            y
        )

        # No improvement
        if gain <= 0:

            return Node(
                value=self._most_common_label(y)
            )

        ################################################
        # SPLIT DATA
        ################################################

        (
            X_left,
            y_left,
            X_right,
            y_right
        ) = self._split(
            X,
            y,
            feature,
            threshold
        )

        ################################################
        # BUILD LEFT SUBTREE
        ################################################

        left_child = self._build_tree(
            X_left,
            y_left,
            depth + 1
        )

        ################################################
        # BUILD RIGHT SUBTREE
        ################################################

        right_child = self._build_tree(
            X_right,
            y_right,
            depth + 1
        )

        ################################################
        # RETURN CURRENT NODE
        ################################################

        return Node(

            feature=feature,

            threshold=threshold,

            left=left_child,

            right=right_child

        )
            ####################################################
    # FIT MODEL
    ####################################################

    def fit(self, X, y):
        """
        Train the Decision Tree.
        """

        X = np.array(X)
        y = np.array(y)

        if self.random_state is not None:
            self.rng_ = np.random.default_rng(self.random_state)
        else:
            self.rng_ = np.random.default_rng()

        self.root = self._build_tree(
            X,
            y
        )

        return self

    ####################################################
    # PREDICT ONE SAMPLE
    ####################################################

    def _predict_sample(
        self,
        x,
        node
    ):
        """
        Predict a single sample by traversing the tree.
        """

        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:

            return self._predict_sample(
                x,
                node.left
            )

        return self._predict_sample(
            x,
            node.right
        )

    ####################################################
    # PREDICT
    ####################################################

    def predict(
        self,
        X
    ):
        """
        Predict labels for multiple samples.
        """

        X = np.array(X)

        predictions = [

            self._predict_sample(
                sample,
                self.root
            )

            for sample in X

        ]

        return np.array(predictions)
            ####################################################
    # SCORE
    ####################################################

    def score(
        self,
        X,
        y
    ):
        """
        Calculate classification accuracy.
        """

        predictions = self.predict(X)

        accuracy = np.mean(
            predictions == y
        )

        return accuracy

    ####################################################
    # PRINT TREE
    ####################################################

    def print_tree(
        self,
        node=None,
        depth=0
    ):
        """
        Print the Decision Tree in a readable format.
        """

        if node is None:
            node = self.root

        indent = "│   " * depth

        if node.is_leaf_node():

            print(
                indent +
                f"Leaf → Class {node.value}"
            )

            return

        print(
            indent +
            f"[Feature {node.feature} <= {node.threshold:.4f}]"
        )

        print(
            indent + "├── Left"
        )

        self.print_tree(
            node.left,
            depth + 1
        )

        print(
            indent + "└── Right"
        )

        self.print_tree(
            node.right,
            depth + 1
        )

    ####################################################
    # REPRESENTATION
    ####################################################

    def __repr__(self):
        return (
            f"DecisionTreeClassifier("
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"criterion='{self.criterion}')"
        )