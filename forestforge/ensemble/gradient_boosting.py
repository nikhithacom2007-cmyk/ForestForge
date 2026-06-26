"""
ForestForge - Gradient Boosting Classifier
Built completely from scratch using NumPy.

Author: Your Name
"""

import numpy as np

from ..tree.decision_tree import DecisionTreeClassifier


class GradientBoostingClassifier:
    """
    Gradient Boosting Classifier
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        criterion="gini",
        random_state=None
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of weak learners.

        learning_rate : float
            Shrinks contribution of each tree.

        max_depth : int
            Maximum depth of each weak learner.

        min_samples_split : int
            Minimum samples required to split.

        criterion : str
            Split criterion.

        random_state : int
            Random seed.
        """

        self.n_estimators = n_estimators

        self.learning_rate = learning_rate

        self.max_depth = max_depth

        self.min_samples_split = min_samples_split

        self.criterion = criterion

        self.random_state = random_state

        self.models = []

        self.initial_prediction = None

        if random_state is not None:
            np.random.seed(random_state)
    ####################################################
    # INITIAL PREDICTION
    ####################################################

    def _initial_prediction(self, y):
        """
        Compute the initial prediction.

        For simplicity, use the most common class.
        """

        classes, counts = np.unique(
            y,
            return_counts=True
        )

        return classes[
            np.argmax(counts)
        ]

    ####################################################
    # RESIDUALS
    ####################################################

    def _compute_residuals(
        self,
        y_true,
        y_pred
    ):
        """
        Compute residuals.

        Residual = Actual - Prediction
        """

        return y_true - y_pred
    ####################################################
    # FIT MODEL
    ####################################################

    def fit(
        self,
        X,
        y
    ):
        """
        Train the Gradient Boosting Classifier.
        """

        X = np.array(X)
        y = np.array(y)

        ################################################
        # INITIAL PREDICTION
        ################################################

        self.initial_prediction = self._initial_prediction(y)

        predictions = np.full(
            len(y),
            self.initial_prediction,
            dtype=float
        )

        ################################################
        # CLEAR OLD MODELS
        ################################################

        self.models = []

        ################################################
        # TRAIN WEAK LEARNERS
        ################################################

        for _ in range(self.n_estimators):

            ############################################
            # Compute Residuals
            ############################################

            residuals = self._compute_residuals(
                y,
                predictions
            )

            ############################################
            # Train Weak Learner
            ############################################

            tree = DecisionTreeClassifier(

                max_depth=self.max_depth,

                min_samples_split=self.min_samples_split,

                criterion=self.criterion

            )

            tree.fit(
                X,
                residuals
            )

            ############################################
            # Predict Residuals
            ############################################

            residual_predictions = tree.predict(
                X
            )

            ############################################
            # Update Predictions
            ############################################

            predictions += (

                self.learning_rate *

                residual_predictions

            )

            ############################################
            # Save Tree
            ############################################

            self.models.append(
                tree
            )

        return self
    ####################################################
    # PREDICT
    ####################################################

    def predict(
        self,
        X
    ):
        """
        Predict class labels.
        """

        X = np.array(X)

        ################################################
        # START WITH INITIAL PREDICTION
        ################################################

        predictions = np.full(

            X.shape[0],

            self.initial_prediction,

            dtype=float

        )

        ################################################
        # ADD CONTRIBUTION OF EACH TREE
        ################################################

        for tree in self.models:

            predictions += (

                self.learning_rate *

                tree.predict(X)

            )

        ################################################
        # CONVERT TO CLASS LABELS
        ################################################

        predictions = np.round(
            predictions
        )

        predictions = predictions.astype(
            int
        )

        return predictions
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
    # NUMBER OF WEAK LEARNERS
    ####################################################

    def __len__(self):
        """
        Return the number of trained trees.
        """

        return len(self.models)

    ####################################################
    # MODEL SUMMARY
    ####################################################

    def __repr__(self):
        """
        String representation of the model.
        """

        return (
            "GradientBoostingClassifier("
            f"n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"criterion='{self.criterion}'"
            ")"
        )

    ####################################################
    # GET PARAMETERS
    ####################################################

    def get_params(self):
        """
        Return model parameters.
        """

        return {

            "n_estimators": self.n_estimators,

            "learning_rate": self.learning_rate,

            "max_depth": self.max_depth,

            "min_samples_split": self.min_samples_split,

            "criterion": self.criterion,

            "random_state": self.random_state

        }