"""
Node class for Decision Tree.

Each node represents either:
1. A decision node (feature + threshold)
2. A leaf node (prediction)
"""


class Node:
    """
    Represents a single node in a Decision Tree.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        """
        Parameters
        ----------
        feature : int
            Index of feature used for splitting.

        threshold : float
            Threshold value for the split.

        left : Node
            Left child node.

        right : Node
            Right child node.

        value : int or float
            Prediction value for a leaf node.
        """

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Returns True if this node is a leaf.
        """
        return self.value is not None