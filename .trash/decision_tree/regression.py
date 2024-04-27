import numpy as np


class Node:
    def __init__(
        self,
        feature_index=None,
        threshold=None,
        left=None,
        right=None,
        var_red=None,
        value=None,
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=None):
        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = np.iinfo(np.int32).max if max_depth is None else max_depth

    def _calculate_leaf_value(self, y):
        """Function to compute leaf node."""

        val = np.mean(y, axis=0)
        return val

    def _split(self, x_col, threshold):
        """Function to split the data"""

        left_idxs = np.argwhere(x_col <= threshold).flatten()
        right_idxs = np.argwhere(x_col > threshold).flatten()
        return left_idxs, right_idxs

    def _variance_reduction(self, y, left_y, right_y):
        """Function to compute variance reduction."""

        weight_l = len(left_y) / len(y)
        weight_r = len(right_y) / len(y)
        reduction = np.var(y) - (weight_l * np.var(left_y) + weight_r * np.var(right_y))
        return reduction

    def _best_split(self, X, y):
        """Function to find the best split."""

        _, num_features = X.shape

        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                left_indices, right_indices = self._split(feature_values, threshold)
                # check if childs are not null
                if len(left_indices) > 0 and len(right_indices) > 0:
                    left_y, right_y = y[left_indices, :], y[right_indices, :]

                    # compute information gain
                    curr_var_red = self._variance_reduction(y, left_y, right_y)

                    # update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_indices"] = left_indices
                        best_split["right_indices"] = right_indices
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red

        # return best split
        return best_split

    def _build_tree(self, X, y, depth=0):
        """Recursive function to build the tree."""

        num_samples, _ = X.shape

        # check the stopping criteria
        if num_samples >= self.min_samples_split and depth <= self.max_depth:
            # find the best split
            best_split = self._best_split(X, y)

            # check if information gain is positive
            if "var_red" in best_split and best_split["var_red"] > 0:
                # recur left
                left_indices = best_split["left_indices"]
                left_subtree = self._build_tree(
                    X[left_indices, :], y[left_indices, :], depth + 1
                )
                # recur right
                right_indices = best_split["right_indices"]
                right_subtree = self._build_tree(
                    X[right_indices, :], y[right_indices, :], depth + 1
                )
                # return decision node
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left_subtree,
                    right_subtree,
                    best_split["var_red"],
                )
        # compute leaf node
        leaf_value = self._calculate_leaf_value(y)
        # return leaf node
        return Node(value=leaf_value)

    def fit(self, X, y):
        """Function to train the tree."""
        y = y.reshape(-1, 1) if y.ndim == 1 else y
        self.root = self._build_tree(X, y)

    def print_tree(self, tree=None, indent=" "):
        """Function to print the tree."""

        if tree is None:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print(f"X_{tree.feature_index} <= {tree.threshold} ? {tree.var_red}")
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + indent)
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + indent)

    def _make_prediction(self, x, tree):
        """Function to predict for new examples."""

        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)

    def predict(self, X):
        """Function to predict a single data point."""

        preditions = np.stack([self._make_prediction(x, self.root) for x in X])
        return preditions


# # Import the necessary modules and libraries
# import matplotlib.pyplot as plt
# import numpy as np

# # Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.stack([np.sin(X).ravel(), np.sin(X).ravel()], axis=1)
# y[::5, :] += 3 * (0.5 - rng.rand(16, 2))
# y = y[:, 0:1]

# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=None)
# regr_1.fit(X, y)
# regr_2.fit(X, y)

# # Predict
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)

# # Plot the results
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()
