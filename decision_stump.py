from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART
    algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature
        is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        base_error = np.inf
        n_features = X.shape[1]
        sign_values = [-1, 1]
        for j in range(n_features):
            for sign in sign_values:
                threshold, error = self._find_threshold(X[:, j], y,
                                                        sign)
                if error < base_error:
                    base_error = error
                    self.threshold_ = threshold
                    self.sign_ = sign
                    self.j_ = j

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign`
        whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # split input samples into 2 classes based on selected feature and
        # threshold value of the decision stump classifier
        predictions = np.where(X[:, self.j_] < self.threshold_, -self.sign_,
                               self.sign_)
        return predictions

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to
        perform a split
        The threshold is found according to the value minimizing the
        misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are
        predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # sort for better running time and easier and quickly search
        ids = np.argsort(values)
        values = values[ids]
        labels = labels[ids]
        same_sign_mask = np.sign(labels) == sign
        same_sign_abs_labels = labels[same_sign_mask]
        abs_labels = np.abs(same_sign_abs_labels)
        loss = np.sum(abs_labels)
        cumulative_sum = np.cumsum(labels * sign)
        difference = loss - cumulative_sum
        loss = np.append(loss, difference)
        id = np.argmin(loss)
        concatenated_array = [-np.inf]
        concatenated_array.extend(values[1:])
        concatenated_array.append(np.inf)
        threshold = concatenated_array[id]
        threshold_loss = loss[id]
        return threshold, threshold_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y_true=y, y_pred=y_pred)
