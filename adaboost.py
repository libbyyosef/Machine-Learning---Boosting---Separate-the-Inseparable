import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.D_ = np.full(len(y), 1 / len(y), dtype=np.float64)
        self.weights_ = np.zeros(self.iterations_)
        self.models_ = []
        for i in range(self.iterations_):
            # invoke base learner
            weighted_labels = self.D_ * y
            weak_learner = self.wl_()
            fitted_learner = weak_learner.fit(X, weighted_labels)
            self.models_.append(fitted_learner)
            y_pred = weak_learner.predict(X)
            # compute epsilon t
            errors = np.sum(self.D_ * (y != y_pred))
            # set w t
            w_t = np.log((1 - errors) / errors) / 2
            self.weights_[i] = w_t
            # update sample weights
            self.D_ *= np.exp(-y * w_t * y_pred)
            # normalize
            self.D_ /= np.sum(self.D_)

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        partial_predict = self.partial_predict(X, T=self.iterations_)
        return partial_predict

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
        partial_loss = self.partial_loss(X, y, T=self.iterations_)
        return partial_loss

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        weighted_sum = 0.0
        for t in range(min(T, self.iterations_)):
            weak_learner = self.models_[t]
            weight = self.weights_[t]
            prediction = weak_learner.predict(X)
            weighted_sum += weight * prediction
        return np.sign(weighted_sum)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ..metrics import misclassification_error
        y_pred = self.partial_predict(X, T)
        return misclassification_error(y_true=y,
                                       y_pred=y_pred)
