from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the
        cross-validated model.
        When called, the scoring function receives the true- and predicted
        values for each sample
        and potentially additional arguments. The function returns the score
        for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    folds = np.array_split(indices, cv)
    train_score = 0.0
    validation_score = 0.0

    for fold_indices in folds:
        train_mask = ~np.isin(indices, fold_indices)
        estimator_copy = deepcopy(estimator).fit(X[train_mask], y[train_mask])

        train_predictions = estimator_copy.predict(X[train_mask])
        train_score += scoring(y[train_mask], train_predictions)

        validation_predictions = estimator_copy.predict(X[fold_indices])
        validation_score += scoring(y[fold_indices], validation_predictions)

    avg_train_score = train_score / cv
    avg_validation_score = validation_score / cv

    return avg_train_score, avg_validation_score
