import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers.decision_stump import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape 
    (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_errors, test_errors = [], []
    adaboost = AdaBoost(DecisionStump, n_learners)
    model = adaboost.fit(train_X, train_y)
    for t in range(n_learners):
        train_error1 = model.partial_loss(train_X, train_y, t + 1)
        train_errors.append(train_error1)
        test_error1 = model.partial_loss(test_X, test_y, t + 1)
        test_errors.append(test_error1)
    list_learners = list(range(1, n_learners + 1))
    trace_train = go.Scatter(x=list_learners, y=train_errors, name="Train "
                                                                   "Errors",
                             mode="lines")
    trace_test = go.Scatter(x=list_learners, y=test_errors, name="Test "
                                                                 "Errors",
                            mode="lines")

    layout = go.Layout(
        width=600, height=600,
        title={"x": 0.5,
               "text": r"$\text{'Misclassification Error in AdaBoost with "
                       r"Increasing Classifiers'}$"},
        xaxis={"title": r"$\text{Iteration}$"},
        yaxis={"title": r"$\text{Misclassification Error}$"})
    graph = go.Figure(data=[trace_train, trace_test], layout=layout)
    graph.write_image(f"adaboost_{noise}.png")

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    graph = make_subplots(rows=1, cols=4,
                          subplot_titles=[rf"$\text{{{t} Classifiers}}$" for t
                                          in T])

    for i, t in enumerate(T):
        traces = [
            decision_surface(lambda X: model.partial_predict(X, t), lims[0],
                             lims[1], density=60, showscale=False),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers",
                       showlegend=False,
                       marker=dict(color=test_y,
                                   symbol=np.where(test_y == 1, "circle",
                                                   "x")))
        ]
        graph.add_traces(traces, rows=1, cols=i + 1)
    graph.update_layout(height=500, width=2000)
    graph.update_xaxes(visible=False)
    graph.update_yaxes(visible=False)

    graph.write_image(f"adaboost_{noise}_decision_boundaries.png")


    # Question 3: Decision surface of best performing ensemble
    min_t = np.argmin(test_errors) + 1
    decision_trace = decision_surface(
        lambda X: model.partial_predict(X, min_t),
        lims[0],
        lims[1],
        density=60,
        showscale=False)
    scatter_trace = go.Scatter(
        x=test_X[:, 0],
        y=test_X[:, 1],
        mode="markers",
        showlegend=False,
        marker=dict(
            color=test_y,
            symbol=np.where(test_y == 1, "circle", "x")
        )
    )
    layout = go.Layout(
        width=500,
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=f"Best Performing Ensemble<br>Size: {min_t}, Accuracy: {1 - round(test_errors[min_t - 1], 2)}"
    )
    graph = go.Figure(
        data=[decision_trace, scatter_trace],
        layout=layout
    )
    graph.write_image(f"adaboost_{noise}_best_over_test.png")

    # Question 4: Decision surface with weighted samples
    D = 20 * model.D_ / model.D_.max()
    decision_trace = decision_surface(model.predict, lims[0], lims[1],
                                      density=60, showscale=False)
    scatter_trace = go.Scatter(
        x=train_X[:, 0],
        y=train_X[:, 1],
        mode="markers",
        showlegend=False,
        marker=dict(
            size=D,
            color=train_y,
            symbol=np.where(train_y == 1, "circle", "x")
        )
    )
    layout = go.Layout(
        width=500,
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title="Final AdaBoost Sample Distribution"
    )
    graph = go.Figure(
        data=[decision_trace, scatter_trace],
        layout=layout
    )
    graph.write_image(f"adaboost_{noise}_weighted_samples.png")

if __name__ == '__main__':
    np.random.seed(0)
    [fit_and_evaluate_adaboost(noise) for noise in [0, 0.4]]

