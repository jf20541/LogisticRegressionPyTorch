import numpy as np
from sklearn.metrics import log_loss


def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce


predictions = np.array([[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.97]])
targets = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])

print(cross_entropy(predictions, targets))
# 0.7083767843022996

print(log_loss(targets, predictions))
# 0.7083767843022996

log_loss(targets, predictions) == cross_entropy(predictions, targets)
# True
