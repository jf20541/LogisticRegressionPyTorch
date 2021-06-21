import numpy as np

class LogisticRegression():
    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, features, target):
        # initalize weights and bias
        n_samples, n_features = features.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # for-loop to update the weights (gradient decent)
        for _ in range(self.n_iters):
            log_model = 1 / (1 + np.exp(-(np.dot(features, self.weights) + self.bias)))

            # derivate of weights and bias 
            dw = (1 / n_samples) * np.dot(features.T, (log_model-target))
            db = (1 / n_samples) * np.sum(log_model-target)

            # updates weights 
            self.weights -= self.learning_rate * dw
            self.bias -= self.bias * db

    def predict(self, features):
        log_model = 1 / (1 + np.exp(-(np.dot(features, self.weights) + self.bias)))
        pred_class = [1 if i > 0.5 else 0 for i in log_model]
        return pred_class