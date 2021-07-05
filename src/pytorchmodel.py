import torch
import torch.nn as nn
import pandas as pd
import config
import numpy as np
from sklearn.model_selection import train_test_split


# import data and convert to numpy arrays
df = pd.read_csv(config.CLEAN_FILE)
features = df.drop("RainTomorrow", axis=1).values
targets = df.RainTomorrow.values

# convert to tensors
features = torch.from_numpy(features.astype(np.float32))
targets = torch.from_numpy(targets.astype(np.float32))

# split the data 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2)

# reshape the tensors
y_test = y_test.view(y_test.shape[0], 1)
y_train = y_train.view(y_train.shape[0], 1)


# build Logistic Regression with PyTorch
class LogisticRegressionPytorch(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegressionPytorch, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegressionPytorch(features.shape[1])

# define Binary Cross Entropy Loss and Stochastic GD optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# for loop training data
for epoch in range(500):
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    print(f"Epoch {epoch + 1} and  Loss: {loss.item():.4f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    y_pred = model(x_test).round()
    acc = (y_pred.eq(y_test).sum() / float(y_test.shape[0])) * 100
    print(f"Logistic Regression using Pytorch Accuracy: {acc:0.2f}%")
