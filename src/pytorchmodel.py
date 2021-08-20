import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import config


if __name__ == "__main__":
    # import data and convert to numpy arrays
    df = pd.read_csv(config.CLEAN_FILE)
    features = df.drop("RainTomorrow", axis=1).values
    targets = df.RainTomorrow.values

    # convert to tensors
    features = torch.from_numpy(features.astype(np.float32))
    targets = torch.from_numpy(targets.astype(np.float32))

    # split the data 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.25
    )

    # reshape the tensors
    y_test = y_test.view(y_test.shape[0], 1)
    y_train = y_train.view(y_train.shape[0], 1)

    # build Logistic Regression with PyTorch
    class LogisticRegressionPytorch(nn.Module):
        def __init__(self, n_features):
            super(LogisticRegressionPytorch, self).__init__()
            self.linear = nn.Linear(n_features, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.linear(x)
            return self.sigmoid(x)

    model = LogisticRegressionPytorch(features.shape[1])

    # define Binary Cross Entropy Loss and AdamW
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # for loop training data
    for epoch in range(500):
        output = model(x_train)
        loss = criterion(output, y_train)
        print(f"Epoch {epoch + 1} and  Loss: {loss.item():.4f}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        output = model(x_test).round()
        acc = (output.eq(y_test).sum() / float(y_test.shape[0])) * 100
        print(f"Logistic Regression using Pytorch Accuracy: {acc:0.2f}%")
