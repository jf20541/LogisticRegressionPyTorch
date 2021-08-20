import pandas as pd
from model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import argparse
import joblib
import os
import config


def run(fold):
    df = pd.read_csv(config.TRAIN_FOLDS)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    # drop the label column from df and convert to numpy array
    x_train = df_train.drop("RainTomorrow", axis=1).values
    y_train = df_train.RainTomorrow.values
    x_test = df_test.drop("RainTomorrow", axis=1).values
    y_test = df_test.RainTomorrow.values

    # initiate the Logistic Regression model
    model = LogisticRegression(0.0001, 200)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred) * 100
    print(f"Logistic Regression Accuracy: {acc:0.2f}% for Fold={fold}")
    # save the model
    joblib.dump(model, os.path.join(config.MODEL_PATH, f"LR_fold{fold}.bin"))


if __name__ == "__main__":
    # initializing Argument Parser class
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    # read the arguments from CL
    args = parser.parse_args()
    # run the fold specified by CL arguments
    run(fold=args.fold)
