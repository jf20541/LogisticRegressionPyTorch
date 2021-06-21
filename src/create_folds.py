import pandas as pd
from sklearn import model_selection
import config


if __name__ == "__main__":
    df = pd.read_csv(config.CLEAN_FILE)
    # create new column as -1 and randomize the data
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    # initiate the kfold 5-folds and fill kfold column
    kf = model_selection.KFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, "kfold"] = fold
    # save the new csv with kofld column
    df.to_csv(config.TRAIN_FOLDS, index=False)
