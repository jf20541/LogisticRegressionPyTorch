import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import optuna
import config


# import dataset and define target/feature values
df = pd.read_csv(config.CLEAN_FILE)
features = df.drop("RainTomorrow", axis=1).values
targets = df.RainTomorrow.values

# split the data 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, shuffle=False)


def create_model(trial):
    """ Find the optimal ML Model (Decision Tree, SVM, Log-Reg) with optimal hyperparameters
    Args:
        trial: instance represents a process of evaluating an objective function
    Raises:
        optuna.TrialPruned: automatically stops unpromising trials at the early stages of the training (
    Returns: return optimal model with optimal hyperparameters
    """
    model_type = trial.suggest_categorical(
        "model_type", ["logistic-regression", "decision-tree", "svm"]
    )

    # Support Vector Machines with suggested hyper-parameters
    if model_type == "svm":
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        regularization = trial.suggest_uniform("svm-regularization", 0.01, 10)
        degree = trial.suggest_discrete_uniform("degree", 1, 5, 1)
        model = SVC(kernel=kernel, C=regularization, degree=degree)

    # Logistic Regression with suggested hyper-parameters
    if model_type == "logistic-regression":
        penalty = trial.suggest_categorical("penalty", ["l2", "l1"])
        if penalty == "l1":
            solver = "saga"
        else:
            solver = "lbfgs"
        regularization = trial.suggest_uniform("logistic-regularization", 0.01, 10)
        model = LogisticRegression(penalty=penalty, C=regularization, solver=solver)

    # Decision Tree with suggested hyper-parameters
    if model_type == "decision-tree":
        max_depth = trial.suggest_int("max_depth", 5, x_train.shape[1])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 20)
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    # prune trial if model doesn't improve
    if trial.should_prune():
        raise optuna.TrialPruned()

    return model


def model_performance(model, features=x_test, target=y_test):
    # calculate accuracy score
    pred = model.predict(features)
    return accuracy_score(pred, target)


def objective(trial):
    model = create_model(trial)
    model.fit(x_train, y_train)
    return model_performance(model)


if __name__ == "__main__":
    # maximize the model's performace (accuracy)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=3)

    # get optimal model and its hyper-parameters
    best_model = create_model(study.best_trial)
    best_model.fit(x_train, y_train)
    print("Performance: ", model_performance(best_model))
