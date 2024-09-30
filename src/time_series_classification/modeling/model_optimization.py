import warnings

import mlflow
import optuna
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

from time_series_classification.data_processing.data_preprocessing import load_data
from time_series_classification.modeling.KPIs import (
    get_classification_report,
    get_confusion_matrix_plot,
    get_metrics,
)
from time_series_classification.modeling.model import build_model

warnings.filterwarnings("ignore")

X_train, X_valid, y_train, y_valid = load_data()

mlflow.autolog()


def objective(trial):
    params = {
        "activation": trial.suggest_categorical("activation", ["relu", "selu"]),
        "filters": trial.suggest_int("filters", 16, 64, step=8),
        "number_of_layers": trial.suggest_int("number_of_layers", 1, 3),
        "dense_units": trial.suggest_int("dense_units", 4, 40, step=8),
    }

    with mlflow.start_run(nested=True):

        model = KerasClassifier(
            model=lambda: build_model(params), epochs=30, batch_size=32, verbose=0
        )
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=4,
            scoring=make_scorer(f1_score, average="weighted"),
            n_jobs=-1,
        )

        mlflow.log_metric("mean_f1_score", scores.mean())

    return 1 - scores.mean()


def optimize(n_trials=10):
    mlflow.set_experiment("Optimization")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial


if __name__ == "__main__":
    best = optimize(n_trials=10)
    mlflow.set_experiment("Optimization")
    with mlflow.start_run() as run:
        model = build_model(best.params)
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
        y_pred = model.predict(X_valid).argmax(axis=1)

        get_classification_report(y_valid, y_pred)
        get_confusion_matrix_plot(y_valid, y_pred)

        mlflow.log_params(best.params)
        mlflow.log_metrics(get_metrics(y_valid, y_pred))
        mlflow.log_artifact("reports/classification_report.csv")
        mlflow.log_artifact("reports/confusion_matrix.png")
        mlflow.sklearn.save_model(model, "src/time_series_classification/deployment/best_model")
