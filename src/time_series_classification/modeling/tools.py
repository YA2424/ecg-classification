import mlflow


def load_best_model():
    """Load best model from mlflow

    Returns:
        model: best model
    """
    run_id = get_best_model_id()
    model = mlflow.pyfunc.load_model("runs:/" + run_id + "/model")
    return model


def get_best_model_id():
    """Get best model id from mlflow

    Returns:
        run_id: best model id
    """
    mlflow.set_experiment("Deployment")
    df = mlflow.search_runs()
    run_id = df.iloc[df["metrics.f1"].idxmax()]["run_id"]
    return run_id
