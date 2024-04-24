import mlflow
from src.model import get_best_model_id






if __name__ == "__main__":
    # run_id=get_best_model_id()
    # print(run_id)
    model = mlflow.pyfunc.load_model("models:/tsclassifier/production")
    # model.serve(port=5000, enable_mlserver=True)