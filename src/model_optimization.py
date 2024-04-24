import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import mlflow
import warnings
warnings.filterwarnings("ignore")
from KPIs import *
from model import build_model
from scikeras.wrappers import KerasClassifier

X_train=np.load("data/X_train.npy")
X_valid=np.load("data/X_valid.npy")
y_train=np.load("data/y_train.npy")
y_valid=np.load("data/y_valid.npy")

def objective(trial):

  activation=trial.suggest_categorical("activation", ["relu", "selu"])
  filters=trial.suggest_int("filters", 16, 64, step=8)
  number_of_layers=trial.suggest_int("number_of_layers", 1, 3)
  dense_units=trial.suggest_int("dense_units", 4, 40,step=8)
  
  params = {
    "activation": activation,
    "filters": filters,
    "number_of_layers": number_of_layers,
    "dense_units": dense_units,
  }
    
  with mlflow.start_run():
    
    def create_model():
      return build_model(params)
    
    model = KerasClassifier( model = create_model, epochs=20, batch_size=32, verbose=0)
    scores=cross_val_score(model, X_train, y_train, cv=4,
                          scoring=make_scorer(f1_score),
                          n_jobs=-1)
    mlflow.log_params(params)
    mlflow.log_metric("mean_f1_score", scores.mean())
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
  return 1 - scores.mean()

def optimize():
  mlflow.set_experiment("optimization")
  study = optuna.create_study()
  study.optimize(objective, n_trials=20)
  best = study.best_trial
  return best



if __name__ == "__main__":
  best = optimize()
  mlflow.autolog()
  mlflow.set_experiment("Deployment")
  with mlflow.start_run() as run:
      model = build_model(best.params)

      model.fit(X_train, y_train,epochs=20,batch_size=32,verbose=0,)
      y_pred = np.round(model.predict(X_valid))
      
      get_classification_report(y_valid, y_pred)
      get_confusion_matrix_plot(y_valid, y_pred)
      
      mlflow.log_params(best.params)
      mlflow.log_metrics(get_metrics(y_valid, y_pred))
      mlflow.log_artifact("src/reports/classification_report.csv")
      mlflow.log_artifact("src/reports/confusion_matrix.png")
      mlflow.end_run()
    