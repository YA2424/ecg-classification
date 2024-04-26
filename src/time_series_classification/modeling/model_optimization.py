import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
import mlflow
import warnings
warnings.filterwarnings("ignore")
from time_series_classification.modeling.KPIs import *
from time_series_classification.modeling.model import build_model
from scikeras.wrappers import KerasClassifier
from time_series_classification.data_processing.data_preprocessing import load_data




X_train, X_valid, y_train, y_valid = load_data()

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
    
    model = KerasClassifier( model = create_model, epochs=30, batch_size=32, verbose=0)
    scores=cross_val_score(model, X_train, y_train, cv=4,
                          scoring=make_scorer(f1_score,average='weighted'),
                          n_jobs=-1)
    mlflow.log_params(params)
    mlflow.log_metric("mean_f1_score", scores.mean())
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()
  return 1 - scores.mean()

def optimize():
  mlflow.set_experiment("Optimization")
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
      model.fit(X_train, y_train,epochs=30,batch_size=32,verbose=0,)
      y_pred = np.argmax(model.predict(X_valid),axis=1)
      
      get_classification_report(y_valid, y_pred)
      get_confusion_matrix_plot(y_valid, y_pred)
      
      mlflow.log_params(best.params)
      mlflow.log_metrics(get_metrics(y_valid, y_pred))
      mlflow.log_artifact("reports/classification_report.csv")
      mlflow.log_artifact("reports/confusion_matrix.png")
      mlflow.end_run()
    
    
