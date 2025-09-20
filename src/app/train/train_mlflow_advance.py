from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import optuna
import warnings
warnings.filterwarnings('ignore')

import optuna

class TrainOptuna:

    def __init__(self, X_train, X_test, y_train, y_test, model=LogisticRegression, metric="f1"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.metric = metric
        self.study = None
        self.best_model = None

    def _objective(self, trial):
        # Definir hiperparámetros para LogisticRegression
        C = trial.suggest_loguniform("C", 1e-4, 1e2)
        max_iter = trial.suggest_int("max_iter", 100, 1000)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga", "liblinear"])
        
        model = self.model(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        if self.metric == "f1":
            return f1_score(self.y_test, y_pred, average="weighted")
        else:
            return accuracy_score(self.y_test, y_pred)

    def run_study(self, n_trials=30):
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self._objective, n_trials=n_trials)

        print("Mejores hiperparámetros:", self.study.best_params)
        print("Mejor score:", self.study.best_value)

        # Entrenar modelo final
        self.best_model = self.model(**self.study.best_params, random_state=42)
        self.best_model.fit(self.X_train, self.y_train)

        return self.best_model
