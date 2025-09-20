from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn


class Trainermlflow:
    
    def __init__(self, df: pd.DataFrame, numeric_features: list, categorical_features: list, target_column: str):
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.pipeline = None

        # Configurar MLflow para guardar runs en carpeta local
        mlflow.set_tracking_uri("file:./tracking/mlruns")
        mlflow.set_experiment("experiment_ventas")

    def train_test_split(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def build_pipeline(self):
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features),
        ])

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ])

        return self.pipeline

    def train(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = self.train_test_split(test_size, random_state)
        self.build_pipeline()

        # Entrenar con MLflow tracking
        with mlflow.start_run():
            self.pipeline.fit(X_train, y_train)

            score = self.pipeline.score(X_test, y_test)

            # Log params
            mlflow.log_param("numeric_features", self.numeric_features)
            mlflow.log_param("categorical_features", self.categorical_features)
            mlflow.log_param("classifier", "LogisticRegression")

            # Log metrics
            mlflow.log_metric("accuracy", score)

            # Log modelo
            mlflow.sklearn.log_model(self.pipeline, "model")

            print(f"✅ Modelo entrenado con accuracy: {score:.4f}")

        return self.pipeline, score

    def save_model(self, path="models/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"✅ Modelo guardado en {path}")
        return path
