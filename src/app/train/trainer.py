from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
import os


class Trainer:

    def __init__(self, df: pd.DataFrame, numeric_features: list, categorical_features: list, target_column: str):
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.pipeline = None

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

    def save_model(self, path="models/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"✅ Modelo guardado en {path}")
        return path