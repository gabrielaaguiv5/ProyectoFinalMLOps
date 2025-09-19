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
import warnings
warnings.filterwarnings('ignore')

class TrainMlflow:
    def __init__(
        self, df, numeric_features, categorical_features, 
        target_column, model, 
        test_size=0.2, model_params=None, mlflow_setup=None
    ):
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.test_size = test_size
        self.model = model
        self.model_params = model_params if model_params is not None else {}

        # Set up MLflow tracking
        self.setup = mlflow_setup
        

    def train_test_split(self):
        