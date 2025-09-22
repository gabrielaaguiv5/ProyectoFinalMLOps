import json
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Importar las clases y funciones personalizadas del proyecto
from etl import UserGenerator
from feature_engineer import FeatureEngineer
from train_mlflow import TrainMlflow
from train_mlflow_advance import TrainOptuna

# Configuración inicial
def main():
    # 1. Generación de datos ETL
    user_generator = UserGenerator(n_samples=25000)
    ds = user_generator.create_dataset()
    print(f"Tipo de dataset: {type(ds)}, ¿es una tupla? {isinstance(ds, tuple)}")
    
    # 2. Ejecutar ETL (si es necesario)
    ds = user_generator.run_etl()
    print(f"Info Dataset:\n{ds.info()}")
    
    # 3. Exploración básica
    print(f"Fecha mínima: {ds['InvoiceDate'].min()}")
    print(f"Fecha máxima: {ds['InvoiceDate'].max()}")
    print(f"Clientes únicos: {ds['CustomerID'].nunique()}")
    print(f"Productos únicos: {ds['Description'].nunique()}")
    print(f"Países: {ds['Country'].nunique()}")
    print(f"Rango de fechas: {ds['InvoiceDate'].min().date()} → {ds['InvoiceDate'].max().date()}")

    # 4. Feature Engineering
    feature_engineer = FeatureEngineer(ds)
    df_engineered = feature_engineer.run()
    # Mostrar algunas filas del dataframe generado
    print(df_engineered.head())

    # 5. Configuración MLflow y entrenamiento del modelo
    import mlflow
    import mlflow.sklearn

    experiment_name = "recompra-LogReg"
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment_name)

    # Definir features numéricas y categóricas
    num_feats = [
        'recency_days', 'n_past_invoices', 'spend_prior', 'qty_prior',
        'avg_ticket_prior', 'avg_qty_per_invoice_prior',
        'UnitPrice', 'Quantity', 'Revenue'
    ]
    cat_feats = ['Country']

    # 6. Entrenamiento con MLflow
    model = LogisticRegression(max_iter=500)
    trainer = TrainMlflow(
        df=df_engineered,
        numeric_features=num_feats,
        categorical_features=cat_feats,
        target_column='y_repurchase_30d',
        model=model,
        mlflow_setup={
            "tracking_uri": "file:./mlruns",
            "experiment_name": "OnlineRetail"
        }
    )
    pipeline, run_id = trainer.train()

    # Guardar modelo
    trainer.pipeline = pipeline
    trainer.save_model("models/model.pkl")

    # 7. Optimización con Optuna
    from sklearn.linear_model import LogisticRegression
    
    params = {
        'solver': ('categorical', ['lbfgs', 'liblinear', 'saga']),
        'C': ('float', 1e-3, 1e2, True),
        'max_iter': ('int', 300, 1500),
        'class_weight': ('categorical', [None, 'balanced']),
        'penalty': ('categorical', ['l2'])
    }

    best_pipeline, best_run_id, study = TrainOptuna(
        df=df_engineered,
        numeric_features=num_feats,
        categorical_features=cat_feats,
        target_column='y_repurchase_30d',
        model_class=LogisticRegression,
        model_params={},
        n_trials=30,
        optimization_metric='roc_auc',
        param_distributions=params
    ).train()

    # Guardar la mejor solución
    TrainOptuna(
        df=df_engineered,
        numeric_features=num_feats,
        categorical_features=cat_feats,
        target_column='y_repurchase_30d',
        model_class=LogisticRegression,
        model_params=study.best_params,
        n_trials=0,  # ya no es necesario
        optimization_metric='roc_auc'
    ).save_model("models/modeloptuna.pkl")

if __name__ == "__main__":
    main()
