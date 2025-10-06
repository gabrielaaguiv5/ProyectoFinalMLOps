import os, sys, joblib
import numpy as np
import pandas as pd
from src.app.train.etl import UserGenerator 
from src.app.train.feature_engineer import FeatureEngineer

# ---------------------------
# Configuración de rutas
# ---------------------------
# Calcula REPO_ROOT relativo al archivo actual
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = r"C:\Users\usuari\documents\repo\ProyectoFinalMLOps"

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f" Proyecto raíz detectado en: {REPO_ROOT}")

MODEL_PATH = os.path.join(REPO_ROOT, "src", "app", "train", "models", "modeloptuna.pkl")
OUT_PATH = os.path.join(REPO_ROOT, "data", "predictions", "predictions_repurchase.csv")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# ---------------------------
# Importar pipeline de ETL y Feature Engineering
# ---------------------------

def main():
    # ETL
    ug = UserGenerator()
    ug.run_etl()
    df_raw = ug.df.copy()

    # Feature engineering
    fe = FeatureEngineer(df_raw)
    df_features = fe.run()

    assert pd.api.types.is_datetime64_any_dtype(df_features["InvoiceDate"])

    # ---------------------------
    # Cargar modelo entrenado
    # ---------------------------
    from joblib import load
    pipe = load(MODEL_PATH)

    pre = pipe.named_steps.get("preprocessor") or pipe.named_steps.get("pre")
    if pre is None:
        raise RuntimeError("No se encontró el preprocesador en el pipeline (esperado 'preprocessor' o 'pre').")

    # ---------------------------
    # Hacer predicciones
    # ---------------------------
    y_pred = pipe.predict(df_features)
    y_proba = pipe.predict_proba(df_features)[:, 1]

    # Guardar predicciones
    df_features["prediction"] = y_pred
    df_features["probability"] = y_proba
    df_features.to_csv(OUT_PATH, index=False)
    print(f"Predicciones guardadas en {OUT_PATH}")
    
    print("DEBUG - __file__:", __file__)
    print("DEBUG - CURRENT_DIR:", CURRENT_DIR)
    print("DEBUG - REPO_ROOT:", REPO_ROOT)
    print("DEBUG - MODEL_PATH:", MODEL_PATH)



if __name__ == "__main__":
    main()