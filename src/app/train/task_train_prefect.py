import os
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd
import joblib

from prefect import flow, task, get_run_logger
from pydantic import BaseModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.app.train.etl import UserGenerator
from src.app.train.feature_engineer import FeatureEngineer
from src.app.train.train_mlflow_advance import TrainOptuna  

class PredictConfig(BaseModel):
    model_path: str = os.path.join(REPO_ROOT, "src", "app", "train", "models", "modeloptuna.pkl")
    threshold_path: Optional[str] = os.path.join(REPO_ROOT, "models", "threshold.joblib")
    out_csv_path: str = os.path.join(REPO_ROOT, "data", "predictions", "predictions_repurchase.csv")
    # estrategia de umbral: "fixed" | "topk" | "f1"
    threshold_policy: Literal["fixed", "topk", "f1"] = "f1"
    fixed_threshold: float = 0.5
    topk_rate: float = 0.05  
    save_threshold_if_f1: bool = True  # guardar el threshold calculado por F1
    mlflow_log: bool = True  # loggear corrida de predicción en MLflow

# Prefect
@task
def load_pipeline(cfg):
    logger = get_run_logger()
    logger.info(f"Cargando pipeline desde: {cfg.model_path}")
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {cfg.model_path}")
    return joblib.load(cfg.model_path)

@task
def build_inference_features():
    """
    Usa TU ETL + FE para construir el DF de inferencia (mismas transformaciones que en training).
    """
    logger = get_run_logger()
    logger.info("Ejecutando ETL (UserGenerator.run_etl)")
    ug = UserGenerator()
    ug.run_etl()            
    df_raw = ug.df.copy()

    logger.info("Ejecutando FeatureEngineering (FeatureEngineer.run)")
    fe = FeatureEngineer(df_raw)
    df_features = fe.run()  
    return df_features

@task
def extract_feat_cols_from_pipeline(pipe):
    """
    Extrae las columnas exactas que espera el ColumnTransformer del pipeline.
    """
    logger = get_run_logger()
    pre = pipe.named_steps.get("preprocessor") or pipe.named_steps.get("pre")
    if pre is None:
        raise RuntimeError("No se encontró el step 'preprocessor'/'pre' en el pipeline.")

    transformer_map = {name: cols for name, trans, cols in pre.transformers_}
    num_cols = list(transformer_map.get("num", []))
    cat_cols = list(transformer_map.get("cat", []))
    feat_cols = num_cols + cat_cols

    logger.info(f"Columnas num: {len(num_cols)} | cat: {len(cat_cols)} | total: {len(feat_cols)}")
    return num_cols, cat_cols, feat_cols

@task
def choose_threshold(cfg: PredictConfig, pipe, df_features: pd.DataFrame, num_cols, cat_cols) -> float:
    """
    Selecciona t_star según policy:
        - fixed: usa cfg.fixed_threshold
        - topk: toma k = rate * N sobre probabilidades del test temporal (o en batch si prefieres)
        - f1: hace split temporal con TrainOptuna y maximiza F1 en el test
    """
    logger = get_run_logger()
    policy = cfg.threshold_policy

    if cfg.threshold_path and os.path.exists(cfg.threshold_path) and policy != "fixed":
        try:
            th = joblib.load(cfg.threshold_path)
            if isinstance(th, dict) and "decision_threshold" in th:
                t_star = float(th["decision_threshold"])
                logger.info(f"Umbral cargado desde {cfg.threshold_path}: t={t_star:.4f}")
                return t_star
        except Exception as e:
            logger.warning(f"No se pudo cargar threshold de {cfg.threshold_path}: {e}")

    if policy == "fixed":
        logger.info(f"Usando umbral fijo: t={cfg.fixed_threshold}")
        return float(cfg.fixed_threshold)

    target_column = "y_repurchase_30d"
    if target_column not in df_features.columns and policy == "f1":
        raise ValueError(f"No existe la columna target '{target_column}' en df_features (requerida para policy='f1').")

    trainer = TrainOptuna(
        df=df_features,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        target_column=target_column,
        n_trials=1,
        optimization_metric="roc_auc",
    )
    X_train, X_test, y_train, y_test = trainer.train_test_split_by_quantiles()

    proba_test = pipe.predict_proba(X_test)[:, 1]

    if policy == "topk":
        rate = max(min(cfg.topk_rate, 1.0), 0.0)
        k = int(len(proba_test) * rate)
        t_star = float(np.partition(proba_test, -k)[-k]) if k > 0 else 1.0
        logger.info(f"Umbral por capacidad top-{rate*100:.1f}%: t={t_star:.4f}")
        return t_star

    # "f1": elegimos t que maximiza F1 en test
    from sklearn.metrics import precision_recall_curve, f1_score
    prec, rec, thr = precision_recall_curve(y_test.astype(int), proba_test)
    if len(thr) > 0:
        f1s = [f1_score(y_test, (proba_test >= t).astype(int)) for t in thr]
        t_star = float(thr[int(np.argmax(f1s))])
    else:
        t_star = 0.5
    
    if t_star <= 1e-6:
    # fallback razonable si la clase positiva domina
        t_star = 0.5    

    logger.info(f"Umbral por F1 óptimo en test: t={t_star:.4f}")

    if cfg.save_threshold_if_f1:
        os.makedirs(os.path.join(REPO_ROOT, "models"), exist_ok=True)
        joblib.dump({"decision_threshold": t_star},
                os.path.join(REPO_ROOT, "models", "threshold.joblib"),
                compress=3)

    if cfg.save_threshold_if_f1:
        os.makedirs(os.path.join(REPO_ROOT, "models"), exist_ok=True)
        joblib.dump({"decision_threshold": t_star},
                    os.path.join(REPO_ROOT, "models", "threshold.joblib"),
                    compress=3)
        logger.info("Threshold guardado en models/threshold.joblib")
    return t_star

@task
def score_and_save(cfg: PredictConfig, pipe, df_features: pd.DataFrame, feat_cols, t_star: float):
    """
    Construye X_new con columnas exactas, predice, aplica umbral y guarda CSV.
    """
    logger = get_run_logger()
    X_new = df_features.reindex(columns=feat_cols)
    proba = pipe.predict_proba(X_new)[:, 1]
    yhat = (proba >= t_star).astype(int)

    pred = df_features.copy()
    pred["p_repurchase_30d"] = proba
    pred["repurchase_flag"] = yhat

    os.makedirs(os.path.dirname(cfg.out_csv_path), exist_ok=True)
    pred.to_csv(cfg.out_csv_path, index=False, encoding="utf-8")
    logger.info(f"Predicciones guardadas en: {cfg.out_csv_path}")

    logger.info(pred[["CustomerID", "InvoiceDate", "p_repurchase_30d", "repurchase_flag"]].head().to_string())
    return cfg.out_csv_path

@task
def mlflow_log_inference(cfg: PredictConfig, model_path: str, threshold: float, n_scored: int, artifact_path: str):
    """Logging opcional en MLflow."""
    import mlflow
    logger = get_run_logger()
    try:
        with mlflow.start_run(run_name="batch_predict"):
            mlflow.log_param("stage", "batch_inference")
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("threshold", float(threshold))
            mlflow.log_param("n_scored", int(n_scored))
            if os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)
        logger.info(f"Batch scoring loggeado en MLflow ({mlflow.get_tracking_uri()})")
    except Exception as e:
        logger.warning(f"No se pudo loggear en MLflow: {e}")
@task
def mlflow_log_inference(cfg, model_path: str, threshold: float, n_scored: int, artifact_path: str):
    """Log the batch inference run to MLflow."""
    import os
    import mlflow
    log = get_run_logger()
    try:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("batch_predict")
        with mlflow.start_run(run_name="batch_predict"):
            mlflow.log_param("stage", "batch_inference")
            mlflow.log_param("model_path", model_path)
            mlflow.log_param("threshold", float(threshold))
            mlflow.log_param("n_scored", int(n_scored))
            if artifact_path and os.path.exists(artifact_path):
                mlflow.log_artifact(artifact_path)
        log.info("Batch scoring logged to MLflow.")
    except Exception as e:
        log.warning(f"Could not log to MLflow: {e}")


@flow(name="predict-batch")
def predict_batch_flow(
    model_path: Optional[str] = None,
    threshold_policy: Optional[str] = None,
    fixed_threshold: Optional[float] = None,
    topk_rate: Optional[float] = None,
    out_csv_path: Optional[str] = None,
    mlflow_log: Optional[bool] = None
):
    """
    Ejecuta el flujo de predicción en batch con overrides opcionales.
    """
    cfg = PredictConfig()
    # Submit ONCE, keep the futures
    pipe_fut        = load_pipeline.submit(cfg)
    df_feat_fut     = build_inference_features.submit()
    cols_fut        = extract_feat_cols_from_pipeline.submit(pipe_fut)

    # Unpack the tuple returned by cols_fut
    num_cols, cat_cols, feat_cols = cols_fut.result()

    # Reuse the SAME df_feat_fut future everywhere (do NOT resubmit the task)
    t_star_fut = choose_threshold.submit(cfg, pipe_fut, df_feat_fut, num_cols, cat_cols)
    out_path_fut = score_and_save.submit(cfg, pipe_fut, df_feat_fut, feat_cols, t_star_fut)

    if cfg.mlflow_log:
        # Reuse df_feat_fut and resolve to get the row count
        n_scored = int(df_feat_fut.result().shape[0])
        mlflow_log_inference.submit(
            cfg,
            model_path=cfg.model_path,
            threshold=t_star_fut,       # can pass future here
            n_scored=n_scored,          # resolved value
            artifact_path=out_path_fut  # future is OK
        )

    return out_path_fut

# Ejecutable local
if __name__ == "__main__":
    # Ejecución local:
    # Umbral F1 (default)
    predict_batch_flow()

