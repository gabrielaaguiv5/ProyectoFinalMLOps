# =========================
# Librerias
# =========================
import os, sys, json
from typing import Optional, Literal, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import joblib

from prefect import flow, task, get_run_logger
from pydantic import BaseModel

import mlflow, mlflow.sklearn
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.app.train.etl import UserGenerator
from src.app.train.feature_engineer import FeatureEngineer
from src.app.train.train_mlflow_advance import TrainOptuna


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
class FlowCfg(BaseModel):
    target_column: str = "y_repurchase_30d"
    num_feats: List[str] = [
        "recency_days","n_past_invoices","spend_prior","qty_prior",
        "avg_ticket_prior","avg_qty_per_invoice_prior","UnitPrice","Quantity","Revenue"
    ]
    cat_feats: List[str] = ["Country"]

    # MLflow 
    tracking_uri_http: str = "http://127.0.0.1:5000"
    experiment_baseline: str = "recompra-LogReg"
    experiment_optuna: str = "recompra-optuna"
    enable_autolog: bool = True
    autolog_max_tuning_runs: int = 5

    # Baseline
    run_baseline: bool = True
    baseline_max_iter: int = 500

    # Optuna
    n_trials: int = 30
    optimization_metric: Literal["roc_auc","pr_auc","f1","accuracy","recall","precision"] = "roc_auc"
    random_state: int = 42
    param_distributions: Dict[str, Any] = {
        "solver": ("categorical", ["lbfgs","liblinear","saga"]),
        "C": ("float", 1e-3, 1e2, True),
        "max_iter": ("int", 300, 1500),
        "class_weight": ("categorical", [None, "balanced"]),
        "penalty": ("categorical", ["l2"]),
    }

    # Artefactos
    models_dir: str = os.path.join(REPO_ROOT, "models")
    baseline_model_path: str = os.path.join(REPO_ROOT, "models", "model.pkl")
    optuna_model_path: str = os.path.join(REPO_ROOT, "models", "modeloptuna.pkl")
    best_params_path: str = os.path.join(REPO_ROOT, "models", "best_params.json")
    study_csv_path: str = os.path.join(REPO_ROOT, "models", "study_summary.csv")

    # Guardar según la mejora
    min_improvement: float = 1e-4

# ------------------------------------------------------------------
# Validaciones
# ------------------------------------------------------------------
def fijar_columnas(df, num_fixed, cat_fixed):
    present_num = [c for c in num_fixed if c in df.columns]
    present_cat = [c for c in cat_fixed if c in df.columns]
    missing_num = sorted(set(num_fixed) - set(present_num))
    missing_cat = sorted(set(cat_fixed) - set(present_cat))
    if missing_num or missing_cat:
        get_run_logger().warning(f"Faltan columnas -> num:{missing_num} cat:{missing_cat}. Usando intersección.")
    return present_num, present_cat

def guardar_regla(best_score, baseline_score, min_gain=0.0):
    if baseline_score is None or not np.isfinite(baseline_score):
        return True, float("nan")
    params = best_score - baseline_score
    return (params >= min_gain), params

# ------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------
@task
def build_features() -> pd.DataFrame:
    log = get_run_logger()
    ug = UserGenerator(); ug.run_etl()
    fe = FeatureEngineer(ug.df.copy())
    df = fe.run()
    if "y_repurchase_30d" not in df.columns:
        raise ValueError("No existe la columna y_repurchase_30d en df_features.")
    log.info(f"Features shape={df.shape}")
    return df

@task
def setup_mlflow_and_autolog(cfg: FlowCfg, experiment: str):
    log = get_run_logger()
    # Intento HTTP y fallback a file://
    try:
        mlflow.set_tracking_uri(cfg.tracking_uri_http)
        mlflow.set_experiment(experiment)
    except Exception as e:
        local_store = os.path.abspath(os.path.join(REPO_ROOT, "mlruns"))
        mlflow.set_tracking_uri(f"file:///{local_store}")
        mlflow.set_experiment(experiment)
        log.warning(f"No HTTP tracking server. Usando file:// {local_store}. Error: {e}")

    if cfg.enable_autolog:
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True,
            disable=False,
            exclusive=False,
            disable_for_unsupported_versions=False,
            silent=False,
            max_tuning_runs=cfg.autolog_max_tuning_runs,
        )
    log.info(f"MLflow listo: {mlflow.get_tracking_uri()} exp={experiment}")

@task
def train_baseline(df: pd.DataFrame, cfg: FlowCfg, num_cols: List[str], cat_cols: List[str]) -> Dict[str, Any]:
    """
    Baseline mínimo con LogisticRegression + autolog (sin clase externa).
    Si prefieres tu TrainMlflow, reemplaza este task por el tuyo.
    """
    log = get_run_logger()
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    model = LogisticRegression(max_iter=cfg.baseline_max_iter)
    pipe = Pipeline([("pre", pre), ("clf", model)])

    # CV simple para baseline_score
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Nota: usa roc_auc para alinear con optimization_metric por defecto
    scores = cross_val_score(pipe, df[num_cols + cat_cols], df[cfg.target_column], cv=cv, scoring="roc_auc")
    baseline_score = float(np.mean(scores))

    with mlflow.start_run(run_name="baseline_cv"):
        mlflow.log_param("stage", "baseline")
        mlflow.log_metric("cv_roc_auc", baseline_score)

        # Entrena en todo el dataset para guardar el modelo baseline
        pipe.fit(df[num_cols + cat_cols], df[cfg.target_column])
        os.makedirs(cfg.models_dir, exist_ok=True)
        joblib.dump(pipe, cfg.baseline_model_path, compress=3)
        mlflow.log_param("artifact_model_path", cfg.baseline_model_path)
        mlflow.log_artifact(cfg.baseline_model_path)

    log.info(f"Baseline guardado en {cfg.baseline_model_path} cv_roc_auc={baseline_score:.4f}")
    return {"cv_score": baseline_score, "model_path": cfg.baseline_model_path}

@task
def train_optuna(df: pd.DataFrame, cfg: FlowCfg, num_cols: List[str], cat_cols: List[str]) -> Dict[str, Any]:
    log = get_run_logger()
    # Cambia experimento a Optuna
    mlflow.set_experiment(cfg.experiment_optuna)

    trainer = TrainOptuna(
        df=df,
        numeric_features=num_cols,
        categorical_features=cat_cols,
        target_column=cfg.target_column,
        model_class=LogisticRegression,
        model_params={},  # defaults
        n_trials=cfg.n_trials,
        optimization_metric=cfg.optimization_metric,
        param_distributions=cfg.param_distributions,
        random_state=cfg.random_state,
    )

    # API estilo notebook: retorna (best_pipeline, best_run_id, study)
    best_pipeline, best_run_id, study = trainer.train()
    best_score = getattr(study, "best_value", None)
    best_params = getattr(study, "best_params", {})
    study_df = getattr(study, "trials_dataframe", lambda: pd.DataFrame())()

    return {
        "best_pipeline": best_pipeline,
        "best_run_id": best_run_id,
        "best_score": best_score,
        "best_params": best_params,
        "study_df": study_df,
    }

@task
def persist_and_verify(results: Dict[str, Any], baseline_info: Dict[str, Any], cfg: FlowCfg) -> Dict[str, Any]:
    log = get_run_logger()
    os.makedirs(cfg.models_dir, exist_ok=True)

    baseline_score = baseline_info.get("cv_score") if baseline_info else None
    ok, delta = guardar_regla(results["best_score"], baseline_score, cfg.min_improvement)

    saved_model_path = None
    if ok and results.get("best_pipeline") is not None:
        joblib.dump(results["best_pipeline"], cfg.optuna_model_path, compress=3)
        saved_model_path = cfg.optuna_model_path
        log.info(f"Optuna model guardado: {saved_model_path} (Δ vs baseline: {delta:+.4f})")
    else:
        log.warning(f"No se guarda Optuna model (best={results['best_score']}, baseline={baseline_score}, Δ={delta}).")

    # best_params.json con evidencia
    payload = {
        "metric": cfg.optimization_metric,
        "best_score": results["best_score"],
        "baseline_score": baseline_score,
        "improved": bool(saved_model_path),
        "mlflow_run_id": results.get("best_run_id"),
        "best_params": results.get("best_params", {}),
        "saved_model_path": saved_model_path,
    }
    with open(cfg.best_params_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Trials CSV
    if isinstance(results.get("study_df"), pd.DataFrame) and not results["study_df"].empty:
        results["study_df"].to_csv(cfg.study_csv_path, index=False, encoding="utf-8")

    # Trazas en MLflow del artefacto final
    mlflow.log_param("artifact_model_path", saved_model_path or "")
    mlflow.set_tag("artifact_model_path", saved_model_path or "")
    mlflow.set_tag("improved_vs_baseline", str(bool(saved_model_path)))

    return {
        "saved_model_path": saved_model_path,
        "best_params_path": cfg.best_params_path,
        "study_csv_path": cfg.study_csv_path,
    }

# ------------------------------------------------------------------
# Flow
# ------------------------------------------------------------------
@flow(name="train-with-optuna")
def train_with_optuna_flow(
    run_baseline: Optional[bool] = None,
    n_trials: Optional[int] = None,
    optimization_metric: Optional[str] = None,
):
    cfg = FlowCfg()
    if run_baseline is not None:
        cfg.run_baseline = bool(run_baseline)
    if n_trials is not None:
        cfg.n_trials = int(n_trials)
    if optimization_metric is not None:
        cfg.optimization_metric = optimization_metric  # type: ignore

    # 1) Features
    df = build_features.submit().result()
    num_cols, cat_cols = fijar_columnas(df, cfg.num_feats, cfg.cat_feats)

    # 2) MLflow setup + autolog (baseline)
    setup_mlflow_and_autolog.submit(cfg, cfg.experiment_baseline)

    # 3) Baseline (opcional)
    baseline_info = {"cv_score": None, "model_path": None}
    if cfg.run_baseline:
        baseline_info = train_baseline.submit(df, cfg, num_cols, cat_cols).result()

    # 4) Optuna
    optuna_results = train_optuna.submit(df, cfg, num_cols, cat_cols).result()

    # 5) Guardar artefactos y evidencias
    saved = persist_and_verify.submit(optuna_results, baseline_info, cfg).result()

    return {
        "tracking_uri": mlflow.get_tracking_uri(),
        "baseline_experiment": cfg.experiment_baseline,
        "optuna_experiment": cfg.experiment_optuna,
        "baseline_cv_score": baseline_info.get("cv_score"),
        "optuna_best_score": optuna_results.get("best_score"),
        "saved_model_path": saved.get("saved_model_path"),
        "best_params_path": saved.get("best_params_path"),
        "study_csv_path": saved.get("study_csv_path"),
    }

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    out = train_with_optuna_flow(
        run_baseline=True,
        n_trials=30,
        optimization_metric="roc_auc",
    )
    print(out)