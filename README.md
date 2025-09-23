# ProyectoFinalMLOps

## Integrantes
- Gabriela Aguilar  
- Ximena Patiño Henao  
- Julián Ruiz  

---

## Descripción General
# ProyectoFinalMLOps

Pipeline de MLOps para modelar \*\*recompra a 30 días\*\* (`y\_repurchase\_30d`) sobre el dataset de Online Retail. El proyecto incluye ETL, \*feature engineering\*, división temporal, entrenamiento trazable con MLflow, serialización del pipeline e inferencia por lotes.

Este proyecto lo organizamos en varias carpetas dentro de `src/`, cada una con un rol muy claro en el ciclo de vida del modelo.  
La idea fue estructurarlo de manera que todo el flujo, desde que recibimos los datos hasta que el modelo está disponible en producción, quedara ordenado, fácil de reproducir y entendible para cualquier persona del equipo.

---

## Estructura del Proyecto

### 📂 `src/app`
Este módulo es el encargado de tomar el modelo entrenado y exponerlo a través de una API. No se trata solo de devolver predicciones, sino de garantizar que todo el flujo sea confiable y consistente.
Nos enfocamos en que la API:
- Reciba peticiones externas y aplique el mismo preprocesamiento y devuelva predicciones listas para usar.
- Valide correctamente los datos de entrada, evitando errores o inconsistencias 
- Mantenga la trazabilidad mediante logging.
- Ofrezca endpoints adicionales de salud y versión del modelo, para asegurar disponibilidad y control de cambios.

---
## 📂 Estructura del repositorio

\```

src/

└─ app/

├─ train/

│  ├─ mlartifacts/              # Artefactos/experimentos (MLflow/auxiliares)

│  ├─ models/                   # Modelos/pipelines serializados (e.g., model.pkl)

│  ├─ Orchestrator.py           # Orquestador de ETL → FE → train

│  ├─ \_\_init\_\_.py

│  ├─ backend.db                # DB/SQLite auxiliar (si aplica)

│  ├─ etl.py                    # Extracción/limpieza y conformado del dataset base

│  ├─ feature\_engineer.py       # Construcción de variables numéricas/categóricas

│  ├─ task\_train.ipynb          # Notebook guía del proceso de entrenamiento

│  ├─ train\_mlflow.py           # Entrenamiento básico + tracking MLflow

│  └─ train\_mlflow\_advance.py   # Entrenamiento avanzado (p. ej., Optuna)

└─ pred/

├─ etl.py                    # Preparación de datos para scoring

├─ feature\_engineer.py       # Misma lógica de FE para inferencia

└─ (otros módulos de predicción)

data/                               # Datos crudos/procesados

notebooks/                          # EDA/otros notebooks

reports/                            # Salidas, figuras o informes
\---

## Resumen del Flujo
## 🔄 Flujo end-to-end (`src/app/train/task\_train.ipynb`)

1. **Ingesta y ETL (`etl.py`)**
- Carga del histórico transaccional.
- Limpieza de nulos/duplicados y estandarización de tipos.
- Conformado de claves (cliente, factura, fecha/hora).

2. **Feature Engineering (`feature\_engineer.py`)**

- Variables \*\*numéricas\*\*: frecuencia, recencia, monetización, conteos/ratios.
- Variables \*\*categóricas\*\*: banderas, \*bucketizaciones\*, etc.
- Listas `numeric\_features` y `categorical\_features`.
- Definición de la etiqueta binaria `y\_repurchase\_30d`.

3. **División temporal**

- Split \*\*por fechas\*\* (no aleatorio) para evitar \*leakage\*:
- Fechas tipo `train\_end` / `valid\_end` / `cutoff`.
- Evaluación por conjuntos \*\*train/test\*\* y verificación de \*base rates\*.

4. **Entrenamiento con MLflow (`train\_mlflow.py`)**

- Modelo base: `LogisticRegression(max\_iter=500)`.
- Pipeline: `ColumnTransformer` (imputación + escalado + OHE) → clasificador.
- Tracking MLflow: parámetros, métricas y artefactos.
- Serialización del pipeline completo en `src/app/train/models/model.pkl` (+ `run\_id`).

5. **Entrenamiento avanzado (`train\_mlflow\_advance.py`)**

- Búsqueda de hiperparámetros (p. ej., \*\*Optuna\*\*: `solver`, `C`, regularización).
- Registro de ejecuciones y artefactos en MLflow.
- Serialización del mejor pipeline (p. ej., `models/modeloptuna.pkl`).

## Decisiones de diseño

- Split temporal estricto para simular despliegue real (evita data leakage).
- Pipeline único (prepro + modelo) para garantizar paridad tren-serving.
- MLflow para comparabilidad y gobernanza de experimentos.
- Repos separados de train/ y pred/ para claridad operativa.
- Serialización en models/ para facilitar CI/CD y deployment.


## 🗂️ Archivos clave

src/app/train/etl.py: limpieza y conformado del dataset base.

src/app/train/feature_engineer.py: construcción de features y definición de listas de columnas.

src/app/train/train_mlflow.py: entrenamiento con MLflow + guardado del pipeline.

src/app/train/train_mlflow_advance.py: búsqueda/optimización + tracking.

src/app/train/Orchestrator.py: orquestación end-to-end.

src/app/train/models/: salida de modelos (model.pkl, etc.).

src/app/train/task_train.ipynb: cuaderno guía del proceso de entrenamiento (misma lógica que los scripts).


## Resultados en MLFlow

<img width="921" height="398" alt="image" src="https://github.com/user-attachments/assets/9ac4771d-f228-4bde-a410-2537c21121fc" />

1. **Logistic regression:**

experiment_name = "recompra-LogReg"
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name)

/# Enable autologging for sklearn models
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5
)
...

model = LogisticRegression(max_iter=500)

Instancia y entrena
trainer = TrainMlflow(
    df=df_engineered,
    numeric_features=num_feats,
    categorical_features=cat_feats,
    target_column='y_repurchase_30d',
    model=model,
    mlflow_setup={"tracking_uri": "file:./mlruns", "experiment_name": "OnlineRetail"}
)

pipeline, run_id = trainer.train()
trainer.pipeline = pipeline                  # <- necesario para save_model()
trainer.save_model("models/model.pkl")       # ✅ Modelo guardado en models/model.pkl

<img width="921" height="391" alt="image" src="https://github.com/user-attachments/assets/8fa22b60-614c-4207-86e1-53d9691194b3" />
<img width="921" height="317" alt="image" src="https://github.com/user-attachments/assets/89d779ce-1af9-462a-bec8-213c06e6db3e" />
<img width="921" height="449" alt="image" src="https://github.com/user-attachments/assets/a1c08465-40c2-4289-8d1e-29be5e548a4a" />
<img width="921" height="452" alt="image" src="https://github.com/user-attachments/assets/5c7e2c00-ddca-4a8b-aed4-349898ab85c1" />
<img width="921" height="458" alt="image" src="https://github.com/user-attachments/assets/b8a69611-c77d-4b01-9051-be93858fca2c" />
<img width="921" height="457" alt="image" src="https://github.com/user-attachments/assets/59a3ba40-6557-43ac-bdb4-595d87c1c1b8" />

2. **Logistic regression con Optuna:**

mlflow.set_experiment("recompra-optuna")

params = {
            'solver': ('categorical', ['lbfgs', 'liblinear', 'saga']),
            'C':      ('float', 1e-3, 1e2, True),
            'max_iter': ('int', 300, 1500),
            'class_weight': ('categorical', [None, 'balanced']),
            # solver-specific penalties are tricky to encode generically—start simple with l2
            'penalty': ('categorical', ['l2']),
}

trainer = TrainOptuna(
    df=df_engineered,
    numeric_features=num_feats,
    categorical_features=cat_feats,
    target_column='y_repurchase_30d',
    model_class=LogisticRegression,
    model_params={},                 
    n_trials=30,                     
    optimization_metric='roc_auc',   
    param_distributions=params,
)

best_pipeline, best_run_id, study = trainer.train()   # runs Optuna + logs to MLflow
trainer.save_model("models/modeloptuna.pkl")
<img width="921" height="335" alt="image" src="https://github.com/user-attachments/assets/42437eeb-5cc7-410b-a97a-e06de71caa65" />
<img width="921" height="296" alt="image" src="https://github.com/user-attachments/assets/ad47696b-cea1-471e-bd7f-c5ad5df1cffc" />



