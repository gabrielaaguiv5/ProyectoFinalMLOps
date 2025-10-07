# ProyectoFinalMLOps

## Integrantes
- Gabriela Aguilar  
- Ximena Patiño Henao  
- Julián Ruiz  

---

## Descripción General
# ProyectoFinalMLOps

Pipeline de MLOps para modelar **recompra a 30 días** (`y\_repurchase\_30d`) sobre el dataset de Online Retail. El proyecto incluye ETL, \*feature engineering\*, división temporal, entrenamiento trazable con MLflow, serialización del pipeline e inferencia por lotes.

Este proyecto lo organizamos en varias carpetas dentro de `src/`, cada una con un rol muy claro en el ciclo de vida del modelo.  
La idea fue estructurarlo de manera que todo el flujo, desde que recibimos los datos hasta que el modelo está disponible en producción, quedara ordenado, fácil de reproducir y entendible para cualquier persona del equipo.

---

## Estructura del Proyecto

### 📂 `src/app`
El directorio src/ contiene todo el código de la aplicación (entrenamiento e inferencia) organizado para asegurar trazabilidad con MLflow y fácil interacción entre los diferentes componentes del pipeline. 
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
<img width="921" height="453" alt="image" src="https://github.com/user-attachments/assets/11a8fc30-350c-4660-9c1a-56fe19a65357" />
<img width="1358" height="663" alt="image" src="https://github.com/user-attachments/assets/c929ae06-0100-4864-aaf4-52e0b818e489" />
<img width="1355" height="662" alt="image" src="https://github.com/user-attachments/assets/9b870bac-0170-4781-b7dc-64a1cab77205" />
<img width="1365" height="664" alt="image" src="https://github.com/user-attachments/assets/870bfe54-00bf-4c6d-8ee7-9adbc2b31379" />
<img width="1355" height="663" alt="image" src="https://github.com/user-attachments/assets/ea632561-fdf6-4545-93d3-2bf89fee004b" />

**Desempeño del modelo**

**1.Matriz de confusion**:
El modelo logra identificar correctamente el 90% de los casos negativos (clase 0), mientras que en la clase positiva (1) alcanza una detección del 44%, evidenciando un buen desempeño general
<img width="585" height="433" alt="image" src="https://github.com/user-attachments/assets/9427141f-82cf-41ba-9031-1d3606c319e5" />

**2.Curva ROC (AUC = 0.74)**:
Indica una capacidad de discriminación aceptable entre ambas clases, lo que demuestra que el modelo puede diferenciar razonablemente bien los eventos positivos y negativos.
<img width="514" height="429" alt="image" src="https://github.com/user-attachments/assets/8d7af8c3-295d-45eb-944d-c9d3f7d5aac7" />

**3.**Curva Precision–Recall (AP = 0.72)**
Muestra un equilibrio adecuado entre precisión y cobertura, reflejando que el modelo mantiene una buena calidad de predicción
<img width="501" height="417" alt="image" src="https://github.com/user-attachments/assets/bedb1353-316f-492d-bb83-a6303456a661" />

**Estructura del pipeline**
El pipeline implementado integra los pasos principales de un proceso de aprendizaje supervisado:
Imputación de valores faltantes (SimpleImputer)
Escalado de variables numéricas (StandardScaler)
Codificación de variables categóricas (OneHotEncoder)
Entrenamiento con regresión logística optimizada

Durante la orquestación, Prefect permitió ejecutar de manera controlada las tareas de configuración, construcción de características, entrenamiento base, optimización y validación final.
Cada ejecución fue monitoreada y registrada en MLflow, asegurando reproducibilidad, versionamiento y trazabilidad completa de los experimentos.
<img width="855" height="415" alt="image" src="https://github.com/user-attachments/assets/d27312fb-425d-4a73-aebf-a48d2e07981b" />
<img width="869" height="422" alt="image" src="https://github.com/user-attachments/assets/d3d2761a-ebdf-4f27-80c5-accdf5168a8c" />

**Resultados del modelo optimizado**
El mejor modelo identificado por Optuna fue una Regresión Logística con los siguientes parámetros destacados:
Regularización: L2
Coeficiente C ≈ 0.8199
Iteraciones máximas: 393
Optimizador: saga
En MLflow, las métricas promedio obtenidas fueron:
Accuracy: 0.70
F1-score: 0.68
Precision: 0.70
Recall: 0.70

Estos resultados reafirman que el modelo mantiene un equilibrio adecuado entre precisión y sensibilidad, alcanzando un rendimiento confiable dentro del conjunto de prueba

<img width="893" height="431" alt="image" src="https://github.com/user-attachments/assets/6fc31e83-31de-4c8e-9748-527ef1c633db" />
<img width="862" height="429" alt="image" src="https://github.com/user-attachments/assets/52f9b902-806f-49b9-9fc6-4a0d8adbba79" />





