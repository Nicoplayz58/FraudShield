# Databricks setup for FraudShield

This folder packages the FraudShield workflow for Databricks with MLflow tracking, reusable source code, Spark-based feature engineering, and Databricks-ready notebooks.

## What is inside

- `src/fraudshield_databricks/feature_engineering.py`: pandas feature engineering logic reused from the current project.
- `src/fraudshield_databricks/spark_feature_engineering.py`: Spark DataFrame feature engineering for the Databricks exercise.
- `src/fraudshield_databricks/modeling.py`: LightGBM training pipeline with MLflow logging.
- `scripts/01_feature_engineering.py`: loads raw data with Spark and writes the engineered dataset.
- `scripts/02_train_mlflow.py`: reads the engineered data with Spark, converts to pandas for the final trainer, logs metrics, and registers the run in MLflow.
- `notebooks/01_feature_engineering.ipynb` and `notebooks/02_train_mlflow.ipynb`: legacy interactive versions; keep them only if you want exploratory work.
- `requirements.txt`: local dependency list for reproducing the same environment outside Databricks.

## Recommended Databricks flow

1. Upload the raw CSV to DBFS or a Unity Catalog Volume.
2. Run feature engineering with Spark to generate the model input dataset.
3. Train the model with MLflow tracking enabled.
4. Compare runs in MLflow UI and register the best model if you want a managed version.

## Where to store the data

For your current setup, use the Unity Catalog Volume directly:

- Raw file or directory: `/Volumes/ml/fraudshield/data`
- Engineered output: `/Volumes/ml/fraudshield/data/credit_card_transactions_fe.parquet`

Spark can read the raw CSV from the volume and write the parquet output back into the same volume namespace.

## How to upload the data

### Option 1: Databricks UI

1. Open Databricks.
2. Go to **Data** or **Catalog**.
3. Upload `credit_card_transactions.csv` into the Unity Catalog Volume mounted at `/Volumes/ml/fraudshield/data`.
4. Keep the notebook variables pointed at the same volume path.

### Option 2: Databricks CLI

If you prefer terminal upload:

```bash
databricks fs cp credit_card_transactions.csv dbfs:/Volumes/ml/fraudshield/data/credit_card_transactions.csv --overwrite
```

### Option 3: From a notebook

If the file is already on the cluster driver, copy it into the Volume path:

```python
dbutils.fs.cp("file:/tmp/credit_card_transactions.csv", "dbfs:/Volumes/ml/fraudshield/data/credit_card_transactions.csv")
```

## How to run the files

Run the files in this order:

1. `scripts/01_feature_engineering.py`
2. `scripts/02_train_mlflow.py`

The scripts expect the repo to be available as a Databricks Repo or as a workspace folder with the `src` package on the Python path. The feature engineering script uses Spark directly; the training script keeps Spark for loading and then switches to pandas only for the final LightGBM fit.

## MLflow in plain language

MLflow is the tool that lets you keep every training run versioned.

- `experiment`: the container for related runs.
- `run`: one execution of the notebook or training script.
- `parameters`: the model settings used in that run.
- `metrics`: the quality numbers for that run, such as PR-AUC.
- `artifacts`: files saved by the run, such as the trained model.
- `registered model`: the curated model version you decide to promote.

For this project, PR-AUC remains the main metric because the fraud class is highly imbalanced.

## How to connect VS Code to Databricks

The simplest path for a beginner is:

1. Install the **Databricks** extension in VS Code.
2. Sign in to your Databricks workspace from the extension.
3. Install the **Python** extension in VS Code for local editing and linting.
4. Install **Pylance** for type checking and IntelliSense.
5. Install **Jupyter** if you want to open notebooks locally in VS Code.
6. Optional: install **GitLens** if you want a better Git review workflow.
7. Open this repository locally and also create a Databricks Repo that points to the same Git remote.
8. Use Git sync to keep both sides aligned.
9. Edit notebooks in VS Code or in Databricks, then commit and push from either side.

If you want to run code from VS Code against a Databricks cluster, use Databricks Connect or the Databricks extension workflow that your workspace supports.

## What to check in MLflow

After a run finishes, open the MLflow experiment and review:

- PR-AUC on the test split.
- ROC-AUC as a secondary metric.
- Precision and recall if you want a threshold view.
- The logged model artifact.

The notebook also writes a `training_summary.json` artifact so you can inspect the run configuration later.

## Practical next step

Start with the notebooks, verify the raw CSV path, and use the MLflow experiment UI to compare the first few runs. After that, register the best model version and decide whether to move it to a serving endpoint or keep it as a batch scorer.
