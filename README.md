# FraudShield

Pipeline para deteccion de fraude en transacciones con analisis exploratorio, feature engineering sin leakage, seleccion de variables y benchmarking de modelos.

## Flujo del proyecto

1. `EDA.ipynb`
   - Analisis exploratorio del dataset raw.
   - Revision de desbalance, distribuciones, nulos y concentracion por usuario.

2. `feature_engineering_script.py`
   - Genera las features `A*`, `T*` y `L*`.
   - Usa solo historial previo, sin look-forward.
   - Exporta `credit_card_transactions_fe.csv`.

3. `VariableSelection.ipynb`
   - Evalua variables originales con calidad de dato, Mutual Information e Information Value.
   - Mantiene todas las engineered features y descarta identificadores, trazabilidad y variables de baja senal.
   - Exporta `credit_card_transactions_model_input.csv` y `variable_selection_report.csv`.

4. `Modeling.ipynb`
   - Benchmark de XGBoost, RandomForest, LightGBM y CatBoost.
   - Preprocesamiento, SMOTENC, cross-validation y evaluacion en holdout.
   - Ranking por PR-AUC para un problema altamente desbalanceado.

5. `save_lightgbm_model.py`
   - Entrena y guarda el pipeline final de LightGBM listo para produccion.

## Archivos principales

- `credit_card_transactions.csv`: dataset raw.
- `credit_card_transactions_fe.csv`: dataset enriquecido con features engineered.
- `credit_card_transactions_model_input.csv`: dataset final para modelado.
- `variable_selection_report.csv`: reporte de seleccion de variables.

## Requisitos

Python 3.10+ y, al menos:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost jupyter
```

## Como ejecutar

```bash
python feature_engineering_script.py --input credit_card_transactions.csv --output credit_card_transactions_fe.csv
```

Luego abrir y correr en orden:

1. `EDA.ipynb`
2. `VariableSelection.ipynb`
3. `Modeling.ipynb`

Para generar el artefacto de produccion:

```bash
python save_lightgbm_model.py --input credit_card_transactions_model_input.csv --output artifacts/lightgbm_pipeline.joblib
```

## Notas metodologicas

- Las features temporales y de comportamiento se construyen con look-back estricto.
- Se excluyen identificadores y variables con alto riesgo de leakage en la seleccion final.
- La metrica principal del modelado es PR-AUC por el fuerte desbalance de clases.

## Resultado destacado

En el benchmark actual, LightGBM fue el mejor modelo en holdout por PR-AUC.

## Databricks

Para llevar el flujo a Databricks, usa la carpeta [databricks/](databricks/) y empieza por su guía principal en [databricks/README.md](databricks/README.md). Ese flujo usa Spark para la preparación de datos dentro de Databricks y MLflow para versionar los entrenamientos.

Ahí tienes:

1. Los scripts ejecutables para feature engineering y entrenamiento con MLflow.
2. Las rutas sugeridas para subir el CSV al Unity Catalog Volume `/Volumes/ml/fraudshield/data`.
3. Los pasos para conectar VS Code con Databricks sin perder el control del repo.
4. La forma de revisar los runs y comparar modelos en MLflow.

## Backend de fraud detection

Para ejecutar el pipeline backend completo:

1. Copiar `.env.example` a `.env` y configurar `DATABASE_URL`.
2. Instalar dependencias del serving:

```bash
pip install -r serving_api/requirements.txt
```

3. Levantar la API de prediccion.
4. Ejecutar simulacion realtime desde PostgreSQL o CSV:

```bash
python simulations/generate_api_simulations.py --mode realtime --source db --batch-size 500 --iterations 20
```

5. Generar reporte de evaluacion sobre predicciones almacenadas:

```bash
python -m evaluation.report --database-url %DATABASE_URL%
```

Esquema SQL disponible en `sql/predictions_schema.sql`.

## Dashboard Streamlit

Para levantar el dashboard de monitoreo en tiempo real:

```bash
pip install -r requirements-dashboard.txt
streamlit run app.py
```

El dashboard usa `DATABASE_URL` y consulta PostgreSQL de forma incremental con filtros, `LIMIT` y caché de datos.
