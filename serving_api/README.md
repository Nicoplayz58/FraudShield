# FraudShield Serving API (FastAPI)

API de produccion para inferencia de fraude usando el modelo LightGBM preentrenado en `artifacts/lightgbm_pipeline.joblib`.

## Estructura

```text
serving_api/
  app/
    __init__.py
    exceptions.py
    feature_builder.py
    main.py
    model_loader.py
    schemas.py
  requirements.txt
  README.md
```

## Que hace la API

- Carga el modelo una sola vez al iniciar (lifespan FastAPI).
- Recibe 1 o N transacciones en `POST /predict`.
- Valida payload con Pydantic.
- Si `trans_num` no llega en el input, la API genera un id interno automaticamente.
- Calcula features engineered (`A*`, `T*`, `L*`) usando logica look-back.
- Aplica post-procesamiento estricto para dejar el dataset de inferencia con el mismo schema final usado en entrenamiento (`credit_card_transactions_model_input.csv` sin `is_fraud`).
- Ejecuta inferencia con el pipeline LightGBM.
- Retorna transacciones ordenadas de mayor a menor riesgo.

## Dependencias

Recomendado: Python 3.11 o 3.12 para evitar compilacion manual de paquetes cientificos.

```bash
pip install -r serving_api/requirements.txt
```

## Ejecucion local con Uvicorn

Desde la raiz del proyecto (`FraudShield`):

```bash
uvicorn serving_api.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger:
- `http://localhost:8000/docs`

Healthcheck:
- `http://localhost:8000/health`

Schema esperado por el modelo:
- `http://localhost:8000/schema`

## Endpoint de inferencia

`POST /predict`

- Acepta objeto unico o lista de objetos.
- Parametro opcional `threshold` (default `0.5`).
- `is_fraud` no forma parte del contrato de inferencia.

Ejemplo `curl` (una transaccion):

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "trans_date_trans_time": "2019-01-01 00:00:18",
    "cc_num": 2703186189652095,
    "merchant": "fraud_Rippin, Kub and Mann",
    "category": "misc_net",
    "amt": 4.97,
    "first": "Jennifer",
    "last": "Banks",
    "gender": "F",
    "street": "561 Perry Cove",
    "city": "Moravian Falls",
    "state": "NC",
    "zip": 28654,
    "lat": 36.0788,
    "long": -81.1781,
    "city_pop": 3495,
    "job": "Psychologist, counselling",
    "dob": "1988-03-09",
    "unix_time": 1325376018,
    "merch_lat": 36.0113,
    "merch_long": -82.0483,
    "merch_zipcode": 28705.0
  }'
```

## Variables de entorno opcionales

- `MODEL_PATH`: ruta al `.joblib` (default: `artifacts/lightgbm_pipeline.joblib`).
- `MODEL_METADATA_PATH`: metadata del modelo (default: `artifacts/lightgbm_pipeline_metadata.json`).
- `TRAIN_SCHEMA_PATH`: CSV final de entrenamiento para validar schema (default: `credit_card_transactions_model_input.csv`).
