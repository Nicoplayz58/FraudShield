from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import Body, FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.persistence import persist_prediction_results
from .exceptions import AppException
from .feature_builder import build_model_ready_features, transactions_to_dataframe
from db.connection import create_database_engine, get_database_url
from db.schema import ensure_prediction_schema
from .model_loader import ModelBundle, load_model_bundle
from .schemas import EXAMPLE_TRANSACTION, PredictResponse, PredictResponseItem, TransactionInput


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "lightgbm_pipeline.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "lightgbm_pipeline_metadata.json"
DEFAULT_TRAIN_SCHEMA_PATH = PROJECT_ROOT / "credit_card_transactions_model_input.csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()
    metadata_path = Path(os.getenv("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH))).resolve()
    train_schema_path = Path(os.getenv("TRAIN_SCHEMA_PATH", str(DEFAULT_TRAIN_SCHEMA_PATH))).resolve()
    database_url = get_database_url(os.getenv("DATABASE_URL"))

    model_bundle = load_model_bundle(model_path=model_path, metadata_path=metadata_path)
    db_engine = create_database_engine(database_url)
    ensure_prediction_schema(db_engine)

    if train_schema_path.exists():
        train_cols = pd.read_csv(train_schema_path, nrows=0).columns.tolist()
        train_cols = [c for c in train_cols if c != "is_fraud"]
        if train_cols != model_bundle.expected_columns:
            raise RuntimeError(
                "El schema del modelo no coincide con el dataset final de entrenamiento. "
                "Reentrena/exporta el modelo o revisa el post-procesamiento."
            )

    app.state.model_bundle = model_bundle
    app.state.db_engine = db_engine
    app.state.train_schema_checked = train_schema_path.exists()
    yield


app = FastAPI(
    title="FraudShield Serving API",
    version="1.0.0",
    description="API de inferencia para riesgo de fraude transaccional con LightGBM.",
    lifespan=lifespan,
)


def _to_list(payload: TransactionInput | list[TransactionInput]) -> list[TransactionInput]:
    if isinstance(payload, list):
        if len(payload) == 0:
            raise AppException("La lista de transacciones no puede estar vacia.", status_code=422, code="empty_payload")
        return payload
    return [payload]


def _predict_internal(bundle: ModelBundle, db_engine, transactions: list[TransactionInput], threshold: float) -> PredictResponse:
    transaction_rows: list[dict] = []
    for t in transactions:
        tx = t.model_dump(by_alias=True)
        if not tx.get("trans_num"):
            tx["trans_num"] = f"srv_{uuid4().hex}"
        transaction_rows.append(tx)

    raw_df = transactions_to_dataframe(transaction_rows)
    model_df = build_model_ready_features(raw_df, bundle.expected_columns)

    probabilities = bundle.pipeline.predict_proba(model_df)
    if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
        risk_scores = probabilities[:, 1]
    else:
        risk_scores = probabilities.astype(float)

    predictions = (risk_scores >= threshold).astype(int)

    persist_prediction_results(
        db_engine,
        model_name=bundle.model_name,
        threshold=threshold,
        transactions=transaction_rows,
        risk_scores=[float(score) for score in np.asarray(risk_scores, dtype=float)],
        predictions=[int(value) for value in np.asarray(predictions, dtype=int)],
    )

    items: list[PredictResponseItem] = []
    for i, tx in enumerate(transaction_rows):
        items.append(
            PredictResponseItem(
                rank=0,
                transaction_id=str(tx["trans_num"]),
                risk_score=float(risk_scores[i]),
                prediction=int(predictions[i]),
                transaction=tx,
            )
        )

    items.sort(key=lambda x: x.risk_score, reverse=True)
    for rank, item in enumerate(items, start=1):
        item.rank = rank

    return PredictResponse(
        model=bundle.model_name,
        threshold=threshold,
        total_transactions=len(items),
        predictions=items,
    )


@app.get("/health")
async def health(request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise AppException("Modelo no cargado.", status_code=503, code="model_unavailable")
    return {
        "status": "ok",
        "model": bundle.model_name,
        "features": len(bundle.expected_columns),
        "train_schema_checked": bool(getattr(request.app.state, "train_schema_checked", False)),
    }


@app.get("/schema")
async def schema(request: Request):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise AppException("Modelo no cargado.", status_code=503, code="model_unavailable")
    return {
        "feature_count": len(bundle.expected_columns),
        "features": bundle.expected_columns,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    payload: TransactionInput | list[TransactionInput] = Body(
        ...,
        examples={
            "single": {
                "summary": "Una transaccion",
                "value": EXAMPLE_TRANSACTION,
            },
            "batch": {
                "summary": "Batch de transacciones",
                "value": [EXAMPLE_TRANSACTION, {**EXAMPLE_TRANSACTION, "Unnamed: 0": 1, "amt": 540.35}],
            },
        },
    ),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Umbral de clasificacion"),
):
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise AppException("Modelo no cargado.", status_code=503, code="model_unavailable")

    transactions = _to_list(payload)
    db_engine = getattr(request.app.state, "db_engine", None)
    if db_engine is None:
        raise AppException("Persistencia no disponible.", status_code=503, code="database_unavailable")
    return await run_in_threadpool(_predict_internal, bundle, db_engine, transactions, threshold)


@app.exception_handler(AppException)
async def app_exception_handler(_request: Request, exc: AppException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Error de validacion en el payload.",
                "details": jsonable_encoder(exc.errors()),
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Error interno del servidor.",
                "details": str(exc),
            }
        },
    )
