from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    trans_date_trans_time: datetime = Field(..., description="Fecha-hora transaccion")
    cc_num: int = Field(..., ge=1, description="Numero de tarjeta")
    merchant: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    amt: float = Field(..., ge=0)
    first: str = Field(..., min_length=1)
    last: str = Field(..., min_length=1)
    gender: str = Field(..., min_length=1, max_length=1)
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    state: str = Field(..., min_length=1, max_length=2)
    zip: int = Field(..., ge=0)
    lat: float = Field(..., ge=-90, le=90)
    long: float = Field(..., ge=-180, le=180)
    city_pop: int = Field(..., ge=0)
    job: str = Field(..., min_length=1)
    dob: date = Field(..., description="Fecha de nacimiento")
    trans_num: str | None = Field(default=None, min_length=1, description="Id externo opcional")
    unix_time: int = Field(..., ge=0)
    merch_lat: float = Field(..., ge=-90, le=90)
    merch_long: float = Field(..., ge=-180, le=180)
    merch_zipcode: float | None = Field(default=None, ge=0)

    @field_validator("merch_zipcode", mode="before")
    @classmethod
    def parse_merch_zipcode(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            cleaned = value.replace(",", "").strip()
            return float(cleaned)
        return value


class PredictResponseItem(BaseModel):
    rank: int
    transaction_id: str
    risk_score: float
    prediction: int
    transaction: dict[str, Any]


class PredictResponse(BaseModel):
    model: str
    threshold: float
    total_transactions: int
    predictions: list[PredictResponseItem]


EXAMPLE_TRANSACTION = {
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
    "merch_zipcode": 28705.0,
}
