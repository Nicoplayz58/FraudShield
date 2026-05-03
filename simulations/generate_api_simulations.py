from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from faker import Faker
from dotenv import load_dotenv
from joblib import load as joblib_load
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from simulation.realtime import SimulationConfig, run_realtime_simulation  # noqa: E402


API_FIELDS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "merch_zipcode",
]


FRAUD_PATTERNS = [
    "high_amount_anomaly",
    "geolocation_anomaly",
    "time_anomaly",
    "merchant_category_inconsistency",
]


@dataclass
class Knobs:
    amount_volatility: float = 1.0
    weekend_boost: float = 1.25
    recurrent_ratio: float = 0.85
    travel_boost: float = 1.0


def _build_trans_num(run_id: str, sequence: int) -> str:
    return f"sim_{run_id}_{sequence:08d}_{uuid4().hex}"


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    total = float(s.sum())
    if total <= 0:
        return pd.Series(np.ones(len(s)) / len(s), index=s.index)
    return s / total


def _estimate_amount_profile(amounts: pd.Series) -> dict[str, float | str]:
    clean = pd.to_numeric(amounts, errors="coerce").dropna()
    clean = clean[clean > 0]

    if clean.empty:
        return {
            "dist": "lognormal",
            "mu": 3.5,
            "sigma": 0.7,
            "shape": 2.0,
            "scale": 20.0,
            "q01": 1.0,
            "q99": 500.0,
            "q995": 700.0,
            "tail_rate": 0.01,
        }

    log_vals = np.log(clean.clip(lower=0.01))
    mu = float(log_vals.mean())
    sigma = float(max(log_vals.std(ddof=0), 0.08))

    mean = float(clean.mean())
    var = float(clean.var(ddof=0))

    if mean <= 0 or var <= 0:
        shape, scale = 2.0, max(mean / 2.0, 1.0)
    else:
        shape = float(max((mean * mean) / var, 0.2))
        scale = float(max(var / mean, 0.5))

    std = float(clean.std(ddof=0))
    skew = 0.0 if std <= 0 else float((((clean - mean) ** 3).mean()) / (std**3))
    dist = "lognormal" if skew >= 1.5 else "gamma"

    q01 = float(clean.quantile(0.01))
    q99 = float(clean.quantile(0.99))
    q995 = float(clean.quantile(0.995))
    tail_rate = float(np.clip((clean > q99).mean(), 0.005, 0.05))

    return {
        "dist": dist,
        "mu": mu,
        "sigma": sigma,
        "shape": shape,
        "scale": scale,
        "q01": max(0.01, q01),
        "q99": max(q99, q01 + 1.0),
        "q995": max(q995, q99 + 1.0),
        "tail_rate": tail_rate,
    }


def _build_distance_profiles(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    dist_ok = df[["category", "lat", "long", "merch_lat", "merch_long"]].dropna()
    profiles: dict[str, dict[str, float]] = {}

    for cat, sub in dist_ok.groupby("category"):
        d = haversine_km(
            sub["lat"].to_numpy(),
            sub["long"].to_numpy(),
            sub["merch_lat"].to_numpy(),
            sub["merch_long"].to_numpy(),
        )
        if len(d) == 0:
            continue
        local = d[d <= 80]
        sigma_km = np.std(local) if len(local) > 5 else 3.0
        profiles[str(cat)] = {
            "travel_rate": float(np.clip((d > 80).mean(), 0.01, 0.5)),
            "sigma_km": float(np.clip(sigma_km, 1.5, 20.0)),
        }

    return profiles


def _build_profiles(df: pd.DataFrame, knobs: Knobs) -> dict[str, Any]:
    prof: dict[str, Any] = {}

    prof["category_probs"] = _normalize(df["category"].value_counts())
    prof["state_probs"] = _normalize(df["state"].value_counts())

    city_cols = ["city", "state", "zip", "lat", "long", "city_pop"]
    city_ref = (
        df[city_cols]
        .dropna(subset=["city", "state", "lat", "long"])
        .groupby(["city", "state"], as_index=False)
        .agg(
            zip=("zip", "median"),
            lat=("lat", "median"),
            long=("long", "median"),
            city_pop=("city_pop", "median"),
        )
    )

    city_counts = df.groupby(["city", "state"], as_index=False).size().rename(columns={"size": "cnt"})
    city_ref = city_ref.merge(city_counts, on=["city", "state"], how="left")
    city_ref["prob"] = city_ref["cnt"] / max(city_ref["cnt"].sum(), 1)
    prof["city_ref"] = city_ref.reset_index(drop=True)

    prof["merchant_by_cat"] = {
        str(cat): sub["merchant"].dropna().astype(str).unique().tolist()
        for cat, sub in df.groupby("category")
    }

    prof["amount_by_cat"] = {
        str(cat): _estimate_amount_profile(sub["amt"])
        for cat, sub in df.groupby("category")
    }
    prof["amount_global"] = _estimate_amount_profile(df["amt"])

    dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce").dropna()
    if dt.empty:
        dow_probs = pd.Series(np.ones(7) / 7, index=np.arange(7))
        weekday_hours = np.ones(24) / 24
        weekend_hours = np.ones(24) / 24
    else:
        dow_probs = _normalize(dt.dt.dayofweek.value_counts().reindex(np.arange(7), fill_value=0) + 1.0)
        weekday = dt[dt.dt.dayofweek < 5]
        weekend = dt[dt.dt.dayofweek >= 5]
        weekday_hours = weekday.dt.hour.value_counts().reindex(np.arange(24), fill_value=0).to_numpy(dtype=float) + 1.0
        weekend_hours = weekend.dt.hour.value_counts().reindex(np.arange(24), fill_value=0).to_numpy(dtype=float) + 1.0
        weekday_hours /= weekday_hours.sum()
        weekend_hours /= weekend_hours.sum()

    prof["dow_probs"] = dow_probs
    prof["weekday_hours"] = weekday_hours
    prof["weekend_hours"] = weekend_hours

    dist_ok = df[["lat", "long", "merch_lat", "merch_long"]].dropna()
    if dist_ok.empty:
        base_travel_rate = 0.07
        local_sigma_km = 3.0
    else:
        d = haversine_km(
            dist_ok["lat"].to_numpy(),
            dist_ok["long"].to_numpy(),
            dist_ok["merch_lat"].to_numpy(),
            dist_ok["merch_long"].to_numpy(),
        )
        base_travel_rate = float((d > 80).mean())
        local = d[d <= 80]
        local_sigma_km = float(np.clip(np.std(local) if len(local) > 5 else 3.0, 1.5, 20.0))

    prof["travel_rate"] = float(np.clip(base_travel_rate * knobs.travel_boost, 0.01, 0.5))
    prof["local_sigma_km"] = local_sigma_km
    prof["travel_boost"] = knobs.travel_boost

    user_cols = ["cc_num", "first", "last", "gender", "street", "city", "state", "zip", "lat", "long", "city_pop", "job", "dob"]
    user_pool = (
        df[user_cols]
        .dropna(subset=["cc_num", "city", "state", "lat", "long"])
        .drop_duplicates(subset=["cc_num"])
        .copy()
    )
    user_pool["dob"] = pd.to_datetime(user_pool["dob"], errors="coerce").dt.date.astype(str)

    prof["user_pool"] = user_pool.reset_index(drop=True)
    prof["distance_by_cat"] = _build_distance_profiles(df)
    return prof


def _sample_timestamp(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, prof: dict[str, Any], knobs: Knobs) -> pd.Timestamp:
    days = pd.date_range(start.normalize(), end.normalize(), freq="D")
    if len(days) == 0:
        return pd.Timestamp(datetime.utcnow())

    dow = days.dayofweek
    dow_w = prof["dow_probs"].reindex(np.arange(7), fill_value=1 / 7).to_numpy(dtype=float)
    weights = dow_w[dow]
    weights = np.where(dow >= 5, weights * knobs.weekend_boost, weights)
    weights /= weights.sum()

    day = pd.Timestamp(rng.choice(days, p=weights))
    hour_probs = prof["weekend_hours"] if day.dayofweek >= 5 else prof["weekday_hours"]
    hour = int(rng.choice(np.arange(24), p=hour_probs))

    return pd.Timestamp(
        day.year,
        day.month,
        day.day,
        hour,
        int(rng.integers(0, 60)),
        int(rng.integers(0, 60)),
    )


def _sample_amount(rng: np.random.Generator, cat: str, prof: dict[str, Any], knobs: Knobs) -> float:
    p = prof["amount_by_cat"].get(cat, prof["amount_global"])
    vol = max(0.2, knobs.amount_volatility)

    if p["dist"] == "lognormal":
        x = rng.lognormal(p["mu"], max(0.05, p["sigma"] * vol))
    else:
        x = rng.gamma(max(0.2, p["shape"]), max(0.2, p["scale"] * vol))

    if rng.random() < np.clip(p["tail_rate"] * vol, 0.005, 0.08):
        x *= rng.lognormal(1.0, 0.55)

    return round(float(np.clip(x, p["q01"], max(p["q995"] * 1.8, p["q01"] + 5.0))), 2)


def _km_to_deg_lat(km: float) -> float:
    return km / 111.0


def _km_to_deg_lon(km: float, lat: float) -> float:
    return km / (111.320 * max(np.cos(np.radians(lat)), 0.2))


def _sample_merchant_geo(rng: np.random.Generator, user: dict[str, Any], prof: dict[str, Any], category: str) -> tuple[float, float, float]:
    city_ref = prof["city_ref"]
    cat_profile = prof["distance_by_cat"].get(category)

    if cat_profile:
        travel_rate = float(np.clip(cat_profile["travel_rate"] * prof["travel_boost"], 0.01, 0.7))
        sigma_km = float(cat_profile["sigma_km"])
    else:
        travel_rate = float(prof["travel_rate"])
        sigma_km = float(prof["local_sigma_km"])

    travel = rng.random() < travel_rate
    if travel and not city_ref.empty:
        row = city_ref.loc[int(rng.choice(city_ref.index, p=city_ref["prob"].to_numpy()))]
        base_lat, base_lon, mzip = float(row["lat"]), float(row["long"]), float(row["zip"])
    else:
        base_lat, base_lon, mzip = float(user["lat"]), float(user["long"]), float(user["zip"])

    return (
        base_lat + rng.normal(0.0, _km_to_deg_lat(sigma_km)),
        base_lon + rng.normal(0.0, _km_to_deg_lon(sigma_km, base_lat)),
        mzip,
    )


def _sample_city_row(rng: np.random.Generator, city_ref: pd.DataFrame) -> pd.Series:
    if city_ref.empty:
        return pd.Series({"city": "Unknown", "state": "NA", "zip": 0, "lat": 0.0, "long": 0.0, "city_pop": 0})
    return city_ref.loc[int(rng.choice(city_ref.index, p=city_ref["prob"].to_numpy()))]


def _build_new_user(fake: Faker, rng: np.random.Generator, city_ref: pd.DataFrame) -> dict[str, Any]:
    city = _sample_city_row(rng, city_ref)
    gender = str(rng.choice(["M", "F"]))
    first = fake.first_name_male() if gender == "M" else fake.first_name_female()

    return {
        "cc_num": int(rng.integers(10**15, 10**16 - 1)),
        "first": first,
        "last": fake.last_name(),
        "gender": gender,
        "street": fake.street_address(),
        "city": str(city["city"]),
        "state": str(city["state"]),
        "zip": int(float(city["zip"])),
        "lat": float(city["lat"]),
        "long": float(city["long"]),
        "city_pop": int(float(city["city_pop"])),
        "job": fake.job(),
        "dob": fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
    }


def generate_records(
    base_csv: Path,
    n_transactions: int,
    knobs: Knobs,
    seed: int,
    start_date: str | None,
    end_date: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    fake = Faker("en_US")
    Faker.seed(seed)
    run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")

    df = pd.read_csv(base_csv)
    prof = _build_profiles(df, knobs)

    dt = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    start = pd.Timestamp(start_date) if start_date else pd.Timestamp(dt.min())
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(dt.max())

    if end < start:
        raise ValueError("end_date debe ser mayor o igual que start_date")

    users = prof["user_pool"].to_dict(orient="records") if not prof["user_pool"].empty else []
    cats = prof["category_probs"].index.to_list()
    probs = prof["category_probs"].to_numpy()

    records: list[dict[str, Any]] = []
    for _ in range(n_transactions):
        if users and rng.random() < knobs.recurrent_ratio:
            user = users[int(rng.integers(0, len(users)))]
        else:
            user = _build_new_user(fake, rng, prof["city_ref"])

        ts = _sample_timestamp(rng, start, end, prof, knobs)
        cat = str(rng.choice(cats, p=probs))
        amt = _sample_amount(rng, cat, prof, knobs)

        merchants = prof["merchant_by_cat"].get(cat, [])
        merchant = str(rng.choice(merchants)) if merchants else f"merchant_{cat}_{fake.company().split(' ')[0]}"

        mlat, mlon, mzip = _sample_merchant_geo(rng, user, prof, cat)

        records.append(
            {
                "trans_date_trans_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "cc_num": int(user["cc_num"]),
                "merchant": merchant,
                "category": cat,
                "amt": float(amt),
                "first": user["first"],
                "last": user["last"],
                "gender": user["gender"],
                "street": user["street"],
                "city": user["city"],
                "state": user["state"],
                "zip": int(user["zip"]),
                "lat": float(user["lat"]),
                "long": float(user["long"]),
                "city_pop": int(user["city_pop"]),
                "job": user["job"],
                "dob": user["dob"],
                "trans_num": _build_trans_num(run_id, len(records) + 1),
                "unix_time": int(ts.timestamp()),
                "merch_lat": round(mlat, 6),
                "merch_long": round(mlon, 6),
                "merch_zipcode": float(mzip),
            }
        )

    return records, prof


def _build_user_hour_profile(records: list[dict[str, Any]]) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for r in records:
        cc = int(r["cc_num"])
        hour = pd.Timestamp(r["trans_date_trans_time"]).hour
        if cc not in out:
            out[cc] = np.zeros(24, dtype=int)
        out[cc][hour] += 1
    return out


def _pick_far_city(city_ref: pd.DataFrame, lat: float, lon: float, min_km: float, rng: np.random.Generator) -> pd.Series:
    if city_ref.empty:
        return pd.Series({"zip": 0.0, "lat": lat + 8.0, "long": lon + 8.0})

    d = haversine_km(
        np.full(len(city_ref), lat, dtype=float),
        np.full(len(city_ref), lon, dtype=float),
        city_ref["lat"].to_numpy(dtype=float),
        city_ref["long"].to_numpy(dtype=float),
    )
    idx = np.where(d > min_km)[0]
    if len(idx) == 0:
        return city_ref.loc[int(rng.choice(city_ref.index.to_numpy(), p=city_ref["prob"].to_numpy()))]

    subset = city_ref.iloc[idx]
    probs = subset["prob"].to_numpy(dtype=float)
    probs = probs / probs.sum()
    chosen = int(rng.choice(subset.index.to_numpy(), p=probs))
    return city_ref.loc[chosen]


def _inject_high_amount_anomaly(rec: dict[str, Any], prof: dict[str, Any], rng: np.random.Generator) -> None:
    category = str(rec["category"])
    p = prof["amount_by_cat"].get(category, prof["amount_global"])
    threshold = float(p["q99"])
    extreme = threshold * float(rng.uniform(1.15, 2.1))
    rec["amt"] = round(float(max(rec["amt"], extreme)), 2)


def _inject_geolocation_anomaly(rec: dict[str, Any], prof: dict[str, Any], rng: np.random.Generator) -> None:
    far = _pick_far_city(prof["city_ref"], float(rec["lat"]), float(rec["long"]), 500.0, rng)
    base_lat = float(far["lat"])
    base_lon = float(far["long"])
    rec["merch_lat"] = round(float(base_lat + rng.normal(0.0, 0.25)), 6)
    rec["merch_long"] = round(float(base_lon + rng.normal(0.0, 0.25)), 6)
    rec["merch_zipcode"] = float(far.get("zip", rec.get("zip", 0)))


def _inject_time_anomaly(rec: dict[str, Any], user_hours: dict[int, np.ndarray], rng: np.random.Generator) -> None:
    cc = int(rec["cc_num"])
    user_hist = user_hours.get(cc, np.ones(24, dtype=int))
    least_hours = np.argsort(user_hist)[:6]
    odd_candidates = [h for h in least_hours if h in {0, 1, 2, 3, 4}]
    if not odd_candidates:
        odd_candidates = [0, 1, 2, 3, 4]

    dt = pd.Timestamp(rec["trans_date_trans_time"])
    new_hour = int(rng.choice(odd_candidates))
    dt = pd.Timestamp(dt.year, dt.month, dt.day, new_hour, int(rng.integers(0, 60)), int(rng.integers(0, 60)))
    rec["trans_date_trans_time"] = dt.strftime("%Y-%m-%d %H:%M:%S")
    rec["unix_time"] = int(dt.timestamp())


def _inject_merchant_category_inconsistency(rec: dict[str, Any], prof: dict[str, Any], rng: np.random.Generator) -> None:
    current_cat = str(rec["category"])
    options = [c for c in prof["merchant_by_cat"].keys() if c != current_cat and prof["merchant_by_cat"].get(c)]
    if not options:
        rec["merchant"] = f"inconsistent_{uuid4().hex[:10]}"
        return

    wrong_cat = str(rng.choice(options))
    rec["merchant"] = str(rng.choice(prof["merchant_by_cat"][wrong_cat]))


def inject_fraud_labels(
    records: list[dict[str, Any]],
    profiles: dict[str, Any],
    fraud_ratio: float = 0.02,
    seed: int = 42,
) -> list[dict[str, Any]]:
    out = [dict(r) for r in records]

    for r in out:
        r["is_fraud"] = 0
        r["fraud_type"] = None

    n = len(out)
    if n == 0:
        return out

    ratio = float(np.clip(fraud_ratio, 0.0, 1.0))
    target = int(round(n * ratio))
    if ratio > 0.0 and target == 0:
        target = 1

    if target == 0:
        return out

    rng = np.random.default_rng(seed + 101)
    idx = rng.choice(np.arange(n), size=min(target, n), replace=False)
    user_hours = _build_user_hour_profile(out)

    pattern_cycle = np.resize(np.array(FRAUD_PATTERNS, dtype=object), len(idx))
    rng.shuffle(pattern_cycle)

    for i, pattern in zip(idx, pattern_cycle):
        rec = out[int(i)]

        if pattern == "high_amount_anomaly":
            _inject_high_amount_anomaly(rec, profiles, rng)
        elif pattern == "geolocation_anomaly":
            _inject_geolocation_anomaly(rec, profiles, rng)
        elif pattern == "time_anomaly":
            _inject_time_anomaly(rec, user_hours, rng)
        else:
            _inject_merchant_category_inconsistency(rec, profiles, rng)

        rec["is_fraud"] = 1
        rec["fraud_type"] = str(pattern)

    return out


def _to_api_record(record: dict[str, Any]) -> dict[str, Any]:
    return {k: record[k] for k in API_FIELDS}


def to_api_payload(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_to_api_record(r) for r in records]


def generate_payload(
    base_csv: Path,
    n_transactions: int,
    output_txt: Path,
    knobs: Knobs,
    seed: int,
    start_date: str | None,
    end_date: str | None,
) -> Path:
    records, _ = generate_records(
        base_csv=base_csv,
        n_transactions=n_transactions,
        knobs=knobs,
        seed=seed,
        start_date=start_date,
        end_date=end_date,
    )

    payload = to_api_payload(records)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    return output_txt


def _select_model_input(records: list[dict[str, Any]], model: Any) -> pd.DataFrame:
    df = pd.DataFrame(records)

    expected: list[str] | None = None
    if hasattr(model, "expected_columns"):
        expected = [str(c) for c in getattr(model, "expected_columns")]
    elif hasattr(model, "feature_names_in_"):
        expected = [str(c) for c in getattr(model, "feature_names_in_")]

    if expected is None:
        cols = [c for c in API_FIELDS if c in df.columns]
        return df[cols].copy()

    if all(c in df.columns for c in expected):
        return df[expected].copy()

    # Si el modelo espera features engineered, las construimos explicitamente.
    from feature_engineering_script import build_feature_engineered_dataset

    df_fe = build_feature_engineered_dataset(df)
    missing_after_fe = [c for c in expected if c not in df_fe.columns]
    if missing_after_fe:
        raise ValueError(
            "No fue posible construir el schema esperado por el modelo tras feature engineering. "
            f"Columnas faltantes: {missing_after_fe}"
        )

    return df_fe[expected].copy()


def run_model_inference(records: list[dict[str, Any]], model: Any) -> list[dict[str, Any]]:
    if len(records) == 0:
        return []

    x = _select_model_input(records, model)
    proba = model.predict_proba(x)
    arr = np.asarray(proba)

    if arr.ndim == 2 and arr.shape[1] >= 2:
        scores = arr[:, 1].astype(float)
    else:
        scores = arr.reshape(-1).astype(float)

    y_pred = (scores >= 0.5).astype(int)

    out: list[dict[str, Any]] = []
    for i, rec in enumerate(records):
        row = dict(rec)
        row["y_pred"] = int(y_pred[i])
        row["y_score"] = float(scores[i])
        out.append(row)
    return out


def run_model_inference_with_latency(records: list[dict[str, Any]], model: Any) -> tuple[list[dict[str, Any]], dict[str, float]]:
    start = time.perf_counter()
    out = run_model_inference(records, model)
    total = time.perf_counter() - start

    n = max(len(records), 1)
    latency = {
        "total_time_sec": float(total),
        "avg_latency_ms": float((total / n) * 1000.0),
        "throughput_txn_per_sec": float(len(records) / max(total, 1e-9)),
    }
    return out, latency


def compute_evaluation_metrics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    y_true = np.asarray([int(r["is_fraud"]) for r in records], dtype=int)
    y_pred = np.asarray([int(r.get("y_pred", 0)) for r in records], dtype=int)
    y_score = np.asarray([float(r.get("y_score", 0.0)) for r in records], dtype=float)

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
    }

    unique_classes = np.unique(y_true)
    if unique_classes.size > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_score)
        metrics["pr_auc"] = float(auc(pr_recall, pr_precision))

    return metrics


def save_summary_log(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    return path


def save_detailed_log(path: Path, records: list[dict[str, Any]], output_format: str = "csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return path

    pd.DataFrame(records).to_csv(path, index=False)
    return path


def save_fraud_only_txt(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fraud_payload = [
        _to_api_record(r)
        for r in records
        if int(r.get("is_fraud", 0)) == 1
    ]
    path.write_text(json.dumps(fraud_payload, ensure_ascii=True), encoding="utf-8")
    return path


def evaluate_model_pipeline(
    base_csv: Path,
    n_transactions: int,
    knobs: Knobs,
    fraud_ratio: float,
    seed: int = 42,
    start_date: str | None = None,
    end_date: str | None = None,
    model: Any = None,
    model_path: Path | None = None,
    output_dir: Path = Path("simulations"),
    request_time: str | None = None,
    details_format: str = "csv",
    payload_output: Path | None = None,
    fraud_output: Path | None = None,
    summary_output: Path | None = None,
    details_output: Path | None = None,
) -> dict[str, Any]:
    req_time = request_time or datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_records, profiles = generate_records(
        base_csv=base_csv,
        n_transactions=n_transactions,
        knobs=knobs,
        seed=seed,
        start_date=start_date,
        end_date=end_date,
    )

    labeled_records = inject_fraud_labels(
        records=raw_records,
        profiles=profiles,
        fraud_ratio=fraud_ratio,
        seed=seed,
    )

    if model is None and model_path is not None:
        model = joblib_load(model_path)

    if model is not None:
        scored_records, latency = run_model_inference_with_latency(labeled_records, model)
        metrics = compute_evaluation_metrics(scored_records)
    else:
        scored_records = [dict(r, y_pred=None, y_score=None) for r in labeled_records]
        latency = {
            "total_time_sec": 0.0,
            "avg_latency_ms": 0.0,
            "throughput_txn_per_sec": 0.0,
        }
        metrics = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "roc_auc": None,
            "pr_auc": None,
        }

    payload_path = payload_output or (output_dir / f"simulations_{req_time}.txt")
    fraud_path = fraud_output or (output_dir / f"fraud_test_{req_time}.txt")
    summary_path = summary_output or (output_dir / f"summary_{req_time}.json")
    details_suffix = "jsonl" if details_format == "jsonl" else "csv"
    details_path = details_output or (output_dir / f"details_{req_time}.{details_suffix}")

    payload = to_api_payload(scored_records)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    save_fraud_only_txt(fraud_path, scored_records)
    save_detailed_log(details_path, scored_records, output_format=details_format)

    observed_fraud_ratio = float(np.mean([int(r["is_fraud"]) for r in scored_records])) if scored_records else 0.0

    summary: dict[str, Any] = {
        "request_time": req_time,
        "n_transactions": int(len(scored_records)),
        "fraud_ratio_target": float(np.clip(fraud_ratio, 0.0, 1.0)),
        "fraud_ratio_observed": observed_fraud_ratio,
        "fraud_count": int(sum(int(r["is_fraud"]) for r in scored_records)),
        "metrics": metrics,
        "latency": latency,
        "outputs": {
            "payload_txt": str(payload_path),
            "fraud_test_txt": str(fraud_path),
            "summary_json": str(summary_path),
            "details": str(details_path),
        },
    }

    save_summary_log(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Synthetic transaction simulator + controlled fraud injection + evaluation pipeline")
    p.add_argument("--mode", choices=["offline", "realtime"], default="offline")
    p.add_argument("--base-csv", type=Path, default=Path("credit_card_transactions.csv"))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-date", type=str, default=None)
    p.add_argument("--end-date", type=str, default=None)
    p.add_argument("--amount-volatility", type=float, default=1.0)
    p.add_argument("--weekend-boost", type=float, default=1.25)
    p.add_argument("--recurrent-ratio", type=float, default=0.85)
    p.add_argument("--travel-boost", type=float, default=1.0)
    p.add_argument("--fraud-ratio", type=float, default=0.02)
    p.add_argument("--model-path", type=Path, default=None, help="Modelo con metodo predict_proba (opcional)")
    p.add_argument("--request-time", type=str, default=None, help="Timestamp para nombres de output")
    p.add_argument("--output", type=Path, default=None, help="Alias de salida principal (simulations_{requesttime}.txt)")
    p.add_argument("--fraud-output", type=Path, default=None, help="Salida de fraudes: fraud_test_{requesttime}.txt")
    p.add_argument("--summary-output", type=Path, default=None)
    p.add_argument("--details-output", type=Path, default=None)
    p.add_argument("--details-format", choices=["csv", "jsonl"], default="csv")
    p.add_argument("--api-url", type=str, default="http://127.0.0.1:8000")
    p.add_argument("--source", choices=["db", "csv"], default="db")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--pause-seconds", type=float, default=0.0)
    p.add_argument("--timeout-seconds", type=float, default=30.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--database-url", type=str, default=None)
    p.add_argument("--table-name", type=str, default="transactions")
    p.add_argument("--order-by", type=str, default="trans_num")
    p.add_argument("--csv-path", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "realtime":
        config = SimulationConfig(
            source=str(args.source),
            batch_size=int(args.batch_size),
            iterations=int(args.iterations),
            continuous=bool(args.continuous),
            pause_seconds=float(args.pause_seconds),
            endpoint_url=str(args.api_url),
            timeout_seconds=float(args.timeout_seconds),
            retries=int(args.retries),
            csv_path=args.csv_path or args.base_csv,
            database_url=args.database_url,
            table_name=str(args.table_name),
            order_by=str(args.order_by),
        )

        summary = run_realtime_simulation(config)
        print(json.dumps(summary, indent=2, ensure_ascii=True))
        return

    req_time = args.request_time or datetime.now().strftime("%Y%m%d_%H%M%S")

    knobs = Knobs(
        amount_volatility=float(args.amount_volatility),
        weekend_boost=float(args.weekend_boost),
        recurrent_ratio=float(np.clip(args.recurrent_ratio, 0.0, 1.0)),
        travel_boost=float(max(args.travel_boost, 0.1)),
    )

    payload_output = args.output or (Path("simulations") / f"simulations_{req_time}.txt")
    fraud_output = args.fraud_output or (Path("simulations") / f"fraud_test_{req_time}.txt")

    summary = evaluate_model_pipeline(
        base_csv=args.base_csv,
        n_transactions=int(args.n),
        knobs=knobs,
        fraud_ratio=float(args.fraud_ratio),
        seed=int(args.seed),
        start_date=args.start_date,
        end_date=args.end_date,
        model_path=args.model_path,
        output_dir=Path("simulations"),
        request_time=req_time,
        details_format=args.details_format,
        payload_output=payload_output,
        fraud_output=fraud_output,
        summary_output=args.summary_output,
        details_output=args.details_output,
    )

    print(f"Simulaciones generadas: {summary['outputs']['payload_txt']}")
    print(f"Fraudes (solo positivos): {summary['outputs']['fraud_test_txt']}")
    print(f"Resumen: {summary['outputs']['summary_json']}")
    print(f"Detalle: {summary['outputs']['details']}")


if __name__ == "__main__":
    main()
