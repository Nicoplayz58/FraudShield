from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True, slots=True)
class PredictionHttpResult:
    payload: dict[str, Any]
    status_code: int
    latency_seconds: float


def _predict_url(endpoint_url: str) -> str:
    base = endpoint_url.rstrip("/")
    return base if base.endswith("/predict") else f"{base}/predict"


def post_prediction_batch(
    endpoint_url: str,
    payload: list[dict[str, Any]],
    timeout_seconds: float = 30.0,
    retries: int = 3,
    backoff_seconds: float = 0.75,
) -> PredictionHttpResult:
    url = _predict_url(endpoint_url)
    body = json.dumps(payload, ensure_ascii=True, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_error: Exception | None = None

    for attempt in range(max(retries, 0) + 1):
        start = time.perf_counter()
        try:
            req = request.Request(url=url, data=body, headers=headers, method="POST")
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8")
                payload_out = json.loads(raw) if raw else {}
                return PredictionHttpResult(
                    payload=payload_out,
                    status_code=int(resp.status),
                    latency_seconds=time.perf_counter() - start,
                )
        except error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
            if exc.code < 500 and exc.code != 429:
                raise RuntimeError(f"Predict endpoint returned HTTP {exc.code}: {body_text}") from exc
            last_error = RuntimeError(f"Predict endpoint returned HTTP {exc.code}: {body_text}")
        except error.URLError as exc:
            last_error = exc

        if attempt >= retries:
            break

        time.sleep(backoff_seconds * (2**attempt))

    raise RuntimeError(f"No fue posible llamar al endpoint de prediccion: {last_error}") from last_error
