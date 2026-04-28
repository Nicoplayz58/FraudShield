from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from db.connection import create_database_engine, get_database_url


def _get_engine(engine: Engine | None = None, database_url: str | None = None) -> Engine:
    if engine is not None:
        return engine
    return create_database_engine(get_database_url(database_url))


def _latest_predictions_scope_sql(body_sql: str) -> str:
    return f"""
        WITH latest_predictions AS (
            SELECT
                p.trans_num,
                p.batch_id,
                p.risk_score,
                p.prediction,
                p.created_at,
                p.id,
                ROW_NUMBER() OVER (
                    PARTITION BY p.trans_num
                    ORDER BY p.created_at DESC, p.id DESC
                ) AS rn
            FROM predictions p
        )
        {body_sql}
    """


def _fetch_single_column_list(sql: str, *, engine: Engine | None = None, database_url: str | None = None) -> list[str]:
    engine = _get_engine(engine, database_url)
    with engine.connect() as connection:
        frame = pd.read_sql_query(text(sql), connection)
    return [str(value) for value in frame.iloc[:, 0].dropna().astype(str).tolist() if str(value).strip()]


def get_unique_merchants(*, engine: Engine | None = None, database_url: str | None = None) -> list[str]:
    return _fetch_single_column_list(
        """
        SELECT DISTINCT t.merchant
        FROM transactions t
        WHERE t.merchant IS NOT NULL
          AND BTRIM(t.merchant) <> ''
        ORDER BY t.merchant ASC
        """,
        engine=engine,
        database_url=database_url,
    )


def get_unique_categories(*, engine: Engine | None = None, database_url: str | None = None) -> list[str]:
    return _fetch_single_column_list(
        """
        SELECT DISTINCT t.category
        FROM transactions t
        WHERE t.category IS NOT NULL
          AND BTRIM(t.category) <> ''
        ORDER BY t.category ASC
        """,
        engine=engine,
        database_url=database_url,
    )


def _prediction_filters(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
    created_at_alias: str = "p.created_at",
    prediction_alias: str = "p",
) -> tuple[str, dict[str, Any]]:
    clauses = ["1 = 1"]
    params: dict[str, Any] = {}

    if start_date is not None:
        clauses.append(f"{created_at_alias} >= :start_date")
        params["start_date"] = start_date

    if end_date is not None:
        clauses.append(f"{created_at_alias} < (CAST(:end_date AS DATE) + INTERVAL '1 day')")
        params["end_date"] = end_date

    if min_risk_score is not None:
        clauses.append(f"{prediction_alias}.risk_score >= :min_risk_score")
        params["min_risk_score"] = float(min_risk_score)

    if merchant:
        clauses.append("t.merchant = :merchant")
        params["merchant"] = merchant

    if category:
        clauses.append("t.category = :category")
        params["category"] = category

    if only_fraud:
        clauses.append(f"{prediction_alias}.prediction = TRUE")

    return " AND ".join(clauses), params


def get_date_bounds(*, engine: Engine | None = None, database_url: str | None = None) -> tuple[date | None, date | None]:
    engine = _get_engine(engine, database_url)
    sql = text(
        """
        SELECT MIN(p.created_at)::date AS min_date, MAX(p.created_at)::date AS max_date
        FROM predictions p
        """
    )
    with engine.connect() as connection:
        row = connection.execute(sql).mappings().one()
    return row["min_date"], row["max_date"]


def load_recent_transactions_with_predictions(
    *,
    limit: int = 100,
    offset: int = 0,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
        created_at_alias="COALESCE(lp.created_at, t.trans_date_trans_time)",
        prediction_alias="lp",
    )
    params["limit"] = int(limit)
    params["offset"] = max(int(offset), 0)

    sql = text(
        f"""
        WITH latest_predictions AS (
            SELECT
                p.trans_num,
                p.batch_id,
                p.risk_score,
                p.prediction,
                p.created_at,
                p.id,
                ROW_NUMBER() OVER (
                    PARTITION BY p.trans_num
                    ORDER BY p.created_at DESC, p.id DESC
                ) AS rn
            FROM predictions p
        )
        SELECT
            t.trans_num,
            t.trans_date_trans_time,
            t.amt,
            t.merchant,
            t.category,
            lp.batch_id,
            lp.risk_score,
            COALESCE(
                CASE
                    WHEN lp.prediction IS TRUE THEN 'fraud'
                    WHEN lp.prediction IS FALSE THEN 'legit'
                    ELSE NULL
                END,
                'pending'
            ) AS prediction,
            lp.prediction AS prediction_raw,
            ROW_NUMBER() OVER (
                ORDER BY COALESCE(lp.risk_score, -1) DESC, COALESCE(lp.created_at, t.trans_date_trans_time) DESC, t.trans_num ASC
            ) AS rank,
            COALESCE(lp.created_at, t.trans_date_trans_time) AS created_at
        FROM transactions t
        LEFT JOIN latest_predictions lp
            ON lp.trans_num = t.trans_num
           AND lp.rn = 1
        WHERE {where_clause}
        ORDER BY COALESCE(lp.risk_score, -1) DESC, COALESCE(lp.created_at, t.trans_date_trans_time) DESC, t.trans_num ASC
        LIMIT :limit OFFSET :offset
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame


def load_latest_predictions(
    *,
    limit: int = 100,
    offset: int = 0,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    return load_recent_transactions_with_predictions(
        limit=limit,
        offset=offset,
        engine=engine,
        database_url=database_url,
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
    )


def get_kpis(
    *,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> dict[str, float]:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
        created_at_alias="COALESCE(lp.created_at, t.trans_date_trans_time)",
        prediction_alias="lp",
    )

    sql = text(
        f"""
        WITH latest_predictions AS (
            SELECT
                p.trans_num,
                p.risk_score,
                p.prediction,
                p.created_at,
                p.id,
                ROW_NUMBER() OVER (
                    PARTITION BY p.trans_num
                    ORDER BY p.created_at DESC, p.id DESC
                ) AS rn
            FROM predictions p
        )
        SELECT
            COUNT(*) AS total_transactions,
            COALESCE(SUM(CASE WHEN lp.prediction IS TRUE THEN 1 ELSE 0 END), 0) AS fraud_count,
            COALESCE(AVG(lp.risk_score), 0) AS avg_risk_score,
            COALESCE(MAX(lp.risk_score), 0) AS max_risk_score
        FROM transactions t
        LEFT JOIN latest_predictions lp
            ON lp.trans_num = t.trans_num
           AND lp.rn = 1
        WHERE {where_clause}
        """
    )
    with engine.connect() as connection:
        row = connection.execute(sql, params).mappings().one()

    return {
        "total_transactions": float(row["total_transactions"] or 0),
        "fraud_count": float(row["fraud_count"] or 0),
        "avg_risk_score": float(row["avg_risk_score"] or 0),
        "max_risk_score": float(row["max_risk_score"] or 0),
    }


def get_fraud_over_time(
    *,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
    )

    sql = text(
        f"""
        SELECT
            DATE(p.created_at) AS date,
            SUM(CASE WHEN p.prediction THEN 1 ELSE 0 END) AS fraud_count,
            AVG(p.risk_score) AS avg_risk_score
        FROM predictions p
        INNER JOIN transactions t
            ON t.trans_num = p.trans_num
        WHERE {where_clause}
        GROUP BY DATE(p.created_at)
        ORDER BY date
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame


def get_merchant_ranking(
    *,
    top_n: int = 10,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
    )
    params["top_n"] = int(top_n)

    sql = text(
        f"""
        SELECT
            t.merchant,
            SUM(CASE WHEN p.prediction THEN 1 ELSE 0 END) AS fraud_count,
            AVG(p.risk_score) AS avg_risk_score
        FROM predictions p
        INNER JOIN transactions t
            ON t.trans_num = p.trans_num
        WHERE {where_clause}
        GROUP BY t.merchant
        HAVING SUM(CASE WHEN p.prediction THEN 1 ELSE 0 END) > 0
        ORDER BY fraud_count DESC, avg_risk_score DESC, t.merchant ASC
        LIMIT :top_n
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame


def get_risk_distribution(
    *,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
    )

    sql = text(
        f"""
        WITH bucketed AS (
            SELECT
                LEAST(9, GREATEST(0, FLOOR(COALESCE(p.risk_score, 0) * 10)::int)) AS bucket_index
            FROM predictions p
            INNER JOIN transactions t
                ON t.trans_num = p.trans_num
            WHERE {where_clause}
        )
        SELECT
            CONCAT(
                TO_CHAR(bucket_index / 10.0, 'FM0.0'),
                ' - ',
                TO_CHAR((bucket_index + 1) / 10.0, 'FM0.0')
            ) AS risk_score_bucket,
            COUNT(*) AS count,
            bucket_index
        FROM bucketed
        GROUP BY bucket_index
        ORDER BY bucket_index
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame


def get_transaction_detail(
    trans_num: str,
    *,
    engine: Engine | None = None,
    database_url: str | None = None,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    sql = text(
        """
        SELECT
            t.trans_num,
            t.trans_date_trans_time,
            t.unix_time,
            t.cc_num,
            t.merchant,
            t.category,
            t.amt,
            t.first,
            t.last,
            t.gender,
            t.job,
            t.dob,
            t.street,
            t.city,
            t.state,
            t.zip,
            t.lat,
            t.longitude,
            t.city_pop,
            t.merch_lat,
            t.merch_long,
            t.merch_zipcode,
            p.batch_id,
            p.risk_score,
            COALESCE(
                CASE
                    WHEN p.prediction IS TRUE THEN 'fraud'
                    WHEN p.prediction IS FALSE THEN 'legit'
                    ELSE NULL
                END,
                'pending'
            ) AS prediction,
            p.prediction AS prediction_raw,
            p.created_at AS prediction_created_at
        FROM transactions t
        LEFT JOIN LATERAL (
            SELECT
                p.batch_id,
                p.risk_score,
                p.prediction,
                p.created_at
            FROM predictions p
            WHERE p.trans_num = t.trans_num
            ORDER BY p.created_at DESC, p.id DESC
            LIMIT 1
        ) p ON TRUE
        WHERE t.trans_num = :trans_num
        LIMIT 1
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params={"trans_num": trans_num})
    return frame


def get_category_summary(
    *,
    top_n: int = 15,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
        created_at_alias="COALESCE(lp.created_at, t.trans_date_trans_time)",
        prediction_alias="lp",
    )
    params["top_n"] = int(top_n)

    sql = text(
        _latest_predictions_scope_sql(
            f"""
            SELECT
                t.category,
                COUNT(*) AS transaction_count,
                COALESCE(SUM(CASE WHEN lp.prediction IS TRUE THEN 1 ELSE 0 END), 0) AS fraud_count,
                COALESCE(
                    SUM(CASE WHEN lp.prediction IS TRUE THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0),
                    0
                ) AS fraud_rate,
                COALESCE(AVG(lp.risk_score), 0) AS avg_risk_score,
                COALESCE(MAX(lp.risk_score), 0) AS max_risk_score,
                COALESCE(SUM(t.amt), 0) AS total_amount
            FROM transactions t
            LEFT JOIN latest_predictions lp
                ON lp.trans_num = t.trans_num
               AND lp.rn = 1
            WHERE {where_clause}
            GROUP BY t.category
            ORDER BY fraud_count DESC, fraud_rate DESC, avg_risk_score DESC, t.category ASC
            LIMIT :top_n
            """
        )
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame


def get_merchant_summary(
    *,
    top_n: int = 15,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
        created_at_alias="COALESCE(lp.created_at, t.trans_date_trans_time)",
        prediction_alias="lp",
    )
    params["top_n"] = int(top_n)

    sql = text(
        _latest_predictions_scope_sql(
            f"""
            SELECT
                t.merchant,
                COUNT(*) AS transaction_count,
                COALESCE(SUM(CASE WHEN lp.prediction IS TRUE THEN 1 ELSE 0 END), 0) AS fraud_count,
                COALESCE(
                    SUM(CASE WHEN lp.prediction IS TRUE THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0),
                    0
                ) AS fraud_rate,
                COALESCE(AVG(lp.risk_score), 0) AS avg_risk_score,
                COALESCE(MAX(lp.risk_score), 0) AS max_risk_score,
                COALESCE(SUM(t.amt), 0) AS total_amount
            FROM transactions t
            LEFT JOIN latest_predictions lp
                ON lp.trans_num = t.trans_num
               AND lp.rn = 1
            WHERE {where_clause}
            GROUP BY t.merchant
            ORDER BY fraud_count DESC, fraud_rate DESC, avg_risk_score DESC, t.merchant ASC
            LIMIT :top_n
            """
        )
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame


def get_hourly_summary(
    *,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
        created_at_alias="COALESCE(lp.created_at, t.trans_date_trans_time)",
        prediction_alias="lp",
    )

    sql = text(
        _latest_predictions_scope_sql(
            f"""
            SELECT
                EXTRACT(HOUR FROM COALESCE(lp.created_at, t.trans_date_trans_time))::int AS hour_of_day,
                COUNT(*) AS transaction_count,
                COALESCE(SUM(CASE WHEN lp.prediction IS TRUE THEN 1 ELSE 0 END), 0) AS fraud_count,
                COALESCE(
                    SUM(CASE WHEN lp.prediction IS TRUE THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0),
                    0
                ) AS fraud_rate,
                COALESCE(AVG(lp.risk_score), 0) AS avg_risk_score,
                COALESCE(MAX(lp.risk_score), 0) AS max_risk_score
            FROM transactions t
            LEFT JOIN latest_predictions lp
                ON lp.trans_num = t.trans_num
               AND lp.rn = 1
            WHERE {where_clause}
            GROUP BY EXTRACT(HOUR FROM COALESCE(lp.created_at, t.trans_date_trans_time))
            ORDER BY hour_of_day ASC
            """
        )
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame


def get_amount_bucket_summary(
    *,
    bucket_size: float = 100.0,
    engine: Engine | None = None,
    database_url: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    min_risk_score: float | None = None,
    merchant: str | None = None,
    category: str | None = None,
    only_fraud: bool = False,
) -> pd.DataFrame:
    engine = _get_engine(engine, database_url)
    where_clause, params = _prediction_filters(
        start_date=start_date,
        end_date=end_date,
        min_risk_score=min_risk_score,
        merchant=merchant,
        category=category,
        only_fraud=only_fraud,
        created_at_alias="COALESCE(lp.created_at, t.trans_date_trans_time)",
        prediction_alias="lp",
    )
    params["bucket_size"] = float(bucket_size)

    sql = text(
        f"""
        WITH latest_predictions AS (
            SELECT
                p.trans_num,
                p.batch_id,
                p.risk_score,
                p.prediction,
                p.created_at,
                p.id,
                ROW_NUMBER() OVER (
                    PARTITION BY p.trans_num
                    ORDER BY p.created_at DESC, p.id DESC
                ) AS rn
            FROM predictions p
        ),
        bucketed AS (
            SELECT
                FLOOR(COALESCE(t.amt, 0) / :bucket_size) * :bucket_size AS bucket_start,
                lp.risk_score,
                lp.prediction,
                t.trans_num
            FROM transactions t
            LEFT JOIN latest_predictions lp
                ON lp.trans_num = t.trans_num
               AND lp.rn = 1
            WHERE {where_clause}
        )
        SELECT
            bucket_start,
            COUNT(*) AS transaction_count,
            COALESCE(SUM(CASE WHEN prediction IS TRUE THEN 1 ELSE 0 END), 0) AS fraud_count,
            COALESCE(
                SUM(CASE WHEN prediction IS TRUE THEN 1 ELSE 0 END)::numeric / NULLIF(COUNT(*), 0),
                0
            ) AS fraud_rate,
            COALESCE(AVG(risk_score), 0) AS avg_risk_score,
            COALESCE(MAX(risk_score), 0) AS max_risk_score
        FROM bucketed
        GROUP BY bucket_start
        ORDER BY bucket_start ASC
        LIMIT 24
        """
    )
    with engine.connect() as connection:
        frame = pd.read_sql_query(sql, connection, params=params)
    return frame
