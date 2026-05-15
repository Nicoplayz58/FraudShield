from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window


EPS = 1e-9


def _ordered_windows(order_cols: list[F.Column]) -> tuple[Window, Window]:
    base = Window.orderBy(*order_cols)
    return (
        base.rowsBetween(Window.unboundedPreceding, -1),
        base.rowsBetween(Window.unboundedPreceding, 0),
    )


def build_feature_engineered_dataset_spark(df_raw: DataFrame) -> DataFrame:
    df = (
        df_raw.withColumn("__row_id", F.monotonically_increasing_id())
        .withColumn("trans_date_trans_time", F.to_timestamp(F.col("trans_date_trans_time")))
        .withColumn("amt", F.col("amt").cast("double"))
        .dropna(subset=["cc_num", "trans_date_trans_time", "amt"])
    )

    if "country" in df.columns:
        df = df.withColumn("location_country", F.col("country").cast("string"))
    elif "state" in df.columns:
        df = df.withColumn("location_country", F.col("state").cast("string"))
    else:
        df = df.withColumn("location_country", F.lit("UNK"))

    df = (
        df.withColumn("location_city", F.when(F.col("city").isNotNull(), F.col("city").cast("string")).otherwise(F.lit("UNK")))
        .withColumn("location_key", F.concat_ws("|", F.col("location_country"), F.col("location_city")))
        .withColumn("_ts_sec", F.col("trans_date_trans_time").cast("long"))
    )

    order_cols = [F.col("trans_date_trans_time")]
    if "trans_num" in df.columns:
        order_cols.append(F.col("trans_num"))

    user_order = Window.partitionBy("cc_num").orderBy(*order_cols)
    user_prev = user_order.rowsBetween(Window.unboundedPreceding, -1)
    user_incl = user_order.rowsBetween(Window.unboundedPreceding, 0)

    df = df.withColumn("A1_amt_diff_prev", F.abs(F.col("amt") - F.lag("amt", 1).over(user_order)).cast("double"))

    user_prev_count = F.count(F.lit(1)).over(user_prev)
    user_prev_sum = F.sum("amt").over(user_prev)
    user_prev_mean = F.when(user_prev_count > 0, user_prev_sum / user_prev_count).otherwise(F.lit(None).cast("double"))
    user_prev_median = F.percentile_approx("amt", F.lit(0.5), F.lit(100)).over(user_prev)
    global_amt_median = df.approxQuantile("amt", [0.5], 0.001)[0]
    user_prev_mean_filled = F.coalesce(user_prev_mean, F.lit(float(global_amt_median)))
    user_prev_median_filled = F.coalesce(user_prev_median, F.lit(float(global_amt_median)))

    df = (
        df.withColumn("A2_amt_user_cum_mean", user_prev_mean_filled)
        .withColumn("A3_amt_dev_user_cum_median", F.abs(F.col("amt") - user_prev_median_filled))
        .withColumn(
            "A4_identical_amt_cnt_24h",
            F.count(F.lit(1)).over(
                Window.partitionBy("cc_num", "amt")
                .orderBy(F.col("_ts_sec"))
                .rangeBetween(-24 * 3600, -1)
            ),
        )
    )

    user_prev_std = F.stddev_samp("amt").over(user_prev)
    user_prev_std_filled = F.coalesce(user_prev_std, F.lit(0.0))
    df = (
        df.withColumn("A5_amt_user_std_score", F.abs(F.col("amt") - user_prev_mean_filled) / (user_prev_std_filled + F.lit(EPS)))
        .withColumn("A6_amt_within_1std_user", F.when((F.col("amt") >= user_prev_mean_filled - user_prev_std_filled) & (F.col("amt") <= user_prev_mean_filled + user_prev_std_filled), F.lit(1)).otherwise(F.lit(0)))
    )

    if "merchant" not in df.columns:
        df = df.withColumn("merchant", F.lit("UNK"))

    merch_order = Window.partitionBy("merchant").orderBy(*order_cols)
    merch_prev = merch_order.rowsBetween(Window.unboundedPreceding, -1)
    merch_prev_count = F.count(F.lit(1)).over(merch_prev)
    merch_prev_sum = F.sum("amt").over(merch_prev)
    merch_prev_mean = F.when(merch_prev_count > 0, merch_prev_sum / merch_prev_count).otherwise(F.lit(None).cast("double"))
    merch_prev_mean_filled = F.coalesce(merch_prev_mean, F.lit(float(global_amt_median)))
    merch_prev_std = F.coalesce(F.stddev_samp("amt").over(merch_prev), F.lit(0.0))

    df = (
        df.withColumn("A7_amt_diff_merchant_mean", F.abs(F.col("amt") - merch_prev_mean_filled))
        .withColumn("A8_merchant_amt_std", merch_prev_std)
    )

    um_order = Window.partitionBy("cc_num", "merchant").orderBy(*order_cols)
    um_prev = um_order.rowsBetween(Window.unboundedPreceding, -1)
    um_prev_count = F.count(F.lit(1)).over(um_prev)
    um_prev_sum = F.sum("amt").over(um_prev)
    um_prev_mean = F.when(um_prev_count > 0, um_prev_sum / um_prev_count).otherwise(F.lit(None).cast("double"))
    um_prev_mean_filled = F.coalesce(um_prev_mean, F.lit(float(global_amt_median)))
    um_prev_std = F.coalesce(F.stddev_samp("amt").over(um_prev), F.lit(0.0))

    df = (
        df.withColumn("A9_user_merchant_cum_mean", um_prev_mean_filled)
        .withColumn("A10_user_merchant_cum_std", um_prev_std)
        .withColumn("A11_user_merchant_cum_sum", F.coalesce(um_prev_sum, F.lit(0.0)))
        .withColumn("A12_user_merchant_global_mean", F.col("A9_user_merchant_cum_mean"))
        .withColumn("A13_user_merchant_global_std", F.col("A10_user_merchant_cum_std"))
        .withColumn("A14_user_merchant_global_sum", F.col("A11_user_merchant_cum_sum"))
        .withColumn(
            "A15_amt_within_2std_user_merchant",
            F.when(
                (F.col("amt") >= F.col("A9_user_merchant_cum_mean") - F.lit(2.0) * F.col("A10_user_merchant_cum_std"))
                & (F.col("amt") <= F.col("A9_user_merchant_cum_mean") + F.lit(2.0) * F.col("A10_user_merchant_cum_std")),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn(
            "A16_amt_within_confidence_user_merchant",
            F.when(
                ((F.col("amt") >= F.col("A9_user_merchant_cum_mean") - F.col("A10_user_merchant_cum_std"))
                 & (F.col("amt") <= F.col("A9_user_merchant_cum_mean") + F.col("A10_user_merchant_cum_std")))
                & (um_prev_count >= 3),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
    )

    df = (
        df.withColumn("T1_time_diff_sec", (F.col("_ts_sec") - F.lag("_ts_sec", 1).over(user_order)).cast("double"))
        .withColumn("T1_time_diff_sec_safe", F.when(F.col("T1_time_diff_sec").isNull() | (F.col("T1_time_diff_sec") <= 0), F.lit(1.0)).otherwise(F.col("T1_time_diff_sec")))
        .withColumn("T2_amt_time_angle", F.atan2(F.col("amt"), F.col("T1_time_diff_sec_safe")))
        .withColumn("T3_hour_sin", F.sin(F.lit(2.0 * 3.141592653589793) * (F.hour("trans_date_trans_time") / F.lit(24.0))))
        .withColumn("T4_hour_cos", F.cos(F.lit(2.0 * 3.141592653589793) * (F.hour("trans_date_trans_time") / F.lit(24.0))))
        .withColumn("T5_is_madrugada", F.when(F.hour("trans_date_trans_time") < 6, F.lit(1)).otherwise(F.lit(0)))
    )

    user_prev_madrugada = F.sum("T5_is_madrugada").over(user_prev)
    df = df.withColumn(
        "T6_user_madrugada_ratio_cum",
        F.when(user_prev_count > 0, user_prev_madrugada / user_prev_count).otherwise(F.lit(0.0)),
    )

    user_prev_sin = F.sum("T3_hour_sin").over(user_prev)
    user_prev_cos = F.sum("T4_hour_cos").over(user_prev)
    df = df.withColumn(
        "T7_user_hour_vonmises_proxy",
        F.when(
            user_prev_count > 0,
            F.sqrt(F.pow(user_prev_sin, F.lit(2.0)) + F.pow(user_prev_cos, F.lit(2.0))) / user_prev_count,
        ).otherwise(F.lit(0.0)),
    )

    for window_seconds, column_name in [
        (1, "T8_txn_count_1s"),
        (10, "T8_txn_count_10s"),
        (60, "T8_txn_count_60s"),
        (3600, "T8_txn_count_1h"),
        (7200, "T8_txn_count_2h"),
    ]:
        df = df.withColumn(
            column_name,
            F.count(F.lit(1)).over(
                Window.partitionBy("cc_num")
                .orderBy(F.col("_ts_sec"))
                .rangeBetween(-window_seconds, -1)
            ),
        )

    user_t1_order = Window.partitionBy("cc_num").orderBy(*order_cols)
    user_t1_prev = user_t1_order.rowsBetween(Window.unboundedPreceding, -1)
    user_t1_prev_count = F.count(F.lit(1)).over(user_t1_prev)
    user_t1_mean_prev = F.coalesce(F.avg("T1_time_diff_sec").over(user_t1_prev), F.lit(0.0))
    user_t1_std_prev = F.coalesce(F.stddev_samp("T1_time_diff_sec").over(user_t1_prev), F.lit(0.0))
    df = df.withColumn(
        "T9_time_interval_coherent",
        F.when(
            (F.col("T1_time_diff_sec") >= user_t1_mean_prev - F.lit(2.0) * user_t1_std_prev)
            & (F.col("T1_time_diff_sec") <= user_t1_mean_prev + F.lit(2.0) * user_t1_std_prev)
            & (user_t1_prev_count > 0),
            F.lit(1),
        ).otherwise(F.lit(0)),
    )

    city_order = Window.partitionBy("cc_num", "location_city").orderBy(*order_cols)
    city_prev = city_order.rowsBetween(Window.unboundedPreceding, -1)
    country_order = Window.partitionBy("cc_num", "location_country").orderBy(*order_cols)
    country_prev = country_order.rowsBetween(Window.unboundedPreceding, -1)

    city_first_flag = F.when(F.row_number().over(city_order) == 1, F.lit(1)).otherwise(F.lit(0))
    country_first_flag = F.when(F.row_number().over(country_order) == 1, F.lit(1)).otherwise(F.lit(0))

    df = (
        df.withColumn("L1_user_city_visit_count", F.count(F.lit(1)).over(city_prev))
        .withColumn("L2_user_country_visit_count", F.count(F.lit(1)).over(country_prev))
        .withColumn("L3_user_unique_city_count", F.sum(city_first_flag).over(user_prev))
        .withColumn("L3_user_unique_country_count", F.sum(country_first_flag).over(user_prev))
        .withColumn("L4_user_city_txn_ratio", F.when(user_prev_count > 0, F.col("L1_user_city_visit_count") / user_prev_count).otherwise(F.lit(0.0)))
        .withColumn("L4_user_country_txn_ratio", F.when(user_prev_count > 0, F.col("L2_user_country_visit_count") / user_prev_count).otherwise(F.lit(0.0)))
    )

    prev_loc = F.lag("location_key", 1).over(user_order)
    user_location_order = Window.partitionBy("cc_num", "location_key").orderBy(*order_cols)
    user_location_prev = user_location_order.rowsBetween(Window.unboundedPreceding, -1)

    df = (
        df.withColumn("L5_same_location_as_prev", F.when(F.col("location_key") == prev_loc, F.lit(1)).otherwise(F.lit(0)))
        .withColumn("L6_location_seen_before", F.when(F.count(F.lit(1)).over(user_location_prev) > 0, F.lit(1)).otherwise(F.lit(0)))
    )

    return (
        df.orderBy("__row_id")
        .drop("__row_id", "_ts_sec", "T1_time_diff_sec_safe")
    )
