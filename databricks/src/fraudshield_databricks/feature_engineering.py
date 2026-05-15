from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-9


def _past_count_in_window(times_sec: np.ndarray, window_sec: int) -> np.ndarray:
    counts = np.zeros(len(times_sec), dtype=np.int32)
    for i, t in enumerate(times_sec):
        left = np.searchsorted(times_sec, t - window_sec, side="left")
        right = np.searchsorted(times_sec, t, side="left")
        counts[i] = max(0, right - left)
    return counts


def _user_window_counts(df: pd.DataFrame, window_sec: int) -> pd.Series:
    out = np.zeros(len(df), dtype=np.int32)
    for _, idx in df.groupby("cc_num", sort=False).groups.items():
        pos = np.asarray(idx)
        t = df.loc[pos, "trans_date_trans_time"].astype("int64").to_numpy() // 10**9
        out[pos] = _past_count_in_window(t, window_sec)
    return pd.Series(out, index=df.index)


def _identical_amt_24h_counts(df: pd.DataFrame) -> pd.Series:
    out = np.zeros(len(df), dtype=np.int32)
    for _, idx in df.groupby(["cc_num", "amt"], sort=False).groups.items():
        pos = np.asarray(idx)
        t = df.loc[pos, "trans_date_trans_time"].astype("int64").to_numpy() // 10**9
        out[pos] = _past_count_in_window(t, 24 * 3600)
    return pd.Series(out, index=df.index)


def build_feature_engineered_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
    df["amt"] = pd.to_numeric(df["amt"], errors="coerce")
    df = df.dropna(subset=["cc_num", "trans_date_trans_time", "amt"]).copy()

    if "country" in df.columns:
        df["location_country"] = df["country"].astype(str)
    elif "state" in df.columns:
        df["location_country"] = df["state"].astype(str)
    else:
        df["location_country"] = "UNK"

    df["location_city"] = df["city"].astype(str) if "city" in df.columns else "UNK"
    df["location_key"] = df["location_country"] + "|" + df["location_city"]

    df["__row_id"] = np.arange(len(df), dtype=np.int64)
    sort_cols = ["trans_date_trans_time", "cc_num"]
    if "trans_num" in df.columns:
        sort_cols.append("trans_num")
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    g_user = df.groupby("cc_num", sort=False)
    user_txn_idx = g_user.cumcount() + 1
    user_txn_prev = (user_txn_idx - 1).clip(lower=0)

    prev_amt = g_user["amt"].shift(1)
    df["A1_amt_diff_prev"] = (df["amt"] - prev_amt).abs().fillna(0.0)

    user_cum_sum = g_user["amt"].cumsum()
    user_cum_sum_prev = user_cum_sum - df["amt"]
    user_mean_prev = np.where(user_txn_prev > 0, user_cum_sum_prev / user_txn_prev, np.nan)

    user_median_cum = g_user["amt"].expanding().median().reset_index(level=0, drop=True)
    user_median_prev = user_median_cum.groupby(df["cc_num"], sort=False).shift(1)

    global_amt_median = df["amt"].median()
    user_mean_prev_s = pd.Series(user_mean_prev, index=df.index).fillna(global_amt_median)
    user_median_prev_s = user_median_prev.fillna(global_amt_median)

    df["A2_amt_user_cum_mean"] = user_mean_prev_s
    df["A3_amt_dev_user_cum_median"] = (df["amt"] - user_median_prev_s).abs()
    df["A4_identical_amt_cnt_24h"] = _identical_amt_24h_counts(df)

    user_mean_cum = g_user["amt"].expanding().mean().reset_index(level=0, drop=True)
    user_std_cum = g_user["amt"].expanding().std().reset_index(level=0, drop=True).fillna(0.0)

    user_mean_prev2 = user_mean_cum.groupby(df["cc_num"], sort=False).shift(1).fillna(global_amt_median)
    user_std_prev = user_std_cum.groupby(df["cc_num"], sort=False).shift(1).fillna(0.0)

    df["A5_amt_user_std_score"] = (df["amt"] - user_mean_prev2).abs() / (user_std_prev + EPS)
    amt_arr = df["amt"].to_numpy()
    user_lo = (user_mean_prev2 - user_std_prev).to_numpy()
    user_hi = (user_mean_prev2 + user_std_prev).to_numpy()
    df["A6_amt_within_1std_user"] = ((amt_arr >= user_lo) & (amt_arr <= user_hi)).astype(int)

    if "merchant" not in df.columns:
        df["merchant"] = "UNK"

    g_merch = df.groupby("merchant", sort=False)
    merch_txn_idx = g_merch.cumcount() + 1
    merch_prev_cnt = (merch_txn_idx - 1).clip(lower=0)
    merch_cum_sum = g_merch["amt"].cumsum()
    merch_cum_sum_prev = merch_cum_sum - df["amt"]
    merch_mean_prev = np.where(merch_prev_cnt > 0, merch_cum_sum_prev / merch_prev_cnt, np.nan)
    merch_mean_prev_s = pd.Series(merch_mean_prev, index=df.index).fillna(global_amt_median)

    merch_std_cum = g_merch["amt"].expanding().std().reset_index(level=0, drop=True).fillna(0.0)
    merch_std_prev = merch_std_cum.groupby(df["merchant"], sort=False).shift(1).fillna(0.0)

    df["A7_amt_diff_merchant_mean"] = (df["amt"] - merch_mean_prev_s).abs()
    df["A8_merchant_amt_std"] = merch_std_prev

    g_um = df.groupby(["cc_num", "merchant"], sort=False)
    um_txn_idx = g_um.cumcount() + 1
    um_prev_cnt = (um_txn_idx - 1).clip(lower=0)
    um_cum_sum = g_um["amt"].cumsum()
    um_cum_sum_prev = um_cum_sum - df["amt"]
    um_cum_mean_prev = np.where(um_prev_cnt > 0, um_cum_sum_prev / um_prev_cnt, np.nan)
    um_cum_mean_prev_s = pd.Series(um_cum_mean_prev, index=df.index).fillna(global_amt_median)

    um_cum_std = g_um["amt"].expanding().std().reset_index(level=[0, 1], drop=True).fillna(0.0)
    um_cum_std_prev = um_cum_std.groupby([df["cc_num"], df["merchant"]], sort=False).shift(1).fillna(0.0)

    df["A9_user_merchant_cum_mean"] = um_cum_mean_prev_s
    df["A10_user_merchant_cum_std"] = um_cum_std_prev
    df["A11_user_merchant_cum_sum"] = um_cum_sum_prev.clip(lower=0)
    df["A12_user_merchant_global_mean"] = df["A9_user_merchant_cum_mean"]
    df["A13_user_merchant_global_std"] = df["A10_user_merchant_cum_std"]
    df["A14_user_merchant_global_sum"] = df["A11_user_merchant_cum_sum"]

    um_lo_2 = (df["A9_user_merchant_cum_mean"] - 2 * df["A10_user_merchant_cum_std"]).to_numpy()
    um_hi_2 = (df["A9_user_merchant_cum_mean"] + 2 * df["A10_user_merchant_cum_std"]).to_numpy()
    df["A15_amt_within_2std_user_merchant"] = ((amt_arr >= um_lo_2) & (amt_arr <= um_hi_2)).astype(int)

    um_lo_1 = (df["A9_user_merchant_cum_mean"] - df["A10_user_merchant_cum_std"]).to_numpy()
    um_hi_1 = (df["A9_user_merchant_cum_mean"] + df["A10_user_merchant_cum_std"]).to_numpy()
    df["A16_amt_within_confidence_user_merchant"] = (((amt_arr >= um_lo_1) & (amt_arr <= um_hi_1)) & (um_prev_cnt >= 3)).astype(int)

    prev_time = g_user["trans_date_trans_time"].shift(1)
    df["T1_time_diff_sec"] = (df["trans_date_trans_time"] - prev_time).dt.total_seconds().fillna(np.nan)

    t1_safe = df["T1_time_diff_sec"].fillna(np.inf).clip(lower=1.0)
    df["T2_amt_time_angle"] = np.arctan2(df["amt"], t1_safe)

    hour = df["trans_date_trans_time"].dt.hour
    hour_angle = 2 * np.pi * (hour / 24.0)
    df["T3_hour_sin"] = np.sin(hour_angle)
    df["T4_hour_cos"] = np.cos(hour_angle)
    df["T5_is_madrugada"] = (hour < 6).astype(int)

    user_madrugada_cum = g_user["T5_is_madrugada"].cumsum()
    user_madrugada_prev = user_madrugada_cum - df["T5_is_madrugada"]
    df["T6_user_madrugada_ratio_cum"] = np.where(user_txn_prev > 0, user_madrugada_prev / user_txn_prev, 0.0)

    user_cum_sin = g_user["T3_hour_sin"].cumsum()
    user_cum_cos = g_user["T4_hour_cos"].cumsum()
    user_cum_sin_prev = user_cum_sin - df["T3_hour_sin"]
    user_cum_cos_prev = user_cum_cos - df["T4_hour_cos"]
    df["T7_user_hour_vonmises_proxy"] = np.where(
        user_txn_prev > 0,
        np.sqrt(user_cum_sin_prev.pow(2) + user_cum_cos_prev.pow(2)) / user_txn_prev,
        0.0,
    )

    df["T8_txn_count_1s"] = _user_window_counts(df, 1)
    df["T8_txn_count_10s"] = _user_window_counts(df, 10)
    df["T8_txn_count_60s"] = _user_window_counts(df, 60)
    df["T8_txn_count_1h"] = _user_window_counts(df, 3600)
    df["T8_txn_count_2h"] = _user_window_counts(df, 7200)

    user_t1_mean = g_user["T1_time_diff_sec"].expanding().mean().reset_index(level=0, drop=True)
    user_t1_std = g_user["T1_time_diff_sec"].expanding().std().reset_index(level=0, drop=True).fillna(0.0)
    user_t1_mean_prev = user_t1_mean.groupby(df["cc_num"], sort=False).shift(1)
    user_t1_std_prev = user_t1_std.groupby(df["cc_num"], sort=False).shift(1).fillna(0.0)
    t1 = df["T1_time_diff_sec"].to_numpy()
    t1_lo = (user_t1_mean_prev - 2 * user_t1_std_prev).to_numpy()
    t1_hi = (user_t1_mean_prev + 2 * user_t1_std_prev).to_numpy()
    coherent = (t1 >= t1_lo) & (t1 <= t1_hi)
    df["T9_time_interval_coherent"] = pd.Series(coherent, index=df.index).fillna(False).astype(int)

    df["L1_user_city_visit_count"] = df.groupby(["cc_num", "location_city"], sort=False).cumcount()
    df["L2_user_country_visit_count"] = df.groupby(["cc_num", "location_country"], sort=False).cumcount()

    new_city_flag = (~df.duplicated(subset=["cc_num", "location_city"])).astype(int)
    new_country_flag = (~df.duplicated(subset=["cc_num", "location_country"])).astype(int)

    city_unique_cum = new_city_flag.groupby(df["cc_num"]).cumsum()
    country_unique_cum = new_country_flag.groupby(df["cc_num"]).cumsum()
    df["L3_user_unique_city_count"] = (city_unique_cum - new_city_flag).clip(lower=0)
    df["L3_user_unique_country_count"] = (country_unique_cum - new_country_flag).clip(lower=0)

    df["L4_user_city_txn_ratio"] = np.where(user_txn_prev > 0, df["L1_user_city_visit_count"] / user_txn_prev, 0.0)
    df["L4_user_country_txn_ratio"] = np.where(user_txn_prev > 0, df["L2_user_country_visit_count"] / user_txn_prev, 0.0)

    prev_loc = df.groupby("cc_num", sort=False)["location_key"].shift(1)
    df["L5_same_location_as_prev"] = (df["location_key"] == prev_loc).fillna(False).astype(int)
    df["L6_location_seen_before"] = df.duplicated(subset=["cc_num", "location_key"]).astype(int)

    df = df.sort_values("__row_id", kind="mergesort").drop(columns=["__row_id"]).reset_index(drop=True)
    return df
