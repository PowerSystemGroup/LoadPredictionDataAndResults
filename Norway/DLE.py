import os
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import pywt
from scipy.signal import medfilt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))


def trimmed_stats(residuals, weights, trim_ratio=0.05):
    r = np.asarray(residuals, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(r) & np.isfinite(w)
    r = r[mask]
    w = w[mask]
    n = len(r)
    if n == 0:
        return np.array([0.0]), np.array([1.0])
    k = int(n * trim_ratio)
    if k < 1 or n <= 2 * k:
        return r, w
    idx = np.argsort(np.abs(r))
    keep_idx = idx[k:n - k]
    return r[keep_idx], w[keep_idx]


def compute_huber_alpha(residuals, scale=2.0):
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) == 0:
        return 1.0
    med = np.median(np.abs(residuals)) + 1e-6
    sigma = med * 1.4826 * scale
    return float(max(sigma, 1e-3))


def compute_dynamic_max_depth_advanced(
    residuals, weights,
    d0=6, delta_up=1.35, delta_down=1.1,
    min_depth=3, max_depth_limit=12,
    prev_depth=None, trim_ratio=0.06, smooth_factor=0.45
):
    r_trim, w_trim = trimmed_stats(residuals, weights, trim_ratio=trim_ratio)
    w_sum = np.sum(w_trim) + 1e-12
    mu = np.sum(w_trim * r_trim) / w_sum
    var_w = np.sum(w_trim * (r_trim - mu) ** 2) / w_sum
    sigma_w = np.sqrt(max(var_w, 0.0))

    skew_val = abs(skew(r_trim)) if len(r_trim) >= 3 else 0.0
    kurt_val = abs(kurtosis(r_trim, fisher=True)) if len(r_trim) >= 4 else 0.0

    med_abs = np.median(np.abs(r_trim - np.median(r_trim))) + 1e-6
    mad_w = med_abs * 1.4826

    f = (sigma_w * (1.0 + 4.5 * skew_val + 4.5 * kurt_val)) ** (1.0 / 3.0) + (mad_w) ** (1.0 / 3.0)

    raw = d0 + delta_up * f if f >= 1.0 else d0 - delta_down * f
    if prev_depth is not None:
        raw = smooth_factor * raw + (1.0 - smooth_factor) * prev_depth

    depth = int(round(raw))
    return max(min_depth, min(depth, max_depth_limit))


def compute_dynamic_gamma_advanced(
    residuals, weights,
    base=1.0, alpha=0.8,
    min_gamma=0.02, max_gamma=6.0,
    prev_gamma=None, trim_ratio=0.06, smooth_factor=0.45
):
    r_arr = np.asarray(residuals, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    r_trim, w_trim = trimmed_stats(r_arr, w_arr, trim_ratio=trim_ratio)

    idx = np.argsort(r_trim)
    r_sorted = r_trim[idx]
    w_sorted = w_trim[idx]
    w_cum = np.cumsum(w_sorted)
    w_sum = w_cum[-1] + 1e-12

    q1_pos = 0.25 * w_sum
    q3_pos = 0.75 * w_sum
    q1_idx = np.searchsorted(w_cum, q1_pos)
    q3_idx = np.searchsorted(w_cum, q3_pos)
    r_q1 = r_sorted[min(q1_idx, len(r_sorted) - 1)]
    r_q3 = r_sorted[min(q3_idx, len(r_sorted) - 1)]
    iqr_w = abs(r_q3 - r_q1)

    med_abs = np.median(np.abs(r_trim - np.median(r_trim))) + 1e-6
    mad_w = med_abs * 1.4826

    skew_val = abs(skew(r_trim)) if len(r_trim) >= 3 else 0.0
    kurt_val = abs(kurtosis(r_trim, fisher=True)) if len(r_trim) >= 4 else 0.0

    f_gamma = (iqr_w / (mad_w + 1e-12)) * (1.0 + 4.5 * skew_val + 4.5 * kurt_val)
    raw = base / (1.0 + alpha * f_gamma)

    if prev_gamma is not None:
        raw = smooth_factor * raw + (1.0 - smooth_factor) * prev_gamma

    gamma = float(max(min_gamma, min(raw, max_gamma)))
    return gamma


def sanitize_target(y, lower=None, upper=None, fill_value=None):
    y = np.asarray(y, dtype=float)
    if lower is not None:
        y = np.maximum(y, lower)
    if upper is not None:
        y = np.minimum(y, upper)
    bad = ~np.isfinite(y)
    if np.any(bad):
        if fill_value is None:
            good = y[np.isfinite(y)]
            fill_value = np.median(good) if len(good) else 0.0
        y[bad] = fill_value
    return y


def safe_wavelet_denoise(arr, wave="db4", level=3, lower=None):
    arr = np.asarray(arr, dtype=float)
    fill_value = np.median(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0
    arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)
    if len(arr) < 8:
        if lower is not None:
            arr = np.maximum(arr, lower)
        return arr

    coeffs = pywt.wavedec(arr, wave, level=min(level, pywt.dwt_max_level(len(arr), pywt.Wavelet(wave).dec_len)))
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) else 0.0
    thr = sigma * np.sqrt(2 * np.log(max(len(arr), 2)))
    coeffs[1:] = [pywt.threshold(c, thr, "soft") for c in coeffs[1:]]
    rec = pywt.waverec(coeffs, wave)[:len(arr)]
    if lower is not None:
        rec = np.maximum(rec, lower)
    return rec


def ensure_numeric_column(df, col, default=0.0):
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_static_columns(df):
    defaults = {
        "is_weekend": 0, "connectionTime_is_holiday": 0, "is_holiday_weekend": 0,
        "hour": 0, "weekday": 0, "month": 1, "dayofyear": 1, "weekofyear": 1, "season": 1,
        "hour_decimal": 0.0,
        "hour_sin": 0.0, "hour_cos": 1.0,
        "weekday_sin": 0.0, "weekday_cos": 1.0,
        "month_sin": 0.0, "month_cos": 1.0,
        "dayofyear_sin": 0.0, "dayofyear_cos": 1.0,
        "weekofyear_sin": 0.0, "weekofyear_cos": 1.0,
        "season_sin": 0.0, "season_cos": 1.0,
        "pt1": np.nan, "pt2": np.nan, "rt3": np.nan,
        "pk1": np.nan, "pk2": np.nan, "rk3": np.nan,
        "station_roll_kWh3": np.nan, "station_roll_park3": np.nan,
        "user_avg_kWh": np.nan, "station_avg_kWh": np.nan,
        "user_avg_park": np.nan, "station_avg_park": np.nan,
        "user_hist_avg_power": np.nan, "station_hist_avg_power": np.nan,
        "station_hour_avg_kWh": np.nan, "user_hour_avg_kWh": np.nan, "station_user_hour_avg_kWh": np.nan,
        "station_hour_avg_park": np.nan, "user_hour_avg_park": np.nan, "station_user_hour_avg_park": np.nan,
        "sim_kwh": np.nan, "sim_park": np.nan,
        "user_session_order": np.nan, "station_session_order": np.nan,
        "hours_since_prev_user": np.nan, "hours_since_prev_station": np.nan,
        "user_freq": np.nan, "station_freq": np.nan,
        "user_lag1_kWhDelivered": np.nan, "user_lag2_kWhDelivered": np.nan, "user_roll3_kWhDelivered": np.nan,
        "station_lag1_kWhDelivered": np.nan, "station_lag2_kWhDelivered": np.nan, "station_roll3_kWhDelivered": np.nan,
        "user_lag1_parking_time": np.nan, "user_lag2_parking_time": np.nan, "user_roll3_parking_time": np.nan,
        "station_lag1_parking_time": np.nan, "station_lag2_parking_time": np.nan, "station_roll3_parking_time": np.nan
    }
    for col, default in defaults.items():
        ensure_numeric_column(df, col, default)

    if "location" in df.columns:
        df["location"] = df["location"].astype(str)
        df["location_code"] = LabelEncoder().fit_transform(df["location"])
    else:
        df["location_code"] = 0

    df["is_holiday"] = pd.to_numeric(df["connectionTime_is_holiday"], errors="coerce").fillna(0).astype(int)
    df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)
    df["is_holiday_weekend"] = pd.to_numeric(df["is_holiday_weekend"], errors="coerce").fillna(0).astype(int)

    for col in ["hours_since_prev_user", "hours_since_prev_station", "user_hist_avg_power", "station_hist_avg_power",
                "user_session_order", "station_session_order"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df["log_gap_user"] = np.log1p(df["hours_since_prev_user"].clip(lower=0).fillna(0))
    df["log_gap_station"] = np.log1p(df["hours_since_prev_station"].clip(lower=0).fillna(0))
    df["log_user_order"] = np.log1p(df["user_session_order"].clip(lower=0).fillna(0))
    df["log_station_order"] = np.log1p(df["station_session_order"].clip(lower=0).fillna(0))

    return df


def add_cluster(df, test_start, seed):
    hist_all = df[df.connection_time_copy < test_start].copy()
    grp = hist_all.groupby("userID").agg(
        avg_kwh=("kWhDelivered", "mean"),
        avg_park=("parking_time", "mean"),
        avg_power=("user_hist_avg_power", "mean"),
        freq=("kWhDelivered", "count")
    )
    grp["freq"] = np.log1p(grp["freq"])
    grp = grp.replace([np.inf, -np.inf], np.nan).fillna(grp.median(numeric_only=True)).fillna(0.0)
    if len(grp) >= 10:
        km = KMeans(n_clusters=10, random_state=seed).fit(grp.values)
        cmap = dict(zip(grp.index, km.labels_))
    else:
        cmap = {uid: 0 for uid in grp.index}
    df["cluster"] = df["userID"].map(cmap).fillna(0).astype(int)
    return df


def add_history_stats(tr, te):
    ua_m = tr.groupby("userID")["kWhDelivered"].mean()
    ua_c = tr["userID"].value_counts()
    sa_m = tr.groupby("stationID")["kWhDelivered"].mean()
    sa_c = tr["stationID"].value_counts()

    ua_p = tr.groupby("userID")["parking_time"].mean()
    sa_p = tr.groupby("stationID")["parking_time"].mean()

    for D in (tr, te):
        D["user_avg_kWh"] = D["userID"].map(ua_m).fillna(ua_m.mean())
        D["user_freq"] = D["userID"].map(ua_c).fillna(0)
        D["station_avg_kWh"] = D["stationID"].map(sa_m).fillna(sa_m.mean())
        D["station_freq"] = D["stationID"].map(sa_c).fillna(0)
        D["user_avg_park"] = D["userID"].map(ua_p).fillna(ua_p.mean())
        D["station_avg_park"] = D["stationID"].map(sa_p).fillna(sa_p.mean())

    tr["hour_of_week"] = tr["weekday"] * 24 + tr["hour"]
    te["hour_of_week"] = te["weekday"] * 24 + te["hour"]

    tr = tr.sort_values("connection_time_copy").reset_index(drop=True)
    te = te.sort_values("connection_time_copy").reset_index(drop=True)

    global_kwh = tr["kWhDelivered"].median()
    global_park = tr["parking_time"].median()

    tr["station_hour_avg_kWh"] = tr.groupby(["stationID", "hour_of_week"])["kWhDelivered"].transform(lambda s: s.shift(1).expanding().mean()).fillna(global_kwh)
    tr["user_hour_avg_kWh"] = tr.groupby(["userID", "hour_of_week"])["kWhDelivered"].transform(lambda s: s.shift(1).expanding().mean()).fillna(global_kwh)
    tr["station_user_hour_avg_kWh"] = tr.groupby(["stationID", "userID", "hour_of_week"])["kWhDelivered"].transform(lambda s: s.shift(1).expanding().mean()).fillna(global_kwh)

    tr["station_hour_avg_park"] = tr.groupby(["stationID", "hour_of_week"])["parking_time"].transform(lambda s: s.shift(1).expanding().mean()).fillna(global_park)
    tr["user_hour_avg_park"] = tr.groupby(["userID", "hour_of_week"])["parking_time"].transform(lambda s: s.shift(1).expanding().mean()).fillna(global_park)
    tr["station_user_hour_avg_park"] = tr.groupby(["stationID", "userID", "hour_of_week"])["parking_time"].transform(lambda s: s.shift(1).expanding().mean()).fillna(global_park)

    sh_kwh = tr.groupby(["stationID", "hour_of_week"])["kWhDelivered"].mean()
    uh_kwh = tr.groupby(["userID", "hour_of_week"])["kWhDelivered"].mean()
    suh_kwh = tr.groupby(["stationID", "userID", "hour_of_week"])["kWhDelivered"].mean()
    sh_park = tr.groupby(["stationID", "hour_of_week"])["parking_time"].mean()
    uh_park = tr.groupby(["userID", "hour_of_week"])["parking_time"].mean()
    suh_park = tr.groupby(["stationID", "userID", "hour_of_week"])["parking_time"].mean()

    te["station_hour_avg_kWh"] = pd.Series(te.set_index(["stationID", "hour_of_week"]).index.map(sh_kwh), index=te.index, dtype=float).fillna(global_kwh)
    te["user_hour_avg_kWh"] = pd.Series(te.set_index(["userID", "hour_of_week"]).index.map(uh_kwh), index=te.index, dtype=float).fillna(global_kwh)
    te["station_user_hour_avg_kWh"] = pd.Series(te.set_index(["stationID", "userID", "hour_of_week"]).index.map(suh_kwh), index=te.index, dtype=float).fillna(global_kwh)
    te["station_hour_avg_park"] = pd.Series(te.set_index(["stationID", "hour_of_week"]).index.map(sh_park), index=te.index, dtype=float).fillna(global_park)
    te["user_hour_avg_park"] = pd.Series(te.set_index(["userID", "hour_of_week"]).index.map(uh_park), index=te.index, dtype=float).fillna(global_park)
    te["station_user_hour_avg_park"] = pd.Series(te.set_index(["stationID", "userID", "hour_of_week"]).index.map(suh_park), index=te.index, dtype=float).fillna(global_park)

    for D in (tr, te):
        if "sim_power" not in D.columns:
            D["sim_power"] = (D["sim_kwh"] / (D["sim_park"] + 1e-6)).replace([np.inf, -np.inf], np.nan)
        D["proxy_req_rate_user"] = D["user_hist_avg_power"].fillna(D["station_hist_avg_power"])
        D["proxy_req_rate_station"] = D["station_hist_avg_power"].fillna(D["user_hist_avg_power"])
        D["proxy_req_rate_mix"] = (0.55 * D["proxy_req_rate_user"] + 0.45 * D["proxy_req_rate_station"]).replace([np.inf, -np.inf], np.nan)
        D["sh_ratio_nor"] = D["station_hour_avg_kWh"] / (D["station_hour_avg_park"] * D["proxy_req_rate_station"] + 1e-6)
        D["uh_ratio_nor"] = D["user_hour_avg_kWh"] / (D["user_hour_avg_park"] * D["proxy_req_rate_user"] + 1e-6)
        D["shur_ratio_nor"] = D["station_user_hour_avg_kWh"] / (D["station_user_hour_avg_park"] * D["proxy_req_rate_mix"] + 1e-6)

    for c in ["sh_ratio_nor", "uh_ratio_nor", "shur_ratio_nor", "proxy_req_rate_mix"]:
        med = tr[c].replace([np.inf, -np.inf], np.nan).median()
        tr[c] = tr[c].replace([np.inf, -np.inf], np.nan).fillna(med)
        te[c] = te[c].replace([np.inf, -np.inf], np.nan).fillna(med)

    return tr, te


def add_similarity_features(tr, te):
    for D in (tr, te):
        D["proxy_recent_park"] = 0.60 * D["rt3"].fillna(D["user_avg_park"]) + 0.40 * D["station_hour_avg_park"].fillna(D["station_avg_park"])
        D["proxy_recent_kwh"] = 0.60 * D["rk3"].fillna(D["user_avg_kWh"]) + 0.40 * D["station_hour_avg_kWh"].fillna(D["station_avg_kWh"])
        D["proxy_req_rate_mix"] = 0.55 * D["user_hist_avg_power"].fillna(D["station_hist_avg_power"]) + 0.45 * D["station_hist_avg_power"].fillna(D["user_hist_avg_power"])
        D["proxy_req_energy_mix"] = D["proxy_recent_park"] * D["proxy_req_rate_mix"]

    sim_feats = ["hour", "weekday", "proxy_req_energy_mix", "proxy_recent_park"]
    tr_base = tr[sim_feats].replace([np.inf, -np.inf], np.nan)
    te_base = te[sim_feats].replace([np.inf, -np.inf], np.nan)
    fill_vals = tr_base.median(numeric_only=True)
    tr_base = tr_base.fillna(fill_vals).fillna(0.0)
    te_base = te_base.fillna(fill_vals).fillna(0.0)

    scaler = StandardScaler().fit(tr_base)
    tr_s = scaler.transform(tr_base)
    te_s = scaler.transform(te_base)

    tr["sim_park"] = 0.0
    tr["sim_kwh"] = 0.0
    te["sim_park"] = 0.0
    te["sim_kwh"] = 0.0

    tr_global_park = tr["parking_time"].median()
    tr_global_kwh = tr["kWhDelivered"].median()

    for i in range(len(tr)):
        prev_idx = tr[(tr.userID == tr.loc[i, "userID"]) & (tr.index < i)].index
        if len(prev_idx) == 0:
            tr.at[i, "sim_park"] = tr_global_park
            tr.at[i, "sim_kwh"] = tr_global_kwh
        else:
            sims = np.dot(tr_s[prev_idx], tr_s[i])
            top_idx = prev_idx[np.argsort(sims)[-5:]]
            w5 = sims[np.argsort(sims)[-5:]]
            tr.at[i, "sim_park"] = np.average(tr.loc[top_idx, "parking_time"], weights=w5 + 1e-6)
            tr.at[i, "sim_kwh"] = np.average(tr.loc[top_idx, "kWhDelivered"], weights=w5 + 1e-6)

    for i in range(len(te)):
        prev_idx = tr[tr.userID == te.loc[i, "userID"]].index
        if len(prev_idx) == 0:
            te.at[i, "sim_park"] = tr_global_park
            te.at[i, "sim_kwh"] = tr_global_kwh
        else:
            sims = np.dot(tr_s[prev_idx], te_s[i])
            top_idx = prev_idx[np.argsort(sims)[-5:]]
            w5 = sims[np.argsort(sims)[-5:]]
            te.at[i, "sim_park"] = np.average(tr.loc[top_idx, "parking_time"], weights=w5 + 1e-6)
            te.at[i, "sim_kwh"] = np.average(tr.loc[top_idx, "kWhDelivered"], weights=w5 + 1e-6)

    return tr, te


def fill_feature_frame(train_df, test_df, feature_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in test_df.columns:
            test_df[col] = np.nan

    train = train_df[feature_cols].copy()
    test = test_df[feature_cols].copy()
    for col in feature_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")
    med = train.median(numeric_only=True)
    train = train.fillna(med).fillna(0.0)
    test = test.fillna(med).fillna(0.0)
    return train, test


def run_dynamic_xgb_regression(X_tr, y_tr, X_va, y_va, sample_w, seed,
                               max_rounds=200, early_stop=30,
                               preheat_rounds=15, dynamic_interval=5,
                               eta=0.05, objective="reg:squarederror",
                               use_huber=False):
    X_tr = np.asarray(X_tr, dtype=float)
    X_va = np.asarray(X_va, dtype=float)
    y_tr = np.asarray(y_tr, dtype=float)
    y_va = np.asarray(y_va, dtype=float)
    sample_w = np.asarray(sample_w, dtype=float)

    X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_va = np.nan_to_num(X_va, nan=0.0, posinf=0.0, neginf=0.0)

    mask_tr = np.isfinite(y_tr) & np.isfinite(sample_w)
    if X_tr.ndim == 2:
        mask_tr &= np.all(np.isfinite(X_tr), axis=1)
    X_tr = X_tr[mask_tr]
    y_tr = y_tr[mask_tr]
    sample_w = sample_w[mask_tr]

    mask_va = np.isfinite(y_va)
    if X_va.ndim == 2:
        mask_va &= np.all(np.isfinite(X_va), axis=1)
    X_va = X_va[mask_va]
    y_va = y_va[mask_va]

    if len(y_tr) == 0 or len(y_va) == 0:
        raise ValueError("Training or validation set became empty after sanitizing labels/features.")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_w)
    dval = xgb.DMatrix(X_va, label=y_va)

    model = None
    val_maes = []
    best_val = np.inf
    rounds_no_improve = 0
    prev_depth = None
    prev_gamma = None
    depth_now = 6
    gamma_now = 1.0

    for m in range(max_rounds):
        if m < preheat_rounds:
            depth_now = 6
            gamma_now = 1.0
        else:
            if (m - preheat_rounds) % dynamic_interval == 0:
                if model is None:
                    residuals = y_tr.copy()
                else:
                    residuals = y_tr - model.predict(dtrain)

                depth_now = compute_dynamic_max_depth_advanced(
                    residuals, sample_w,
                    d0=6, delta_up=1.35, delta_down=1.1,
                    min_depth=3, max_depth_limit=12,
                    prev_depth=prev_depth, trim_ratio=0.06, smooth_factor=0.45
                )
                gamma_now = compute_dynamic_gamma_advanced(
                    residuals, sample_w,
                    base=1.0, alpha=0.8,
                    min_gamma=0.02, max_gamma=6.0,
                    prev_gamma=prev_gamma, trim_ratio=0.06, smooth_factor=0.45
                )
                prev_depth = depth_now
                prev_gamma = gamma_now

        params = {
            "tree_method": "hist",
            "eta": eta,
            "max_depth": depth_now,
            "gamma": gamma_now,
            "objective": objective,
            "eval_metric": "mae",
            "verbosity": 0,
            "subsample": 0.85,
            "colsample_bytree": 0.82,
            "min_child_weight": 5,
            "reg_alpha": 0.12,
            "reg_lambda": 1.4,
            "seed": seed
        }

        if use_huber:
            if model is None:
                residuals = y_tr.copy()
            else:
                residuals = y_tr - model.predict(dtrain)
            params["objective"] = "reg:pseudohubererror"
            params["alpha"] = compute_huber_alpha(residuals, scale=2.0)

        if model is None:
            model = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        else:
            model = xgb.train(params, dtrain, num_boost_round=1, xgb_model=model, verbose_eval=False)

        pred_va = model.predict(dval)
        va_mae = mae(y_va, pred_va)
        val_maes.append(va_mae)

        if va_mae < best_val - 1e-6:
            best_val = va_mae
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
        if rounds_no_improve >= early_stop:
            break

    return model, val_maes


def build_parking_baseline(df, fallback):
    sim = pd.to_numeric(df.get("sim_park", np.nan), errors="coerce")
    rt3 = pd.to_numeric(df.get("rt3", np.nan), errors="coerce")
    user_hour = pd.to_numeric(df.get("user_hour_avg_park", np.nan), errors="coerce")
    station_hour = pd.to_numeric(df.get("station_hour_avg_park", np.nan), errors="coerce")
    user_avg = pd.to_numeric(df.get("user_avg_park", np.nan), errors="coerce")
    pt1 = pd.to_numeric(df.get("pt1", np.nan), errors="coerce")
    station_avg = pd.to_numeric(df.get("station_avg_park", np.nan), errors="coerce")
    base = (
        0.32 * sim.fillna(fallback) +
        0.20 * rt3.fillna(user_avg) +
        0.16 * user_hour.fillna(user_avg) +
        0.12 * station_hour.fillna(station_avg) +
        0.12 * user_avg.fillna(fallback) +
        0.08 * pt1.fillna(user_avg)
    )
    return base.fillna(fallback).values.astype(float)


def build_residual_bias_map(df_hist, residuals, target_df):
    hist = df_hist.copy()
    hist["hour_of_week"] = hist["weekday"] * 24 + hist["hour"]
    tgt = target_df.copy()
    tgt["hour_of_week"] = tgt["weekday"] * 24 + tgt["hour"]

    temp = hist[["userID", "stationID", "hour_of_week"]].copy()
    temp["residual"] = residuals

    user_map = temp.groupby("userID")["residual"].agg(["mean", "count"])
    station_map = temp.groupby("stationID")["residual"].agg(["mean", "count"])
    user_hour_map = temp.groupby(["userID", "hour_of_week"])["residual"].agg(["mean", "count"])
    station_hour_map = temp.groupby(["stationID", "hour_of_week"])["residual"].agg(["mean", "count"])

    user_mean = tgt["userID"].map(user_map["mean"])
    user_cnt = tgt["userID"].map(user_map["count"]).fillna(0)
    station_mean = tgt["stationID"].map(station_map["mean"])
    station_cnt = tgt["stationID"].map(station_map["count"]).fillna(0)

    uh_idx = tgt.set_index(["userID", "hour_of_week"]).index
    sh_idx = tgt.set_index(["stationID", "hour_of_week"]).index
    uh_mean = pd.Series(uh_idx.map(user_hour_map["mean"]), index=tgt.index, dtype=float)
    uh_cnt = pd.Series(uh_idx.map(user_hour_map["count"]), index=tgt.index, dtype=float).fillna(0)
    sh_mean = pd.Series(sh_idx.map(station_hour_map["mean"]), index=tgt.index, dtype=float)
    sh_cnt = pd.Series(sh_idx.map(station_hour_map["count"]), index=tgt.index, dtype=float).fillna(0)

    wu = np.clip(user_cnt.values / 8.0, 0, 1)
    ws = np.clip(station_cnt.values / 10.0, 0, 1)
    wuh = np.clip(uh_cnt.values / 4.0, 0, 1)
    wsh = np.clip(sh_cnt.values / 5.0, 0, 1)

    corr = (
        0.30 * wuh * uh_mean.fillna(0).values +
        0.25 * wsh * sh_mean.fillna(0).values +
        0.25 * wu * user_mean.fillna(0).values +
        0.20 * ws * station_mean.fillna(0).values
    )
    return corr.astype(float)


def choose_parking_residual_blend(y_true, pred_global, pred_base, pred_resid, corr_val,
                                  grid_global=np.arange(0.0, 1.01, 0.2),
                                  grid_resid=np.array([0.6, 0.8, 1.0]),
                                  grid_corr=np.array([0.0, 0.25, 0.5, 0.75, 1.0])):
    best_g = 1.0
    best_r = 0.0
    best_c = 0.0
    best_err = np.inf
    best_pred = pred_global.copy()

    for g in grid_global:
        for r in grid_resid:
            for c in grid_corr:
                route_pred = pred_base + r * pred_resid + c * corr_val
                pred = g * pred_global + (1.0 - g) * route_pred
                err = mae(y_true, pred)
                if err < best_err:
                    best_err = err
                    best_g = float(g)
                    best_r = float(r)
                    best_c = float(c)
                    best_pred = pred.copy()
    return best_g, best_r, best_c, best_pred, best_err


def inverse_positive_power_transform(transformer, arr, eps=1e-3):
    arr = np.asarray(arr, dtype=float).reshape(-1, 1)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 1e-6)
    inv = transformer.inverse_transform(arr).flatten() - eps
    inv = np.where(np.isfinite(inv), inv, 0.0)
    return np.maximum(inv, 0.05)


def choose_pt_dle_blend(y_true, pred_global, pred_dle, pred_base,
                        grid_g=np.arange(0.0, 1.01, 0.2),
                        grid_d=np.arange(0.0, 1.01, 0.2)):
    best_g = 1.0
    best_d = 0.0
    best_err = np.inf
    best_pred = pred_global.copy()
    for g in grid_g:
        for d in grid_d:
            mid = d * pred_dle + (1.0 - d) * pred_base
            pred = g * pred_global + (1.0 - g) * mid
            err = mae(y_true, pred)
            if err < best_err:
                best_err = err
                best_g = float(g)
                best_d = float(d)
                best_pred = pred.copy()
    return best_g, best_d, best_pred, best_err


def build_pt_minimal_baseline(df, fallback):
    sim = pd.to_numeric(df.get("sim_park", np.nan), errors="coerce")
    user_hour = pd.to_numeric(df.get("user_hour_avg_park", np.nan), errors="coerce")
    rt3 = pd.to_numeric(df.get("rt3", np.nan), errors="coerce")
    user_avg = pd.to_numeric(df.get("user_avg_park", np.nan), errors="coerce")
    pt1 = pd.to_numeric(df.get("pt1", np.nan), errors="coerce")
    base = (
        0.36 * sim.fillna(fallback) +
        0.22 * user_hour.fillna(user_avg) +
        0.18 * rt3.fillna(user_avg) +
        0.16 * user_avg.fillna(fallback) +
        0.08 * pt1.fillna(user_avg)
    )
    base = np.where(np.isfinite(base), base, fallback)
    return np.maximum(base, 0.05)


def fit_pt_dle_branch(X_fit, y_fit_raw, X_val, y_val_raw, fit_weights, feature_names, seed,
                      max_rounds=220, early_stop=35, eta=0.045, eps=1e-3,
                      lgb_params=None):
    pt_transform = PowerTransformer(method="box-cox", standardize=False)
    y_fit_safe = sanitize_target(np.asarray(y_fit_raw, dtype=float) + eps, lower=eps)
    y_fit_bc = pt_transform.fit_transform(y_fit_safe.reshape(-1, 1)).flatten()

    y_val_safe = sanitize_target(np.asarray(y_val_raw, dtype=float) + eps, lower=eps)
    y_val_bc = pt_transform.transform(y_val_safe.reshape(-1, 1)).flatten()

    model_xgb, val_maes = run_dynamic_xgb_regression(
        X_fit, y_fit_bc, X_val, y_val_bc, fit_weights, seed=seed,
        max_rounds=max_rounds, early_stop=early_stop,
        preheat_rounds=15, dynamic_interval=5,
        eta=eta, objective="reg:pseudohubererror", use_huber=True
    )

    pred_fit_bc = model_xgb.predict(xgb.DMatrix(np.nan_to_num(X_fit)))
    resid_fit_bc = y_fit_bc - pred_fit_bc

    if lgb_params is None:
        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": 120,
            "learning_rate": 0.03,
            "num_leaves": 15,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "min_data_in_leaf": 25,
            "lambda_l1": 0.05,
            "lambda_l2": 0.60,
            "random_state": 42
        }

    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(pd.DataFrame(X_fit, columns=feature_names), resid_fit_bc, sample_weight=fit_weights)

    pred_val_bc = model_xgb.predict(xgb.DMatrix(np.nan_to_num(X_val))) + model_lgb.predict(pd.DataFrame(X_val, columns=feature_names))
    pred_val = inverse_positive_power_transform(pt_transform, pred_val_bc, eps=eps)

    def predict_fn(X):
        pred_bc = model_xgb.predict(xgb.DMatrix(np.nan_to_num(X))) + model_lgb.predict(pd.DataFrame(X, columns=feature_names))
        return inverse_positive_power_transform(pt_transform, pred_bc, eps=eps)

    return {
        "xgb": model_xgb,
        "lgb": model_lgb,
        "transform": pt_transform,
        "predict": predict_fn,
        "val_pred": pred_val,
        "val_maes": val_maes,
    }


def build_user_ratio_calibration(df_fit, y_true_fit, y_pred_fit, df_target):
    y_true_fit = np.asarray(y_true_fit, dtype=float)
    y_pred_fit = np.asarray(y_pred_fit, dtype=float)
    ratio = y_true_fit / np.maximum(y_pred_fit, 1e-6)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio = np.clip(ratio, 0.6, 1.6)

    tmp = df_fit.copy()
    tmp["ratio_corr"] = ratio
    tmp["hour_of_week"] = tmp["weekday"] * 24 + tmp["hour"]

    user_map = tmp.groupby("userID")["ratio_corr"].median()
    user_hour_map = tmp.groupby(["userID", "hour_of_week"])["ratio_corr"].median()

    tgt = df_target.copy()
    tgt["hour_of_week"] = tgt["weekday"] * 24 + tgt["hour"]

    u = tgt["userID"].map(user_map)
    idx = tgt.set_index(["userID", "hour_of_week"]).index
    uh = pd.Series(idx.map(user_hour_map), index=tgt.index, dtype=float)

    factor = 0.55 * uh.fillna(u) + 0.45 * u.fillna(1.0)
    factor = factor.fillna(1.0).values.astype(float)
    factor = np.clip(factor, 0.75, 1.35)
    return factor


def choose_pt_fourway_blend(y_true, pred_global, pred_core_dle, pred_user_resid, pred_base,
                            step=0.05):
    grid = np.round(np.arange(0.0, 1.0 + step, step), 10)
    best = (1.0, 0.0, 0.0, 0.0)
    best_err = np.inf
    best_pred = pred_global.copy()

    for wg in grid:
        for wc in grid:
            for wr in grid:
                wb = 1.0 - wg - wc - wr
                if wb < -1e-12:
                    continue
                wb = max(wb, 0.0)
                pred = wg * pred_global + wc * pred_core_dle + wr * pred_user_resid + wb * pred_base
                err = mae(y_true, pred)
                if err < best_err:
                    best_err = err
                    best = (float(wg), float(wc), float(wr), float(wb))
                    best_pred = pred.copy()
    return best[0], best[1], best[2], best[3], best_pred, best_err


def main():
    os.makedirs("./results", exist_ok=True)
    metrics_all = []

    df0 = pd.read_csv("Dataset1_charging_reports_featured.csv", parse_dates=["connection_time_copy"])
    df0 = df0.dropna(subset=["connection_time_copy", "parking_time", "kWhDelivered", "userID", "stationID"]).copy()
    df0 = df0[
        (df0["parking_time"] > 0) &
        (df0["parking_time"] <= 24) &
        (df0["kWhDelivered"] >= 0)
        ].reset_index(drop=True)
    df0.sort_values("connection_time_copy", inplace=True)
    df0.reset_index(drop=True, inplace=True)
    df0 = add_static_columns(df0)

    test_start = datetime(2019, 12, 1)
    test_end = datetime(2019, 12, 30)
    windows = [30, 60, 120, 240, 360, 480]
    n_runs = 2
    eps = 1e-3

    PT_FIXED_ETA = 0.05
    PT_MAX_ROUNDS = 200
    PT_EARLY_STOP = 30

    KWH_FIXED_ETA = 0.05
    KWH_MAX_ROUNDS = 200
    KWH_EARLY_STOP = 30

    LGB_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 180,
        "learning_rate": 0.025,
        "num_leaves": 19,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.78,
        "bagging_freq": 1,
        "min_data_in_leaf": 25,
        "lambda_l1": 0.08,
        "lambda_l2": 0.8,
        "random_state": 42
    }

    base_dirs = [
        os.path.join(".", "results", "norway_pt_dual_dle_v16", "parking_time"),
        os.path.join(".", "results", "norway_pt_dual_dle_v16", "kWhDelivered"),
    ]

    for run in range(1, n_runs + 1):
        run_seed = 1000 + run
        for base in base_dirs:
            os.makedirs(os.path.join(base, str(run)), exist_ok=True)

        print(f"\n=== Run {run} ===")
        df = add_cluster(df0.copy(), test_start, seed=run_seed)
        run_summary = []

        for w in windows:
            print(f"\n>>> Window = {w} days")
            t_window_start = time.perf_counter()

            tr0 = test_start - timedelta(days=w)
            tr = df[(df.connection_time_copy >= tr0) & (df.connection_time_copy < test_start)].copy()
            te = df[(df.connection_time_copy >= test_start) & (df.connection_time_copy <= test_end)].copy()
            if tr.empty or te.empty:
                continue

            tr["parking_time"] = np.clip(
                safe_wavelet_denoise(tr["parking_time"].values, lower=0.05), 0.05, 24.0
            )
            te["parking_time"] = np.clip(
                safe_wavelet_denoise(te["parking_time"].values, lower=0.05), 0.05, 24.0
            )

            mask = LocalOutlierFactor(n_neighbors=20).fit_predict(tr[["parking_time", "kWhDelivered"]])
            tr = tr[mask == 1].reset_index(drop=True)

            te["parking_time"] = safe_wavelet_denoise(te["parking_time"].values, lower=0.05)
            te["kWhDelivered"] = safe_wavelet_denoise(te["kWhDelivered"].values, lower=0.0)

            tr["age_days"] = (test_start - tr.connection_time_copy).dt.days
            w_decay = np.exp(-0.014 * tr["age_days"])

            tr.reset_index(drop=True, inplace=True)
            te.reset_index(drop=True, inplace=True)

            tr, te = add_history_stats(tr, te)
            tr, te = add_similarity_features(tr, te)

            pf_core = [
                "hour", "weekday", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "userID", "stationID", "sim_park"
            ]
            pf_core_dle = [
                "hour", "weekday", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "userID", "stationID",
                "sim_park", "user_avg_park", "user_hour_avg_park",
                "pt1", "rt3", "log_gap_user"
            ]
            pf_user_resid = [
                "hour", "weekday", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "userID", "stationID",
                "sim_park", "user_avg_park", "user_hour_avg_park",
                "pt1", "rt3",
                "user_lag1_parking_time", "user_roll3_parking_time",
                "log_gap_user"
            ]

            Xp_core_tr_df, Xp_core_te_df = fill_feature_frame(tr, te, pf_core)
            Xp_cdle_tr_df, Xp_cdle_te_df = fill_feature_frame(tr, te, pf_core_dle)
            Xp_udle_tr_df, Xp_udle_te_df = fill_feature_frame(tr, te, pf_user_resid)

            Xp_core_tr = Xp_core_tr_df.values
            Xp_core_te = Xp_core_te_df.values
            Xp_cdle_tr = Xp_cdle_tr_df.values
            Xp_cdle_te = Xp_cdle_te_df.values
            Xp_udle_tr = Xp_udle_tr_df.values
            Xp_udle_te = Xp_udle_te_df.values

            y_p_tr = sanitize_target(tr["parking_time"].values, lower=0.05)
            y_p_te = sanitize_target(te["parking_time"].values, lower=0.05)

            n_val = max(int(0.2 * len(tr)), 50)

            Xtr_core_p, Xva_core_p = Xp_core_tr[:-n_val], Xp_core_tr[-n_val:]
            Xtr_cdle_p, Xva_cdle_p = Xp_cdle_tr[:-n_val], Xp_cdle_tr[-n_val:]
            Xtr_udle_p, Xva_udle_p = Xp_udle_tr[:-n_val], Xp_udle_tr[-n_val:]
            ytr_p, yva_p = y_p_tr[:-n_val], y_p_tr[-n_val:]
            wtr_p = w_decay.iloc[:-n_val].values

            start_p = time.perf_counter()

            model_p_global, pt_val_maes = run_dynamic_xgb_regression(
                Xtr_core_p, ytr_p, Xva_core_p, yva_p, wtr_p, seed=run_seed,
                max_rounds=PT_MAX_ROUNDS, early_stop=PT_EARLY_STOP,
                preheat_rounds=15, dynamic_interval=5,
                eta=PT_FIXED_ETA, objective="reg:squarederror", use_huber=False
            )
            pred_p_va_global = model_p_global.predict(xgb.DMatrix(np.nan_to_num(Xva_core_p)))
            pred_p_te_global = model_p_global.predict(xgb.DMatrix(np.nan_to_num(Xp_core_te)))

            base_fit = build_pt_minimal_baseline(tr.iloc[:-n_val], np.median(y_p_tr))
            base_va = build_pt_minimal_baseline(tr.iloc[-n_val:], np.median(y_p_tr))
            base_te = build_pt_minimal_baseline(te, np.median(y_p_tr))

            core_branch = fit_pt_dle_branch(
                Xtr_cdle_p, ytr_p, Xva_cdle_p, yva_p, wtr_p, pf_core_dle, seed=run_seed + 17,
                max_rounds=220, early_stop=35, eta=0.045, eps=1e-3,
                lgb_params={
                    "objective": "regression", "metric": "mae", "verbosity": -1, "boosting_type": "gbdt",
                    "n_estimators": 120, "learning_rate": 0.03, "num_leaves": 15,
                    "feature_fraction": 0.75, "bagging_fraction": 0.80, "bagging_freq": 1,
                    "min_data_in_leaf": 25, "lambda_l1": 0.05, "lambda_l2": 0.60, "random_state": 42
                }
            )
            pred_p_va_core = core_branch["val_pred"]
            pred_p_te_core = core_branch["predict"](Xp_cdle_te)

            pred_p_fit_core = core_branch["predict"](Xtr_cdle_p)
            factor_va = build_user_ratio_calibration(tr.iloc[:-n_val], ytr_p, pred_p_fit_core, tr.iloc[-n_val:])
            factor_te = build_user_ratio_calibration(tr.iloc[:-n_val], ytr_p, pred_p_fit_core, te)
            pred_p_va_core = np.clip(pred_p_va_core * factor_va, 0.05, np.quantile(y_p_tr, 0.997))
            pred_p_te_core = np.clip(pred_p_te_core * factor_te, 0.05, np.quantile(y_p_tr, 0.997))

            resid_tr = sanitize_target(ytr_p - base_fit, fill_value=0.0)
            resid_va = sanitize_target(yva_p - base_va, fill_value=0.0)

            tail_scale = np.clip(np.abs(resid_tr) / (np.median(np.abs(resid_tr)) + 1e-6), 0.5, 3.0)
            wtr_resid = wtr_p / np.sqrt(tail_scale)

            user_branch = fit_pt_dle_branch(
                Xtr_udle_p, resid_tr + 10.0, Xva_udle_p, resid_va + 10.0, wtr_resid, pf_user_resid, seed=run_seed + 29,
                max_rounds=180, early_stop=28, eta=0.045, eps=1e-3,
                lgb_params={
                    "objective": "regression", "metric": "mae", "verbosity": -1, "boosting_type": "gbdt",
                    "n_estimators": 110, "learning_rate": 0.03, "num_leaves": 13,
                    "feature_fraction": 0.72, "bagging_fraction": 0.80, "bagging_freq": 1,
                    "min_data_in_leaf": 25, "lambda_l1": 0.05, "lambda_l2": 0.60, "random_state": 42
                }
            )
            pred_p_va_user_resid = user_branch["val_pred"] - 10.0
            pred_p_te_user_resid = user_branch["predict"](Xp_udle_te) - 10.0

            pred_p_va_user = np.maximum(base_va + pred_p_va_user_resid, 0.05)
            pred_p_te_user = np.maximum(base_te + pred_p_te_user_resid, 0.05)

            alpha_g, alpha_c, alpha_u, alpha_b, _, _ = choose_pt_fourway_blend(
                yva_p, pred_p_va_global, pred_p_va_core, pred_p_va_user, base_va, step=0.05
            )

            pred_p_te = (
                alpha_g * pred_p_te_global +
                alpha_c * pred_p_te_core +
                alpha_u * pred_p_te_user +
                alpha_b * base_te
            )

            user_q10 = tr.groupby("userID")["parking_time"].quantile(0.10)
            user_q90 = tr.groupby("userID")["parking_time"].quantile(0.90)
            station_q10 = tr.groupby("stationID")["parking_time"].quantile(0.10)
            station_q90 = tr.groupby("stationID")["parking_time"].quantile(0.90)
            low_te = te["userID"].map(user_q10).fillna(te["stationID"].map(station_q10)).fillna(np.quantile(y_p_tr, 0.10)).values
            high_te = te["userID"].map(user_q90).fillna(te["stationID"].map(station_q90)).fillna(np.quantile(y_p_tr, 0.90)).values
            low_te = np.maximum(low_te - 0.5, 0.05)
            high_te = np.maximum(high_te + 0.5, low_te + 0.5)

            pred_p_te = np.clip(pred_p_te, low_te, high_te)
            pred_p_te = np.clip(pred_p_te, 0.05, np.quantile(y_p_tr, 0.997))

            end_p = time.perf_counter()
            rounds_p = max(len(pt_val_maes), len(core_branch["val_maes"]), len(user_branch["val_maes"]))
            time_p_ms = (end_p - start_p) * 1000

            mae_p = mae(y_p_te, pred_p_te)
            sm_p = smape(y_p_te, pred_p_te)

            folder1 = os.path.join(".", "results", "norway_pt_dual_dle_v16", "parking_time", str(run))
            os.makedirs(folder1, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(1, len(core_branch["val_maes"]) + 1), core_branch["val_maes"], marker="o", linewidth=1)
            ax.set_title(f"Window={w} parking_time Val MAE vs Rounds")
            ax.set_xlabel("Round")
            ax.set_ylabel("Validation MAE")
            plt.tight_layout()
            plt.savefig(os.path.join(folder1, f"convergence_parking_time_window_{w}.png"))
            plt.close(fig)

            tr["pred_parking_time"] = model_p_global.predict(xgb.DMatrix(np.nan_to_num(Xp_core_tr)))
            te["pred_parking_time"] = pred_p_te_global

            tr["pred_parking_time"] = model_p_global.predict(xgb.DMatrix(np.nan_to_num(Xp_core_tr)))
            te["pred_parking_time"] = pred_p_te_global

            tr["pred_parking_time"] = model_p_global.predict(xgb.DMatrix(np.nan_to_num(Xp_core_tr)))
            te["pred_parking_time"] = pred_p_te_global

            pt_transform = PowerTransformer(method="box-cox", standardize=False)
            y_all_k = sanitize_target(tr["kWhDelivered"].values + eps, lower=eps)
            y_bc = pt_transform.fit_transform(y_all_k.reshape(-1, 1)).flatten()

            tr = tr.sort_values("connection_time_copy").reset_index(drop=True)
            te = te.sort_values("connection_time_copy").reset_index(drop=True)

            user_roll3 = tr.groupby("userID")["kWhDelivered"].apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
            tr["user_roll_kWh3"] = user_roll3.fillna(tr["kWhDelivered"].median())

            station_roll3 = tr.groupby("stationID")["kWhDelivered"].apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
            tr["station_roll_kWh3"] = station_roll3.fillna(tr["kWhDelivered"].median())

            user_hist = tr.groupby("userID")["kWhDelivered"].apply(list).to_dict()
            station_hist = tr.groupby("stationID")["kWhDelivered"].apply(list).to_dict()

            te["user_roll_kWh3"] = te["userID"].apply(lambda u: np.mean(user_hist[u][-3:]) if (u in user_hist and len(user_hist[u]) > 0) else tr["kWhDelivered"].median())
            te["station_roll_kWh3"] = te["stationID"].apply(lambda s: np.mean(station_hist[s][-3:]) if (s in station_hist and len(station_hist[s]) > 0) else tr["kWhDelivered"].median())

            med_k = tr["kWhDelivered"].median()
            tr["pk1"] = tr.groupby("userID")["kWhDelivered"].shift(1).fillna(med_k)
            tr["pk2"] = tr.groupby("userID")["kWhDelivered"].shift(2).fillna(med_k)
            tr["rk3"] = tr.groupby("userID")["kWhDelivered"].rolling(3, min_periods=1).mean().reset_index(0, drop=True).fillna(med_k)

            te["pk1"] = med_k
            te["pk2"] = med_k
            te["rk3"] = med_k
            prev_vals = []
            for i in range(len(te)):
                if i >= 1:
                    te.at[i, "pk1"] = prev_vals[i - 1]
                if i >= 2:
                    te.at[i, "pk2"] = prev_vals[i - 2]
                if i >= 3:
                    te.at[i, "rk3"] = np.mean(prev_vals[i - 3:i])
                prev_vals.append(te.at[i, "kWhDelivered"])

            feats_k = [
                "hour", "weekday", "month", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
                "cluster", "sim_kwh", "user_avg_kWh", "user_freq",
                "station_avg_kWh", "station_freq",
                "proxy_req_rate_mix", "hour_of_week",
                "sh_ratio_nor", "uh_ratio_nor", "shur_ratio_nor",
                "pk1", "pk2", "rk3", "station_roll_kWh3"
            ]
            Xk_tr_df, Xk_te_df = fill_feature_frame(tr, te, feats_k)
            X_all_tr_k = Xk_tr_df.values
            X_all_te_k = Xk_te_df.values

            Xtr_k = X_all_tr_k[:-n_val]
            Xva_k = X_all_tr_k[-n_val:]
            ytr_bc = y_bc[:-n_val]
            yva_bc = y_bc[-n_val:]
            wtr_k = w_decay.iloc[:-n_val].values

            dtrain_k = xgb.DMatrix(np.nan_to_num(Xtr_k), label=ytr_bc, weight=wtr_k)
            dval_k = xgb.DMatrix(np.nan_to_num(Xva_k), label=yva_bc)

            best_va_k = np.inf
            kwh_val_maes = []
            model_k = None
            rounds_no_improve_k = 0
            prev_depth_k = None
            prev_gamma_k = None
            start_k = time.perf_counter()

            for m in range(KWH_MAX_ROUNDS):
                if m < 15:
                    depth_k = 6
                    gamma_k = 1.0
                else:
                    if (m - 15) % 5 == 0:
                        if model_k is None:
                            residuals_k = ytr_bc.copy()
                        else:
                            residuals_k = ytr_bc - model_k.predict(xgb.DMatrix(np.nan_to_num(Xtr_k), label=ytr_bc, weight=wtr_k))

                        depth_k = compute_dynamic_max_depth_advanced(
                            residuals_k, wtr_k,
                            d0=6, delta_up=1.35, delta_down=1.1,
                            min_depth=3, max_depth_limit=12,
                            prev_depth=prev_depth_k, trim_ratio=0.06, smooth_factor=0.45
                        )
                        gamma_k = compute_dynamic_gamma_advanced(
                            residuals_k, wtr_k,
                            base=1.0, alpha=0.8,
                            min_gamma=0.02, max_gamma=6.0,
                            prev_gamma=prev_gamma_k, trim_ratio=0.06, smooth_factor=0.45
                        )
                        prev_depth_k = depth_k
                        prev_gamma_k = gamma_k

                if model_k is None:
                    residuals_k = ytr_bc.copy()
                else:
                    residuals_k = ytr_bc - model_k.predict(xgb.DMatrix(np.nan_to_num(Xtr_k), label=ytr_bc, weight=wtr_k))
                huber_alpha = compute_huber_alpha(residuals_k, scale=2.0)

                params_k = {
                    "tree_method": "hist",
                    "eta": KWH_FIXED_ETA,
                    "max_depth": depth_k,
                    "gamma": gamma_k,
                    "objective": "reg:pseudohubererror",
                    "alpha": huber_alpha,
                    "eval_metric": "mae",
                    "verbosity": 0
                }

                if model_k is None:
                    model_k = xgb.train(params_k, dtrain_k, num_boost_round=1, verbose_eval=False)
                else:
                    model_k = xgb.train(params_k, dtrain_k, num_boost_round=1, xgb_model=model_k, verbose_eval=False)

                pred_va_k_now = model_k.predict(dval_k)
                va_k_mae = mae(yva_bc, pred_va_k_now)
                kwh_val_maes.append(va_k_mae)

                if va_k_mae < best_va_k - 1e-6:
                    best_va_k = va_k_mae
                    rounds_no_improve_k = 0
                else:
                    rounds_no_improve_k += 1

                if rounds_no_improve_k >= KWH_EARLY_STOP:
                    break

            end_k = time.perf_counter()
            rounds_k = len(kwh_val_maes)
            time_k_ms = (end_k - start_k) * 1000

            pred_tr_k_full = model_k.predict(xgb.DMatrix(np.nan_to_num(Xtr_k), label=ytr_bc, weight=wtr_k))
            res_tr_bc = ytr_bc - pred_tr_k_full

            Xk_tr_df_lgb = pd.DataFrame(Xtr_k, columns=feats_k)
            X_all_te_df = pd.DataFrame(X_all_te_k, columns=feats_k)

            model_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
            model_lgb.fit(Xk_tr_df_lgb, res_tr_bc, sample_weight=wtr_k)

            pred_te_k_huber = model_k.predict(xgb.DMatrix(np.nan_to_num(X_all_te_k)))
            res_te_bc = model_lgb.predict(X_all_te_df)
            final_huber = pred_te_k_huber + res_te_bc

            clipped = np.maximum(final_huber.reshape(-1, 1), 1e-6)
            inv_te_k = pt_transform.inverse_transform(clipped).flatten() - eps

            order = np.argsort(te.connection_time_copy.values)
            smooth = medfilt(inv_te_k[order], kernel_size=5)
            final_k = np.empty_like(smooth)
            final_k[order] = smooth

            mae_k = mae(te["kWhDelivered"].values, final_k)
            sm_k = smape(te["kWhDelivered"].values, final_k)

            folder2 = os.path.join(".", "results", "norway_pt_dual_dle_v16", "kWhDelivered", str(run))
            os.makedirs(folder2, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(1, rounds_k + 1), kwh_val_maes, marker="o", linewidth=1)
            ax.set_title(f"Window={w} kWhDelivered Val MAE vs Rounds")
            ax.set_xlabel("Round")
            ax.set_ylabel("Validation MAE")
            plt.tight_layout()
            plt.savefig(os.path.join(folder2, f"convergence_kWh_time_window_{w}.png"))
            plt.close(fig)

            t_window_end = time.perf_counter()
            t_window_ms = (t_window_end - t_window_start) * 1000

            print(f"  Window={w}:")
            print(f"    parking_time → MAE={mae_p:.3f}, SMAPE={sm_p:.3f}%, rounds={rounds_p}, time={time_p_ms:.0f}ms")
            print(f"    kWhDelivered → MAE={mae_k:.3f}, SMAPE={sm_k:.3f}%, rounds={rounds_k}, time={time_k_ms:.0f}ms")

            run_summary.append((w, mae_p, sm_p, rounds_p, time_p_ms, mae_k, sm_k, rounds_k, time_k_ms, t_window_ms))
            metrics_all.append({
                "run": run, "window": w,
                "mae_pt": mae_p, "smape_pt": sm_p, "rounds_pt": rounds_p, "time_pt_ms": time_p_ms,
                "mae_kWh": mae_k, "smape_kWh": sm_k, "rounds_kWh": rounds_k, "time_kWh_ms": time_k_ms,
                "window_time_ms": t_window_ms
            })

        print(f"\nResults for run {run}:")
        print("window | MAE_pt | SMAPE_pt | rounds_pt | time_pt(ms) | MAE_kWh | SMAPE_kWh | rounds_kWh | time_kWh(ms)")
        for (w, mpt, spt, rp, tp, mk, sk, rk, tk, _) in run_summary:
            print(f"{w:>6} | {mpt:6.3f} | {spt:7.3f}% | {rp:9d} | {tp:10.0f} | {mk:7.3f} | {sk:9.3f}% | {rk:10d} | {tk:11.0f}")

    result_root = os.path.join(".", "results", "norway_pt_dual_dle_v16")
    os.makedirs(result_root, exist_ok=True)
    pd.DataFrame(metrics_all).to_csv(os.path.join(result_root, "metrics_mixed_dynamic.csv"), index=False)
    print("\nAll runs complete. Results saved to metrics_mixed_dynamic.csv")


if __name__ == "__main__":
    main()
