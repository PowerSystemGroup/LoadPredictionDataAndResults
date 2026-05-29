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


def _ordered_unique(cols):
    out = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out


def get_original_parking_feature_columns():
    pf_core = [
        "hour", "weekday", "month", "is_weekend", "is_holiday",
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
    return _ordered_unique(pf_core + pf_core_dle + pf_user_resid)


def get_original_kwh_feature_columns():
    return [
        "hour", "weekday", "month", "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
        "cluster", "sim_kwh", "user_avg_kWh", "user_freq",
        "station_avg_kWh", "station_freq",
        "proxy_req_rate_mix", "hour_of_week",
        "sh_ratio_nor", "uh_ratio_nor", "shur_ratio_nor",
        "pk1", "pk2", "rk3", "station_roll_kWh3"
    ]


def build_original_kwh_lag_features(tr, te):
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
    return tr, te


def apply_original_parking_clipping(tr, te, y_p_tr, pred_p_te):
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
    return pred_p_te


def apply_original_kwh_smoothing(te, pred_k):
    pred_k = np.asarray(pred_k, dtype=float)
    pred_k = np.where(np.isfinite(pred_k), pred_k, 0.0)
    return np.maximum(pred_k, 0.0)


def save_prediction_detail(folder, window, te, pred_p, pred_k):
    detail = pd.DataFrame({
        "connection_time_copy": te["connection_time_copy"].values,
        "userID": te["userID"].values,
        "stationID": te["stationID"].values,
        "true_parking_time": te["parking_time"].values,
        "pred_parking_time": pred_p,
        "true_kWhDelivered": te["kWhDelivered"].values,
        "pred_kWhDelivered": pred_k
    })
    detail.to_csv(os.path.join(folder, f"prediction_detail_window_{window}.csv"), index=False)


def plot_one_curve(folder, filename, values, title, xlabel="Round", ylabel="Validation MAE"):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        values = np.array([np.nan])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, filename))
    plt.close(fig)


def make_padded_sequences(X, seq_len):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    seq = np.zeros((n, seq_len, d), dtype=float)
    for i in range(n):
        s = max(0, i - seq_len + 1)
        block = X[s:i + 1]
        if len(block) < seq_len:
            pad = np.repeat(block[:1], seq_len - len(block), axis=0)
            block = np.vstack([pad, block])
        seq[i] = block
    return seq


def make_test_sequences_from_train_test(X_train_scaled, X_test_scaled, seq_len):
    all_x = np.vstack([X_train_scaled, X_test_scaled])
    n_train = len(X_train_scaled)
    d = all_x.shape[1]
    seq = np.zeros((len(X_test_scaled), seq_len, d), dtype=float)
    for j in range(len(X_test_scaled)):
        i = n_train + j
        s = max(0, i - seq_len + 1)
        block = all_x[s:i + 1]
        if len(block) < seq_len:
            pad = np.repeat(block[:1], seq_len - len(block), axis=0)
            block = np.vstack([pad, block])
        seq[j] = block
    return seq


def fit_predict_recurrent_single_fixed(X_train_df, X_test_df, y_train, n_val, seed, model_type="LSTM",
                                       seq_len=12, max_epochs=50, batch_size=32,
                                       lower=0.0, upper=None):
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers
    except Exception as e:
        raise ImportError("This baseline needs TensorFlow. Please run: pip install tensorflow") from e

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    X_train = np.asarray(X_train_df.values, dtype=float)
    X_test = np.asarray(X_test_df.values, dtype=float)
    y_train = sanitize_target(y_train, lower=lower, upper=upper).reshape(-1, 1)

    X_fit_raw = X_train[:-n_val]
    y_fit_raw = y_train[:-n_val]

    x_scaler = StandardScaler().fit(X_fit_raw)
    X_train_scaled = x_scaler.transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = StandardScaler().fit(y_fit_raw)
    y_scaled = y_scaler.transform(y_train)

    X_seq = make_padded_sequences(X_train_scaled, seq_len)
    X_test_seq = make_test_sequences_from_train_test(X_train_scaled, X_test_scaled, seq_len)

    X_fit, X_val = X_seq[:-n_val], X_seq[-n_val:]
    y_fit, y_val = y_scaled[:-n_val], y_scaled[-n_val:]

    recurrent_layer = layers.LSTM if model_type.upper() == "LSTM" else layers.GRU

    inp = layers.Input(shape=(seq_len, X_seq.shape[-1]))
    x = recurrent_layer(32, return_sequences=False)(inp)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, name="target_output")(x)
    model = models.Model(inp, out, name=f"{model_type.upper()}_Fixed_Baseline")
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss="mae")

    hist = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        verbose=0,
        shuffle=False
    )

    pred_test_scaled = model.predict(X_test_seq, verbose=0)
    pred_val_scaled = model.predict(X_val, verbose=0)
    pred_test = y_scaler.inverse_transform(pred_test_scaled).flatten()
    pred_val = y_scaler.inverse_transform(pred_val_scaled).flatten()

    pred_test = np.where(np.isfinite(pred_test), pred_test, np.median(y_fit_raw),)
    pred_val = np.where(np.isfinite(pred_val), pred_val, np.median(y_fit_raw),)
    pred_test = np.maximum(pred_test, lower)
    pred_val = np.maximum(pred_val, lower)
    if upper is not None:
        pred_test = np.minimum(pred_test, upper)
        pred_val = np.minimum(pred_val, upper)

    y_val_true = sanitize_target(y_train[-n_val:].flatten(), lower=lower, upper=upper)
    val_curve = [mae(y_val_true, pred_val)]
    return pred_test, pred_val, val_curve, max_epochs


def main():
    os.makedirs("./results", exist_ok=True)
    metrics_all = []

    ROBUST_TYPE = "missing"
    ROBUST_RATIO = 5
    ROBUST_DATA_DIR = "./norway_robust_datasets"
    EXPERIMENT_NAME = f"norway_{ROBUST_TYPE}_{ROBUST_RATIO}_wavePT_trainMedianFill_keepLOF_woMedian"

    def find_window_dataset(window):
        filename = f"norway_{ROBUST_TYPE}_w{window}_{ROBUST_RATIO}.csv"
        candidates = [
            os.path.join(ROBUST_DATA_DIR, filename),
            filename,
            os.path.join("/kaggle/working", "norway_robust_datasets", filename),
            os.path.join("/kaggle/input", filename),
            os.path.join("/mnt/data", "norway_robust_datasets", filename),
            os.path.join("/mnt/data", filename),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

        for root in ["/kaggle/input", "/mnt/data", "."]:
            if not os.path.exists(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                if filename in filenames:
                    return os.path.join(dirpath, filename)

        raise FileNotFoundError(
            f"{filename} was not found. Please run the Norway Robust Data Generator first,"
            f"Place the files in {ROBUST_DATA_DIR}, the current directory, or the Kaggle input."
        )

    def load_window_robust_data(window):
        data_file = find_window_dataset(window)
        dfw = pd.read_csv(data_file, parse_dates=["connection_time_copy"])

        required_cols = ["connection_time_copy", "parking_time", "kWhDelivered", "userID", "stationID"]
        missing_cols = [c for c in required_cols if c not in dfw.columns]
        if missing_cols:
            raise ValueError(f"{data_file} is missing the required fields: {missing_cols}")

        for c in ["parking_time", "kWhDelivered", "userID", "stationID"]:
            dfw[c] = pd.to_numeric(dfw[c], errors="coerce")

        before = len(dfw)
        dfw = dfw.replace([np.inf, -np.inf], np.nan)

        train_start_for_fill = test_start - timedelta(days=window)
        train_mask_for_fill = (
            (dfw["connection_time_copy"] >= train_start_for_fill) &
            (dfw["connection_time_copy"] < test_start)
        )

        pt_median = dfw.loc[train_mask_for_fill, "parking_time"].dropna().median()
        kwh_median = dfw.loc[train_mask_for_fill, "kWhDelivered"].dropna().median()

        if not np.isfinite(pt_median):
            pt_median = dfw["parking_time"].dropna().median()
        if not np.isfinite(kwh_median):
            kwh_median = dfw["kWhDelivered"].dropna().median()
        if not np.isfinite(pt_median):
            pt_median = 1.0
        if not np.isfinite(kwh_median):
            kwh_median = 1.0

        missing_pt_before = int(dfw["parking_time"].isna().sum())
        missing_kwh_before = int(dfw["kWhDelivered"].isna().sum())

        dfw["parking_time"] = dfw["parking_time"].fillna(float(pt_median))
        dfw["kWhDelivered"] = dfw["kWhDelivered"].fillna(float(kwh_median))

        id_time_cols = ["connection_time_copy", "userID", "stationID"]
        dfw = dfw.dropna(subset=id_time_cols).copy()

        dfw = dfw[(dfw["parking_time"] > 0) & (dfw["kWhDelivered"] >= 0)].copy()
        dfw.sort_values("connection_time_copy", inplace=True)
        dfw.reset_index(drop=True, inplace=True)
        dfw = add_static_columns(dfw)

        return dfw, data_file

    test_start = datetime(2019, 12, 1)
    test_end = datetime(2019, 12, 30)
    windows = [30, 60, 120, 240, 360, 480]
    n_runs = 2

    result_root = os.path.join(".", "results", f"{EXPERIMENT_NAME}_norway12_lstm_fixed_only_model")
    base_dirs = [
        os.path.join(result_root, "parking_time"),
        os.path.join(result_root, "kWhDelivered"),
    ]

    for run in range(1, n_runs + 1):
        run_seed = 1000 + run
        for base in base_dirs:
            os.makedirs(os.path.join(base, str(run)), exist_ok=True)

        print(f"\n=== Run {run} ===")
        run_summary = []

        for w in windows:
            print(f"\n>>> Window = {w} days")
            t_window_start = time.perf_counter()

            dfw, data_file = load_window_robust_data(w)
            df = add_cluster(dfw.copy(), test_start, seed=run_seed)

            tr0 = test_start - timedelta(days=w)
            tr = df[(df.connection_time_copy >= tr0) & (df.connection_time_copy < test_start)].copy()
            te = df[(df.connection_time_copy >= test_start) & (df.connection_time_copy <= test_end)].copy()
            if tr.empty or te.empty:
                continue

            tr["parking_time"] = safe_wavelet_denoise(tr["parking_time"].values, lower=0.05)
            tr["kWhDelivered"] = safe_wavelet_denoise(tr["kWhDelivered"].values, lower=0.0)

            if len(tr) >= 25:
                lof_neighbors = min(20, len(tr) - 1)
                if lof_neighbors >= 2:
                    mask = LocalOutlierFactor(n_neighbors=lof_neighbors).fit_predict(tr[["parking_time", "kWhDelivered"]])
                    tr = tr[mask == 1].reset_index(drop=True)

            te["parking_time"] = safe_wavelet_denoise(te["parking_time"].values, lower=0.05)
            te["kWhDelivered"] = safe_wavelet_denoise(te["kWhDelivered"].values, lower=0.0)

            tr["age_days"] = (test_start - tr.connection_time_copy).dt.days
            w_decay = np.exp(-0.014 * tr["age_days"])

            tr.reset_index(drop=True, inplace=True)
            te.reset_index(drop=True, inplace=True)

            tr, te = add_history_stats(tr, te)
            tr, te = add_similarity_features(tr, te)

            pf_pt = get_original_parking_feature_columns()
            Xp_tr_df, Xp_te_df = fill_feature_frame(tr, te, pf_pt)
            y_p_tr = sanitize_target(tr["parking_time"].values, lower=0.05, upper=24.0)
            y_p_te = sanitize_target(te["parking_time"].values, lower=0.05, upper=24.0)

            n_val = max(int(0.2 * len(tr)), 50)
            if len(tr) <= n_val + 5:
                n_val = max(1, len(tr) // 3)

            start_p = time.perf_counter()
            pred_p_te_raw, pred_p_va, val_curve_p, rounds_p = fit_predict_recurrent_single_fixed(Xp_tr_df, Xp_te_df, y_p_tr, n_val, seed=run_seed, model_type="LSTM", seq_len=12, max_epochs=50, batch_size=32, lower=0.05, upper=24.0)
            pred_p_te = apply_original_parking_clipping(tr, te, y_p_tr, pred_p_te_raw)
            end_p = time.perf_counter()
            time_p_ms = (end_p - start_p) * 1000

            mae_p = mae(y_p_te, pred_p_te)
            sm_p = smape(y_p_te, pred_p_te)

            tr["pred_parking_time"] = np.nan
            te["pred_parking_time"] = pred_p_te

            tr, te = build_original_kwh_lag_features(tr, te)

            feats_k = get_original_kwh_feature_columns()
            Xk_tr_df, Xk_te_df = fill_feature_frame(tr, te, feats_k)
            y_k_tr = sanitize_target(tr["kWhDelivered"].values, lower=0.0)
            y_k_te = sanitize_target(te["kWhDelivered"].values, lower=0.0)

            start_k = time.perf_counter()
            pred_k_te_raw, pred_k_va, val_curve_k, rounds_k = fit_predict_recurrent_single_fixed(Xk_tr_df, Xk_te_df, y_k_tr, n_val, seed=run_seed + 77, model_type="LSTM", seq_len=12, max_epochs=50, batch_size=32, lower=0.0, upper=None)
            final_k = apply_original_kwh_smoothing(te, pred_k_te_raw)
            end_k = time.perf_counter()
            time_k_ms = (end_k - start_k) * 1000

            mae_k = mae(y_k_te, final_k)
            sm_k = smape(y_k_te, final_k)

            folder1 = os.path.join(result_root, "parking_time", str(run))
            folder2 = os.path.join(result_root, "kWhDelivered", str(run))
            detail_folder = os.path.join(result_root, "prediction_detail", str(run))
            os.makedirs(folder1, exist_ok=True)
            os.makedirs(folder2, exist_ok=True)
            os.makedirs(detail_folder, exist_ok=True)

            plot_one_curve(folder1, f"convergence_parking_time_window_{w}.png", val_curve_p,
                           f"Window={w} parking_time LSTM Validation MAE")
            plot_one_curve(folder2, f"convergence_kWh_time_window_{w}.png", val_curve_k,
                           f"Window={w} kWhDelivered LSTM Validation MAE")
            save_prediction_detail(detail_folder, w, te, pred_p_te, final_k)

            t_window_end = time.perf_counter()
            t_window_ms = (t_window_end - t_window_start) * 1000

            print(f"  Window={w}:")
            print(f"    parking_time → MAE={mae_p:.3f}, SMAPE={sm_p:.3f}%, rounds={rounds_p}, time={time_p_ms:.0f}ms")
            print(f"    kWhDelivered → MAE={mae_k:.3f}, SMAPE={sm_k:.3f}%, rounds={rounds_k}, time={time_k_ms:.0f}ms")

            run_summary.append((w, mae_p, sm_p, rounds_p, time_p_ms, mae_k, sm_k, rounds_k, time_k_ms, t_window_ms))
            metrics_all.append({
                "robust_type": ROBUST_TYPE, "robust_ratio": ROBUST_RATIO, "data_file": data_file,
                "run": run, "window": w,
                "mae_pt": mae_p, "smape_pt": sm_p, "rounds_pt": rounds_p, "time_pt_ms": time_p_ms,
                "mae_kWh": mae_k, "smape_kWh": sm_k, "rounds_kWh": rounds_k, "time_kWh_ms": time_k_ms,
                "window_time_ms": t_window_ms
            })

        print(f"\nResults for run {run}:")
        print("window | MAE_pt | SMAPE_pt | rounds_pt | time_pt(ms) | MAE_kWh | SMAPE_kWh | rounds_kWh | time_kWh(ms)")
        for (w, mpt, spt, rp, tp, mk, sk, rk, tk, _) in run_summary:
            print(f"{w:>6} | {mpt:6.3f} | {spt:7.3f}% | {rp:9d} | {tp:10.0f} | {mk:7.3f} | {sk:9.3f}% | {rk:10d} | {tk:11.0f}")

    os.makedirs(result_root, exist_ok=True)
    pd.DataFrame(metrics_all).to_csv(os.path.join(result_root, "metrics_lstm_fixed_only_model.csv"), index=False)
    print("\nAll runs complete. Results saved to metrics_lstm_fixed_only_model.csv")


if __name__ == "__main__":
    main()
