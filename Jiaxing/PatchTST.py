import os
import sys
import time
import math
import random
import warnings
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger().setLevel(logging.ERROR)

try:
    import torch
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None
    nn = None

MODEL_NAME = "PatchTST"
MODEL_TAG = MODEL_NAME.lower().replace("transformer", "transformer").replace(" ", "_")

INPUT_CANDIDATES = [
    "Charging_Data_JX_clustered_cleaned_final.csv",
    "Charging_Data_JX_clustered_cleaned_final(1).csv",
    "/mnt/data/Charging_Data_JX_clustered_cleaned_final(1).csv",
]

if os.path.exists("/kaggle/working"):
    OUTPUT_DIR = f"/kaggle/working/jx_{MODEL_TAG}_fixed_noanchor_results"
else:
    OUTPUT_DIR = f"./jx_{MODEL_TAG}_fixed_noanchor_results"

OUTPUT_XLSX = os.path.join(OUTPUT_DIR, f"jx_{MODEL_TAG}_fixed_noanchor_dual_target_metrics.xlsx")
OUTPUT_PRED_CSV = os.path.join(OUTPUT_DIR, f"jx_{MODEL_TAG}_fixed_noanchor_predictions.csv")
OUTPUT_FILTER_CSV = os.path.join(OUTPUT_DIR, f"jx_{MODEL_TAG}_fixed_noanchor_quality_filter_summary.csv")
OUTPUT_STATION_CSV = os.path.join(OUTPUT_DIR, f"jx_{MODEL_TAG}_fixed_noanchor_station_quality_ranking.csv")
OUTPUT_CLEANED_CSV = os.path.join(OUTPUT_DIR, f"jx_{MODEL_TAG}_fixed_noanchor_modeling_data_after_quality_filter.csv")

TEST_START = datetime(2021, 12, 1)
TEST_END = datetime(2021, 12, 30, 23, 59, 59)
WINDOWS = [30, 60, 120, 240, 360, 480]
N_RUNS = 1

CASE_MODE = "strict"
USE_STATION_SELECTION = True
USE_TARGET_QUALITY_FILTER = True
USE_LOF_TRAIN_DENOISE = True
USE_WEATHER_FEATURES = True
USE_ANCHOR_BLEND = False
USE_ONLINE_TEST_HISTORY = False
LOF_CONTAMINATION = 0.05

STRICT_TOP_STATIONS = 20
BALANCED_TOP_STATIONS = 40
SINGLE_STATION_IDS = []
STRICT_KWH_Q = (0.15, 0.85)
STRICT_PARK_Q = (0.05, 0.95)
BALANCED_KWH_Q = (0.10, 0.90)
BALANCED_PARK_Q = (0.03, 0.97)

MIN_STATION_CAL_SAMPLES = 200
MIN_STATION_TEST_SAMPLES = 50
MIN_CLUSTER_TRAIN = 30
MIN_CLUSTER_TEST = 20

TORCH_MAX_EPOCHS = 70
TORCH_PATIENCE = 10
TORCH_BATCH_SIZE = 256
TORCH_LR = 1e-3
TORCH_WEIGHT_DECAY = 1e-4
TORCH_HIDDEN = 64
TORCH_D_MODEL = 64
TORCH_DROPOUT = 0.10
USE_TORCH_EARLY_STOPPING = False

ARIMA_ORDER = (1, 1, 1)
ARIMA_MAX_TRAIN_POINTS = 1200


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return np.nan
    return float(100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6)))


def inverse_log1p(x):
    return np.maximum(np.expm1(np.asarray(x, dtype=float)), 0.0)


def clean_matrix(X):
    X = np.asarray(X, dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def safe_div(a, b, default=0.0):
    b = float(b)
    if abs(b) < 1e-12:
        return default
    return float(a) / b


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def find_input_file():
    for p in INPUT_CANDIDATES:
        if os.path.exists(p):
            return p
    for root in ["/kaggle/input", "/mnt/data", "."]:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                if fname in ["Charging_Data_JX_clustered_cleaned_final.csv", "Charging_Data_JX_clustered_cleaned_final(1).csv"]:
                    return os.path.join(dirpath, fname)
    raise FileNotFoundError("The file Charging_Data_JX_clustered_cleaned_final.csv was not found. Please place it in the current directory, /kaggle/input, or /mnt/data.")


def select_blend_weight(y_true, pred_a, pred_b, step=0.02):
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    best_w = 1.0
    best_mae = np.inf
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        pred = w * pred_a + (1.0 - w) * pred_b
        cur = mae(y_true, pred)
        if cur < best_mae:
            best_mae = cur
            best_w = float(w)
    return best_w, best_mae


def select_blend_weights_greedy(y_true, pred_list, step=0.05, n_iter=4):
    y_true = np.asarray(y_true, dtype=float)
    P = [np.asarray(p, dtype=float) for p in pred_list]
    m = len(P)
    single_scores = [mae(y_true, p) for p in P]
    best_idx = int(np.nanargmin(single_scores))
    weights = np.zeros(m, dtype=float)
    weights[best_idx] = 1.0
    cur_pred = P[best_idx].copy()
    cur_mae = single_scores[best_idx]

    for _ in range(n_iter):
        improved = False
        for j in range(m):
            best_local = (0.0, cur_mae, cur_pred)
            for a in np.arange(0.0, 1.0 + 1e-9, step):
                cand_pred = (1.0 - a) * cur_pred + a * P[j]
                cand_mae = mae(y_true, cand_pred)
                if cand_mae < best_local[1] - 1e-9:
                    best_local = (float(a), cand_mae, cand_pred)
            a, new_mae, new_pred = best_local
            if new_mae < cur_mae - 1e-9:
                weights = (1.0 - a) * weights
                weights[j] += a
                cur_mae = new_mae
                cur_pred = new_pred
                improved = True
        if not improved:
            break
    weights = weights / max(weights.sum(), 1e-12)
    return weights, cur_mae


def load_raw_data():
    input_path = find_input_file()
    df = pd.read_csv(input_path, encoding="utf-8-sig")

    time_col = "start_time" if "start_time" in df.columns else "connection_time_copy"
    end_col = "end_time"
    if time_col not in df.columns:
        raise ValueError("no start_time or connection_time_copy rows")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if end_col in df.columns:
        df[end_col] = pd.to_datetime(df[end_col], errors="coerce")
    df["connection_time_copy"] = df[time_col]

    if "parking_time" not in df.columns:
        if end_col not in df.columns:
            raise ValueError("no parking_time, and no end_time")
        df["parking_time"] = (df[end_col] - df["connection_time_copy"]).dt.total_seconds() / 3600.0

    if "kWhDelivered" not in df.columns:
        raise ValueError("no kWhDelivered ")

    if "cluster_id" not in df.columns:
        raise ValueError("no cluster_id")
    if "scenario_name" not in df.columns:
        df["scenario_name"] = df["cluster_id"].apply(lambda x: f"cluster_{x}")

    for c in ["parking_time", "kWhDelivered", "cluster_id", "userID", "stationID"]:
        if c not in df.columns:
            raise ValueError(f"The input file is missing required fields: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["scenario_name_raw"] = df["scenario_name"].astype(str)
    df = df.dropna(subset=["connection_time_copy", "parking_time", "kWhDelivered", "cluster_id", "userID", "stationID"]).copy()

    before = len(df)
    df = df[(df["parking_time"] > 0) & (df["parking_time"] <= 24) & (df["kWhDelivered"] > 0) & (df["kWhDelivered"] <= 120)].copy()
    df["cluster_id"] = df["cluster_id"].astype(int)
    print(f"Basic Physical Cleaning: {before} -> {len(df)}, delete {before - len(df)}")

    return df.sort_values("connection_time_copy").reset_index(drop=True), input_path


def add_calendar_features(df):
    df = df.copy()
    t = df["connection_time_copy"]
    df["hour"] = t.dt.hour
    df["weekday"] = t.dt.weekday
    df["month"] = t.dt.month
    df["day"] = t.dt.day
    df["dayofyear"] = t.dt.dayofyear
    df["weekofyear"] = t.dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["dayofyear_sin"] = np.sin(2 * np.pi * (df["dayofyear"] - 1) / 366.0)
    df["dayofyear_cos"] = np.cos(2 * np.pi * (df["dayofyear"] - 1) / 366.0)
    df["hour_of_week"] = df["weekday"] * 24 + df["hour"]
    df["is_holiday"] = 0

    for c in ["temperature", "humidity", "precipitation"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].median()
            df[c] = df[c].fillna(med)

    df["charge_rate"] = df["kWhDelivered"] / np.maximum(df["parking_time"], 5 / 60.0)
    df["charge_rate"] = df["charge_rate"].clip(lower=0, upper=df["charge_rate"].quantile(0.995))
    return df


def encode_categories(df):
    df = df.copy()
    for c in ["userID", "stationID", "scenario_name"]:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
    return df


def build_station_quality_table(df):
    cal_start = TEST_START - timedelta(days=max(WINDOWS))
    cal = df[(df["connection_time_copy"] >= cal_start) & (df["connection_time_copy"] < TEST_START)].copy()
    test = df[(df["connection_time_copy"] >= TEST_START) & (df["connection_time_copy"] <= TEST_END)].copy()

    rows = []
    for sid, g in cal.groupby("stationID"):
        gt = test[test["stationID"] == sid]
        if len(g) == 0:
            continue
        k_med = g["kWhDelivered"].median()
        p_med = g["parking_time"].median()
        k_iqr = g["kWhDelivered"].quantile(0.75) - g["kWhDelivered"].quantile(0.25)
        p_iqr = g["parking_time"].quantile(0.75) - g["parking_time"].quantile(0.25)
        k_cv = safe_div(g["kWhDelivered"].std(), g["kWhDelivered"].mean(), default=9.99)
        p_cv = safe_div(g["parking_time"].std(), g["parking_time"].mean(), default=9.99)
        score = 0.50 * k_cv + 0.35 * p_cv + 0.15 * safe_div(k_iqr, k_med + 1e-6, default=9.99) - 0.03 * np.log1p(len(g))
        rows.append({
            "stationID": int(sid),
            "cluster_id": int(g["cluster_id"].mode().iloc[0]),
            "cal_samples": int(len(g)),
            "test_samples": int(len(gt)),
            "kWh_mean": float(g["kWhDelivered"].mean()),
            "kWh_median": float(k_med),
            "kWh_std": float(g["kWhDelivered"].std()),
            "kWh_iqr": float(k_iqr),
            "kWh_cv": float(k_cv),
            "park_mean": float(g["parking_time"].mean()),
            "park_median": float(p_med),
            "park_std": float(g["parking_time"].std()),
            "park_iqr": float(p_iqr),
            "park_cv": float(p_cv),
            "quality_score": float(score),
        })
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise RuntimeError("The site quality table cannot be generated. Please check if there is any historical data prior to the test date.")
    return tab.sort_values(["quality_score", "cal_samples"], ascending=[True, False]).reset_index(drop=True)


def get_quality_config():
    mode = CASE_MODE.lower().strip()
    if mode == "full":
        return {"top_stations": None, "kwh_q": None, "park_q": None}
    if mode == "balanced":
        return {"top_stations": BALANCED_TOP_STATIONS, "kwh_q": BALANCED_KWH_Q, "park_q": BALANCED_PARK_Q}
    if mode == "single":
        return {"top_stations": 1, "kwh_q": STRICT_KWH_Q, "park_q": STRICT_PARK_Q}
    return {"top_stations": STRICT_TOP_STATIONS, "kwh_q": STRICT_KWH_Q, "park_q": STRICT_PARK_Q}


def apply_quality_filter(df, station_quality):
    cfg = get_quality_config()
    filter_rows = []
    selected_stations = sorted(df["stationID"].dropna().unique().astype(int).tolist())

    if USE_STATION_SELECTION and cfg["top_stations"] is not None:
        candidates = station_quality[
            (station_quality["cal_samples"] >= MIN_STATION_CAL_SAMPLES)
            & (station_quality["test_samples"] >= MIN_STATION_TEST_SAMPLES)
        ].copy()
        if candidates.empty:
            print("Warning: No sites meet the minimum sample size requirement; the process is being terminated without site screening.")
        else:
            if CASE_MODE.lower().strip() == "single":
                if SINGLE_STATION_IDS:
                    selected_stations = [int(x) for x in SINGLE_STATION_IDS if int(x) in candidates["stationID"].tolist()]
                    if not selected_stations:
                        selected_stations = [int(candidates.iloc[0]["stationID"])]
                else:
                    selected_stations = [int(candidates.iloc[0]["stationID"])]
            else:
                selected_stations = candidates.head(int(cfg["top_stations"]))["stationID"].astype(int).tolist()

            before = len(df)
            df = df[df["stationID"].isin(selected_stations)].copy()
            filter_rows.append({
                "step": "station_selection",
                "rule": f"CASE_MODE={CASE_MODE}, selected_stations={selected_stations}",
                "before": before,
                "after": len(df),
                "removed": before - len(df),
            })

    if USE_TARGET_QUALITY_FILTER and cfg["kwh_q"] is not None:
        cal_start = TEST_START - timedelta(days=max(WINDOWS))
        cal = df[(df["connection_time_copy"] >= cal_start) & (df["connection_time_copy"] < TEST_START)].copy()
        before = len(df)
        keep = pd.Series(True, index=df.index)
        rules = []

        for sid, g in cal.groupby("stationID"):
            idx_all = df[df["stationID"] == sid].index
            if len(g) < 50:
                continue
            k_lo, k_hi = g["kWhDelivered"].quantile(cfg["kwh_q"][0]), g["kWhDelivered"].quantile(cfg["kwh_q"][1])
            p_lo, p_hi = g["parking_time"].quantile(cfg["park_q"][0]), g["parking_time"].quantile(cfg["park_q"][1])
            p_lo = max(0.02, p_lo)
            keep.loc[idx_all] &= df.loc[idx_all, "kWhDelivered"].between(k_lo, k_hi)
            keep.loc[idx_all] &= df.loc[idx_all, "parking_time"].between(p_lo, p_hi)
            rules.append(f"station {sid}: kWh[{k_lo:.2f},{k_hi:.2f}], park[{p_lo:.2f},{p_hi:.2f}]")

        df = df[keep].copy()
        filter_rows.append({
            "step": "target_quality_filter",
            "rule": f"per-station train-period quantile; kWh_q={cfg['kwh_q']}, park_q={cfg['park_q']}",
            "before": before,
            "after": len(df),
            "removed": before - len(df),
        })

    filter_summary = pd.DataFrame(filter_rows)
    return df.reset_index(drop=True), selected_stations, filter_summary


def count_in_period(df, start_time, end_time, include_end=False):
    if include_end:
        return int(((df["connection_time_copy"] >= start_time) & (df["connection_time_copy"] <= end_time)).sum())
    return int(((df["connection_time_copy"] >= start_time) & (df["connection_time_copy"] < end_time)).sum())


def filter_clusters_before_modeling(df):
    interval_specs = [
        ("0_30d", 0, 30),
        ("30_60d", 30, 60),
        ("60_120d", 60, 120),
        ("120_240d", 120, 240),
        ("240_360d", 240, 360),
        ("360_480d", 360, 480),
    ]
    rows = []
    kept_clusters = []

    for cid in sorted(df["cluster_id"].dropna().astype(int).unique().tolist()):
        g = df[df["cluster_id"] == cid].copy()
        scenario = str(g["scenario_name_raw"].iloc[0]) if len(g) else ""
        row = {
            "cluster_id": cid,
            "scenario_name": scenario,
            "total_samples": int(len(g)),
            "date_min": g["connection_time_copy"].min(),
            "date_max": g["connection_time_copy"].max(),
        }
        reasons = []
        test_count = count_in_period(g, TEST_START, TEST_END, include_end=True)
        row["test_20211201_20211230"] = test_count
        if test_count == 0:
            reasons.append("No data available for the test range")

        for label, newer_days, older_days in interval_specs:
            start = TEST_START - timedelta(days=older_days)
            end = TEST_START - timedelta(days=newer_days)
            cnt = count_in_period(g, start, end, include_end=False)
            row[f"train_{label}"] = cnt
            if cnt == 0:
                reasons.append(f"Training segment {label} has no data")

        if len(reasons) == 0:
            row["keep"] = 1
            row["drop_reason"] = ""
            kept_clusters.append(cid)
        else:
            row["keep"] = 0
            row["drop_reason"] = "; ".join(reasons)
        rows.append(row)

    summary = pd.DataFrame(rows)
    filtered_df = df[df["cluster_id"].isin(kept_clusters)].copy().reset_index(drop=True)

    print("\n========== Category filtering results before modeling ==========")
    print(f"CASE_MODE={CASE_MODE}, retained clusters={kept_clusters}")
    show_cols = [c for c in [
        "cluster_id", "total_samples", "test_20211201_20211230", "train_0_30d",
        "train_30_60d", "train_60_120d", "train_120_240d", "train_240_360d",
        "train_360_480d", "keep", "drop_reason"
    ] if c in summary.columns]
    print(summary[show_cols].to_string(index=False))
    print("========================================\n")
    return filtered_df, kept_clusters, summary


def load_filtered_clustered_data():
    df, input_path = load_raw_data()
    df = add_calendar_features(df)

    station_quality_raw = build_station_quality_table(df)
    print(station_quality_raw.head(15).to_string(index=False))

    df, selected_stations_raw, quality_filter_summary = apply_quality_filter(df, station_quality_raw)
    df.to_csv(OUTPUT_CLEANED_CSV, index=False, encoding="utf-8-sig")

    df["stationID_raw"] = df["stationID"].astype(int)
    df["userID_raw"] = df["userID"].astype(int)
    df = encode_categories(df)

    station_quality = station_quality_raw.copy()
    station_quality["selected_before_encoding"] = station_quality["stationID"].isin(selected_stations_raw).astype(int)

    df = df.sort_values("connection_time_copy").reset_index(drop=True)
    return df, input_path, station_quality, quality_filter_summary



def denoise_train_only(tr):
    tr = tr.copy()
    if not USE_LOF_TRAIN_DENOISE:
        return tr
    if len(tr) > 80:
        n_neighbors = min(25, len(tr) - 1)
        cols = ["parking_time", "kWhDelivered", "charge_rate"]
        mask = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=LOF_CONTAMINATION).fit_predict(tr[cols])
        tr = tr[mask == 1].reset_index(drop=True)
    return tr


def map_from_group(D, stat_series, keys, fill_value=np.nan):
    idx = D.set_index(keys).index
    out = pd.Series(idx.map(stat_series), index=D.index, dtype=float)
    if fill_value is not None:
        out = out.fillna(fill_value)
    return out


def add_group_history_features(tr, te):
    tr = tr.copy()
    te = te.copy()

    global_k = float(tr["kWhDelivered"].mean())
    global_p = float(tr["parking_time"].mean())
    global_r = float(tr["charge_rate"].mean())

    user_stats = tr.groupby("userID").agg(
        user_avg_kWh=("kWhDelivered", "mean"),
        user_med_kWh=("kWhDelivered", "median"),
        user_q25_kWh=("kWhDelivered", lambda x: x.quantile(0.25)),
        user_q75_kWh=("kWhDelivered", lambda x: x.quantile(0.75)),
        user_avg_park=("parking_time", "mean"),
        user_med_park=("parking_time", "median"),
        user_q25_park=("parking_time", lambda x: x.quantile(0.25)),
        user_q75_park=("parking_time", lambda x: x.quantile(0.75)),
        user_avg_rate=("charge_rate", "mean"),
        user_med_rate=("charge_rate", "median"),
        user_freq=("kWhDelivered", "size"),
    )
    station_stats = tr.groupby("stationID").agg(
        station_avg_kWh=("kWhDelivered", "mean"),
        station_med_kWh=("kWhDelivered", "median"),
        station_q25_kWh=("kWhDelivered", lambda x: x.quantile(0.25)),
        station_q75_kWh=("kWhDelivered", lambda x: x.quantile(0.75)),
        station_avg_park=("parking_time", "mean"),
        station_med_park=("parking_time", "median"),
        station_q25_park=("parking_time", lambda x: x.quantile(0.25)),
        station_q75_park=("parking_time", lambda x: x.quantile(0.75)),
        station_avg_rate=("charge_rate", "mean"),
        station_med_rate=("charge_rate", "median"),
        station_freq=("kWhDelivered", "size"),
    )

    for D in (tr, te):
        for col in user_stats.columns:
            default = global_k if "kWh" in col else global_p if "park" in col else global_r if "rate" in col else 0
            D[col] = D["userID"].map(user_stats[col]).fillna(default)
        for col in station_stats.columns:
            default = global_k if "kWh" in col else global_p if "park" in col else global_r if "rate" in col else 0
            D[col] = D["stationID"].map(station_stats[col]).fillna(default)

        D["user_iqr_kWh"] = D["user_q75_kWh"] - D["user_q25_kWh"]
        D["station_iqr_kWh"] = D["station_q75_kWh"] - D["station_q25_kWh"]
        D["user_iqr_park"] = D["user_q75_park"] - D["user_q25_park"]
        D["station_iqr_park"] = D["station_q75_park"] - D["station_q25_park"]

    station_hour = tr.groupby(["stationID", "hour"]).agg(
        kWh=("kWhDelivered", "mean"),
        park=("parking_time", "median"),
        rate=("charge_rate", "mean"),
        cnt=("kWhDelivered", "size"),
    )
    station_weekday = tr.groupby(["stationID", "weekday"]).agg(
        kWh=("kWhDelivered", "mean"),
        park=("parking_time", "median"),
        rate=("charge_rate", "mean"),
    )
    user_hour = tr.groupby(["userID", "hour"]).agg(
        kWh=("kWhDelivered", "mean"),
        park=("parking_time", "median"),
        rate=("charge_rate", "mean"),
        cnt=("kWhDelivered", "size"),
    )

    for D in (tr, te):
        D["station_hour_avg_kWh"] = map_from_group(D, station_hour["kWh"], ["stationID", "hour"]).fillna(D["station_avg_kWh"])
        D["station_hour_avg_park"] = map_from_group(D, station_hour["park"], ["stationID", "hour"]).fillna(D["station_med_park"])
        D["station_hour_avg_rate"] = map_from_group(D, station_hour["rate"], ["stationID", "hour"]).fillna(D["station_avg_rate"])
        D["station_hour_cnt"] = map_from_group(D, station_hour["cnt"], ["stationID", "hour"], fill_value=0)

        D["station_weekday_avg_kWh"] = map_from_group(D, station_weekday["kWh"], ["stationID", "weekday"]).fillna(D["station_avg_kWh"])
        D["station_weekday_avg_park"] = map_from_group(D, station_weekday["park"], ["stationID", "weekday"]).fillna(D["station_med_park"])
        D["station_weekday_avg_rate"] = map_from_group(D, station_weekday["rate"], ["stationID", "weekday"]).fillna(D["station_avg_rate"])

        D["user_hour_avg_kWh"] = map_from_group(D, user_hour["kWh"], ["userID", "hour"]).fillna(D["user_avg_kWh"])
        D["user_hour_avg_park"] = map_from_group(D, user_hour["park"], ["userID", "hour"]).fillna(D["user_med_park"])
        D["user_hour_avg_rate"] = map_from_group(D, user_hour["rate"], ["userID", "hour"]).fillna(D["user_avg_rate"])
        D["user_hour_cnt"] = map_from_group(D, user_hour["cnt"], ["userID", "hour"], fill_value=0)

        D["station_hour_anchor_kWh"] = D["station_hour_avg_rate"] * D["station_hour_avg_park"]
        D["user_hour_anchor_kWh"] = D["user_hour_avg_rate"] * D["user_hour_avg_park"]

    return tr, te


def add_lag_and_gap_features(tr, te):
    tr = tr.copy().sort_values("connection_time_copy").reset_index(drop=True)
    te = te.copy().sort_values("connection_time_copy").reset_index(drop=True)

    med_k = float(tr["kWhDelivered"].median())
    med_p = float(tr["parking_time"].median())
    med_r = float(tr["charge_rate"].median())

    for group_col in ["userID", "stationID"]:
        prefix = "user" if group_col == "userID" else "station"
        tr[f"{prefix}_prev_kWh1"] = tr.groupby(group_col)["kWhDelivered"].shift(1).fillna(med_k)
        tr[f"{prefix}_prev_kWh2"] = tr.groupby(group_col)["kWhDelivered"].shift(2).fillna(med_k)
        tr[f"{prefix}_prev_park1"] = tr.groupby(group_col)["parking_time"].shift(1).fillna(med_p)
        tr[f"{prefix}_prev_park2"] = tr.groupby(group_col)["parking_time"].shift(2).fillna(med_p)
        tr[f"{prefix}_prev_rate1"] = tr.groupby(group_col)["charge_rate"].shift(1).fillna(med_r)
        tr[f"{prefix}_roll_kWh3"] = tr.groupby(group_col)["kWhDelivered"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(med_k)
        tr[f"{prefix}_roll_park3"] = tr.groupby(group_col)["parking_time"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(med_p)
        tr[f"{prefix}_roll_rate3"] = tr.groupby(group_col)["charge_rate"].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(med_r)
        gap_default = 24 * 30 if group_col == "userID" else 24
        gap_cap = 24 * 180 if group_col == "userID" else 24 * 30
        tr[f"{prefix}_gap_hours"] = (tr.groupby(group_col)["connection_time_copy"].diff().dt.total_seconds() / 3600.0).fillna(gap_default).clip(upper=gap_cap)

    if not USE_ONLINE_TEST_HISTORY:
        for group_col in ["userID", "stationID"]:
            prefix = "user" if group_col == "userID" else "station"
            hist_k = tr.groupby(group_col)["kWhDelivered"].apply(list).to_dict()
            hist_p = tr.groupby(group_col)["parking_time"].apply(list).to_dict()
            hist_r = tr.groupby(group_col)["charge_rate"].apply(list).to_dict()
            last_time = tr.groupby(group_col)["connection_time_copy"].max().to_dict()

            te[f"{prefix}_prev_kWh1"] = te[group_col].apply(lambda u: float(hist_k[u][-1]) if (u in hist_k and len(hist_k[u]) >= 1) else med_k)
            te[f"{prefix}_prev_kWh2"] = te[group_col].apply(lambda u: float(hist_k[u][-2]) if (u in hist_k and len(hist_k[u]) >= 2) else med_k)
            te[f"{prefix}_prev_park1"] = te[group_col].apply(lambda u: float(hist_p[u][-1]) if (u in hist_p and len(hist_p[u]) >= 1) else med_p)
            te[f"{prefix}_prev_park2"] = te[group_col].apply(lambda u: float(hist_p[u][-2]) if (u in hist_p and len(hist_p[u]) >= 2) else med_p)
            te[f"{prefix}_prev_rate1"] = te[group_col].apply(lambda u: float(hist_r[u][-1]) if (u in hist_r and len(hist_r[u]) >= 1) else med_r)
            te[f"{prefix}_roll_kWh3"] = te[group_col].apply(lambda u: float(np.mean(hist_k[u][-3:])) if (u in hist_k and len(hist_k[u]) > 0) else med_k)
            te[f"{prefix}_roll_park3"] = te[group_col].apply(lambda u: float(np.mean(hist_p[u][-3:])) if (u in hist_p and len(hist_p[u]) > 0) else med_p)
            te[f"{prefix}_roll_rate3"] = te[group_col].apply(lambda u: float(np.mean(hist_r[u][-3:])) if (u in hist_r and len(hist_r[u]) > 0) else med_r)
            gap_default = 24 * 30 if group_col == "userID" else 24
            gap_cap = 24 * 180 if group_col == "userID" else 24 * 30
            te[f"{prefix}_gap_hours"] = te.apply(
                lambda row: float((row["connection_time_copy"] - last_time[row[group_col]]).total_seconds() / 3600.0)
                if row[group_col] in last_time else gap_default,
                axis=1,
            ).clip(upper=gap_cap)
    else:
        history = pd.concat([tr, te], ignore_index=True).sort_values("connection_time_copy").reset_index(drop=True)
        history, _dummy = add_lag_and_gap_features(history, pd.DataFrame(columns=history.columns))
        te_keys = set(te.index.tolist())
        feature_cols = [c for c in history.columns if any(s in c for s in ["prev_", "roll_", "gap_hours"])]
        merge_cols = ["connection_time_copy", "userID", "stationID", "parking_time", "kWhDelivered"]
        tmp = te.merge(history[merge_cols + feature_cols], on=merge_cols, how="left", suffixes=("", "_online"))
        for c in feature_cols:
            if c in tmp.columns:
                te[c] = tmp[c].values

    return tr, te


def _weighted_knn_value(base_df, scaled_matrix, target_vec, idx_pool, target_col, topk=5):
    if idx_pool is None or len(idx_pool) == 0:
        return None
    dists = np.linalg.norm(scaled_matrix[idx_pool] - target_vec, axis=1)
    order = np.argsort(dists)[:min(topk, len(dists))]
    top_idx = idx_pool[order]
    weights = 1.0 / (1.0 + dists[order])
    return float(np.average(base_df.iloc[top_idx][target_col].values, weights=weights))


def build_similarity_hierarchical(tr, te, target_col, out_col, topk=5):
    tr = tr.copy().sort_values("connection_time_copy").reset_index(drop=True)
    te = te.copy().sort_values("connection_time_copy").reset_index(drop=True)

    sim_feats = [
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
    ]
    scaler = StandardScaler().fit(tr[sim_feats])
    tr_s = scaler.transform(tr[sim_feats])
    te_s = scaler.transform(te[sim_feats]) if len(te) else np.empty((0, len(sim_feats)))

    fallback = float(tr[target_col].median())
    tr[out_col] = fallback
    te[out_col] = fallback

    for i in range(len(tr)):
        row = tr.iloc[i]
        hist = tr.iloc[:i]
        if len(hist) == 0:
            continue
        candidate_masks = [
            (hist["userID"] == row["userID"]) & (hist["stationID"] == row["stationID"]),
            (hist["userID"] == row["userID"]),
            (hist["stationID"] == row["stationID"]) & (hist["hour"] == row["hour"]),
            (hist["stationID"] == row["stationID"]) & (hist["weekday"] == row["weekday"]),
            (hist["cluster_id"] == row["cluster_id"]) & (hist["hour"] == row["hour"]),
            pd.Series([True] * len(hist), index=hist.index),
        ]
        val = None
        for mask in candidate_masks:
            idx_pool = np.where(mask.values)[0]
            val = _weighted_knn_value(hist, tr_s[:i], tr_s[i], idx_pool, target_col, topk=topk)
            if val is not None:
                break
        tr.at[i, out_col] = fallback if val is None else val

    for i in range(len(te)):
        row = te.iloc[i]
        candidate_masks = [
            (tr["userID"] == row["userID"]) & (tr["stationID"] == row["stationID"]),
            (tr["userID"] == row["userID"]),
            (tr["stationID"] == row["stationID"]) & (tr["hour"] == row["hour"]),
            (tr["stationID"] == row["stationID"]) & (tr["weekday"] == row["weekday"]),
            (tr["cluster_id"] == row["cluster_id"]) & (tr["hour"] == row["hour"]),
            pd.Series([True] * len(tr), index=tr.index),
        ]
        val = None
        for mask in candidate_masks:
            idx_pool = np.where(mask.values)[0]
            val = _weighted_knn_value(tr, tr_s, te_s[i], idx_pool, target_col, topk=topk)
            if val is not None:
                break
        te.at[i, out_col] = fallback if val is None else val

    return tr, te


def get_base_time_features():
    feats = [
        "hour", "weekday", "month", "dayofyear", "day",
        "is_weekend", "is_holiday",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
        "hour_of_week", "weekofyear",
    ]
    if USE_WEATHER_FEATURES:
        for c in ["temperature", "humidity", "precipitation"]:
            feats.append(c)
    return feats


def anchor_parking_predictions(D):
    preds = []
    preds.append(D["user_med_park"].values)
    preds.append(D["station_med_park"].values)
    preds.append(D["user_hour_avg_park"].values)
    preds.append(D["station_hour_avg_park"].values)
    preds.append(D["sim_park"].values)
    return np.nanmedian(np.vstack(preds), axis=0)


def anchor_kwh_predictions(D):
    preds = []
    preds.append(D["user_avg_kWh"].values)
    preds.append(D["user_med_kWh"].values)
    preds.append(D["station_avg_kWh"].values)
    preds.append(D["station_med_kWh"].values)
    preds.append(D["user_hour_avg_kWh"].values)
    preds.append(D["station_hour_avg_kWh"].values)
    preds.append(D["sim_kwh"].values)
    if "pred_parking_for_stage2" in D.columns:
        preds.append(D["pred_parking_for_stage2"].values * D["user_avg_rate"].values)
        preds.append(D["pred_parking_for_stage2"].values * D["station_avg_rate"].values)
        preds.append(D["pred_parking_for_stage2"].values * D["user_hour_avg_rate"].values)
        preds.append(D["pred_parking_for_stage2"].values * D["station_hour_avg_rate"].values)
    return np.nanmedian(np.vstack(preds), axis=0)



def get_model_seq_len():
    name = MODEL_NAME.lower()
    if name == "dlinear":
        return 12
    if name in ["patchtst", "itransformer", "autoformer", "tft"]:
        return 12
    return 8


if torch is not None:
    class LSTMRegressor(nn.Module):
        def __init__(self, n_features, hidden=64, dropout=0.1):
            super().__init__()
            self.rnn = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    class GRURegressor(nn.Module):
        def __init__(self, n_features, hidden=64, dropout=0.1):
            super().__init__()
            self.rnn = nn.GRU(n_features, hidden, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.drop(out[:, -1, :])).squeeze(-1)

    class TFTMiniRegressor(nn.Module):
        def __init__(self, n_features, hidden=64, dropout=0.1):
            super().__init__()
            self.input_proj = nn.Linear(n_features, hidden)
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            self.attn = nn.MultiheadAttention(hidden, num_heads=4, dropout=dropout, batch_first=True)
            self.gate = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Sigmoid())
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))
        def forward(self, x):
            z = torch.relu(self.input_proj(x))
            h, _ = self.lstm(z)
            a, _ = self.attn(h, h, h)
            g = self.gate(torch.cat([h, a], dim=-1))
            z = self.norm(g * a + (1.0 - g) * h)
            return self.head(z[:, -1, :]).squeeze(-1)

    class AutoformerMiniRegressor(nn.Module):
        def __init__(self, n_features, d_model=64, dropout=0.1):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        def moving_avg(self, x, k=3):
            pad = k // 2
            xt = x.transpose(1, 2)
            trend = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(xt, (pad, pad), mode="replicate"), kernel_size=k, stride=1)
            return trend.transpose(1, 2)
        def forward(self, x):
            trend = self.moving_avg(x, k=3)
            seasonal = x - trend
            z = self.proj(seasonal + trend)
            h = self.encoder(z)
            return self.head(h.mean(dim=1)).squeeze(-1)

    class PatchTSTMiniRegressor(nn.Module):
        def __init__(self, n_features, seq_len, d_model=64, dropout=0.1, patch_len=4, stride=2):
            super().__init__()
            self.n_features = n_features
            self.seq_len = seq_len
            self.patch_len = min(patch_len, seq_len)
            self.stride = min(stride, self.patch_len)
            self.patch_proj = nn.Linear(n_features * self.patch_len, d_model)
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        def forward(self, x):
            if x.shape[1] < self.patch_len:
                pad_len = self.patch_len - x.shape[1]
                x = torch.cat([x[:, :1, :].repeat(1, pad_len, 1), x], dim=1)
            patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches = patches.permute(0, 1, 3, 2).contiguous().view(x.size(0), -1, self.patch_len * self.n_features)
            z = self.patch_proj(patches)
            h = self.encoder(z)
            return self.head(h.mean(dim=1)).squeeze(-1)

    class DLinearRegressor(nn.Module):
        def __init__(self, n_features, seq_len, dropout=0.1):
            super().__init__()
            self.temporal = nn.Linear(seq_len, 1)
            self.head = nn.Sequential(nn.LayerNorm(n_features), nn.Dropout(dropout), nn.Linear(n_features, max(8, n_features // 2)), nn.ReLU(), nn.Linear(max(8, n_features // 2), 1))
        def forward(self, x):
            z = x.transpose(1, 2)
            z = self.temporal(z).squeeze(-1)
            return self.head(z).squeeze(-1)

    class ITransformerMiniRegressor(nn.Module):
        def __init__(self, n_features, seq_len, d_model=64, dropout=0.1):
            super().__init__()
            self.seq_proj = nn.Linear(seq_len, d_model)
            self.var_pos = nn.Parameter(torch.zeros(1, n_features, d_model))
            enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 2, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1))
        def forward(self, x):
            z = x.transpose(1, 2)
            z = self.seq_proj(z) + self.var_pos[:, :z.shape[1], :]
            h = self.encoder(z)
            return self.head(h.mean(dim=1)).squeeze(-1)


def build_torch_model(n_features, seq_len):
    name = MODEL_NAME.lower()
    if torch is None:
        raise RuntimeError(f"{MODEL_NAME} PyTorch is required; please install torch first.")
    if name == "lstm":
        return LSTMRegressor(n_features, hidden=TORCH_HIDDEN, dropout=TORCH_DROPOUT)
    if name == "gru":
        return GRURegressor(n_features, hidden=TORCH_HIDDEN, dropout=TORCH_DROPOUT)
    if name == "tft":
        return TFTMiniRegressor(n_features, hidden=TORCH_HIDDEN, dropout=TORCH_DROPOUT)
    if name == "autoformer":
        return AutoformerMiniRegressor(n_features, d_model=TORCH_D_MODEL, dropout=TORCH_DROPOUT)
    if name == "patchtst":
        return PatchTSTMiniRegressor(n_features, seq_len=seq_len, d_model=TORCH_D_MODEL, dropout=TORCH_DROPOUT)
    if name == "dlinear":
        return DLinearRegressor(n_features, seq_len=seq_len, dropout=TORCH_DROPOUT)
    if name == "itransformer":
        return ITransformerMiniRegressor(n_features, seq_len=seq_len, d_model=TORCH_D_MODEL, dropout=TORCH_DROPOUT)
    raise ValueError(f"Unknown model: {MODEL_NAME}")


def make_train_sequences(X, seq_len):
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.empty((0, seq_len, X.shape[1]), dtype=np.float32)
    pad = np.repeat(X[:1], seq_len - 1, axis=0)
    Xp = np.vstack([pad, X])
    seqs = [Xp[i:i + seq_len] for i in range(len(X))]
    return np.asarray(seqs, dtype=np.float32)


def make_future_sequences(context_X, future_X, seq_len):
    context_X = np.asarray(context_X, dtype=np.float32)
    future_X = np.asarray(future_X, dtype=np.float32)
    if len(future_X) == 0:
        return np.empty((0, seq_len, context_X.shape[1]), dtype=np.float32)
    if len(context_X) == 0:
        return make_train_sequences(future_X, seq_len)
    hist = context_X[-(seq_len - 1):] if seq_len > 1 else np.empty((0, context_X.shape[1]), dtype=np.float32)
    combined = np.vstack([hist, future_X])
    start = len(hist)
    seqs = []
    for i in range(len(future_X)):
        end = start + i + 1
        begin = max(0, end - seq_len)
        s = combined[begin:end]
        if len(s) < seq_len:
            s = np.vstack([np.repeat(s[:1], seq_len - len(s), axis=0), s])
        seqs.append(s)
    return np.asarray(seqs, dtype=np.float32)


def fit_predict_arima(X_train, y_train_raw, X_test, n_val, target_name, seed=42):
    y_train_raw = np.asarray(y_train_raw, dtype=float)
    y_log = np.log1p(np.maximum(y_train_raw, 0.0))
    n = len(y_log)
    n_core = n - n_val
    y_core = y_log[:n_core]
    horizon = n_val + len(X_test)

    fallback = np.full(horizon, np.median(y_core) if len(y_core) else 0.0, dtype=float)
    fitted_core = np.full(n_core, np.median(y_core) if len(y_core) else 0.0, dtype=float)

    try:
        from statsmodels.tsa.arima.model import ARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_series = y_core[-min(len(y_core), ARIMA_MAX_TRAIN_POINTS):]
            model = ARIMA(fit_series, order=ARIMA_ORDER, enforce_stationarity=False, enforce_invertibility=False)
            try:
                res = model.fit(method_kwargs={"maxiter": 40})
            except TypeError:
                res = model.fit()
            fc = np.asarray(res.forecast(steps=horizon), dtype=float)
            fitted = np.asarray(res.fittedvalues, dtype=float)
            if len(fitted) > 0:
                fitted_core[-len(fitted):] = fitted
            fallback[:len(fc)] = fc
    except Exception:
        if len(y_core) > 0:
            alpha = 0.35
            level = y_core[0]
            fitted_core = []
            for v in y_core:
                fitted_core.append(level)
                level = alpha * v + (1 - alpha) * level
            fitted_core = np.asarray(fitted_core, dtype=float)
            fallback = np.full(horizon, level, dtype=float)

    pred_train_log = np.concatenate([fitted_core, fallback[:n_val]]) if n_val > 0 else fitted_core
    pred_val = inverse_log1p(fallback[:n_val])
    pred_test = inverse_log1p(fallback[n_val:n_val + len(X_test)])
    pred_train_all = inverse_log1p(pred_train_log)
    return pred_train_all, pred_val, pred_test, {"epochs": 0, "best_val_mae": mae(y_train_raw[-n_val:], pred_val) if n_val else np.nan}


def fit_predict_torch(X_train, y_train_raw, X_test, n_val, target_name, seed=42):
    if torch is None:
        raise RuntimeError(f"{MODEL_NAME} PyTorch is required, but torch is not currently installed in this environment.")

    set_seed(seed)
    X_train = clean_matrix(X_train)
    X_test = clean_matrix(X_test)
    y_train_raw = np.asarray(y_train_raw, dtype=float)
    y_log = np.log1p(np.maximum(y_train_raw, 0.0)).astype(np.float32)

    n = len(X_train)
    n_core = n - n_val
    X_core = X_train[:n_core]
    X_val = X_train[n_core:]
    y_core = y_log[:n_core]
    y_val_raw = y_train_raw[n_core:]

    scaler = StandardScaler().fit(X_core)
    X_core_s = scaler.transform(X_core).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32) if len(X_test) else np.empty((0, X_train.shape[1]), dtype=np.float32)

    seq_len = min(get_model_seq_len(), max(2, len(X_core_s)))
    X_core_seq = make_train_sequences(X_core_s, seq_len)
    X_val_seq = make_future_sequences(X_core_s, X_val_s, seq_len)
    X_train_seq = make_train_sequences(X_train_s, seq_len)
    X_test_seq = make_future_sequences(X_train_s, X_test_s, seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(n_features=X_train.shape[1], seq_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=TORCH_LR, weight_decay=TORCH_WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()

    ds = TensorDataset(torch.tensor(X_core_seq, dtype=torch.float32), torch.tensor(y_core, dtype=torch.float32))
    batch_size = min(TORCH_BATCH_SIZE, max(16, len(ds)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    Xv = torch.tensor(X_val_seq, dtype=torch.float32, device=device)

    best_state = None
    best_val = np.inf
    no_imp = 0
    best_epoch = 0

    for epoch in range(1, TORCH_MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            pv_log = model(Xv).detach().cpu().numpy() if len(X_val_seq) else np.array([])
        pv = inverse_log1p(pv_log)
        cur = mae(y_val_raw, pv)
        if cur < best_val - 1e-6:
            best_val = cur
            best_epoch = epoch
            if USE_TORCH_EARLY_STOPPING:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if USE_TORCH_EARLY_STOPPING and no_imp >= TORCH_PATIENCE:
            break

    if USE_TORCH_EARLY_STOPPING and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def predict_seq(Xseq):
        if len(Xseq) == 0:
            return np.array([], dtype=float)
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xseq), TORCH_BATCH_SIZE):
                xb = torch.tensor(Xseq[i:i + TORCH_BATCH_SIZE], dtype=torch.float32, device=device)
                outs.append(model(xb).detach().cpu().numpy())
        return inverse_log1p(np.concatenate(outs))

    pred_train_all = predict_seq(X_train_seq)
    pred_val = pred_train_all[-n_val:] if n_val > 0 else np.array([], dtype=float)
    pred_test = predict_seq(X_test_seq)
    return pred_train_all, pred_val, pred_test, {"epochs": epoch, "best_val_mae": best_val}


def fit_predict_model(X_train, y_train_raw, X_test, n_val, target_name, seed=42):
    if MODEL_NAME.lower() == "arima":
        return fit_predict_arima(X_train, y_train_raw, X_test, n_val, target_name, seed=seed)
    return fit_predict_torch(X_train, y_train_raw, X_test, n_val, target_name, seed=seed)


def fit_one_cluster(cluster_df, window, run):
    scenario_name_label = str(cluster_df["scenario_name_raw"].iloc[0])
    cluster_id = int(cluster_df["cluster_id"].iloc[0])
    t_all_start = time.perf_counter()

    tr0 = TEST_START - timedelta(days=window)
    tr = cluster_df[(cluster_df["connection_time_copy"] >= tr0) & (cluster_df["connection_time_copy"] < TEST_START)].copy()
    te = cluster_df[(cluster_df["connection_time_copy"] >= TEST_START) & (cluster_df["connection_time_copy"] <= TEST_END)].copy()

    if len(tr) < MIN_CLUSTER_TRAIN or len(te) < MIN_CLUSTER_TEST:
        return None, None

    tr = denoise_train_only(tr)
    if len(tr) < MIN_CLUSTER_TRAIN:
        return None, None

    tr = tr.sort_values("connection_time_copy").reset_index(drop=True)
    te = te.sort_values("connection_time_copy").reset_index(drop=True)

    tr, te = add_group_history_features(tr, te)
    tr, te = add_lag_and_gap_features(tr, te)
    tr, te = build_similarity_hierarchical(tr, te, target_col="parking_time", out_col="sim_park", topk=5)
    tr, te = build_similarity_hierarchical(tr, te, target_col="kWhDelivered", out_col="sim_kwh", topk=5)
    tr, te = build_similarity_hierarchical(tr, te, target_col="charge_rate", out_col="sim_rate", topk=5)

    base_time = get_base_time_features()
    n_val = max(int(0.2 * len(tr)), 20)
    n_val = min(n_val, max(1, len(tr) - 10))

    t_parking_start = time.perf_counter()
    pf = base_time + [
        "userID", "stationID", "cluster_id", "sim_park",
        "user_avg_park", "user_med_park", "user_iqr_park", "user_freq",
        "station_avg_park", "station_med_park", "station_iqr_park", "station_freq",
        "user_hour_avg_park", "user_hour_cnt",
        "station_hour_avg_park", "station_hour_cnt",
        "station_weekday_avg_park",
        "user_prev_park1", "user_prev_park2", "user_roll_park3",
        "station_prev_park1", "station_roll_park3",
        "user_gap_hours", "station_gap_hours",
        "user_prev_kWh1", "user_roll_kWh3",
        "station_prev_kWh1", "station_roll_kWh3",
        "station_hour_avg_rate", "user_hour_avg_rate",
    ]

    Xp_tr = clean_matrix(tr[pf].values)
    Xp_te = clean_matrix(te[pf].values)
    y_p_tr_raw = tr["parking_time"].values.astype(float)
    y_p_te = te["parking_time"].values.astype(float)

    pred_p_model_tr, pred_va_p_model, pred_te_p_model, info_p = fit_predict_model(
        Xp_tr, y_p_tr_raw, Xp_te, n_val, target_name="parking", seed=run * 100 + window + cluster_id
    )
    pred_va_p_anchor = anchor_parking_predictions(tr.iloc[-n_val:])
    pred_te_p_anchor = anchor_parking_predictions(te)
    pred_all_p_anchor = anchor_parking_predictions(tr)

    if USE_ANCHOR_BLEND:
        w_p_model, blend_val_mae_p = select_blend_weight(y_p_tr_raw[-n_val:], pred_va_p_model, pred_va_p_anchor, step=0.02)
    else:
        w_p_model, blend_val_mae_p = 1.0, mae(y_p_tr_raw[-n_val:], pred_va_p_model)

    pred_p_tr = w_p_model * pred_p_model_tr + (1.0 - w_p_model) * pred_all_p_anchor
    pred_p_te = w_p_model * pred_te_p_model + (1.0 - w_p_model) * pred_te_p_anchor
    park_upper = float(np.quantile(y_p_tr_raw, 0.995))
    park_upper = max(park_upper, float(np.median(y_p_tr_raw)) + 1.0)
    pred_p_tr = np.clip(pred_p_tr, 0.03, park_upper)
    pred_p_te = np.clip(pred_p_te, 0.03, park_upper)

    mae_pt = mae(y_p_te, pred_p_te)
    smape_pt = smape(y_p_te, pred_p_te)
    time_parking_sec = time.perf_counter() - t_parking_start

    t_kwh_start = time.perf_counter()
    tr["pred_parking_for_stage2"] = pred_p_tr
    te["pred_parking_for_stage2"] = pred_p_te

    for D in (tr, te):
        D["pred_kwh_user_anchor"] = D["pred_parking_for_stage2"] * D["user_avg_rate"]
        D["pred_kwh_station_anchor"] = D["pred_parking_for_stage2"] * D["station_avg_rate"]
        D["pred_kwh_user_hour_anchor"] = D["pred_parking_for_stage2"] * D["user_hour_avg_rate"]
        D["pred_kwh_station_hour_anchor"] = D["pred_parking_for_stage2"] * D["station_hour_avg_rate"]

    feats_k = base_time + [
        "userID", "stationID", "cluster_id",
        "sim_kwh", "sim_rate",
        "user_avg_kWh", "user_med_kWh", "user_iqr_kWh", "user_freq",
        "station_avg_kWh", "station_med_kWh", "station_iqr_kWh", "station_freq",
        "user_avg_park", "station_avg_park",
        "user_avg_rate", "user_med_rate", "station_avg_rate", "station_med_rate",
        "user_hour_avg_kWh", "user_hour_avg_rate", "user_hour_cnt",
        "station_hour_avg_kWh", "station_hour_avg_rate", "station_hour_cnt",
        "station_weekday_avg_kWh", "station_weekday_avg_rate",
        "pred_parking_for_stage2",
        "pred_kwh_user_anchor", "pred_kwh_station_anchor",
        "pred_kwh_user_hour_anchor", "pred_kwh_station_hour_anchor",
        "user_prev_kWh1", "user_prev_kWh2", "user_roll_kWh3",
        "station_prev_kWh1", "station_prev_kWh2", "station_roll_kWh3",
        "user_prev_park1", "user_roll_park3", "station_prev_park1", "station_roll_park3",
        "user_prev_rate1", "user_roll_rate3", "station_prev_rate1", "station_roll_rate3",
        "user_gap_hours", "station_gap_hours",
        "station_hour_anchor_kWh", "user_hour_anchor_kWh",
    ]

    Xk_tr = clean_matrix(tr[feats_k].values)
    Xk_te = clean_matrix(te[feats_k].values)
    y_k_tr_raw = tr["kWhDelivered"].values.astype(float)
    y_k_te = te["kWhDelivered"].values.astype(float)

    pred_k_model_tr, pred_va_k_model, pred_te_k_model, info_k = fit_predict_model(
        Xk_tr, y_k_tr_raw, Xk_te, n_val, target_name="kWh", seed=run * 1000 + window + cluster_id
    )

    rate_upper = float(np.quantile(tr["charge_rate"].values, 0.99))
    rate_upper = max(rate_upper, float(np.median(tr["charge_rate"].values)) + 1.0)
    pred_va_rate = np.nanmedian(np.vstack([
        tr["user_avg_rate"].values[-n_val:], tr["station_avg_rate"].values[-n_val:],
        tr["user_hour_avg_rate"].values[-n_val:], tr["station_hour_avg_rate"].values[-n_val:], tr["sim_rate"].values[-n_val:]
    ]), axis=0)
    pred_te_rate = np.nanmedian(np.vstack([
        te["user_avg_rate"].values, te["station_avg_rate"].values,
        te["user_hour_avg_rate"].values, te["station_hour_avg_rate"].values, te["sim_rate"].values
    ]), axis=0)
    pred_va_rate = np.clip(pred_va_rate, 0.0, rate_upper)
    pred_te_rate = np.clip(pred_te_rate, 0.0, rate_upper)
    pred_va_k_indirect = np.maximum(tr["pred_parking_for_stage2"].values[-n_val:], 0.03) * np.maximum(pred_va_rate, 0.0)
    pred_te_k_indirect = np.maximum(te["pred_parking_for_stage2"].values, 0.03) * np.maximum(pred_te_rate, 0.0)

    pred_va_k_raw_anchor = 0.5 * tr["user_avg_kWh"].values[-n_val:] + 0.5 * tr["station_hour_avg_kWh"].values[-n_val:]
    pred_te_k_raw_anchor = 0.5 * te["user_avg_kWh"].values + 0.5 * te["station_hour_avg_kWh"].values
    pred_va_k_anchor = anchor_kwh_predictions(tr.iloc[-n_val:])
    pred_te_k_anchor = anchor_kwh_predictions(te)

    pred_list_va = [pred_va_k_model, pred_va_k_indirect, pred_va_k_raw_anchor, pred_va_k_anchor]
    pred_list_te = [pred_te_k_model, pred_te_k_indirect, pred_te_k_raw_anchor, pred_te_k_anchor]
    if USE_ANCHOR_BLEND:
        blend_w, blend_val_mae = select_blend_weights_greedy(y_k_tr_raw[-n_val:], pred_list_va, step=0.05, n_iter=4)
    else:
        blend_w = np.array([1.0, 0.0, 0.0, 0.0])
        blend_val_mae = mae(y_k_tr_raw[-n_val:], pred_va_k_model)

    pred_te_k = np.zeros_like(pred_te_k_model, dtype=float)
    for w, p in zip(blend_w, pred_list_te):
        pred_te_k += w * p
    pred_te_k = np.maximum(pred_te_k, 0.0)

    mae_k = mae(y_k_te, pred_te_k)
    smape_k = smape(y_k_te, pred_te_k)
    time_kwh_sec = time.perf_counter() - t_kwh_start
    time_total_sec = time.perf_counter() - t_all_start

    pred_df = pd.DataFrame({
        "model": MODEL_NAME,
        "run": run,
        "window": window,
        "cluster_id": cluster_id,
        "scenario_name": scenario_name_label,
        "time": te["connection_time_copy"].values,
        "stationID_encoded": te["stationID"].values,
        "stationID_raw": te["stationID_raw"].values if "stationID_raw" in te.columns else te["stationID"].values,
        "userID_encoded": te["userID"].values,
        "y_true_parking": y_p_te,
        "y_pred_parking": pred_p_te,
        "y_true_kWh": y_k_te,
        "y_pred_kWh": pred_te_k,
        "parking_blend_w_model": w_p_model,
        "kwh_w_model": blend_w[0],
        "kwh_w_rate_indirect": blend_w[1],
        "kwh_w_raw_anchor": blend_w[2],
        "kwh_w_history_anchor": blend_w[3],
    })

    result_row = {
        "model": MODEL_NAME,
        "run": run,
        "window": window,
        "cluster_id": cluster_id,
        "scenario_name": scenario_name_label,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "mae_parking": mae_pt,
        "smape_parking": smape_pt,
        "mae_kWh": mae_k,
        "smape_kWh": smape_k,
        "parking_blend_w_model": float(w_p_model),
        "parking_blend_val_mae": float(blend_val_mae_p),
        "kwh_w_model": float(blend_w[0]),
        "kwh_w_rate_indirect": float(blend_w[1]),
        "kwh_w_raw_anchor": float(blend_w[2]),
        "kwh_w_history_anchor": float(blend_w[3]),
        "blend_val_mae_kWh": float(blend_val_mae),
        "epochs_parking": int(info_p.get("epochs", 0)),
        "epochs_kWh": int(info_k.get("epochs", 0)),
        "val_mae_parking_model": float(info_p.get("best_val_mae", np.nan)),
        "val_mae_kWh_model": float(info_k.get("best_val_mae", np.nan)),
        "time_parking_sec": time_parking_sec,
        "time_kWh_sec": time_kwh_sec,
        "time_total_sec": time_total_sec,
    }
    return result_row, pred_df


def save_results(overall_df, cluster_detail_df, pred_df, scenario_map, filter_summary, station_quality):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filter_summary.to_csv(OUTPUT_FILTER_CSV, index=False, encoding="utf-8-sig")
    station_quality.to_csv(OUTPUT_STATION_CSV, index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
            overall_df.to_excel(writer, index=False, sheet_name="overall_summary")
            cluster_detail_df.to_excel(writer, index=False, sheet_name="cluster_detail")
            scenario_map.to_excel(writer, index=False, sheet_name="cluster_mapping")
            filter_summary.to_excel(writer, index=False, sheet_name="quality_filter")
            station_quality.to_excel(writer, index=False, sheet_name="station_quality")
    except Exception:
        with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
            overall_df.to_excel(writer, index=False, sheet_name="overall_summary")
            cluster_detail_df.to_excel(writer, index=False, sheet_name="cluster_detail")
            scenario_map.to_excel(writer, index=False, sheet_name="cluster_mapping")
            filter_summary.to_excel(writer, index=False, sheet_name="quality_filter")
            station_quality.to_excel(writer, index=False, sheet_name="station_quality")

    if not pred_df.empty:
        pred_df.to_csv(OUTPUT_PRED_CSV, index=False, encoding="utf-8-sig")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, input_path, station_quality, quality_filter_summary = load_filtered_clustered_data()
    df, retained_cluster_ids, cluster_filter_summary = filter_clusters_before_modeling(df)
    if len(retained_cluster_ids) == 0:
        raise RuntimeError("No categories meet the filter criteria; modeling is not possible. Please reduce the filter strength for station/top/quantile.")

    filter_summary = pd.concat([quality_filter_summary, cluster_filter_summary], ignore_index=True, sort=False)
    scenario_map = (
        df[["cluster_id", "scenario_name_raw"]]
        .drop_duplicates()
        .sort_values(["cluster_id", "scenario_name_raw"])
        .reset_index(drop=True)
    )

    overall_rows = []
    cluster_rows = []
    pred_frames = []

    print(f"\n==========  JX-{MODEL_NAME}  ==========")

    for run in range(1, N_RUNS + 1):
        print(f"\n================ Run {run} ================")
        for window in WINDOWS:
            print(f"\n---- Training window = {window} days ----")
            cluster_pred_frames = []
            cluster_metric_rows = []

            for cid in retained_cluster_ids:
                cluster_df = df[df["cluster_id"] == cid].copy()
                out_row, pred_one = fit_one_cluster(cluster_df, window=window, run=run)
                if out_row is None:
                    print(f"cluster {cid:>2}: skipped, train/test samples not enough after filtering.")
                    continue
                cluster_metric_rows.append(out_row)
                cluster_pred_frames.append(pred_one)
                print(
                    f"cluster {cid:>2} | n_train={out_row['n_train']:>5}, n_test={out_row['n_test']:>5} | "
                    f"MAE_park={out_row['mae_parking']:.4f}, SMAPE_park={out_row['smape_parking']:.2f}% | "
                    f"MAE_kWh={out_row['mae_kWh']:.4f}, SMAPE_kWh={out_row['smape_kWh']:.2f}% | "
                    f"wP_model={out_row['parking_blend_w_model']:.2f}, "
                    f"wK=[{out_row['kwh_w_model']:.2f},{out_row['kwh_w_rate_indirect']:.2f},{out_row['kwh_w_raw_anchor']:.2f},{out_row['kwh_w_history_anchor']:.2f}] | "
                    f"epochs=({out_row['epochs_parking']},{out_row['epochs_kWh']}) | "
                    f"time_total={out_row['time_total_sec']:.2f}s"
                )

            if not cluster_metric_rows:
                print(f"window={window}: no valid cluster results.")
                continue

            cluster_df_metrics = pd.DataFrame(cluster_metric_rows)
            all_preds = pd.concat(cluster_pred_frames, ignore_index=True)
            overall_row = {
                "model": MODEL_NAME,
                "run": run,
                "window": window,
                "case_mode": CASE_MODE,
                "clusters_retained_before_modeling": ",".join(map(str, retained_cluster_ids)),
                "clusters_used": int(cluster_df_metrics["cluster_id"].nunique()),
                "test_samples": int(len(all_preds)),
                "overall_mae_parking": mae(all_preds["y_true_parking"], all_preds["y_pred_parking"]),
                "overall_smape_parking": smape(all_preds["y_true_parking"], all_preds["y_pred_parking"]),
                "overall_mae_kWh": mae(all_preds["y_true_kWh"], all_preds["y_pred_kWh"]),
                "overall_smape_kWh": smape(all_preds["y_true_kWh"], all_preds["y_pred_kWh"]),
                "mean_cluster_mae_parking": float(cluster_df_metrics["mae_parking"].mean()),
                "mean_cluster_smape_parking": float(cluster_df_metrics["smape_parking"].mean()),
                "mean_cluster_mae_kWh": float(cluster_df_metrics["mae_kWh"].mean()),
                "mean_cluster_smape_kWh": float(cluster_df_metrics["smape_kWh"].mean()),
                "sum_time_parking_sec": float(cluster_df_metrics["time_parking_sec"].sum()),
                "sum_time_kWh_sec": float(cluster_df_metrics["time_kWh_sec"].sum()),
                "sum_time_total_sec": float(cluster_df_metrics["time_total_sec"].sum()),
            }
            overall_rows.append(overall_row)
            cluster_rows.append(cluster_df_metrics)
            pred_frames.append(all_preds)
            print(
                f"[OVERALL window={window}] clusters_used={overall_row['clusters_used']}, "
                f"test_samples={overall_row['test_samples']} | "
                f"MAE_park={overall_row['overall_mae_parking']:.4f}, "
                f"SMAPE_park={overall_row['overall_smape_parking']:.2f}% | "
                f"MAE_kWh={overall_row['overall_mae_kWh']:.4f}, "
                f"SMAPE_kWh={overall_row['overall_smape_kWh']:.2f}% | "
                f"time_total={overall_row['sum_time_total_sec']:.2f}s"
            )

    overall_df = pd.DataFrame(overall_rows)
    cluster_detail_df = pd.concat(cluster_rows, ignore_index=True) if cluster_rows else pd.DataFrame()
    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    save_results(overall_df, cluster_detail_df, pred_df, scenario_map, filter_summary, station_quality)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
