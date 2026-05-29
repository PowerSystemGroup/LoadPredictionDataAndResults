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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.getLogger("xgboost").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true, y_pred):
    return 100 * np.mean(
        np.abs(y_pred - y_true) /
        (np.abs(y_pred) + np.abs(y_true) + 1e-6)
    )


def trimmed_stats(residuals, weights, trim_ratio=0.05):
    r = np.array(residuals)
    w = np.array(weights)
    n = len(r)
    k = int(n * trim_ratio)
    if k < 1:
        return r, w
    idx = np.argsort(np.abs(r))
    keep_idx = idx[k: n - k]
    return r[keep_idx], w[keep_idx]


def compute_huber_alpha(residuals, scale=2.0):
    med = np.median(np.abs(residuals)) + 1e-6
    sigma = med * 1.4826 * scale
    return float(max(sigma, 1e-3))


def compute_dynamic_max_depth_advanced(residuals, weights,
                                       d0=6, δ=1.5, δp=1.2,
                                       min_depth=2, max_depth_limit=20,
                                       prev_depth=None,
                                       trim_ratio=0.05,
                                       smooth_factor=0.3):
    r_trim, w_trim = trimmed_stats(residuals, weights, trim_ratio=trim_ratio)
    w_sum = np.sum(w_trim) + 1e-12
    mu = np.sum(w_trim * r_trim) / w_sum
    var_w = np.sum(w_trim * (r_trim - mu) ** 2) / w_sum
    σ_w = np.sqrt(var_w)

    skew_val = abs(skew(r_trim))
    kurt_val = abs(kurtosis(r_trim, fisher=True))

    med_abs = np.median(np.abs(r_trim - np.median(r_trim))) + 1e-6
    MAD_w = med_abs * 1.4826

    f = (σ_w * (1.0 + 5*skew_val + 5*kurt_val)) ** (1/3) + (MAD_w) ** (1/3)

    if f >= 1.0:
        raw = d0 + δ * f
    else:
        raw = d0 - δp * f

    if prev_depth is not None:
        raw = smooth_factor * raw + (1.0 - smooth_factor) * prev_depth

    depth = int(round(raw))
    depth = max(min_depth, min(depth, max_depth_limit))
    return depth


def compute_dynamic_gamma_advanced(residuals, weights,
                                   base=1.0, α=0.7,
                                   min_gamma=0.001, max_gamma=20.0,
                                   prev_gamma=None,
                                   trim_ratio=0.05,
                                   smooth_factor=0.3):
    r_arr = np.array(residuals)
    w_arr = np.array(weights)
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
    IQR_w = abs(r_q3 - r_q1)

    med_abs = np.median(np.abs(r_trim - np.median(r_trim))) + 1e-6
    MAD_w = med_abs * 1.4826

    skew_val = abs(skew(r_trim))
    kurt_val = abs(kurtosis(r_trim, fisher=True))

    f_gamma = (IQR_w / MAD_w) * (1.0 + 5*skew_val + 5*kurt_val)

    raw = base / (1.0 + α * f_gamma)
    if prev_gamma is not None:
        raw = smooth_factor * raw + (1.0 - smooth_factor) * prev_gamma

    gamma = float(max(min_gamma, min(raw, max_gamma)))
    return gamma


INPUT_XLSX = "caltech_station_scalability_splits.xlsx"
RESULT_ROOT = os.path.join(".", "results", "caltech_scalability")

DESIRED_SHEET_ORDER = [
    "Caltech_53_stations",
    "Caltech_40_stations",
    "Caltech_30_stations",
    "Caltech_20_stations",
    "Caltech_10_stations",
]

BASE_RANDOM_SEED = 20260505


def safe_name(name):
    s = str(name)
    for ch in ["\\", "/", ":", "*", "?", '"', "<", ">", "|"]:
        s = s.replace(ch, "_")
    return s.strip()[:60]


def get_station_count_from_sheet(sheet_name):
    import re
    m = re.search(r"Caltech_(\d+)_stations", str(sheet_name))
    return int(m.group(1)) if m else None


def make_seed(*values):
    seed = BASE_RANDOM_SEED
    for v in values:
        seed = (seed * 1315423911 + int(v) * 2654435761) % (2**31 - 1)
    return int(seed)


def get_sheet_seed(sheet_name, run, station_count):
    try:
        sheet_rank = DESIRED_SHEET_ORDER.index(sheet_name) + 1
    except ValueError:
        sheet_rank = 99
    return make_seed(run, station_count, sheet_rank)


def find_input_xlsx(input_name=INPUT_XLSX):
    candidates = [
        input_name,
        os.path.join(".", input_name),
        os.path.join("/mnt/data", input_name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    if os.path.exists("/kaggle/input"):
        for root, _, files in os.walk("/kaggle/input"):
            if input_name in files:
                return os.path.join(root, input_name)

    raise FileNotFoundError(
        f"{input_name} was not found. Please place this Excel file in the same directory as this program,"
        f"or upload it to Kaggle as a dataset."
    )


def preprocess_caltech_dataframe(raw_df, sheet_name):
    df0 = raw_df.copy()

    if "connection_time_copy" not in df0.columns:
        raise KeyError(f"The `connection_time_copy` field was not found in `sheet={sheet_name}`.")

    df0["connection_time_copy"] = pd.to_datetime(df0["connection_time_copy"], errors="coerce")

    df0 = df0[
        (df0.parking_time <= df0.Requested_parking_time + 2) &
        (df0.kWhRequested <= 150) &
        (df0.kWhDelivered <= df0.kWhRequested)
    ].dropna(subset=["connection_time_copy"]).reset_index(drop=True)

    df0["hour"] = df0.connection_time_copy.dt.hour
    df0["weekday"] = df0.connection_time_copy.dt.weekday
    df0["month"] = df0.connection_time_copy.dt.month
    df0["dayofyear"] = df0.connection_time_copy.dt.dayofyear
    for c in ["hour", "weekday", "month", "dayofyear"]:
        df0[f"{c}_sin"] = np.sin(2 * np.pi * df0[c] / df0[c].max())
        df0[f"{c}_cos"] = np.cos(2 * np.pi * df0[c] / df0[c].max())

    for c in ["userID", "stationID"]:
        df0[c] = LabelEncoder().fit_transform(df0[c].astype(str))

    df0.sort_values("connection_time_copy", inplace=True)
    df0.reset_index(drop=True, inplace=True)
    return df0


def load_scalability_sheets(input_xlsx=INPUT_XLSX):
    input_path = find_input_xlsx(input_xlsx)
    xls = pd.ExcelFile(input_path)

    available = set(xls.sheet_names)
    sheet_names = [s for s in DESIRED_SHEET_ORDER if s in available]

    missing = [s for s in DESIRED_SHEET_ORDER if s not in available]
    if missing:
        print(f"Warning: The following sheet was not found in Excel and will be skipped: {missing}")

    if not sheet_names:
        sheet_names = [
            s for s in xls.sheet_names
            if str(s).startswith("Caltech_") and str(s).endswith("_stations")
        ]
        sheet_names = sorted(
            sheet_names,
            key=lambda s: get_station_count_from_sheet(s) if get_station_count_from_sheet(s) is not None else -1,
            reverse=True
        )

    if not sheet_names:
        raise ValueError(
            f"No sheet in the format Caltech_*_stations was found in {input_path}."
        )

    df0_by_sheet = {}
    sheet_station_counts = {}

    print("================ Scalability Analysis: Reading Excel Data ================")
    print(f"Input file: {input_path}")
    print(f"Identified modeling sheets: {sheet_names}")

    for sheet in sheet_names:
        raw = pd.read_excel(input_path, sheet_name=sheet)
        df0 = preprocess_caltech_dataframe(raw, sheet)
        station_count = get_station_count_from_sheet(sheet)
        if station_count is None:
            station_count = int(df0["stationID"].nunique())

        df0_by_sheet[sheet] = df0
        sheet_station_counts[sheet] = station_count
        print(f"  {sheet}: stations={station_count}, samples_after_filter={len(df0)}")

    return df0_by_sheet, sheet_station_counts


def print_run_summary(run, sheet_name, station_count, run_summary):
    print(f"\nResults for run {run} | sheet={sheet_name} | stations={station_count}:")
    print("window | MAE_pt | SMAPE_pt | rounds_pt | time_pt(ms) | MAE_kWh | SMAPE_kWh | rounds_kWh | time_kWh(ms)")
    for (w, mpt, spt, rp, tp, mk, sk, rk, tk, _) in run_summary:
        print(f"{w:>6} | {mpt:6.3f} | {spt:7.3f}% | {rp:9d} | {tp:10.0f} | {mk:7.3f} | {sk:9.3f}% | {rk:10d} | {tk:11.0f}")


def print_collected_run_summaries(run, run_all_sheet_summaries):
    print(f"\n================ Run {run}: Summary of results across five sheets ================")
    for sheet_name, (station_count, run_summary) in run_all_sheet_summaries.items():
        print_run_summary(run, sheet_name, station_count, run_summary)


def print_and_save_final_mean_tables(metrics_all):
    if not metrics_all:
        print("No results to display.")
        return

    df_metrics = pd.DataFrame(metrics_all)
    os.makedirs(RESULT_ROOT, exist_ok=True)

    metric_cols = [
        "mae_pt", "smape_pt", "rounds_pt", "time_pt_ms",
        "mae_kWh", "smape_kWh", "rounds_kWh", "time_kWh_ms", "window_time_ms"
    ]

    print("\n================ All rounds completed: Output a table of average results by sheet ================")
    for sheet_name in df_metrics["sheet"].drop_duplicates().tolist():
        sub = df_metrics[df_metrics["sheet"] == sheet_name].copy()
        station_count = int(sub["number_of_stations"].iloc[0])
        mean_table = (
            sub.groupby("window", as_index=False)[metric_cols]
               .mean()
               .sort_values("window")
        )

        out_path = os.path.join(RESULT_ROOT, f"final_mean_summary_{safe_name(sheet_name)}.csv")
        mean_table.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\nFinal mean table | sheet={sheet_name} | stations={station_count}")
        print("window | MAE_pt | SMAPE_pt | rounds_pt | time_pt(ms) | MAE_kWh | SMAPE_kWh | rounds_kWh | time_kWh(ms)")
        for _, row in mean_table.iterrows():
            print(
                f"{int(row['window']):>6} | "
                f"{row['mae_pt']:6.3f} | {row['smape_pt']:7.3f}% | "
                f"{int(round(row['rounds_pt'])):9d} | {row['time_pt_ms']:10.0f} | "
                f"{row['mae_kWh']:7.3f} | {row['smape_kWh']:9.3f}% | "
                f"{int(round(row['rounds_kWh'])):10d} | {row['time_kWh_ms']:11.0f}"
            )

def main():
    os.makedirs(RESULT_ROOT, exist_ok=True)
    metrics_all = []

    df0_by_sheet, sheet_station_counts = load_scalability_sheets(INPUT_XLSX)

    test_start = datetime(2019, 12, 1)
    test_end = datetime(2019, 12, 30)
    windows = [30, 60, 120, 240, 360, 480]
    n_runs = 1
    eps = 1e-3

    PT_FIXED_DEPTH  = 6
    PT_FIXED_GAMMA  = 1.0
    PT_FIXED_ETA    = 0.05
    PT_MAX_ROUNDS   = 200
    PT_EARLY_STOP   = 30
    PT_PREHEAT_ROUNDS = 15
    PT_DYNAMIC_INTERVAL = 5

    KWH_FIXED_DEPTH  = 6
    KWH_FIXED_GAMMA  = 1.0
    KWH_FIXED_ETA    = 0.05
    KWH_FIXED_ALPHA  = 1.0
    KWH_MAX_ROUNDS   = 200
    KWH_EARLY_STOP   = 30
    KWH_PREHEAT_ROUNDS = 15
    KWH_DYNAMIC_INTERVAL = 5

    LGB_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": 200,
        "learning_rate": 0.02,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "random_state": 42
    }

    for run in range(1, n_runs + 1):
        run_all_sheet_summaries = {}

        for sheet_name, df0 in df0_by_sheet.items():
            station_count = sheet_station_counts[sheet_name]
            sheet_key = safe_name(sheet_name)
            sheet_seed = get_sheet_seed(sheet_name, run, station_count)
            np.random.seed(sheet_seed)
            run_str = str(run)

            base_dirs = [
                os.path.join(RESULT_ROOT, sheet_key, "parking_time"),
                os.path.join(RESULT_ROOT, sheet_key, "kWhDelivered"),
            ]
            for base in base_dirs:
                folder_path = os.path.join(base, run_str)
                os.makedirs(folder_path, exist_ok=True)
                print(f"The folder exists:{folder_path}")

            print(f"\n=== Run {run} | Sheet={sheet_name} | Stations={station_count} | Seed={sheet_seed} ===")
            df = df0.copy()

            hist_all = df[df.connection_time_copy < test_start]
            ua = hist_all.groupby("userID").kWhDelivered.mean() / (
                 hist_all.groupby("userID").kWhRequested.mean() + 1e-6
            )
            km = KMeans(n_clusters=10, random_state=sheet_seed).fit(ua.values.reshape(-1, 1))
            cluster_map = dict(zip(ua.index, km.labels_))
            df["cluster"] = df.userID.map(cluster_map).fillna(0).astype(int)

            run_summary = []

            for w in windows:
                print(f"\n>>> Window = {w} days")
                t_window_start = time.perf_counter()

                tr0 = test_start - timedelta(days=w)
                tr = df[(df.connection_time_copy >= tr0) & (df.connection_time_copy < test_start)].copy()
                te = df[(df.connection_time_copy >= test_start) & (df.connection_time_copy <= test_end)].copy()
                if tr.empty or te.empty:
                    continue

                for col in ["parking_time", "kWhDelivered"]:
                    coeffs = pywt.wavedec(tr[col], 'db4', level=3)
                    σ = np.median(np.abs(coeffs[-1])) / 0.6745
                    thr = σ * np.sqrt(2 * np.log(len(tr)))
                    coeffs[1:] = [pywt.threshold(c, thr, 'soft') for c in coeffs[1:]]
                    tr[col] = pywt.waverec(coeffs, 'db4')[:len(tr)]
                mask = LocalOutlierFactor(n_neighbors=20).fit_predict(tr[["parking_time", "kWhDelivered"]])
                tr   = tr[mask == 1].reset_index(drop=True)
                for col in ["parking_time", "kWhDelivered"]:
                    coeffs = pywt.wavedec(te[col], 'db4', level=3)
                    coeffs[1:] = [pywt.threshold(c, thr, 'soft') for c in coeffs[1:]]
                    te[col] = pywt.waverec(coeffs, 'db4')[:len(te)]

                tr["age_days"] = (test_start - tr.connection_time_copy).dt.days
                w_decay = np.exp(-0.015 * tr["age_days"])

                tr.reset_index(drop=True, inplace=True)
                te.reset_index(drop=True, inplace=True)

                tr["is_weekend"] = tr.weekday.isin([5, 6]).astype(int)
                te["is_weekend"] = te.weekday.isin([5, 6]).astype(int)
                tr["is_holiday"] = tr.get("connectionTime_is_holiday", 0)
                te["is_holiday"] = te.get("connectionTime_is_holiday", 0)

                sim_feats = ["hour", "weekday", "kWhRequested", "Requested_parking_time"]
                ss = StandardScaler().fit(tr[sim_feats])
                tr_s = ss.transform(tr[sim_feats])
                te_s = ss.transform(te[sim_feats])

                tr["sim_park"] = tr["sim_kwh"] = 0.0
                for i in range(len(tr)):
                    prev_idx = tr[(tr.userID == tr.loc[i, "userID"]) & (tr.index < i)].index
                    if len(prev_idx) == 0:
                        tr.at[i, "sim_park"] = tr.parking_time.median()
                        tr.at[i, "sim_kwh"] = tr.kWhDelivered.median()
                    else:
                        sims = np.dot(tr_s[prev_idx], tr_s[i])
                        top_idx = prev_idx[np.argsort(sims)[-5:]]     # Top5
                        w5 = sims[np.argsort(sims)[-5:]]
                        tr.at[i, "sim_park"] = np.average(tr.loc[top_idx, "parking_time"], weights=w5 + 1e-6)
                        tr.at[i, "sim_kwh"] = np.average(tr.loc[top_idx, "kWhDelivered"],   weights=w5 + 1e-6)

                te["sim_park"] = te["sim_kwh"] = 0.0
                for i in range(len(te)):
                    prev_idx = tr[tr.userID == te.loc[i, "userID"]].index
                    if len(prev_idx) == 0:
                        te.at[i, "sim_park"] = tr.parking_time.median()
                        te.at[i, "sim_kwh"] = tr.kWhDelivered.median()
                    else:
                        sims = np.dot(tr_s[prev_idx], te_s[i])
                        top_idx = prev_idx[np.argsort(sims)[-5:]]      # Top5
                        w5 = sims[np.argsort(sims)[-5:]]
                        te.at[i, "sim_park"] = np.average(tr.loc[top_idx, "parking_time"], weights=w5 + 1e-6)
                        te.at[i, "sim_kwh"] = np.average(tr.loc[top_idx, "kWhDelivered"],   weights=w5 + 1e-6)

                ua_m = tr.groupby("userID").kWhDelivered.mean()
                ua_c = tr.userID.value_counts()
                sa_m = tr.groupby("stationID").kWhDelivered.mean()
                sa_c = tr.stationID.value_counts()
                for D in (tr, te):
                    D["user_avg_kWh"] = D.userID.map(ua_m).fillna(ua_m.mean())
                    D["user_freq"] = D.userID.map(ua_c).fillna(0)
                    D["station_avg_kWh"] = D.stationID.map(sa_m).fillna(sa_m.mean())
                    D["station_freq"] = D.stationID.map(sa_c).fillna(0)

                tr["hour_of_week"] = tr.weekday * 24 + tr.hour
                te["hour_of_week"] = te.weekday * 24 + te.hour

                tr["req_rate"] = tr.kWhRequested / (tr.Requested_parking_time + 1e-6)
                te["req_rate"] = te.kWhRequested / (te.Requested_parking_time + 1e-6)

                sh = tr.groupby(["stationID", "hour_of_week"]).kWhDelivered.sum() / \
                     (tr.groupby(["stationID", "hour_of_week"]).kWhRequested.sum() + 1e-6)
                uh = tr.groupby(["userID", "hour_of_week"]).kWhDelivered.sum() / \
                     (tr.groupby(["userID", "hour_of_week"]).kWhRequested.sum() + 1e-6)
                shu = tr.groupby(["stationID", "userID", "hour_of_week"]).kWhDelivered.sum() / \
                      (tr.groupby(["stationID", "userID", "hour_of_week"]).kWhRequested.sum() + 1e-6)

                tr["sh_ratio"] = tr.set_index(["stationID", "hour_of_week"]).index.map(sh)
                tr["uh_ratio"] = tr.set_index(["userID", "hour_of_week"]).index.map(uh)
                tr["shur_ratio"] = tr.set_index(["stationID", "userID", "hour_of_week"]).index.map(shu)

                te["sh_ratio"] = te.set_index(["stationID", "hour_of_week"]).index.map(sh).fillna(sh.mean())
                te["uh_ratio"] = te.set_index(["userID", "hour_of_week"]).index.map(uh).fillna(uh.mean())
                te["shur_ratio"] = te.set_index(["stationID", "userID", "hour_of_week"]).index.map(shu).fillna(shu.mean())

                pf = [
                    "hour", "weekday", "month", "is_weekend", "is_holiday",
                    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                    "month_sin", "month_cos", "userID", "stationID", "sim_park"
                ]
                Xp_tr, y_p_tr = tr[pf].values, tr.parking_time.values
                Xp_te, y_p_te = te[pf].values, te.parking_time.values

                n_val = max(int(0.2 * len(tr)), 50)
                Xtr_p, Xva_p = Xp_tr[:-n_val], Xp_tr[-n_val:]
                ytr_p, yva_p = y_p_tr[:-n_val], y_p_tr[-n_val:]
                wtr_p = w_decay[:-n_val]

                dtrain_p = xgb.DMatrix(Xtr_p, label=ytr_p, weight=wtr_p)
                dval_p = xgb.DMatrix(Xva_p, label=yva_p)

                pt_val_maes = []
                model_p = None
                best_va_p = np.inf
                rounds_no_improve_p = 0

                prev_depth_p = None
                prev_gamma_p = None

                fixed_depth_p = PT_FIXED_DEPTH
                fixed_gamma_p = PT_FIXED_GAMMA

                start_p = time.perf_counter()
                for m in range(PT_MAX_ROUNDS):
                    if m < PT_PREHEAT_ROUNDS:
                        depth_p = fixed_depth_p
                        gamma_p = fixed_gamma_p
                    else:
                        if (m - PT_PREHEAT_ROUNDS) % PT_DYNAMIC_INTERVAL == 0:
                            if model_p is None:
                                residuals_p = ytr_p.copy()
                            else:
                                residuals_p = ytr_p - model_p.predict(dtrain_p)

                            depth_p = compute_dynamic_max_depth_advanced(
                                residuals_p, wtr_p,
                                d0=PT_FIXED_DEPTH, δ=1.5, δp=1.2,
                                min_depth=2, max_depth_limit=20,
                                prev_depth=prev_depth_p,
                                trim_ratio=0.05,
                                smooth_factor=0.3
                            )
                            gamma_p = compute_dynamic_gamma_advanced(
                                residuals_p, wtr_p,
                                base=1.0, α=0.7,
                                min_gamma=0.001, max_gamma=20.0,
                                prev_gamma=prev_gamma_p,
                                trim_ratio=0.05,
                                smooth_factor=0.3
                            )
                            prev_depth_p = depth_p
                            prev_gamma_p = gamma_p

                    params_p = {
                        "tree_method": "hist",
                        "eta": PT_FIXED_ETA,
                        "max_depth": depth_p,
                        "gamma": gamma_p,
                        "objective": "reg:squarederror",
                        "eval_metric": "mae",
                        "verbosity": 0,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "seed": make_seed(sheet_seed, w, m, 101)
                    }

                    if model_p is None:
                        model_p = xgb.train(params_p, dtrain_p, num_boost_round=1, verbose_eval=False)
                    else:
                        model_p = xgb.train(params_p, dtrain_p, num_boost_round=1,
                                            xgb_model=model_p, verbose_eval=False)

                    pred_va_p = model_p.predict(dval_p)
                    va_mae_p = mae(yva_p, pred_va_p)
                    pt_val_maes.append(va_mae_p)

                    if va_mae_p < best_va_p - 1e-6:
                        best_va_p = va_mae_p
                        rounds_no_improve_p = 0
                    else:
                        rounds_no_improve_p += 1

                    if rounds_no_improve_p >= PT_EARLY_STOP:
                        break

                end_p = time.perf_counter()
                rounds_p = len(pt_val_maes)
                time_p_ms = (end_p - start_p) * 1000

                pred_p_te = model_p.predict(xgb.DMatrix(Xp_te))
                mae_p, sm_p = mae(y_p_te, pred_p_te), smape(y_p_te, pred_p_te)

                fig, ax = plt.subplots(figsize=(6,4))
                ax.plot(np.arange(1, rounds_p+1), pt_val_maes, marker='o', linewidth=1)
                ax.set_title(f"Window={w}  parking_time  Val MAE vs Rounds")
                ax.set_xlabel("Round")
                ax.set_ylabel("Validation MAE")
                plt.tight_layout()
                folder1 = os.path.join(".", "results", "caltech_scalability", sheet_key, "parking_time", str(run))
                os.makedirs(folder1, exist_ok=True)

                filename = f"convergence_parking_time_window_{w}.png"
                save_path = os.path.join(folder1, filename)

                plt.savefig(save_path)
                plt.close(fig)

                pt_transform = PowerTransformer(method='box-cox', standardize=False)
                y_all_k = tr.kWhDelivered.values + eps
                y_bc = pt_transform.fit_transform(y_all_k.reshape(-1, 1)).flatten()

                tr = tr.sort_values("connection_time_copy").reset_index(drop=True)
                te = te.sort_values("connection_time_copy").reset_index(drop=True)

                user_roll3 = tr.groupby("userID")["kWhDelivered"] \
                                .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean()) \
                                .reset_index(level=0, drop=True)
                tr["user_roll_kWh3"] = user_roll3.fillna(tr.kWhDelivered.median())

                station_roll3 = tr.groupby("stationID")["kWhDelivered"] \
                                   .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean()) \
                                   .reset_index(level=0, drop=True)
                tr["station_roll_kWh3"] = station_roll3.fillna(tr.kWhDelivered.median())

                user_hist = tr.groupby("userID")["kWhDelivered"].apply(list).to_dict()
                station_hist = tr.groupby("stationID")["kWhDelivered"].apply(list).to_dict()

                te["user_roll_kWh3"] = te["userID"].apply(lambda u: (
                    np.mean(user_hist[u][-3:]) if (u in user_hist and len(user_hist[u]) > 0)
                    else tr.kWhDelivered.median()
                ))
                te["station_roll_kWh3"] = te["stationID"].apply(lambda s: (
                    np.mean(station_hist[s][-3:]) if (s in station_hist and len(station_hist[s]) > 0)
                    else tr.kWhDelivered.median()
                ))

                med_k = tr.kWhDelivered.median()
                tr["pk1"] = tr.groupby("userID").kWhDelivered.shift(1).fillna(med_k)
                tr["pk2"] = tr.groupby("userID").kWhDelivered.shift(2).fillna(med_k)
                tr["rk3"] = tr.groupby("userID").kWhDelivered.rolling(3, min_periods=1).mean() \
                               .reset_index(0, drop=True).fillna(med_k)

                te["pk1"] = med_k
                te["pk2"] = med_k
                te["rk3"] = med_k
                prev_vals = []
                for i in range(len(te)):
                    if i >= 1: te.at[i, "pk1"] = prev_vals[i-1]
                    if i >= 2: te.at[i, "pk2"] = prev_vals[i-2]
                    if i >= 3: te.at[i, "rk3"] = np.mean(prev_vals[i-3:i])
                    prev_vals.append(te.at[i, "kWhDelivered"])

                feats_k = [
                    "hour", "weekday", "month", "is_weekend", "is_holiday",
                    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                    "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
                    "cluster", "sim_kwh", "user_avg_kWh", "user_freq",
                    "station_avg_kWh", "station_freq",
                    "req_rate", "hour_of_week",
                    "sh_ratio", "uh_ratio", "shur_ratio",
                    "pk1", "pk2", "rk3",
                    "station_roll_kWh3"
                ]

                X_all_tr_k = tr[feats_k].values
                X_all_te_k = te[feats_k].values

                Xk_tr = X_all_tr_k[:-n_val]
                Xk_va = X_all_tr_k[-n_val:]
                ytr_bc = y_bc[:-n_val]
                yva_bc = y_bc[-n_val:]
                wtr_k = w_decay[:-n_val]

                dtrain_k = xgb.DMatrix(Xk_tr, label=ytr_bc, weight=wtr_k)
                dval_k = xgb.DMatrix(Xk_va, label=yva_bc)

                best_va_k = np.inf
                best_round_k = 0
                kwh_val_maes = []
                model_k = None
                best_va_k = np.inf
                rounds_no_improve_k = 0

                prev_depth_k = None
                prev_gamma_k = None

                fixed_depth_k = KWH_FIXED_DEPTH
                fixed_gamma_k = KWH_FIXED_GAMMA

                start_k = time.perf_counter()
                for m in range(KWH_MAX_ROUNDS):
                    if m < KWH_PREHEAT_ROUNDS:
                        depth_k = fixed_depth_k
                        gamma_k = fixed_gamma_k
                    else:
                        if (m - KWH_PREHEAT_ROUNDS) % KWH_DYNAMIC_INTERVAL == 0:
                            if model_k is None:
                                residuals_k = ytr_bc.copy()
                            else:
                                residuals_k = ytr_bc - model_k.predict(xgb.DMatrix(Xk_tr, label=ytr_bc, weight=wtr_k))

                            depth_k = compute_dynamic_max_depth_advanced(
                                residuals_k, wtr_k,
                                d0=KWH_FIXED_DEPTH, δ=1.5, δp=1.2,
                                min_depth=2, max_depth_limit=20,
                                prev_depth=prev_depth_k,
                                trim_ratio=0.05,
                                smooth_factor=0.3
                            )
                            gamma_k = compute_dynamic_gamma_advanced(
                                residuals_k, wtr_k,
                                base=1.0, α=0.7,
                                min_gamma=0.001, max_gamma=20.0,
                                prev_gamma=prev_gamma_k,
                                trim_ratio=0.05,
                                smooth_factor=0.3
                            )
                            prev_depth_k = depth_k
                            prev_gamma_k = gamma_k

                    if model_k is None:
                        residuals_k = ytr_bc.copy()
                    else:
                        residuals_k = ytr_bc - model_k.predict(xgb.DMatrix(Xk_tr, label=ytr_bc, weight=wtr_k))
                    huber_alpha = compute_huber_alpha(residuals_k, scale=2.0)

                    params_k = {
                        "tree_method": "hist",
                        "eta": KWH_FIXED_ETA,
                        "max_depth": depth_k,
                        "gamma": gamma_k,
                        "objective": "reg:pseudohubererror",
                        "alpha": huber_alpha,
                        "eval_metric": "mae",
                        "verbosity": 0,
                        "seed": make_seed(sheet_seed, w, m, 202)
                    }

                    if model_k is None:
                        model_k = xgb.train(params_k, dtrain_k, num_boost_round=1, verbose_eval=False)
                    else:
                        model_k = xgb.train(params_k, dtrain_k, num_boost_round=1,
                                            xgb_model=model_k, verbose_eval=False)

                    pred_va_k_now = model_k.predict(dval_k)
                    va_k_mae = mae(yva_bc, pred_va_k_now)
                    kwh_val_maes.append(va_k_mae)

                    if va_k_mae < best_va_k - 1e-6:
                        best_va_k = va_k_mae
                        best_round_k = m + 1
                        rounds_no_improve_k = 0
                    else:
                        rounds_no_improve_k += 1

                    if rounds_no_improve_k >= KWH_EARLY_STOP:
                        break

                end_k = time.perf_counter()
                rounds_k = len(kwh_val_maes)
                time_k_ms = (end_k - start_k) * 1000

                pred_tr_k_full = model_k.predict(xgb.DMatrix(Xk_tr, label=ytr_bc, weight=wtr_k))
                res_tr_bc = ytr_bc - pred_tr_k_full

                Xk_tr_df = pd.DataFrame(Xk_tr, columns=feats_k)
                Xk_va_df = pd.DataFrame(Xk_va, columns=feats_k)
                X_all_te_df = pd.DataFrame(X_all_te_k, columns=feats_k)

                lgb_params_this = dict(LGB_PARAMS)
                lgb_params_this["random_state"] = make_seed(sheet_seed, w, 0, 303)
                model_lgb = lgb.LGBMRegressor(**lgb_params_this)
                model_lgb.fit(Xk_tr_df, res_tr_bc, sample_weight=wtr_k)

                pred_te_k_huber = model_k.predict(xgb.DMatrix(X_all_te_k))
                res_te_bc = model_lgb.predict(X_all_te_df)
                final_huber = pred_te_k_huber + res_te_bc

                clipped = np.maximum(final_huber.reshape(-1, 1), 1e-6)
                inv_te_k = pt_transform.inverse_transform(clipped).flatten() - eps

                order = np.argsort(te.connection_time_copy.values)
                smooth = medfilt(inv_te_k[order], kernel_size=5)
                final_k = np.empty_like(smooth); final_k[order] = smooth

                mae_k, sm_k = mae(te.kWhDelivered.values, final_k), smape(te.kWhDelivered.values, final_k)

                fig, ax = plt.subplots(figsize=(6,4))
                x = np.arange(1, rounds_k + 1)
                y = kwh_val_maes
                ax.plot(x, y, marker='o', linewidth=1, color='orange')
                ax.set_title(f"Window={w}  kWhDelivered  Val MAE vs Rounds")
                ax.set_xlabel("Round")
                ax.set_ylabel("Validation MAE")
                plt.tight_layout()
                folder2 = os.path.join(".", "results", "caltech_scalability", sheet_key, "kWhDelivered", str(run))
                os.makedirs(folder2, exist_ok=True)

                filename = f"convergence_kWh_time_window_{w}.png"
                save_path = os.path.join(folder2, filename)

                plt.savefig(save_path)
                plt.close(fig)

                t_window_end = time.perf_counter()
                t_window_ms = (t_window_end - t_window_start) * 1000

                print(f"  Window={w}:")
                print(f"    parking_time → MAE={mae_p:.3f}, SMAPE={sm_p:.3f}%, rounds={rounds_p}, time={time_p_ms:.0f}ms")
                print(f"    kWhDelivered → MAE={mae_k:.3f}, SMAPE={sm_k:.3f}%, rounds={rounds_k}, time={time_k_ms:.0f}ms")

                run_summary.append((
                    w,
                    mae_p, sm_p, rounds_p, time_p_ms,
                    mae_k, sm_k, rounds_k, time_k_ms,
                    t_window_ms
                ))
                metrics_all.append({
                    "sheet": sheet_name,
                    "number_of_stations": station_count,
                    "sheet_seed": sheet_seed,
                    "run": run,
                    "window": w,
                    "mae_pt": mae_p,
                    "smape_pt": sm_p,
                    "rounds_pt": rounds_p,
                    "time_pt_ms": time_p_ms,
                    "mae_kWh": mae_k,
                    "smape_kWh": sm_k,
                    "rounds_kWh": rounds_k,
                    "time_kWh_ms": time_k_ms,
                    "window_time_ms": t_window_ms
                })

            print_run_summary(run, sheet_name, station_count, run_summary)
            run_all_sheet_summaries[sheet_name] = (station_count, list(run_summary))

        print_collected_run_summaries(run, run_all_sheet_summaries)

    metrics_df = pd.DataFrame(metrics_all)
    metrics_path = os.path.join(RESULT_ROOT, "metrics_mixed_dynamic_scalability.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    for sheet_name in metrics_df["sheet"].drop_duplicates().tolist():
        sheet_metrics = metrics_df[metrics_df["sheet"] == sheet_name].copy()
        sheet_metrics.to_csv(
            os.path.join(RESULT_ROOT, f"metrics_{safe_name(sheet_name)}.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print_and_save_final_mean_tables(metrics_all)
    print(f"\nAll scalability runs complete. Results saved to {metrics_path}")


if __name__ == "__main__":
    main()
