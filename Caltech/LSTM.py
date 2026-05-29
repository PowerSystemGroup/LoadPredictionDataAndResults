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
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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


def main():
    os.makedirs("./results", exist_ok=True)
    metrics_all = []

    df0 = pd.read_csv("caltech_test_data.csv", parse_dates=["connection_time_copy"])
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

    test_start = datetime(2019, 12, 1)
    test_end = datetime(2019, 12, 30)
    windows = [30, 60, 120, 240, 360, 480]
    n_runs = 20
    eps = 1e-3

    base_dirs = [
        os.path.join(".", "results", "LSTM", "parking_time"),
        os.path.join(".", "results", "LSTM", "kWhDelivered"),
    ]

    for run in range(1, n_runs + 1):
        run_str = str(run)
        for base in base_dirs:
            folder_path = os.path.join(base, run_str)
            os.makedirs(folder_path, exist_ok=True)
            print(f"The folder has been verified: {folder_path}")
        print(f"\n=== Run {run} ===")
        df = df0.copy()

        hist_all = df[df.connection_time_copy < test_start]
        ua = hist_all.groupby("userID").kWhDelivered.mean() / (
             hist_all.groupby("userID").kWhRequested.mean() + 1e-6
        )
        km = KMeans(n_clusters=10, random_state=run).fit(ua.values.reshape(-1, 1))
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
                    top_idx = prev_idx[np.argsort(sims)[-5:]]
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
                    top_idx = prev_idx[np.argsort(sims)[-5:]]
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

            Xtr_tensor = torch.tensor(Xtr_p, dtype=torch.float32).unsqueeze(1)
            ytr_tensor = torch.tensor(ytr_p, dtype=torch.float32).unsqueeze(1)
            Xva_tensor = torch.tensor(Xva_p, dtype=torch.float32).unsqueeze(1)
            yva_tensor = torch.tensor(yva_p, dtype=torch.float32).unsqueeze(1)

            train_dataset = TensorDataset(Xtr_tensor, ytr_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

            class LSTMRegressor(nn.Module):
                def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
                    super(LSTMRegressor, self).__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                    self.drop = nn.Dropout(dropout)
                    self.out = nn.Linear(hidden_dim, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    out = self.drop(out[:, -1, :])
                    return self.out(out)

            model_p = LSTMRegressor(input_dim=Xtr_p.shape[1]).to("cpu")
            loss_fn = nn.L1Loss()
            optimizer = torch.optim.Adam(model_p.parameters(), lr=0.01)

            start_p = time.perf_counter()
            val_losses = []
            best_va_p = float("inf")
            rounds_no_improve = 0
            rounds_p = 0
            MAX_EPOCHS = 200
            EARLY_STOP = 20

            for epoch in range(MAX_EPOCHS):
                model_p.train()
                for xb, yb in train_loader:
                    pred = model_p(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model_p.eval()
                with torch.no_grad():
                    pred_val = model_p(Xva_tensor).squeeze().numpy()
                val_mae = mae(yva_p, pred_val)
                val_losses.append(val_mae)
                rounds_p += 1

                if val_mae < best_va_p - 1e-6:
                    best_va_p = val_mae
                    best_model_state = model_p.state_dict()
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if rounds_no_improve >= EARLY_STOP:
                    break

            end_p = time.perf_counter()
            time_p_ms = (end_p - start_p) * 1000

            model_p.load_state_dict(best_model_state)
            model_p.eval()
            with torch.no_grad():
                pred_te_p = model_p(torch.tensor(Xp_te, dtype=torch.float32).unsqueeze(1)).squeeze().numpy()

            mae_p = mae(y_p_te, pred_te_p)
            sm_p = smape(y_p_te, pred_te_p)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(1, len(val_losses) + 1), val_losses, marker='o', linewidth=1)
            ax.set_title(f"Window={w}  parking_time (LSTM) Val MAE")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation MAE")
            plt.tight_layout()

            folder1 = os.path.join(".", "results", "jpl", "parking_time", str(run))
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
                if i >= 1: te.at[i, "pk1"] = prev_vals[i - 1]
                if i >= 2: te.at[i, "pk2"] = prev_vals[i - 2]
                if i >= 3: te.at[i, "rk3"] = np.mean(prev_vals[i - 3:i])
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

            n_val = max(int(0.2 * len(tr)), 50)
            Xk_tr = X_all_tr_k[:-n_val]
            Xk_va = X_all_tr_k[-n_val:]
            ytr_bc = y_bc[:-n_val]
            yva_bc = y_bc[-n_val:]

            Xtr_tensor = torch.tensor(Xk_tr, dtype=torch.float32).unsqueeze(1)
            ytr_tensor = torch.tensor(ytr_bc, dtype=torch.float32).unsqueeze(1)
            Xva_tensor = torch.tensor(Xk_va, dtype=torch.float32).unsqueeze(1)
            yva_tensor = torch.tensor(yva_bc, dtype=torch.float32).unsqueeze(1)
            Xte_tensor = torch.tensor(X_all_te_k, dtype=torch.float32).unsqueeze(1)

            train_loader = DataLoader(TensorDataset(Xtr_tensor, ytr_tensor), batch_size=64, shuffle=True)

            class LSTMRegressor(nn.Module):
                def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                    self.drop = nn.Dropout(dropout)
                    self.out = nn.Linear(hidden_dim, 1)

                def forward(self, x):
                    out, _ = self.lstm(x)
                    out = self.drop(out[:, -1, :])
                    return self.out(out)

            model_k = LSTMRegressor(input_dim=Xk_tr.shape[1])
            loss_fn = nn.L1Loss()
            optimizer = torch.optim.Adam(model_k.parameters(), lr=0.01)

            start_k = time.perf_counter()
            val_losses = []
            best_va_k = float("inf")
            rounds_k = 0
            rounds_no_improve = 0
            MAX_EPOCHS = 200
            EARLY_STOP = 20

            for epoch in range(MAX_EPOCHS):
                model_k.train()
                for xb, yb in train_loader:
                    pred = model_k(xb)
                    loss = loss_fn(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model_k.eval()
                with torch.no_grad():
                    pred_va_k = model_k(Xva_tensor).squeeze().numpy()
                va_mae = mae(yva_bc, pred_va_k)
                val_losses.append(va_mae)
                rounds_k += 1

                if va_mae < best_va_k - 1e-6:
                    best_va_k = va_mae
                    best_model_state_k = model_k.state_dict()
                    rounds_no_improve = 0
                else:
                    rounds_no_improve += 1

                if rounds_no_improve >= EARLY_STOP:
                    break

            end_k = time.perf_counter()
            time_k_ms = (end_k - start_k) * 1000

            model_k.load_state_dict(best_model_state_k)
            model_k.eval()
            with torch.no_grad():
                pred_te_bc = model_k(Xte_tensor).squeeze().numpy()

            clipped = np.maximum(pred_te_bc.reshape(-1, 1), 1e-6)
            inv_te_k = pt_transform.inverse_transform(clipped).flatten() - eps
            order = np.argsort(te.connection_time_copy.values)
            smooth = medfilt(inv_te_k[order], kernel_size=5)
            final_k = np.empty_like(smooth);
            final_k[order] = smooth

            mae_k = mae(te.kWhDelivered.values, final_k)
            sm_k = smape(te.kWhDelivered.values, final_k)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(np.arange(1, len(val_losses) + 1), val_losses, marker='o', linewidth=1, color='orange')
            ax.set_title(f"Window={w}  kWhDelivered (LSTM) Val MAE")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation MAE")
            plt.tight_layout()

            folder2 = os.path.join(".", "results", "jpl", "kWhDelivered", str(run))
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

        print(f"\nResults for run {run}:")
        print("window | MAE_pt | SMAPE_pt | rounds_pt | time_pt(ms) | MAE_kWh | SMAPE_kWh | rounds_kWh | time_kWh(ms)")
        for (w, mpt, spt, rp, tp, mk, sk, rk, tk, _) in run_summary:
            print(f"{w:>6} | {mpt:6.3f} | {spt:7.3f}% | {rp:9d} | {tp:10.0f} | {mk:7.3f} | {sk:9.3f}% | {rk:10d} | {tk:11.0f}")

    pd.DataFrame(metrics_all).to_csv("./results/LSTM/metrics_mixed_dynamic.csv", index=False)
    print("\nAll runs complete. Results saved to metrics_mixed_dynamic.csv")


if __name__ == "__main__":
    main()
