import os
import gc
import copy
import random
import warnings
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import medfilt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denoise_train_test(tr: pd.DataFrame, te: pd.DataFrame):
    tr = tr.copy()
    te = te.copy()

    for col in ["parking_time", "kWhDelivered"]:
        coeffs = pywt.wavedec(tr[col], "db4", level=3)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        thr = sigma * np.sqrt(2 * np.log(len(tr)))
        coeffs[1:] = [pywt.threshold(c, thr, "soft") for c in coeffs[1:]]
        tr[col] = pywt.waverec(coeffs, "db4")[: len(tr)]

    mask = LocalOutlierFactor(n_neighbors=20).fit_predict(tr[["parking_time", "kWhDelivered"]])
    tr = tr[mask == 1].reset_index(drop=True)

    for col in ["parking_time", "kWhDelivered"]:
        coeffs = pywt.wavedec(te[col], "db4", level=3)
        coeffs[1:] = [pywt.threshold(c, thr, "soft") for c in coeffs[1:]]
        te[col] = pywt.waverec(coeffs, "db4")[: len(te)]

    return tr, te


def save_curve_from_list(values, save_path: str, title: str):
    if values is None or len(values) == 0:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def choose_encoder_length(n_train_samples: int, base_len: int = 32):
    if n_train_samples <= 6:
        return max(2, n_train_samples - 2)
    candidate = max(8, n_train_samples // 5)
    return max(4, min(base_len, n_train_samples - 2, candidate))


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        if self.kernel_size <= 1:
            return x
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_avg = self.avg(x_padded.transpose(1, 2)).transpose(1, 2)
        return x_avg


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinearOneStep(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        moving_avg: int = 25,
        dropout: float = 0.0,
        individual: bool = False,
        hidden_head: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.individual = individual
        self.decomp = SeriesDecomp(moving_avg)

        if individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(input_dim)])
            self.linear_trend = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(input_dim)])
        else:
            self.linear_seasonal = nn.Linear(seq_len, 1)
            self.linear_trend = nn.Linear(seq_len, 1)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_head, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.individual:
            for lin in self.linear_seasonal:
                nn.init.constant_(lin.weight, 1.0 / self.seq_len)
                nn.init.zeros_(lin.bias)
            for lin in self.linear_trend:
                nn.init.constant_(lin.weight, 1.0 / self.seq_len)
                nn.init.zeros_(lin.bias)
        else:
            nn.init.constant_(self.linear_seasonal.weight, 1.0 / self.seq_len)
            nn.init.zeros_(self.linear_seasonal.bias)
            nn.init.constant_(self.linear_trend.weight, 1.0 / self.seq_len)
            nn.init.zeros_(self.linear_trend.bias)

    def forward(self, x):
        seasonal_init, trend_init = self.decomp(x)
        seasonal_init = seasonal_init.transpose(1, 2)
        trend_init = trend_init.transpose(1, 2)

        if self.individual:
            seasonal_out = torch.cat(
                [self.linear_seasonal[i](seasonal_init[:, i, :]).unsqueeze(1) for i in range(self.input_dim)],
                dim=1,
            )
            trend_out = torch.cat(
                [self.linear_trend[i](trend_init[:, i, :]).unsqueeze(1) for i in range(self.input_dim)],
                dim=1,
            )
        else:
            bsz = seasonal_init.shape[0]
            seasonal_out = self.linear_seasonal(seasonal_init.reshape(-1, self.seq_len)).reshape(bsz, self.input_dim, 1)
            trend_out = self.linear_trend(trend_init.reshape(-1, self.seq_len)).reshape(bsz, self.input_dim, 1)

        out = (seasonal_out + trend_out).squeeze(-1)  # [B, C]
        out = self.dropout(out)
        out = self.head(out).squeeze(-1)
        return out


class SequenceForecastDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_model_dataframe(df_part: pd.DataFrame, target_col: str, target_out_col: str):
    data = df_part.copy().reset_index(drop=True)
    data["time_idx"] = np.arange(len(data), dtype=int)
    data[target_out_col] = data[target_col].astype(float)
    return data


def build_supervised_samples(
    df_model: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    index_list,
    encoder_length: int,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
):
    sequences = []
    targets = []

    for t in index_list:
        if t < encoder_length:
            continue
        hist = df_model.iloc[t - encoder_length:t].copy()
        cur = df_model.iloc[t].copy()

        hist_feat = feature_scaler.transform(hist[feature_cols])
        cur_feat = feature_scaler.transform(cur[feature_cols].to_frame().T)[0]
        hist_tgt = target_scaler.transform(hist[[target_col]]).reshape(-1)
        cur_tgt = float(target_scaler.transform(np.array([[cur[target_col]]], dtype=float))[0, 0])

        seq = np.zeros((encoder_length + 1, len(feature_cols) + 1), dtype=np.float32)
        seq[:encoder_length, :-1] = hist_feat.astype(np.float32)
        seq[:encoder_length, -1] = hist_tgt.astype(np.float32)
        seq[encoder_length, :-1] = cur_feat.astype(np.float32)
        seq[encoder_length, -1] = hist_tgt[-1].astype(np.float32)

        sequences.append(seq)
        targets.append(cur_tgt)

    if len(sequences) == 0:
        return np.empty((0, encoder_length + 1, len(feature_cols) + 1), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def build_single_inference_sequence(
    hist_df: pd.DataFrame,
    row: pd.Series,
    feature_cols: list,
    target_col: str,
    encoder_length: int,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
):
    hist = hist_df.sort_values("time_idx").tail(encoder_length).copy()
    if len(hist) < encoder_length:
        raise ValueError(f"There are not enough historical samples; currently, there are only {len(hist)} entries, "
                         f"which is insufficient to construct an input sequence of length {encoder_length}.")

    hist_feat = feature_scaler.transform(hist[feature_cols])
    cur_feat = feature_scaler.transform(row[feature_cols].to_frame().T)[0]
    hist_tgt = target_scaler.transform(hist[[target_col]]).reshape(-1)

    seq = np.zeros((encoder_length + 1, len(feature_cols) + 1), dtype=np.float32)
    seq[:encoder_length, :-1] = hist_feat.astype(np.float32)
    seq[:encoder_length, -1] = hist_tgt.astype(np.float32)
    seq[encoder_length, :-1] = cur_feat.astype(np.float32)
    seq[encoder_length, -1] = hist_tgt[-1].astype(np.float32)
    return seq


def predict_array(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device):
    if len(X) == 0:
        return np.array([], dtype=float)

    loader = DataLoader(
        SequenceForecastDataset(X, np.zeros((len(X),), dtype=np.float32)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            pred = model(xb).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    if len(preds) == 0:
        return np.array([], dtype=float)
    return np.concatenate(preds, axis=0)


def fit_dlinear_model(
    df_model: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    train_end_idx: int,
    val_start_idx: int,
    val_end_idx: int,
    seed: int,
    max_epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    moving_avg: int = 25,
    dropout: float = 0.0,
    individual: bool = False,
    hidden_head: int = 64,
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = df_model[df_model["time_idx"] <= train_end_idx].copy()
    encoder_length = choose_encoder_length(len(train_data), base_len=32)
    if encoder_length < 4:
        raise ValueError(f"There are too few training samples to construct a DLinear sequence: {len(train_data)}")

    feature_scaler = StandardScaler().fit(train_data[feature_cols])
    target_scaler = StandardScaler().fit(train_data[[target_col]])

    train_indices = list(range(encoder_length, train_end_idx + 1))
    val_indices = list(range(max(val_start_idx, encoder_length), val_end_idx + 1))

    X_train, y_train = build_supervised_samples(
        df_model=df_model,
        feature_cols=feature_cols,
        target_col=target_col,
        index_list=train_indices,
        encoder_length=encoder_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )
    X_val, y_val = build_supervised_samples(
        df_model=df_model,
        feature_cols=feature_cols,
        target_col=target_col,
        index_list=val_indices,
        encoder_length=encoder_length,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
    )

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("The training or validation set is empty; DLinear cannot be trained.")

    seq_len = X_train.shape[1]

    train_loader = DataLoader(SequenceForecastDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(SequenceForecastDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)

    model = DLinearOneStep(
        input_dim=X_train.shape[-1],
        seq_len=seq_len,
        moving_avg=moving_avg,
        dropout=dropout,
        individual=individual,
        hidden_head=hidden_head,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.L1Loss()

    best_state = None
    best_val_mae = np.inf
    val_mae_history = []
    epochs_no_improve = 0

    t0 = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_preds_scaled = []
        val_true_scaled = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).detach().cpu().numpy().reshape(-1)
                val_preds_scaled.append(pred)
                val_true_scaled.append(yb.numpy().reshape(-1))

        val_preds_scaled = np.concatenate(val_preds_scaled, axis=0)
        val_true_scaled = np.concatenate(val_true_scaled, axis=0)

        val_preds = target_scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).reshape(-1)
        val_true = target_scaler.inverse_transform(val_true_scaled.reshape(-1, 1)).reshape(-1)
        val_mae = mae(val_true, val_preds)
        val_mae_history.append(val_mae)
        scheduler.step(val_mae)

        if val_mae < best_val_mae - 1e-4:
            best_val_mae = val_mae
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= 5:
            break

    time_ms = (time.perf_counter() - t0) * 1000

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_val_scaled = predict_array(model, X_val, batch_size=batch_size, device=device)
    y_val_scaled = y_val.copy()
    pred_val = target_scaler.inverse_transform(pred_val_scaled.reshape(-1, 1)).reshape(-1)
    y_val_actual = target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).reshape(-1)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "encoder_length": encoder_length,
        "pred_val": pred_val,
        "y_val": y_val_actual,
        "time_ms": time_ms,
        "rounds": len(val_mae_history),
        "best_val_mae": float(best_val_mae),
        "val_mae_history": val_mae_history,
        "device": device,
    }


def recursive_forecast_dlinear(
    model: nn.Module,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    encoder_length: int,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    device: torch.device,
    row_prepare_fn=None,
    row_finalize_fn=None,
):
    hist = history_df.copy().sort_values("time_idx").reset_index(drop=True)
    future = future_df.copy().sort_values("time_idx").reset_index(drop=True)
    preds = []

    model.eval()
    for i in range(len(future)):
        row = future.iloc[i].copy()
        if row_prepare_fn is not None:
            row = row_prepare_fn(hist, row)

        seq = build_single_inference_sequence(
            hist_df=hist,
            row=row,
            feature_cols=feature_cols,
            target_col=target_col,
            encoder_length=encoder_length,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
        )
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = float(model(seq_tensor).detach().cpu().item())
        pred_actual = float(target_scaler.inverse_transform(np.array([[pred_scaled]], dtype=float))[0, 0])
        preds.append(pred_actual)

        row[target_col] = pred_actual
        if row_finalize_fn is not None:
            row = row_finalize_fn(hist, row, pred_actual)
        hist = pd.concat([hist, pd.DataFrame([row])], axis=0, ignore_index=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.asarray(preds, dtype=float)


def prepare_parking_row_from_history(hist_df: pd.DataFrame, row: pd.Series):
    row = row.copy()
    row["parking_time"] = np.nan
    return row


def finalize_parking_row(hist_df: pd.DataFrame, row: pd.Series, pred: float):
    row = row.copy()
    row["parking_time"] = float(pred)
    row["target_parking_time"] = float(pred)
    return row


def prepare_kwh_row_from_history(hist_df: pd.DataFrame, row: pd.Series, med_k: float):
    row = row.copy()
    user = row["userID"]
    station = row["stationID"]

    user_hist = hist_df.loc[hist_df["userID"] == user, "kWhDelivered"].dropna().tolist()
    station_hist = hist_df.loc[hist_df["stationID"] == station, "kWhDelivered"].dropna().tolist()

    row["user_roll_kWh3"] = float(np.mean(user_hist[-3:])) if len(user_hist) > 0 else float(med_k)
    row["station_roll_kWh3"] = float(np.mean(station_hist[-3:])) if len(station_hist) > 0 else float(med_k)
    row["pk1"] = float(user_hist[-1]) if len(user_hist) >= 1 else float(med_k)
    row["pk2"] = float(user_hist[-2]) if len(user_hist) >= 2 else float(med_k)
    row["rk3"] = float(np.mean(user_hist[-3:])) if len(user_hist) > 0 else float(med_k)

    row["kWhDelivered"] = np.nan
    row["kWhDelivered_bc"] = np.nan
    row["target_kwh_bc"] = np.nan
    return row


def make_finalize_kwh_row(pt_transform, eps: float):
    def _finalize(hist_df: pd.DataFrame, row: pd.Series, pred_bc: float):
        row = row.copy()
        pred_kwh = float(pt_transform.inverse_transform(np.array([[pred_bc]], dtype=float)).reshape(-1)[0] - eps)
        pred_kwh = max(pred_kwh, 0.0)
        row["kWhDelivered_bc"] = float(pred_bc)
        row["target_kwh_bc"] = float(pred_bc)
        row["kWhDelivered"] = pred_kwh
        return row

    return _finalize


def main():
    os.makedirs("./Caltech/results", exist_ok=True)
    os.makedirs("./Caltech/results/DLinear_strict", exist_ok=True)
    metrics_all = []

    df0 = pd.read_csv("caltech_test_data.csv", parse_dates=["connection_time_copy"])
    df0 = df0[
        (df0.parking_time <= df0.Requested_parking_time + 2)
        & (df0.kWhRequested <= 150)
        & (df0.kWhDelivered <= df0.kWhRequested)
    ].dropna(subset=["connection_time_copy"]).reset_index(drop=True)

    df0["hour"] = df0.connection_time_copy.dt.hour
    df0["weekday"] = df0.connection_time_copy.dt.weekday
    df0["month"] = df0.connection_time_copy.dt.month
    df0["dayofyear"] = df0.connection_time_copy.dt.dayofyear
    for c in ["hour", "weekday", "month", "dayofyear"]:
        denom = max(df0[c].max(), 1)
        df0[f"{c}_sin"] = np.sin(2 * np.pi * df0[c] / denom)
        df0[f"{c}_cos"] = np.cos(2 * np.pi * df0[c] / denom)

    for c in ["userID", "stationID"]:
        df0[c] = LabelEncoder().fit_transform(df0[c].astype(str))

    df0.sort_values("connection_time_copy", inplace=True)
    df0.reset_index(drop=True, inplace=True)

    test_start = datetime(2019, 12, 1)
    test_end = datetime(2019, 12, 30)
    windows = [30, 60, 120, 240, 360, 480]
    n_runs = 5
    eps = 1e-3

    base_dirs = [
        os.path.join(".", "Caltech", "results", "DLinear_strict", "parking_time"),
        os.path.join(".", "Caltech", "results", "DLinear_strict", "kWhDelivered"),
    ]

    for run in range(1, n_runs + 1):
        run_str = str(run)
        for base in base_dirs:
            os.makedirs(os.path.join(base, run_str), exist_ok=True)
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

            tr, te = denoise_train_test(tr, te)

            tr["age_days"] = (test_start - tr.connection_time_copy).dt.days
            w_decay = np.exp(-0.015 * tr["age_days"])
            _ = w_decay

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

            tr["sim_park"] = 0.0
            tr["sim_kwh"] = 0.0
            for i in range(len(tr)):
                prev_idx = tr[(tr.userID == tr.loc[i, "userID"]) & (tr.index < i)].index
                if len(prev_idx) == 0:
                    tr.at[i, "sim_park"] = tr.parking_time.median()
                    tr.at[i, "sim_kwh"] = tr.kWhDelivered.median()
                else:
                    sims = np.dot(tr_s[prev_idx], tr_s[i])
                    order = np.argsort(sims)[-5:]
                    top_idx = prev_idx[order]
                    w5 = sims[order]
                    tr.at[i, "sim_park"] = np.average(tr.loc[top_idx, "parking_time"], weights=w5 + 1e-6)
                    tr.at[i, "sim_kwh"] = np.average(tr.loc[top_idx, "kWhDelivered"], weights=w5 + 1e-6)

            te["sim_park"] = 0.0
            te["sim_kwh"] = 0.0
            for i in range(len(te)):
                prev_idx = tr[tr.userID == te.loc[i, "userID"]].index
                if len(prev_idx) == 0:
                    te.at[i, "sim_park"] = tr.parking_time.median()
                    te.at[i, "sim_kwh"] = tr.kWhDelivered.median()
                else:
                    sims = np.dot(tr_s[prev_idx], te_s[i])
                    order = np.argsort(sims)[-5:]
                    top_idx = prev_idx[order]
                    w5 = sims[order]
                    te.at[i, "sim_park"] = np.average(tr.loc[top_idx, "parking_time"], weights=w5 + 1e-6)
                    te.at[i, "sim_kwh"] = np.average(tr.loc[top_idx, "kWhDelivered"], weights=w5 + 1e-6)

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

            sh = tr.groupby(["stationID", "hour_of_week"]).kWhDelivered.sum() / (
                tr.groupby(["stationID", "hour_of_week"]).kWhRequested.sum() + 1e-6
            )
            uh = tr.groupby(["userID", "hour_of_week"]).kWhDelivered.sum() / (
                tr.groupby(["userID", "hour_of_week"]).kWhRequested.sum() + 1e-6
            )
            shu = tr.groupby(["stationID", "userID", "hour_of_week"]).kWhDelivered.sum() / (
                tr.groupby(["stationID", "userID", "hour_of_week"]).kWhRequested.sum() + 1e-6
            )

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

            tr_pt = tr.sort_values("connection_time_copy").reset_index(drop=True).copy()
            te_pt = te.sort_values("connection_time_copy").reset_index(drop=True).copy()
            all_pt = pd.concat([tr_pt, te_pt], axis=0, ignore_index=True)

            df_pt = build_model_dataframe(
                all_pt,
                target_col="parking_time",
                target_out_col="target_parking_time",
            )

            n_val_pt = max(int(0.2 * len(tr_pt)), 50)
            n_val_pt = min(n_val_pt, max(1, len(tr_pt) - 5))
            train_end_idx_pt = len(tr_pt) - n_val_pt - 1
            val_start_idx_pt = train_end_idx_pt + 1
            val_end_idx_pt = len(tr_pt) - 1
            test_start_idx_pt = len(tr_pt)

            try:
                pt_fit = fit_dlinear_model(
                    df_model=df_pt,
                    feature_cols=pf,
                    target_col="target_parking_time",
                    train_end_idx=train_end_idx_pt,
                    val_start_idx=val_start_idx_pt,
                    val_end_idx=val_end_idx_pt,
                    seed=run * 100 + w,
                    max_epochs=30,
                    batch_size=128,
                    lr=1e-3,
                    moving_avg=25,
                    dropout=0.0,
                    individual=False,
                    hidden_head=64,
                )

                hist_pt = df_pt[df_pt["time_idx"] < test_start_idx_pt].copy()
                future_pt = df_pt[df_pt["time_idx"] >= test_start_idx_pt].copy()
                pred_p_te = recursive_forecast_dlinear(
                    model=pt_fit["model"],
                    history_df=hist_pt,
                    future_df=future_pt,
                    feature_cols=pf,
                    target_col="target_parking_time",
                    encoder_length=pt_fit["encoder_length"],
                    feature_scaler=pt_fit["feature_scaler"],
                    target_scaler=pt_fit["target_scaler"],
                    device=pt_fit["device"],
                    row_prepare_fn=prepare_parking_row_from_history,
                    row_finalize_fn=finalize_parking_row,
                )

                y_p_te = te_pt["parking_time"].values
                mae_p = mae(y_p_te, pred_p_te)
                sm_p = smape(y_p_te, pred_p_te)
                rounds_p = pt_fit["rounds"]
                time_p_ms = pt_fit["time_ms"]

                save_curve_from_list(
                    pt_fit["val_mae_history"],
                    os.path.join(".", "Caltech", "results", "DLinear_strict", "parking_time", run_str, f"convergence_parking_time_window_{w}.png"),
                    f"Window={w}  parking_time (DLinear strict) Val MAE vs Epoch",
                )
            except Exception as e:
                print(f"[DLINEAR STRICT ERROR - parking_time] Run={run}, Window={w}, Error: {e}")
                mae_p, sm_p, rounds_p, time_p_ms = np.nan, np.nan, 0, 0.0

            pt_transform = PowerTransformer(method="box-cox", standardize=False)
            y_all_k = tr.kWhDelivered.values + eps
            _ = pt_transform.fit_transform(y_all_k.reshape(-1, 1)).flatten()

            tr_k = tr.sort_values("connection_time_copy").reset_index(drop=True).copy()
            te_k = te.sort_values("connection_time_copy").reset_index(drop=True).copy()

            user_roll3 = (
                tr_k.groupby("userID")["kWhDelivered"]
                .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            tr_k["user_roll_kWh3"] = user_roll3.fillna(tr_k.kWhDelivered.median())

            station_roll3 = (
                tr_k.groupby("stationID")["kWhDelivered"]
                .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
            tr_k["station_roll_kWh3"] = station_roll3.fillna(tr_k.kWhDelivered.median())

            user_hist = tr_k.groupby("userID")["kWhDelivered"].apply(list).to_dict()
            station_hist = tr_k.groupby("stationID")["kWhDelivered"].apply(list).to_dict()

            te_k["user_roll_kWh3"] = te_k["userID"].apply(
                lambda u: np.mean(user_hist[u][-3:]) if (u in user_hist and len(user_hist[u]) > 0) else tr_k.kWhDelivered.median()
            )
            te_k["station_roll_kWh3"] = te_k["stationID"].apply(
                lambda s: np.mean(station_hist[s][-3:]) if (s in station_hist and len(station_hist[s]) > 0) else tr_k.kWhDelivered.median()
            )

            med_k = float(tr_k.kWhDelivered.median())
            tr_k["pk1"] = tr_k.groupby("userID").kWhDelivered.shift(1).fillna(med_k)
            tr_k["pk2"] = tr_k.groupby("userID").kWhDelivered.shift(2).fillna(med_k)
            tr_k["rk3"] = (
                tr_k.groupby("userID").kWhDelivered.rolling(3, min_periods=1).mean().reset_index(0, drop=True).fillna(med_k)
            )

            te_k["pk1"] = te_k["userID"].apply(
                lambda u: float(user_hist[u][-1]) if (u in user_hist and len(user_hist[u]) >= 1) else med_k
            )
            te_k["pk2"] = te_k["userID"].apply(
                lambda u: float(user_hist[u][-2]) if (u in user_hist and len(user_hist[u]) >= 2) else med_k
            )
            te_k["rk3"] = te_k["userID"].apply(
                lambda u: float(np.mean(user_hist[u][-3:])) if (u in user_hist and len(user_hist[u]) >= 1) else med_k
            )

            feats_k = [
                "hour", "weekday", "month", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
                "cluster", "sim_kwh", "user_avg_kWh", "user_freq",
                "station_avg_kWh", "station_freq",
                "req_rate", "hour_of_week",
                "sh_ratio", "uh_ratio", "shur_ratio",
                "pk1", "pk2", "rk3",
                "user_roll_kWh3", "station_roll_kWh3"
            ]

            all_k = pd.concat([tr_k, te_k], axis=0, ignore_index=True).copy()
            all_k["kWhDelivered_bc"] = pt_transform.transform((all_k["kWhDelivered"].values + eps).reshape(-1, 1)).flatten()

            df_k = build_model_dataframe(
                all_k,
                target_col="kWhDelivered_bc",
                target_out_col="target_kwh_bc",
            )

            n_val_k = max(int(0.2 * len(tr_k)), 50)
            n_val_k = min(n_val_k, max(1, len(tr_k) - 5))
            train_end_idx_k = len(tr_k) - n_val_k - 1
            val_start_idx_k = train_end_idx_k + 1
            val_end_idx_k = len(tr_k) - 1
            test_start_idx_k = len(tr_k)

            try:
                k_fit = fit_dlinear_model(
                    df_model=df_k,
                    feature_cols=feats_k,
                    target_col="target_kwh_bc",
                    train_end_idx=train_end_idx_k,
                    val_start_idx=val_start_idx_k,
                    val_end_idx=val_end_idx_k,
                    seed=run * 1000 + w,
                    max_epochs=35,
                    batch_size=128,
                    lr=1e-3,
                    moving_avg=25,
                    dropout=0.0,
                    individual=False,
                    hidden_head=64,
                )

                hist_k = df_k[df_k["time_idx"] < test_start_idx_k].copy()
                future_k = df_k[df_k["time_idx"] >= test_start_idx_k].copy()
                pred_te_bc = recursive_forecast_dlinear(
                    model=k_fit["model"],
                    history_df=hist_k,
                    future_df=future_k,
                    feature_cols=feats_k,
                    target_col="target_kwh_bc",
                    encoder_length=k_fit["encoder_length"],
                    feature_scaler=k_fit["feature_scaler"],
                    target_scaler=k_fit["target_scaler"],
                    device=k_fit["device"],
                    row_prepare_fn=lambda hist, row: prepare_kwh_row_from_history(hist, row, med_k),
                    row_finalize_fn=make_finalize_kwh_row(pt_transform, eps),
                ).reshape(-1, 1)

                inv_te_k = pt_transform.inverse_transform(pred_te_bc).flatten() - eps

                order = np.argsort(te_k.connection_time_copy.values)
                smooth = medfilt(inv_te_k[order], kernel_size=5)
                final_k = np.empty_like(smooth)
                final_k[order] = smooth

                mae_k = mae(te_k.kWhDelivered.values, final_k)
                sm_k = smape(te_k.kWhDelivered.values, final_k)
                rounds_k = k_fit["rounds"]
                time_k_ms = k_fit["time_ms"]

                save_curve_from_list(
                    k_fit["val_mae_history"],
                    os.path.join(".", "Caltech", "results", "DLinear_strict", "kWhDelivered", run_str, f"convergence_kWh_time_window_{w}.png"),
                    f"Window={w}  kWhDelivered (DLinear strict) Val MAE vs Epoch",
                )
            except Exception as e:
                print(f"[DLINEAR STRICT ERROR - kWhDelivered] Run={run}, Window={w}, Error: {e}")
                mae_k, sm_k, rounds_k, time_k_ms = np.nan, np.nan, 0, 0.0

            t_window_ms = (time.perf_counter() - t_window_start) * 1000

            print(f"  Window={w}:")
            print(f"    parking_time → MAE={mae_p:.3f}, SMAPE={sm_p:.3f}%, epochs={rounds_p}, time={time_p_ms:.0f}ms")
            print(f"    kWhDelivered → MAE={mae_k:.3f}, SMAPE={sm_k:.3f}%, epochs={rounds_k}, time={time_k_ms:.0f}ms")

            run_summary.append((
                w,
                mae_p, sm_p, rounds_p, time_p_ms,
                mae_k, sm_k, rounds_k, time_k_ms,
                t_window_ms,
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
                "window_time_ms": t_window_ms,
            })

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\nResults for run {run}:")
        print("window | MAE_pt | SMAPE_pt | epochs_pt | time_pt(ms) | MAE_kWh | SMAPE_kWh | epochs_kWh | time_kWh(ms)")
        for (w, mpt, spt, rp, tp, mk, sk, rk, tk, _) in run_summary:
            print(f"{w:>6} | {mpt:6.3f} | {spt:7.3f}% | {rp:9d} | {tp:10.0f} | {mk:7.3f} | {sk:9.3f}% | {rk:10d} | {tk:11.0f}")

    pd.DataFrame(metrics_all).to_csv("./Caltech/results/DLinear_strict/metrics_mixed_dynamic.csv", index=False)
    print("\nAll runs complete. Results saved to ./Caltech/results/DLinear_strict/metrics_mixed_dynamic.csv")


if __name__ == "__main__":
    main()
