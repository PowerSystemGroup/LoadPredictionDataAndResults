import os
import gc
import warnings
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
from scipy.signal import medfilt
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
import io
import contextlib

warnings.filterwarnings("ignore")
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.accelerators").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.plugins").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric.utilities.seed").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.seed").setLevel(logging.ERROR)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))



def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return 100 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-6))


def import_tft_dependencies():
    try:
        import lightning.pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
        from lightning.pytorch.loggers import CSVLogger
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        try:
            from pytorch_forecasting.data.encoders import NaNLabelEncoder
        except Exception:
            from pytorch_forecasting.data import NaNLabelEncoder
        from pytorch_forecasting.metrics import MAE
        return {
            "pl": pl,
            "EarlyStopping": EarlyStopping,
            "ModelCheckpoint": ModelCheckpoint,
            "CSVLogger": CSVLogger,
            "TimeSeriesDataSet": TimeSeriesDataSet,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "GroupNormalizer": GroupNormalizer,
            "PF_MAE": MAE,
            "NaNLabelEncoder": NaNLabelEncoder,
        }
    except Exception as e:
        raise ImportError(
            "Missing TFT runtime dependencies. Please install the following first: \n"
            "pip install lightning pytorch-forecasting\n"
            "If you are using a conda environment, we also recommend verifying that PyTorch is installed first. \n"
            f"Current error: {e}"
        )


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


def save_curve_from_metrics(metrics_csv_path: str, save_path: str, title: str):
    if not os.path.exists(metrics_csv_path):
        return

    dfm = pd.read_csv(metrics_csv_path)
    candidates = ["val_loss", "valid_loss", "val_loss_epoch"]
    metric_col = None
    for c in candidates:
        if c in dfm.columns:
            metric_col = c
            break
    if metric_col is None:
        return

    plot_df = dfm[["epoch", metric_col]].dropna().copy()
    if plot_df.empty:
        return

    plot_df = plot_df.groupby("epoch", as_index=False)[metric_col].last()

    plt.figure(figsize=(6, 4))
    plt.plot(plot_df["epoch"] + 1, plot_df[metric_col], marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def extract_targets_from_dataloader(dataloader):
    ys = []
    for _, y in dataloader:
        if isinstance(y, (tuple, list)):
            target = y[0]
        else:
            target = y
        if isinstance(target, list):
            target = target[0]
        if torch.is_tensor(target):
            arr = target.detach().cpu().numpy()
        else:
            arr = np.asarray(target)
        ys.append(arr)
    if not ys:
        return np.array([])
    y = np.concatenate(ys, axis=0)
    return y.reshape(-1)


def choose_encoder_length(n_train_samples: int, base_len: int = 32):
    if n_train_samples <= 6:
        return max(2, n_train_samples - 2)
    candidate = max(8, n_train_samples // 5)
    return max(4, min(base_len, n_train_samples - 2, candidate))


def tft_predict_compat(model, dataloader):
    attempts = [
        {"mode": "prediction"},
        {"mode": "prediction", "trainer_kwargs": {"accelerator": "auto", "devices": 1}},
        {},
    ]
    last_err = None
    silent_buf = io.StringIO()

    for kwargs in attempts:
        try:
            with contextlib.redirect_stdout(silent_buf), contextlib.redirect_stderr(silent_buf):
                pred = model.predict(dataloader, **kwargs)
            if torch.is_tensor(pred):
                return pred.detach().cpu().numpy().reshape(-1)
            return np.asarray(pred).reshape(-1)
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue
    raise last_err


def build_tft_dataframe(df_part: pd.DataFrame, target_col: str, target_out_col: str, categorical_cols: list):
    data = df_part.copy().reset_index(drop=True)
    data["series_id"] = "global_series"
    data["time_idx"] = np.arange(len(data), dtype=int)
    for c in categorical_cols + ["series_id"]:
        data[c] = data[c].astype(str)
    data[target_out_col] = data[target_col].astype(float)
    return data


def fit_tft_model(
    df_model: pd.DataFrame,
    target_col: str,
    known_reals: list,
    known_categoricals: list,
    train_end_idx: int,
    val_start_idx: int,
    val_end_idx: int,
    logger_root: str,
    model_tag: str,
    seed: int,
    max_epochs: int = 30,
    batch_size: int = 128,
):
    deps = import_tft_dependencies()
    pl = deps["pl"]
    EarlyStopping = deps["EarlyStopping"]
    ModelCheckpoint = deps["ModelCheckpoint"]
    CSVLogger = deps["CSVLogger"]
    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    TemporalFusionTransformer = deps["TemporalFusionTransformer"]
    GroupNormalizer = deps["GroupNormalizer"]
    PF_MAE = deps["PF_MAE"]
    NaNLabelEncoder = deps["NaNLabelEncoder"]

    silent_buf = io.StringIO()

    with contextlib.redirect_stdout(silent_buf), contextlib.redirect_stderr(silent_buf):
        pl.seed_everything(seed, workers=True)

    train_data = df_model[df_model["time_idx"] <= train_end_idx].copy()
    val_data = df_model[df_model["time_idx"] <= val_end_idx].copy()

    encoder_length = choose_encoder_length(len(train_data), base_len=32)
    if encoder_length < 4:
        raise ValueError(f"There are too few training samples to construct a TFT sequence: {len(train_data)}")

    categorical_encoders = {"series_id": NaNLabelEncoder(add_nan=True)}
    for c in known_categoricals:
        categorical_encoders[c] = NaNLabelEncoder(add_nan=True)

    training = TimeSeriesDataSet(
        train_data,
        time_idx="time_idx",
        target=target_col,
        group_ids=["series_id"],
        min_encoder_length=encoder_length,
        max_encoder_length=encoder_length,
        min_prediction_length=1,
        max_prediction_length=1,
        static_categoricals=["series_id"],
        time_varying_known_categoricals=known_categoricals,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=False,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        val_data,
        min_prediction_idx=val_start_idx,
        stop_randomization=True,
        predict=False,
    )

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    logger = CSVLogger(save_dir=logger_root, name=model_tag)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="best-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=5,
        verbose=False,
        mode="min",
    )

    with contextlib.redirect_stdout(silent_buf), contextlib.redirect_stderr(silent_buf):
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            gradient_clip_val=0.1,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=logger,
            callbacks=[early_stop_callback, checkpoint_callback],
            log_every_n_steps=20,
        )

        t0 = time.perf_counter()
        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=32,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=16,
            output_size=1,
            loss=PF_MAE(),
            log_interval=-1,
            reduce_on_plateau_patience=3,
            lstm_layers=1,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    time_ms = (time.perf_counter() - t0) * 1000

    best_model = model
    if checkpoint_callback.best_model_path:
        best_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_callback.best_model_path)

    pred_val = tft_predict_compat(best_model, val_loader)
    y_val = extract_targets_from_dataloader(val_loader)

    metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
    rounds = 0
    best_val_mae = np.nan
    if os.path.exists(metrics_csv):
        dfm = pd.read_csv(metrics_csv)
        for c in ["val_loss", "valid_loss", "val_loss_epoch"]:
            if c in dfm.columns:
                tmp = dfm[["epoch", c]].dropna().copy()
                if not tmp.empty:
                    rounds = int(tmp["epoch"].max()) + 1
                    best_val_mae = tmp[c].min()
                    break
    if rounds == 0:
        rounds = max_epochs
        if len(y_val) == len(pred_val) and len(y_val) > 0:
            best_val_mae = mae(y_val, pred_val)

    del trainer, train_loader, val_loader, validation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": best_model,
        "training_dataset": training,
        "pred_val": pred_val,
        "y_val": y_val,
        "time_ms": time_ms,
        "rounds": rounds,
        "best_val_mae": best_val_mae,
        "metrics_csv": metrics_csv,
        "encoder_length": encoder_length,
    }


def recursive_forecast_tft(
    model,
    training_dataset,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_col: str,
    batch_size: int = 64,
    row_prepare_fn=None,
    row_finalize_fn=None,
):
    deps = import_tft_dependencies()
    TimeSeriesDataSet = deps["TimeSeriesDataSet"]

    hist = history_df.copy().sort_values("time_idx").reset_index(drop=True)
    future = future_df.copy().sort_values("time_idx").reset_index(drop=True)

    preds = []
    for i in range(len(future)):
        row = future.iloc[i].copy()

        if row_prepare_fn is not None:
            row = row_prepare_fn(hist, row)

        if len(hist) > 0:
            row[target_col] = float(hist[target_col].iloc[-1])
        else:
            row[target_col] = 0.0

        pred_input = pd.concat([hist, pd.DataFrame([row])], axis=0, ignore_index=True)
        pred_ds = TimeSeriesDataSet.from_dataset(
            training_dataset,
            pred_input,
            min_prediction_idx=int(row["time_idx"]),
            stop_randomization=True,
            predict=True,
        )
        pred_loader = pred_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        pred = float(tft_predict_compat(model, pred_loader).reshape(-1)[-1])
        preds.append(pred)

        row[target_col] = pred
        if row_finalize_fn is not None:
            row = row_finalize_fn(hist, row, pred)
        hist = pd.concat([hist, pd.DataFrame([row])], axis=0, ignore_index=True)

        del pred_ds, pred_loader
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
        pred_bc = max(float(pred_bc), 1e-6)
        pred_kwh = float(pt_transform.inverse_transform(np.array([[pred_bc]])).reshape(-1)[0] - eps)
        pred_kwh = max(pred_kwh, 0.0)
        row["kWhDelivered_bc"] = pred_bc
        row["target_kwh_bc"] = pred_bc
        row["kWhDelivered"] = pred_kwh
        return row

    return _finalize


def main():
    os.makedirs("./Caltech/results", exist_ok=True)
    os.makedirs("./Caltech/results/TFT_strict", exist_ok=True)
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
    n_runs = 20
    eps = 1e-3

    base_dirs = [
        os.path.join(".", "Caltech", "results", "TFT_strict", "parking_time"),
        os.path.join(".", "Caltech", "results", "TFT_strict", "kWhDelivered"),
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

            pt_categoricals = ["userID", "stationID"]
            pt_known_reals = [c for c in pf if c not in pt_categoricals]

            df_pt = build_tft_dataframe(
                all_pt,
                target_col="parking_time",
                target_out_col="target_parking_time",
                categorical_cols=pt_categoricals,
            )

            n_val_pt = max(int(0.2 * len(tr_pt)), 50)
            n_val_pt = min(n_val_pt, max(1, len(tr_pt) - 5))
            train_end_idx_pt = len(tr_pt) - n_val_pt - 1
            val_start_idx_pt = train_end_idx_pt + 1
            val_end_idx_pt = len(tr_pt) - 1
            test_start_idx_pt = len(tr_pt)

            try:
                pt_fit = fit_tft_model(
                    df_model=df_pt,
                    target_col="target_parking_time",
                    known_reals=pt_known_reals,
                    known_categoricals=pt_categoricals,
                    train_end_idx=train_end_idx_pt,
                    val_start_idx=val_start_idx_pt,
                    val_end_idx=val_end_idx_pt,
                    logger_root=os.path.join(".", "Caltech", "results", "TFT_strict", "parking_time", run_str),
                    model_tag=f"window_{w}",
                    seed=run * 100 + w,
                    max_epochs=30,
                    batch_size=128,
                )

                hist_pt = df_pt[df_pt["time_idx"] < test_start_idx_pt].copy()
                future_pt = df_pt[df_pt["time_idx"] >= test_start_idx_pt].copy()
                pred_p_te = recursive_forecast_tft(
                    model=pt_fit["model"],
                    training_dataset=pt_fit["training_dataset"],
                    history_df=hist_pt,
                    future_df=future_pt,
                    target_col="target_parking_time",
                    batch_size=1,
                    row_prepare_fn=prepare_parking_row_from_history,
                    row_finalize_fn=finalize_parking_row,
                )

                y_p_te = te_pt["parking_time"].values
                mae_p = mae(y_p_te, pred_p_te)
                sm_p = smape(y_p_te, pred_p_te)
                rounds_p = pt_fit["rounds"]
                time_p_ms = pt_fit["time_ms"]

                save_curve_from_metrics(
                    pt_fit["metrics_csv"],
                    os.path.join(".", "Caltech", "results", "TFT_strict", "parking_time", run_str, f"convergence_parking_time_window_{w}.png"),
                    f"Window={w}  parking_time (TFT strict) Val MAE vs Epoch",
                )
            except Exception as e:
                print(f"[TFT STRICT ERROR - parking_time] Run={run}, Window={w}, Error: {e}")
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

            k_categoricals = ["cluster"]
            k_known_reals = [c for c in feats_k if c not in k_categoricals]

            df_k = build_tft_dataframe(
                all_k,
                target_col="kWhDelivered_bc",
                target_out_col="target_kwh_bc",
                categorical_cols=k_categoricals,
            )

            n_val_k = max(int(0.2 * len(tr_k)), 50)
            n_val_k = min(n_val_k, max(1, len(tr_k) - 5))
            train_end_idx_k = len(tr_k) - n_val_k - 1
            val_start_idx_k = train_end_idx_k + 1
            val_end_idx_k = len(tr_k) - 1
            test_start_idx_k = len(tr_k)

            try:
                k_fit = fit_tft_model(
                    df_model=df_k,
                    target_col="target_kwh_bc",
                    known_reals=k_known_reals,
                    known_categoricals=k_categoricals,
                    train_end_idx=train_end_idx_k,
                    val_start_idx=val_start_idx_k,
                    val_end_idx=val_end_idx_k,
                    logger_root=os.path.join(".", "Caltech", "results", "TFT_strict", "kWhDelivered", run_str),
                    model_tag=f"window_{w}",
                    seed=run * 1000 + w,
                    max_epochs=35,
                    batch_size=128,
                )

                hist_k = df_k[df_k["time_idx"] < test_start_idx_k].copy()
                future_k = df_k[df_k["time_idx"] >= test_start_idx_k].copy()
                pred_te_bc = recursive_forecast_tft(
                    model=k_fit["model"],
                    training_dataset=k_fit["training_dataset"],
                    history_df=hist_k,
                    future_df=future_k,
                    target_col="target_kwh_bc",
                    batch_size=1,
                    row_prepare_fn=lambda hist, row: prepare_kwh_row_from_history(hist, row, med_k),
                    row_finalize_fn=make_finalize_kwh_row(pt_transform, eps),
                ).reshape(-1, 1)

                pred_te_bc = np.maximum(pred_te_bc, 1e-6)
                inv_te_k = pt_transform.inverse_transform(pred_te_bc).flatten() - eps

                order = np.argsort(te_k.connection_time_copy.values)
                smooth = medfilt(inv_te_k[order], kernel_size=5)
                final_k = np.empty_like(smooth)
                final_k[order] = smooth

                mae_k = mae(te_k.kWhDelivered.values, final_k)
                sm_k = smape(te_k.kWhDelivered.values, final_k)
                rounds_k = k_fit["rounds"]
                time_k_ms = k_fit["time_ms"]

                save_curve_from_metrics(
                    k_fit["metrics_csv"],
                    os.path.join(".", "Caltech", "results", "TFT_strict", "kWhDelivered", run_str, f"convergence_kWh_time_window_{w}.png"),
                    f"Window={w}  kWhDelivered (TFT strict) Val MAE vs Epoch",
                )
            except Exception as e:
                print(f"[TFT STRICT ERROR - kWhDelivered] Run={run}, Window={w}, Error: {e}")
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

    pd.DataFrame(metrics_all).to_csv("./Caltech/results/TFT_strict/metrics_mixed_dynamic.csv", index=False)
    print("\nAll runs complete. Results saved to ./Caltech/results/TFT_strict/metrics_mixed_dynamic.csv")


if __name__ == "__main__":
    main()
