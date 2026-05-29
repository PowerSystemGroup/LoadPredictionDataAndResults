import os
import sys
import gc
import math
import random
import logging
import warnings
import contextlib
import time
from datetime import datetime, timedelta

os.environ["LIGHTNING_DISABLE_TIPS"] = "1"
os.environ["LIT_DISABLE_TIPS"] = "1"
os.environ["PL_DISABLE_TIPS"] = "1"
os.environ["LITLOGGER_DISABLE_TIPS"] = "1"
os.environ["LITMODEL_DISABLE_TIPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

warnings.filterwarnings("ignore")

for _logger_name in [
    "lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning",
    "lightning.pytorch.utilities.rank_zero", "lightning.fabric.utilities.rank_zero",
    "lightning.pytorch.accelerators", "lightning.fabric.accelerators",
    "lightning.pytorch.trainer", "lightning.fabric.plugins",
    "pytorch_forecasting",
]:
    _logger = logging.getLogger(_logger_name)
    _logger.setLevel(logging.CRITICAL)
    _logger.propagate = False

USE_GPU = torch.cuda.is_available()
TRAINER_ACCELERATOR = "gpu" if USE_GPU else "cpu"
TRAINER_DEVICES = 1
if USE_GPU:
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass


@contextlib.contextmanager
def suppress_stdout_stderr():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        old_disable_level = logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
        finally:
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
            logging.disable(old_disable_level)


def silence_lightning_rank_zero():
    def _noop(*args, **kwargs):
        return None

    modules = [
        "lightning.pytorch.utilities.rank_zero",
        "lightning.fabric.utilities.rank_zero",
        "pytorch_lightning.utilities.rank_zero",
        "lightning_utilities.core.rank_zero",
    ]
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=["dummy"])
            for fn in ["rank_zero_info", "rank_zero_warn", "rank_zero_debug"]:
                if hasattr(module, fn):
                    setattr(module, fn, _noop)
        except Exception:
            pass


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    return float(100 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / (np.abs(y_pred[mask]) + np.abs(y_true[mask]) + 1e-6)))


def import_tft_dependencies():
    try:
        with suppress_stdout_stderr():
            import lightning.pytorch as pl
            from lightning.pytorch.callbacks import EarlyStopping, Callback
            from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
            from pytorch_forecasting.data import GroupNormalizer
            try:
                from pytorch_forecasting.data.encoders import NaNLabelEncoder
            except Exception:
                from pytorch_forecasting.data import NaNLabelEncoder
            from pytorch_forecasting.metrics import MAE
        silence_lightning_rank_zero()
        return {
            "pl": pl,
            "EarlyStopping": EarlyStopping,
            "Callback": Callback,
            "TimeSeriesDataSet": TimeSeriesDataSet,
            "TemporalFusionTransformer": TemporalFusionTransformer,
            "GroupNormalizer": GroupNormalizer,
            "NaNLabelEncoder": NaNLabelEncoder,
            "PF_MAE": MAE,
        }
    except Exception as e:
        raise ImportError(
            "TFT runtime dependencies are missing. Please install them first: pip install lightning pytorch-forecasting\n"
            f"Current error: {e}"
        )


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_encoder_length(n_train_samples: int, base_len: int = 24):
    if n_train_samples <= 6:
        return max(2, n_train_samples - 2)
    candidate = max(8, n_train_samples // 6)
    return max(4, min(base_len, n_train_samples - 2, candidate))


def clean_numeric_columns(df: pd.DataFrame, cols: list, fill_source: pd.DataFrame = None):
    df = df.copy()
    source = df if fill_source is None else fill_source
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if c in source.columns:
            fill_val = pd.to_numeric(source[c], errors="coerce").replace([np.inf, -np.inf], np.nan).median()
        else:
            fill_val = df[c].median()
        if not np.isfinite(fill_val):
            fill_val = 0.0
        df[c] = df[c].fillna(float(fill_val))
    return df


def safe_weighted_average(values, sims):
    values = np.asarray(values, dtype=float)
    sims = np.asarray(sims, dtype=float)
    if len(values) == 0:
        return np.nan
    if not np.all(np.isfinite(sims)):
        return float(np.nanmean(values))
    weights = sims - np.min(sims) + 1e-6
    if (not np.isfinite(weights).all()) or weights.sum() <= 0:
        return float(np.nanmean(values))
    return float(np.average(values, weights=weights))


def save_curve_from_list(values, save_path: str, title: str):
    if values is None or len(values) == 0:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Validation MAE")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def denoise_train_test(tr: pd.DataFrame, te: pd.DataFrame):
    tr = tr.copy()
    te = te.copy()

    for col in ["kWhDelivered"]:
        tr[col] = pd.to_numeric(tr[col], errors="coerce")
        te[col] = pd.to_numeric(te[col], errors="coerce")
        tr[col] = tr[col].fillna(tr[col].median())
        te[col] = te[col].fillna(tr[col].median())

        if len(tr) >= 16:
            coeffs = pywt.wavedec(tr[col].values, "db4", level=3)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thr = sigma * np.sqrt(2 * np.log(max(len(tr), 2)))
            coeffs[1:] = [pywt.threshold(c, thr, "soft") for c in coeffs[1:]]
            tr[col] = pywt.waverec(coeffs, "db4")[: len(tr)]

            coeffs_te = pywt.wavedec(te[col].values, "db4", level=3)
            coeffs_te[1:] = [pywt.threshold(c, thr, "soft") for c in coeffs_te[1:]]
            te[col] = pywt.waverec(coeffs_te, "db4")[: len(te)]

    tr = clean_numeric_columns(tr, ["parking_time", "kWhDelivered"])
    te = clean_numeric_columns(te, ["parking_time", "kWhDelivered"], fill_source=tr)

    if len(tr) > 25:
        n_neighbors = min(20, len(tr) - 1)
        try:
            mask = LocalOutlierFactor(n_neighbors=n_neighbors).fit_predict(tr[["parking_time", "kWhDelivered"]])
            tr = tr[mask == 1].reset_index(drop=True)
        except Exception:
            tr = tr.reset_index(drop=True)
    else:
        tr = tr.reset_index(drop=True)

    te = te.reset_index(drop=True)
    return tr, te


def build_tft_dataframe(df_part: pd.DataFrame, target_col: str, target_out_col: str, categorical_cols: list):
    data = df_part.copy().reset_index(drop=True)
    data["series_id"] = "global_series"
    data["time_idx"] = np.arange(len(data), dtype=int)
    data[target_out_col] = pd.to_numeric(data[target_col], errors="coerce").astype(float)
    for c in categorical_cols + ["series_id"]:
        data[c] = data[c].astype(str)
    return data


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def extract_prediction_from_output(out):
    if isinstance(out, dict):
        pred = out.get("prediction", None)
        if pred is None:
            pred = next(iter(out.values()))
    elif hasattr(out, "prediction"):
        pred = out.prediction
    elif isinstance(out, (tuple, list)):
        pred = out[0]
    else:
        pred = out
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    return pred


def tft_predict_fast(model, dataloader):
    device = next(model.parameters()).device
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = move_to_device(x, device)
            out = model(x)
            pred = extract_prediction_from_output(out)
            preds.append(pred.detach().cpu().numpy().reshape(-1))
    if not preds:
        return np.array([], dtype=float)
    pred = np.concatenate(preds, axis=0).astype(float)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred


def extract_targets_from_dataloader(dataloader):
    ys = []
    for _, y in dataloader:
        target = y[0] if isinstance(y, (tuple, list)) else y
        if isinstance(target, list):
            target = target[0]
        if torch.is_tensor(target):
            arr = target.detach().cpu().numpy()
        else:
            arr = np.asarray(target)
        ys.append(arr)
    if not ys:
        return np.array([], dtype=float)
    return np.concatenate(ys, axis=0).reshape(-1).astype(float)


def fit_boxcox_safe(y, eps=1e-3):
    y = np.asarray(y, dtype=float)
    y = np.nan_to_num(y, nan=np.nanmedian(y), posinf=np.nanmedian(y), neginf=np.nanmedian(y))
    y = np.maximum(y, eps)
    pt = PowerTransformer(method="box-cox", standardize=False)
    y_bc = pt.fit_transform(y.reshape(-1, 1)).reshape(-1)
    lo = float(np.nanpercentile(y_bc, 0.5))
    hi = float(np.nanpercentile(y_bc, 99.5))
    if not np.isfinite(lo):
        lo = float(np.nanmin(y_bc))
    if not np.isfinite(hi):
        hi = float(np.nanmax(y_bc))
    if lo >= hi:
        lo, hi = float(np.nanmin(y_bc) - 1.0), float(np.nanmax(y_bc) + 1.0)
    return pt, y_bc, (lo, hi)


def boxcox_transform_safe(pt, y, eps=1e-3):
    y = np.asarray(y, dtype=float)
    med = np.nanmedian(y) if np.isfinite(np.nanmedian(y)) else 1.0
    y = np.nan_to_num(y, nan=med, posinf=med, neginf=med)
    y = np.maximum(y, eps)
    return pt.transform(y.reshape(-1, 1)).reshape(-1)


def boxcox_inverse_safe(pt, y_bc, eps=1e-3, bounds=None):
    y_bc = np.asarray(y_bc, dtype=float).reshape(-1)
    if bounds is not None:
        lo, hi = bounds
        fill = (lo + hi) / 2
        y_bc = np.nan_to_num(y_bc, nan=fill, posinf=hi, neginf=lo)
        y_bc = np.clip(y_bc, lo, hi)
    else:
        y_bc = np.nan_to_num(y_bc, nan=0.0, posinf=0.0, neginf=0.0)
    out = pt.inverse_transform(y_bc.reshape(-1, 1)).reshape(-1) - eps
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(out, 0.0)


def fit_tft_model(
    df_model: pd.DataFrame,
    target_col: str,
    known_reals: list,
    known_categoricals: list,
    train_end_idx: int,
    val_start_idx: int,
    val_end_idx: int,
    seed: int,
    max_epochs: int = 15,
    batch_size: int = 256,
    hidden_size: int = 16,
    attention_head_size: int = 2,
    hidden_continuous_size: int = 8,
):
    deps = import_tft_dependencies()
    pl = deps["pl"]
    EarlyStopping = deps["EarlyStopping"]
    Callback = deps["Callback"]
    TimeSeriesDataSet = deps["TimeSeriesDataSet"]
    TemporalFusionTransformer = deps["TemporalFusionTransformer"]
    GroupNormalizer = deps["GroupNormalizer"]
    NaNLabelEncoder = deps["NaNLabelEncoder"]
    PF_MAE = deps["PF_MAE"]

    seed_everything(seed)
    silence_lightning_rank_zero()
    with suppress_stdout_stderr():
        try:
            pl.seed_everything(seed, workers=True, verbose=False)
        except TypeError:
            pl.seed_everything(seed, workers=True)

    train_data = df_model[df_model["time_idx"] <= train_end_idx].copy()
    val_data = df_model[df_model["time_idx"] <= val_end_idx].copy()

    all_numeric = list(set(known_reals + [target_col]))
    train_data = clean_numeric_columns(train_data, all_numeric)
    val_data = clean_numeric_columns(val_data, all_numeric, fill_source=train_data)

    encoder_length = choose_encoder_length(len(train_data), base_len=24)
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

    class ValLossHistory(Callback):
        def __init__(self):
            super().__init__()
            self.values = []

        def on_validation_epoch_end(self, trainer, pl_module):
            v = trainer.callback_metrics.get("val_loss")
            if v is not None:
                try:
                    self.values.append(float(v.detach().cpu().item()))
                except Exception:
                    try:
                        self.values.append(float(v))
                    except Exception:
                        pass

    history_callback = ValLossHistory()
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=3,
        verbose=False,
        mode="min",
    )

    t0 = time.perf_counter()
    with suppress_stdout_stderr():
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=TRAINER_ACCELERATOR,
            devices=TRAINER_DEVICES,
            gradient_clip_val=0.1,
            enable_model_summary=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=[early_stop_callback, history_callback],
            log_every_n_steps=50,
            num_sanity_val_steps=0,
            enable_checkpointing=False,
        )

        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=0.1,
            hidden_continuous_size=hidden_continuous_size,
            output_size=1,
            loss=PF_MAE(),
            log_interval=-1,
            reduce_on_plateau_patience=2,
            lstm_layers=1,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    time_ms = (time.perf_counter() - t0) * 1000

    pred_val = tft_predict_fast(model, val_loader)
    y_val = extract_targets_from_dataloader(val_loader)
    val_mae_history = [v for v in history_callback.values if np.isfinite(v)]
    rounds = len(val_mae_history) if val_mae_history else max_epochs
    best_val_mae = float(np.nanmin(val_mae_history)) if val_mae_history else mae(y_val, pred_val)

    del trainer, train_loader, val_loader, validation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model,
        "training_dataset": training,
        "pred_val": pred_val,
        "y_val": y_val,
        "time_ms": time_ms,
        "rounds": rounds,
        "best_val_mae": best_val_mae,
        "val_mae_history": val_mae_history,
        "encoder_length": encoder_length,
    }


def recursive_forecast_tft(
    model,
    training_dataset,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    target_col: str,
    known_reals: list,
    encoder_length: int,
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

        row[target_col] = float(hist[target_col].iloc[-1]) if len(hist) > 0 else 0.0

        hist_tail = hist.sort_values("time_idx").tail(encoder_length).copy()
        pred_input = pd.concat([hist_tail, pd.DataFrame([row])], axis=0, ignore_index=True)
        pred_input = clean_numeric_columns(pred_input, list(set(known_reals + [target_col])), fill_source=hist_tail)

        pred_ds = TimeSeriesDataSet.from_dataset(
            training_dataset,
            pred_input,
            min_prediction_idx=int(row["time_idx"]),
            stop_randomization=True,
            predict=True,
        )
        pred_loader = pred_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        pred_arr = tft_predict_fast(model, pred_loader)
        if len(pred_arr) == 0:
            pred = float(hist[target_col].iloc[-1])
        else:
            pred = float(pred_arr.reshape(-1)[-1])
        if not np.isfinite(pred):
            pred = float(hist[target_col].iloc[-1])
        preds.append(pred)

        row[target_col] = pred
        if row_finalize_fn is not None:
            row = row_finalize_fn(hist, row, pred)
        hist = pd.concat([hist, pd.DataFrame([row])], axis=0, ignore_index=True)

        del pred_ds, pred_loader
        if (i + 1) % 200 == 0:
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
    pred = max(float(pred), 0.0)
    row["parking_time"] = pred
    row["target_parking_time"] = pred
    return row


def prepare_kwh_row_from_history(hist_df: pd.DataFrame, row: pd.Series, med_k: float):
    row = row.copy()
    user = row["userID"]
    station = row["stationID"]

    user_hist = hist_df.loc[hist_df["userID"] == user, "kWhDelivered"].dropna().astype(float).tolist()
    station_hist = hist_df.loc[hist_df["stationID"] == station, "kWhDelivered"].dropna().astype(float).tolist()

    row["user_roll_kWh3"] = float(np.mean(user_hist[-3:])) if len(user_hist) > 0 else float(med_k)
    row["station_roll_kWh3"] = float(np.mean(station_hist[-3:])) if len(station_hist) > 0 else float(med_k)
    row["pk1"] = float(user_hist[-1]) if len(user_hist) >= 1 else float(med_k)
    row["pk2"] = float(user_hist[-2]) if len(user_hist) >= 2 else float(med_k)
    row["rk3"] = float(np.mean(user_hist[-3:])) if len(user_hist) > 0 else float(med_k)

    row["kWhDelivered"] = np.nan
    row["kWhDelivered_bc"] = np.nan
    row["target_kwh_bc"] = np.nan
    return row


def make_finalize_kwh_row(pt_transform, eps: float, bc_bounds):
    def _finalize(hist_df: pd.DataFrame, row: pd.Series, pred_bc: float):
        row = row.copy()
        pred_kwh = float(boxcox_inverse_safe(pt_transform, np.array([pred_bc]), eps=eps, bounds=bc_bounds)[0])
        pred_bc_clean = float(np.clip(pred_bc, bc_bounds[0], bc_bounds[1])) if np.isfinite(pred_bc) else float(np.mean(bc_bounds))
        row["kWhDelivered_bc"] = pred_bc_clean
        row["target_kwh_bc"] = pred_bc_clean
        row["kWhDelivered"] = pred_kwh
        return row
    return _finalize


def main():
    DATA_FILE = "jpl_extreme_5.csv"
    EXPERIMENT_NAME = os.path.splitext(os.path.basename(DATA_FILE))[0] + "_TFT_keepLOF_woMedian_fast_fixed"
    RESULT_ROOT = os.path.join(".", "results", EXPERIMENT_NAME)
    os.makedirs(RESULT_ROOT, exist_ok=True)

    metrics_all = []

    df0 = pd.read_csv(DATA_FILE, parse_dates=["connection_time_copy"])
    df0 = df0.dropna(subset=["connection_time_copy"]).reset_index(drop=True)

    df0["hour"] = df0.connection_time_copy.dt.hour
    df0["weekday"] = df0.connection_time_copy.dt.weekday
    df0["month"] = df0.connection_time_copy.dt.month
    df0["dayofyear"] = df0.connection_time_copy.dt.dayofyear
    for c in ["hour", "weekday", "month", "dayofyear"]:
        denom = max(float(df0[c].max()), 1.0)
        df0[f"{c}_sin"] = np.sin(2 * np.pi * df0[c] / denom)
        df0[f"{c}_cos"] = np.cos(2 * np.pi * df0[c] / denom)

    for c in ["userID", "stationID"]:
        df0[c] = LabelEncoder().fit_transform(df0[c].astype(str))

    df0.sort_values("connection_time_copy", inplace=True)
    df0.reset_index(drop=True, inplace=True)

    test_start = datetime(2019, 12, 1)
    test_end = datetime(2019, 12, 30)
    windows = [30, 60, 120, 240, 360, 480]
    n_runs = 1
    eps = 1e-3

    base_dirs = [
        os.path.join(RESULT_ROOT, "parking_time"),
        os.path.join(RESULT_ROOT, "kWhDelivered"),
    ]

    for run in range(1, n_runs + 1):
        run_str = str(run)
        for base in base_dirs:
            os.makedirs(os.path.join(base, run_str), exist_ok=True)

        print(f"\n=== Run {run} ===")
        df = df0.copy()

        hist_all = df[df.connection_time_copy < test_start]
        ua = hist_all.groupby("userID").kWhDelivered.mean() / (hist_all.groupby("userID").kWhRequested.mean() + 1e-6)
        ua = ua.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ua) >= 2:
            n_clusters = min(10, len(ua))
            km = KMeans(n_clusters=n_clusters, random_state=run, n_init=10).fit(ua.values.reshape(-1, 1))
            cluster_map = dict(zip(ua.index, km.labels_))
            df["cluster"] = df.userID.map(cluster_map).fillna(0).astype(int)
        else:
            df["cluster"] = 0

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

            tr.reset_index(drop=True, inplace=True)
            te.reset_index(drop=True, inplace=True)

            tr["is_weekend"] = tr.weekday.isin([5, 6]).astype(int)
            te["is_weekend"] = te.weekday.isin([5, 6]).astype(int)
            tr["is_holiday"] = tr.get("connectionTime_is_holiday", 0)
            te["is_holiday"] = te.get("connectionTime_is_holiday", 0)

            sim_feats = ["hour", "weekday", "kWhRequested", "Requested_parking_time"]
            tr = clean_numeric_columns(tr, sim_feats + ["parking_time", "kWhDelivered"])
            te = clean_numeric_columns(te, sim_feats + ["parking_time", "kWhDelivered"], fill_source=tr)
            ss = StandardScaler().fit(tr[sim_feats])
            tr_s = ss.transform(tr[sim_feats])
            te_s = ss.transform(te[sim_feats])

            tr["sim_park"] = 0.0
            tr["sim_kwh"] = 0.0
            for i in range(len(tr)):
                prev_idx = tr[(tr.userID == tr.loc[i, "userID"]) & (tr.index < i)].index
                if len(prev_idx) == 0:
                    tr.at[i, "sim_park"] = float(tr.parking_time.median())
                    tr.at[i, "sim_kwh"] = float(tr.kWhDelivered.median())
                else:
                    sims = np.dot(tr_s[prev_idx], tr_s[i])
                    order = np.argsort(sims)[-5:]
                    top_idx = prev_idx[order]
                    w5 = sims[order]
                    tr.at[i, "sim_park"] = safe_weighted_average(tr.loc[top_idx, "parking_time"], w5)
                    tr.at[i, "sim_kwh"] = safe_weighted_average(tr.loc[top_idx, "kWhDelivered"], w5)

            te["sim_park"] = 0.0
            te["sim_kwh"] = 0.0
            for i in range(len(te)):
                prev_idx = tr[tr.userID == te.loc[i, "userID"]].index
                if len(prev_idx) == 0:
                    te.at[i, "sim_park"] = float(tr.parking_time.median())
                    te.at[i, "sim_kwh"] = float(tr.kWhDelivered.median())
                else:
                    sims = np.dot(tr_s[prev_idx], te_s[i])
                    order = np.argsort(sims)[-5:]
                    top_idx = prev_idx[order]
                    w5 = sims[order]
                    te.at[i, "sim_park"] = safe_weighted_average(tr.loc[top_idx, "parking_time"], w5)
                    te.at[i, "sim_kwh"] = safe_weighted_average(tr.loc[top_idx, "kWhDelivered"], w5)

            ua_m = tr.groupby("userID").kWhDelivered.mean()
            ua_c = tr.userID.value_counts()
            sa_m = tr.groupby("stationID").kWhDelivered.mean()
            sa_c = tr.stationID.value_counts()
            for D in (tr, te):
                D["user_avg_kWh"] = D.userID.map(ua_m).fillna(float(ua_m.mean()))
                D["user_freq"] = D.userID.map(ua_c).fillna(0)
                D["station_avg_kWh"] = D.stationID.map(sa_m).fillna(float(sa_m.mean()))
                D["station_freq"] = D.stationID.map(sa_c).fillna(0)
                D["hour_of_week"] = D.weekday * 24 + D.hour
                D["req_rate"] = D.kWhRequested / (D.Requested_parking_time + 1e-6)

            sh = tr.groupby(["stationID", "hour_of_week"]).kWhDelivered.sum() / (tr.groupby(["stationID", "hour_of_week"]).kWhRequested.sum() + 1e-6)
            uh = tr.groupby(["userID", "hour_of_week"]).kWhDelivered.sum() / (tr.groupby(["userID", "hour_of_week"]).kWhRequested.sum() + 1e-6)
            shu = tr.groupby(["stationID", "userID", "hour_of_week"]).kWhDelivered.sum() / (tr.groupby(["stationID", "userID", "hour_of_week"]).kWhRequested.sum() + 1e-6)
            global_ratio = float(tr.kWhDelivered.sum() / (tr.kWhRequested.sum() + 1e-6))

            tr["sh_ratio"] = tr.set_index(["stationID", "hour_of_week"]).index.map(sh).fillna(global_ratio)
            tr["uh_ratio"] = tr.set_index(["userID", "hour_of_week"]).index.map(uh).fillna(global_ratio)
            tr["shur_ratio"] = tr.set_index(["stationID", "userID", "hour_of_week"]).index.map(shu).fillna(global_ratio)
            te["sh_ratio"] = te.set_index(["stationID", "hour_of_week"]).index.map(sh).fillna(global_ratio)
            te["uh_ratio"] = te.set_index(["userID", "hour_of_week"]).index.map(uh).fillna(global_ratio)
            te["shur_ratio"] = te.set_index(["stationID", "userID", "hour_of_week"]).index.map(shu).fillna(global_ratio)

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
            all_pt = clean_numeric_columns(all_pt, pt_known_reals + ["parking_time"], fill_source=tr_pt)

            df_pt = build_tft_dataframe(all_pt, target_col="parking_time", target_out_col="target_parking_time", categorical_cols=pt_categoricals)
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
                    seed=run * 100 + w,
                    max_epochs=15,
                    batch_size=256,
                )

                hist_pt = df_pt[df_pt["time_idx"] < test_start_idx_pt].copy()
                future_pt = df_pt[df_pt["time_idx"] >= test_start_idx_pt].copy()
                pred_p_te = recursive_forecast_tft(
                    model=pt_fit["model"],
                    training_dataset=pt_fit["training_dataset"],
                    history_df=hist_pt,
                    future_df=future_pt,
                    target_col="target_parking_time",
                    known_reals=pt_known_reals,
                    encoder_length=pt_fit["encoder_length"],
                    batch_size=64,
                    row_prepare_fn=prepare_parking_row_from_history,
                    row_finalize_fn=finalize_parking_row,
                )
                pred_p_te = np.maximum(np.nan_to_num(pred_p_te, nan=np.nanmedian(tr_pt.parking_time)), 0.0)
                y_p_te = te_pt["parking_time"].values
                mae_p = mae(y_p_te, pred_p_te)
                sm_p = smape(y_p_te, pred_p_te)
                rounds_p = pt_fit["rounds"]
                time_p_ms = pt_fit["time_ms"]

                save_curve_from_list(
                    pt_fit["val_mae_history"],
                    os.path.join(RESULT_ROOT, "parking_time", run_str, f"convergence_parking_time_window_{w}.png"),
                    f"Window={w} parking_time TFT Val MAE vs Epoch",
                )
            except Exception as e:
                print(f"[TFT ROBUST ERROR - parking_time] Run={run}, Window={w}, Error: {repr(e)}")
                mae_p, sm_p, rounds_p, time_p_ms = np.nan, np.nan, 0, 0.0

            tr_k = tr.sort_values("connection_time_copy").reset_index(drop=True).copy()
            te_k = te.sort_values("connection_time_copy").reset_index(drop=True).copy()
            tr_k["kWhDelivered"] = np.maximum(pd.to_numeric(tr_k["kWhDelivered"], errors="coerce").fillna(tr_k["kWhDelivered"].median()).values, eps)
            te_k["kWhDelivered"] = np.maximum(pd.to_numeric(te_k["kWhDelivered"], errors="coerce").fillna(tr_k["kWhDelivered"].median()).values, eps)

            pt_transform, y_bc_train, bc_bounds = fit_boxcox_safe(tr_k.kWhDelivered.values, eps=eps)

            user_roll3 = tr_k.groupby("userID")["kWhDelivered"].apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
            tr_k["user_roll_kWh3"] = user_roll3.fillna(tr_k.kWhDelivered.median())
            station_roll3 = tr_k.groupby("stationID")["kWhDelivered"].apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).reset_index(level=0, drop=True)
            tr_k["station_roll_kWh3"] = station_roll3.fillna(tr_k.kWhDelivered.median())

            user_hist = tr_k.groupby("userID")["kWhDelivered"].apply(list).to_dict()
            station_hist = tr_k.groupby("stationID")["kWhDelivered"].apply(list).to_dict()
            med_k = float(tr_k.kWhDelivered.median())

            te_k["user_roll_kWh3"] = te_k["userID"].apply(lambda u: np.mean(user_hist[u][-3:]) if (u in user_hist and len(user_hist[u]) > 0) else med_k)
            te_k["station_roll_kWh3"] = te_k["stationID"].apply(lambda s: np.mean(station_hist[s][-3:]) if (s in station_hist and len(station_hist[s]) > 0) else med_k)

            tr_k["pk1"] = tr_k.groupby("userID").kWhDelivered.shift(1).fillna(med_k)
            tr_k["pk2"] = tr_k.groupby("userID").kWhDelivered.shift(2).fillna(med_k)
            tr_k["rk3"] = tr_k.groupby("userID").kWhDelivered.rolling(3, min_periods=1).mean().reset_index(0, drop=True).fillna(med_k)

            te_k["pk1"] = te_k["userID"].apply(lambda u: float(user_hist[u][-1]) if (u in user_hist and len(user_hist[u]) >= 1) else med_k)
            te_k["pk2"] = te_k["userID"].apply(lambda u: float(user_hist[u][-2]) if (u in user_hist and len(user_hist[u]) >= 2) else med_k)
            te_k["rk3"] = te_k["userID"].apply(lambda u: float(np.mean(user_hist[u][-3:])) if (u in user_hist and len(user_hist[u]) >= 1) else med_k)

            feats_k = [
                "hour", "weekday", "month", "is_weekend", "is_holiday",
                "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                "month_sin", "month_cos", "dayofyear_sin", "dayofyear_cos",
                "cluster", "sim_kwh", "user_avg_kWh", "user_freq",
                "station_avg_kWh", "station_freq", "req_rate", "hour_of_week",
                "sh_ratio", "uh_ratio", "shur_ratio",
                "pk1", "pk2", "rk3", "user_roll_kWh3", "station_roll_kWh3"
            ]

            all_k = pd.concat([tr_k, te_k], axis=0, ignore_index=True).copy()
            all_k = clean_numeric_columns(all_k, feats_k + ["kWhDelivered"], fill_source=tr_k)
            all_k["kWhDelivered_bc"] = boxcox_transform_safe(pt_transform, all_k["kWhDelivered"].values, eps=eps)
            all_k["kWhDelivered_bc"] = np.nan_to_num(all_k["kWhDelivered_bc"], nan=float(np.mean(bc_bounds)))
            all_k["kWhDelivered_bc"] = np.clip(all_k["kWhDelivered_bc"], bc_bounds[0], bc_bounds[1])

            k_categoricals = ["cluster"]
            k_known_reals = [c for c in feats_k if c not in k_categoricals]

            df_k = build_tft_dataframe(all_k, target_col="kWhDelivered_bc", target_out_col="target_kwh_bc", categorical_cols=k_categoricals)
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
                    seed=run * 1000 + w,
                    max_epochs=15,
                    batch_size=256,
                )

                hist_k = df_k[df_k["time_idx"] < test_start_idx_k].copy()
                future_k = df_k[df_k["time_idx"] >= test_start_idx_k].copy()
                pred_te_bc = recursive_forecast_tft(
                    model=k_fit["model"],
                    training_dataset=k_fit["training_dataset"],
                    history_df=hist_k,
                    future_df=future_k,
                    target_col="target_kwh_bc",
                    known_reals=k_known_reals,
                    encoder_length=k_fit["encoder_length"],
                    batch_size=64,
                    row_prepare_fn=lambda hist, row: prepare_kwh_row_from_history(hist, row, med_k),
                    row_finalize_fn=make_finalize_kwh_row(pt_transform, eps, bc_bounds),
                )

                final_k = boxcox_inverse_safe(pt_transform, pred_te_bc, eps=eps, bounds=bc_bounds)
                mae_k = mae(te_k.kWhDelivered.values, final_k)
                sm_k = smape(te_k.kWhDelivered.values, final_k)
                rounds_k = k_fit["rounds"]
                time_k_ms = k_fit["time_ms"]

                save_curve_from_list(
                    k_fit["val_mae_history"],
                    os.path.join(RESULT_ROOT, "kWhDelivered", run_str, f"convergence_kWh_time_window_{w}.png"),
                    f"Window={w} kWhDelivered TFT Val MAE vs Epoch",
                )
            except Exception as e:
                print(f"[TFT ROBUST ERROR - kWhDelivered] Run={run}, Window={w}, Error: {repr(e)}")
                mae_k, sm_k, rounds_k, time_k_ms = np.nan, np.nan, 0, 0.0

            t_window_ms = (time.perf_counter() - t_window_start) * 1000
            print(f"  Window={w}:")
            print(f"    parking_time → MAE={mae_p:.3f}, SMAPE={sm_p:.3f}%, epochs={rounds_p}, time={time_p_ms:.0f}ms")
            print(f"    kWhDelivered → MAE={mae_k:.3f}, SMAPE={sm_k:.3f}%, epochs={rounds_k}, time={time_k_ms:.0f}ms")

            run_summary.append((w, mae_p, sm_p, rounds_p, time_p_ms, mae_k, sm_k, rounds_k, time_k_ms, t_window_ms))
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

    metrics_path = os.path.join(RESULT_ROOT, "metrics_mixed_dynamic.csv")
    pd.DataFrame(metrics_all).to_csv(metrics_path, index=False)
    print(f"\nAll runs complete. Results saved to {metrics_path}")


if __name__ == "__main__":
    main()
