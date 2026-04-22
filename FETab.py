import os
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import optuna
from tabpfn import TabPFNRegressor
import json
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# 1. Read data
file_path = 'F:\FETab.xlsx'
sheet_name = 'Sheet1'
data = pd.read_excel(file_path, sheet_name=sheet_name)
dates = pd.to_datetime(data['month'])
epi_series = pd.Series(data['Ele'].values, index=dates)
split_index = int(len(epi_series) * 0.8)
train_series_raw = epi_series.iloc[:split_index]
test_series_raw = epi_series.iloc[split_index:]
scaler = StandardScaler()
train_values_scaled = scaler.fit_transform(train_series_raw.values.reshape(-1, 1)).flatten()
test_values_scaled = scaler.transform(test_series_raw.values.reshape(-1, 1)).flatten()
train_series = pd.Series(train_values_scaled, index=train_series_raw.index).asfreq('MS')
test_series = pd.Series(test_values_scaled, index=test_series_raw.index).asfreq('MS')
def discretize_data(series, bins_edges):
    return pd.cut(series, bins=bins_edges, labels=False, include_lowest=True)
def create_features_up_to(series, lag=3, window=2, bins_edges=None, start_date=None, end_date=None, fillna=True):
    df = pd.DataFrame(series, columns=['value'])
    for i in range(1, int(lag) + 1):
        df[f'lag_{i}'] = df['value'].shift(i)
    df[f'rolling_mean_{window}'] = df['value'].shift().rolling(window=window).mean()
    df[f'rolling_std_{window}'] = df['value'].shift().rolling(window=window).std()
    df[f'smoothed_value'] = df['value'].shift().ewm(span=window).mean()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    if bins_edges is None:
        bins_edges = pd.cut(series, bins=8, retbins=True)[1]
    df['discrete_values'] = discretize_data(series, bins_edges)
    if fillna:
        df.fillna(0, inplace=True)
    else:
        df.dropna(inplace=True)
    if start_date:
        df = df.loc[start_date:]
    if end_date:
        df = df.loc[:end_date]
    return df, bins_edges
def build_dual_channel_features(y_true, y_pred, lag, window=2, original_features=None, tensor_scaler=None,
                               sequence_scaler=None):
    T = len(y_true)
    tensor_Z = []
    sequence_Z = []
    valid_indices = []


    if original_features is not None:
        n_original_features = original_features.shape[1] - 1
    else:
        n_original_features = 0

    F = n_original_features + 2 * lag

    for t in range(lag + window - 1, T):
        tensor_Z_t = []
        sequence_Z_t = []
        for w in range(window):
            t_w = t - (window - 1) + w
            err_hist = y_pred[max(0, t_w - lag):t_w] - y_true[max(0, t_w - lag):t_w]
            if len(err_hist) < lag:
                err_hist = np.pad(err_hist, (lag - len(err_hist), 0), mode='constant')
            # Confidence
            conf = np.std(y_pred[max(0, t_w - lag):t_w]) if t_w > 0 else 0.0
            conf_vec = np.repeat(conf, lag)
            if original_features is not None and t_w < len(original_features):
                orig_features = original_features.drop(columns=['value']).iloc[t_w].values
            else:
                orig_features = np.zeros(n_original_features)
            z_w = np.concatenate([orig_features, err_hist, conf_vec])  # Shape: [F]
            tensor_Z_t.append(z_w)
            sequence_Z_t.append(z_w)
        tensor_Z.append(tensor_Z_t)
        sequence_Z.append(np.concatenate(tensor_Z_t))
        valid_indices.append(t)

    tensor_Z = np.array(tensor_Z)
    sequence_Z = np.array(sequence_Z)

    if tensor_scaler is None:
        tensor_scaler = StandardScaler()
        tensor_Z_reshaped = tensor_Z.reshape(-1, tensor_Z.shape[-1])
        tensor_scaler.fit(tensor_Z_reshaped)
    if sequence_scaler is None:
        sequence_scaler = StandardScaler()
        sequence_Z_reshaped = sequence_Z.reshape(-1, sequence_Z.shape[-1])
        sequence_scaler.fit(sequence_Z_reshaped)

    tensor_Z_reshaped = tensor_Z.reshape(-1, tensor_Z.shape[-1])
    tensor_Z = tensor_scaler.transform(tensor_Z_reshaped).reshape(tensor_Z.shape)
    sequence_Z = sequence_scaler.transform(sequence_Z).reshape(sequence_Z.shape)

    return tensor_Z, sequence_Z, valid_indices, tensor_scaler, sequence_scaler

def tabpfn_evaluate(trial):
    lag = trial.suggest_int('lag', 1, min(6, len(train_series) // 4))
    window = trial.suggest_int('window', 2, min(6, len(train_series) // 4))
    bins = trial.suggest_int('bins', 5, 15)

    try:
        bins_edges = pd.cut(train_series, bins=bins, retbins=True)[1]
        train_features, _ = create_features_up_to(
            train_series, lag=lag, window=window, bins_edges=bins_edges, fillna=True
        )
        X_train = train_features.drop(columns=['value'])
        y_train = train_features['value']
        model = TabPFNRegressor(device='cpu')
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        tensor_Z_train, sequence_Z_train, train_valid_indices, tensor_scaler, sequence_scaler = build_dual_channel_features(
            y_train.values, y_pred_train, lag=lag, window=window, original_features=train_features
        )
        sequence_Z_train_flat = sequence_Z_train.reshape(len(sequence_Z_train), -1)
        X_train_subset = sequence_Z_train_flat

        tscv = TimeSeriesSplit(n_splits=5)
        mse_scores = []
        for train_idx, val_idx in tscv.split(X_train_subset):
            X_t, X_v = X_train_subset[train_idx], X_train_subset[val_idx]
            y_t, y_v = y_train.iloc[train_valid_indices].iloc[train_idx], y_train.iloc[train_valid_indices].iloc[val_idx]
            model = TabPFNRegressor(device='cpu')
            model.fit(X_t, y_t)
            y_pred_v = model.predict(X_v)
            mse = mean_squared_error(y_v, y_pred_v)
            mse_scores.append(mse)

        return -np.mean(mse_scores) if mse_scores else -float('inf')
    except:
        return -float('inf')
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(tabpfn_evaluate, n_trials=50)


best_params = study.best_params
lag_opt = best_params['lag']
window_opt = best_params['window']
bins_opt = best_params['bins']
print(f"Optimal Parameters: lag={lag_opt}, window={window_opt}, bins={bins_opt}")

with open('tabpfn_best_params.json', 'w') as f:
    json.dump(best_params, f)
bins_edges = pd.cut(train_series, bins=bins_opt, retbins=True)[1]
train_features, _ = create_features_up_to(
    train_series, lag=lag_opt, window=window_opt, bins_edges=bins_edges, fillna=True
)
X_train = train_features.drop(columns=['value'])
y_train = train_features['value']
model = TabPFNRegressor(device='cpu')
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
tensor_Z_train, sequence_Z_train, train_valid_indices, tensor_scaler, sequence_scaler = build_dual_channel_features(
    y_train.values, y_pred_train, lag=lag_opt, window=window_opt, original_features=train_features
)
X_train_augmented = sequence_Z_train.reshape(len(sequence_Z_train), -1)
y_train_subset = y_train.iloc[train_valid_indices]
n_history = lag_opt + window_opt - 1
extended_index = train_series.index[-n_history:].union(test_series.index)
extended_index = pd.date_range(start=extended_index[0], end=extended_index[-1], freq='MS')
extended_series = pd.Series(
    np.concatenate([train_series.values[-n_history:], test_series.values]),
    index=extended_index
)
test_features, _ = create_features_up_to(
    extended_series, lag=lag_opt, window=window_opt, bins_edges=bins_edges, fillna=True
)
test_start_idx = len(train_series.values[-n_history:])
X_test = test_features.drop(columns=['value']).iloc[test_start_idx:]
y_test = test_features['value'].iloc[test_start_idx:]
y_pred_test = model.predict(X_test)
y_pred_extended = np.concatenate([y_pred_train[-n_history:], y_pred_test])
tensor_Z_test, sequence_Z_test, test_valid_indices, _, _ = build_dual_channel_features(
    extended_series.values, y_pred_extended, lag=lag_opt, window=window_opt, original_features=test_features,
    tensor_scaler=tensor_scaler, sequence_scaler=sequence_scaler
)
test_valid_indices = [i - test_start_idx for i in test_valid_indices if i >= test_start_idx]
X_test_augmented = sequence_Z_test.reshape(len(sequence_Z_test), -1)
y_test_subset = y_test.iloc[test_valid_indices]
model.fit(X_train_augmented, y_train_subset)
y_pred_scaled = model.predict(X_test_augmented)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_unscaled = scaler.inverse_transform(y_test_subset.values.reshape(-1, 1)).flatten()
mse = mean_squared_error(y_test_unscaled, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_unscaled, y_pred)
mape = mean_absolute_percentage_error(y_test_unscaled, y_pred) if np.all(np.abs(y_test_unscaled) >= 1e-6) else float('inf')
r2 = r2_score(y_test_unscaled, y_pred)

print(f"Modified TabPFN Performance (lag={lag_opt}, window={window_opt}, bins={bins_opt}):")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2%}" if mape != float('inf') else "MAPE: Unreliable due to near-zero values")
print(f"R²: {r2:.4f}")

# Comparison table
df_compare = pd.DataFrame({
    'ds': test_series.index[test_valid_indices],
    'tr': y_test_unscaled,
    'for': y_pred
})
print("Test Set Prediction Comparison:")
print(df_compare)

# 11. Visualize results
plt.figure(figsize=(10, 5))
plt.plot(train_series_raw.index, train_series_raw, label='trail', color='blue')
plt.plot(test_series_raw.index, test_series_raw, label='trail-text', color='orange')
plt.plot(test_series_raw.index[test_valid_indices], y_pred, label='fore', color='green')
plt.axvline(train_series_raw.index[-1], color='gray', linestyle='--', label='line')
plt.xlabel('month')
plt.ylabel('ele')
plt.title(f'（EU  - lag={lag_opt}, window={window_opt}）')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig('tabpfn_prediction_plot.png')
plt.show()
