import os
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import pywt
from scipy.signal import medfilt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor

import xgboost as xgb
import lightgbm as lgb
import optuna
import time

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

# def compute_dynamic_max_depth(res, X, d0, δ, δp):
#     σ = np.std(res); f = σ if σ>0 else 1.0
#     return max(1, int(round(d0 + (δ*f if f>1 else -δp*f))))
#
# def compute_dynamic_gamma(res, base=1.0, α=0.5):
#     S, K = skew(res), kurtosis(res, fisher=False)
#     return base / (1 + α*(abs(S) + abs(K-3))/3)
#
# def mixed_obj(preds, dtrain):
#     y = dtrain.get_label()
#     w0 = min(0.9, abs(skew(y)) / (abs(skew(y)) + 1))
#     grad = 2*(1-w0)*(preds - y) + w0 * np.sign(preds - y)
#     hess = 2*(1-w0)*np.ones_like(y)
#     return grad, hess

# def tune_dynamic_params(X, y, w, n_trials=100):
#     def objective(trial):
#         d0  = trial.suggest_int("d0", 4, 8)
#         δ   = trial.suggest_float("delta", 0.5, 3.0)
#         δp  = trial.suggest_float("delta_p", 0.5, 3.0)
#         α   = trial.suggest_float("alpha", 0.1, 1.0)
#         depth = compute_dynamic_max_depth(y - y.mean(), X, d0, δ, δp)
#         gamma = compute_dynamic_gamma(y - y.mean(), base=1.0, α=α)
#         dtrain = xgb.DMatrix(X, label=y, weight=w)
#         cv = xgb.cv(
#             {"tree_method":"hist","eta":0.05,"max_depth":depth,"gamma":gamma,
#              "subsample":0.8,"colsample_bytree":0.8,
#              "objective":"reg:squarederror","eval_metric":"mae"},
#             dtrain, num_boost_round=200, nfold=3, seed=42, verbose_eval=False
#         )
#         return cv["test-mae-mean"].min()
#     study = optuna.create_study(direction="minimize")
#     def stop_if_no_improve(study, trial):
#         if trial.number - study.best_trial.number >= 40:
#             study.stop()
#     study.optimize(objective, n_trials=n_trials, n_jobs=5,
#                    callbacks=[stop_if_no_improve], show_progress_bar=False)
#     p = study.best_trial.params
#     return p["d0"], p["delta"], p["delta_p"], p["alpha"]

def main():
    os.makedirs("./results", exist_ok=True)

    metrics_all = []
    params_all  = []

    df0 = pd.read_csv("caltech_test_data.csv", parse_dates=["connection_time_copy"])
    df0 = df0[
        (df0.parking_time <= df0.Requested_parking_time + 2) &
        (df0.kWhRequested <= 150) &
        (df0.kWhDelivered <= df0.kWhRequested)
    ].dropna(subset=["connection_time_copy"]).reset_index(drop=True)

    df0["hour"]      = df0.connection_time_copy.dt.hour
    df0["weekday"]   = df0.connection_time_copy.dt.weekday
    df0["month"]     = df0.connection_time_copy.dt.month
    df0["dayofyear"] = df0.connection_time_copy.dt.dayofyear
    for c in ["hour","weekday","month","dayofyear"]:
        df0[f"{c}_sin"] = np.sin(2*np.pi*df0[c]/df0[c].max())
        df0[f"{c}_cos"] = np.cos(2*np.pi*df0[c]/df0[c].max())
    for c in ["userID","stationID"]:
        df0[c] = LabelEncoder().fit_transform(df0[c].astype(str))
    df0.sort_values("connection_time_copy", inplace=True)
    df0.reset_index(drop=True, inplace=True)

    test_start = datetime(2019,12,1)
    test_end   = datetime(2019,12,30)
    windows    = [30,60,120,240,360,480]
    n_runs     = 1
    eps        = 1e-3

    for run in range(1, n_runs+1):
        run_time = time.perf_counter()
        print(f"\n=== Run {run} : {run_time} ===")
        run_summary = []

        df = df0.copy()

        hist = df[df.connection_time_copy < test_start]
        ua = hist.groupby("userID").kWhDelivered.mean() / (
             hist.groupby("userID").kWhRequested.mean() + 1e-6
        )
        km = KMeans(n_clusters=5, random_state=run).fit(ua.values.reshape(-1,1))
        df["cluster"] = df.userID.map(dict(zip(ua.index, km.labels_)))

        for w in windows:
            print(f"\n>>> Window = {w} days")
            t_window_start = time.perf_counter()

            tr0 = test_start - timedelta(days=w)
            tr  = df[(df.connection_time_copy>=tr0)&(df.connection_time_copy<test_start)].copy()
            te  = df[(df.connection_time_copy>=test_start)&(df.connection_time_copy<=test_end)].copy()
            if tr.empty or te.empty:
                continue

            for col in ["parking_time","kWhDelivered"]:
                coeffs = pywt.wavedec(tr[col], 'db4', level=3)
                σ      = np.median(np.abs(coeffs[-1]))/0.6745
                thr    = σ * np.sqrt(2*np.log(len(tr)))
                coeffs[1:] = [pywt.threshold(c,thr,'soft') for c in coeffs[1:]]
                tr[col]     = pywt.waverec(coeffs,'db4')[:len(tr)]
            mask = LocalOutlierFactor(n_neighbors=20).fit_predict(tr[["parking_time","kWhDelivered"]])
            tr   = tr[mask==1].reset_index(drop=True)
            for col in ["parking_time","kWhDelivered"]:
                coeffs    = pywt.wavedec(te[col], 'db4', level=3)
                coeffs[1:]= [pywt.threshold(c,thr,'soft') for c in coeffs[1:]]
                te[col]    = pywt.waverec(coeffs,'db4')[:len(te)]

            tr["age_days"] = (test_start - tr.connection_time_copy).dt.days
            w_decay       = np.exp(-0.01 * tr["age_days"])
            tr.reset_index(drop=True, inplace=True)
            te.reset_index(drop=True, inplace=True)

            tr["is_weekend"] = tr.weekday.isin([5,6]).astype(int)
            te["is_weekend"] = te.weekday.isin([5,6]).astype(int)
            tr["is_holiday"] = tr.get("connectionTime_is_holiday", 0)
            te["is_holiday"] = te.get("connectionTime_is_holiday", 0)

            sim_feats = ["hour","weekday","kWhRequested","Requested_parking_time"]
            ss = StandardScaler().fit(tr[sim_feats])
            tr_s, te_s = ss.transform(tr[sim_feats]), ss.transform(te[sim_feats])
            tr["sim_park"] = tr["sim_kwh"] = 0.0
            for i in range(len(tr)):
                prev = tr[(tr.userID==tr.loc[i,"userID"])&(tr.index<i)].index
                if prev.empty:
                    tr.at[i,"sim_park"] = tr.parking_time.median()
                    tr.at[i,"sim_kwh"]  = tr.kWhDelivered.median()
                else:
                    sims = np.dot(tr_s[prev], tr_s[i])
                    top = prev[np.argsort(sims)[-3:]]; w3 = sims[np.argsort(sims)[-3:]]
                    tr.at[i,"sim_park"] = np.average(tr.loc[top,"parking_time"], weights=w3)
                    tr.at[i,"sim_kwh"]  = np.average(tr.loc[top,"kWhDelivered"], weights=w3)
            te["sim_park"] = te["sim_kwh"] = 0.0
            for i in range(len(te)):
                prev = tr[tr.userID==te.loc[i,"userID"]].index
                if prev.empty:
                    te.at[i,"sim_park"] = tr.parking_time.median()
                    te.at[i,"sim_kwh"]  = tr.kWhDelivered.median()
                else:
                    sims = np.dot(tr_s[prev], te_s[i])
                    top = prev[np.argsort(sims)[-3:]]; w3 = sims[np.argsort(sims)[-3:]]
                    te.at[i,"sim_park"] = np.average(tr.loc[top,"parking_time"], weights=w3)
                    te.at[i,"sim_kwh"]  = np.average(tr.loc[top,"kWhDelivered"], weights=w3)

            ua_m = tr.groupby("userID").kWhDelivered.mean()
            ua_c = tr.userID.value_counts()
            sa_m = tr.groupby("stationID").kWhDelivered.mean()
            sa_c = tr.stationID.value_counts()
            for D in (tr, te):
                D["user_avg_kWh"]    = D.userID.map(ua_m)
                D["user_freq"]       = D.userID.map(ua_c)
                D["station_avg_kWh"] = D.stationID.map(sa_m)
                D["station_freq"]    = D.stationID.map(sa_c)

            pf = ["hour","weekday","month","is_weekend","is_holiday",
                  "hour_sin","hour_cos","weekday_sin","weekday_cos",
                  "month_sin","month_cos","userID","stationID","sim_park"]
            Xp_tr, y_p_tr = tr[pf].values, tr.parking_time.values
            Xp_te, y_p_te = te[pf].values, te.parking_time.values

            n_val = max(int(0.2*len(tr)), 50)
            Xtr_p, Xva_p = Xp_tr[:-n_val], Xp_tr[-n_val:]
            ytr_p, yva_p = y_p_tr[:-n_val], y_p_tr[-n_val:]
            wtr_p        = w_decay[:-n_val]

            # d0_p, δ_p, δp_p, α_p = tune_dynamic_params(Xtr_p, ytr_p, wtr_p, n_trials=100)
            # dp = compute_dynamic_max_depth(ytr_p - ytr_p.mean(), Xtr_p, d0_p, δ_p, δp_p)
            # gp = compute_dynamic_gamma(ytr_p - ytr_p.mean(), base=1.0, α=α_p)

            fixed_depth_p = 6
            fixed_gamma_p = 0.7

            dtr_p = xgb.DMatrix(Xtr_p, label=ytr_p, weight=wtr_p)
            dva_p = xgb.DMatrix(Xva_p, label=yva_p)
            params_p = {"tree_method":"hist","eta":0.4,"max_depth":fixed_depth_p,"gamma":fixed_gamma_p,
                        "subsample":0.8,"colsample_bytree":0.8,
                        "objective":"reg:squarederror","eval_metric":"mae"}
            bst_p = xgb.train(params_p, dtr_p, num_boost_round=200,
                              evals=[(dtr_p,"train"),(dva_p,"val")],
                              early_stopping_rounds=30, verbose_eval=False)
            pred_p_te = bst_p.predict(xgb.DMatrix(Xp_te))
            mae_p, sm_p = mae(y_p_te, pred_p_te), smape(y_p_te, pred_p_te)

            for D in (tr, te):
                D["req_rate"]     = D.kWhRequested / (D.parking_time + 1e-6)
                D["hour_of_week"] = D.weekday * 24 + D.hour

            sh = tr.groupby(["stationID","hour_of_week"]).kWhDelivered.sum() / \
                 (tr.groupby(["stationID","hour_of_week"]).kWhRequested.sum()+1e-6)
            uh = tr.groupby(["userID","hour_of_week"]).kWhDelivered.sum() / \
                 (tr.groupby(["userID","hour_of_week"]).kWhRequested.sum()+1e-6)
            tr["sh_ratio"] = tr.set_index(["stationID","hour_of_week"]).index.map(sh)
            tr["uh_ratio"] = tr.set_index(["userID","hour_of_week"]).index.map(uh)
            te["sh_ratio"] = te.set_index(["stationID","hour_of_week"]).index.map(sh).fillna(sh.mean())
            te["uh_ratio"] = te.set_index(["userID","hour_of_week"]).index.map(uh).fillna(uh.mean())

            med_k = tr.kWhDelivered.median()
            tr["pk1"] = tr.groupby("userID").kWhDelivered.shift(1).fillna(med_k)
            tr["pk2"] = tr.groupby("userID").kWhDelivered.shift(2).fillna(med_k)
            tr["rk3"] = tr.groupby("userID").kWhDelivered.rolling(3, min_periods=1).mean()\
                           .reset_index(0, drop=True).fillna(med_k)
            te = te.sort_values("connection_time_copy").reset_index(drop=True)
            prev = []
            te["pk1"] = te["pk2"] = te["rk3"] = med_k
            for i in range(len(te)):
                if i>=1: te.at[i,"pk1"] = prev[i-1]
                if i>=2: te.at[i,"pk2"] = prev[i-2]
                if i>=3: te.at[i,"rk3"] = np.mean(prev[i-3:i])
                prev.append(te.at[i,"kWhDelivered"])

            feats = ["hour","weekday","month","is_weekend","is_holiday",
                     "hour_sin","hour_cos","weekday_sin","weekday_cos",
                     "month_sin","month_cos","dayofyear_sin","dayofyear_cos",
                     "cluster","sim_kwh","user_avg_kWh","user_freq",
                     "station_avg_kWh","station_freq",
                     "req_rate","hour_of_week","sh_ratio","uh_ratio",
                     "pk1","pk2","rk3"]

            pt = PowerTransformer(method='box-cox', standardize=False)
            y_all     = tr.kWhDelivered.values + eps
            y_bc      = pt.fit_transform(y_all.reshape(-1,1)).flatten()
            X_all_tr  = tr[feats].values
            X_all_te  = te[feats].values

            Xk_tr     = X_all_tr[:-n_val]
            Xk_va     = X_all_tr[-n_val:]
            ytr_bc    = y_bc[:-n_val]
            yva_bc    = y_bc[-n_val:]
            wtr_k     = w_decay[:-n_val]

            # d0_k, δ_k, δp_k, α_k = tune_dynamic_params(Xk_tr, ytr_bc, wtr_k, n_trials=100)
            # depth_k  = compute_dynamic_max_depth(ytr_bc - ytr_bc.mean(), Xk_tr, d0_k, δ_k, δp_k)
            # gamma_k  = compute_dynamic_gamma(ytr_bc - ytr_bc.mean(), base=1.0, α=α_k)

            fixed_depth_k = 6
            fixed_gamma_k = 0.7

            dtr_k = xgb.DMatrix(Xk_tr, label=ytr_bc, weight=wtr_k)
            dva_k = xgb.DMatrix(Xk_va, label=yva_bc)
            params_k = {"tree_method":"hist","eta":0.4,"max_depth":fixed_depth_k,"gamma":fixed_gamma_k,
                        "subsample":0.8,"colsample_bytree":0.8,
                        "objective":"reg:squarederror","eval_metric":"mae"}
            bst_k = xgb.train(params_k, dtr_k, num_boost_round=200,
                              evals=[(dtr_k,"train"),(dva_k,"val")],
                              early_stopping_rounds=40, verbose_eval=False)
            pred_te_bc = bst_k.predict(xgb.DMatrix(X_all_te))

            res_tr_bc  = ytr_bc - bst_k.predict(xgb.DMatrix(Xk_tr))
            model_lgb  = lgb.LGBMRegressor(objective="regression", metric="mae", verbosity= -1, boosting_type="gbdt",
        n_estimators=200, learning_rate= 0.4,num_leaves=31,feature_fraction=0.8,bagging_fraction= 0.8,bagging_freq=1,
        min_data_in_leaf=20,lambda_l1= 0.1,lambda_l2= 0.1,random_state=42)
            model_lgb.fit(Xk_tr, res_tr_bc, sample_weight=wtr_k)
            res_te_bc  = model_lgb.predict(X_all_te)

            final_bc   = pred_te_bc + res_te_bc
            final_raw  = pt.inverse_transform(final_bc.reshape(-1,1)).flatten() - eps
            order      = np.argsort(te.connection_time_copy.values)
            smooth     = medfilt(final_raw[order], kernel_size=5)
            final_k    = np.empty_like(smooth); final_k[order] = smooth

            mae_k, sm_k = mae(te.kWhDelivered.values, final_k), smape(te.kWhDelivered.values, final_k)

            t_window_end = time.perf_counter()
            t_window_ms = (t_window_end - t_window_start) * 1000

            print(f"  Window={w}: parking_time MAE={mae_p:.3f}, SMAPE={sm_p:.3f}%; "
                  f"kWhDelivered MAE={mae_k:.3f}, SMAPE={sm_k:.3f}%")

            run_summary.append((w, mae_p, sm_p, mae_k, sm_k))

            metrics_all.append({
                "run": run,
                "window": w,
                "mae_pt": mae_p,
                "smape_pt": sm_p,
                "mae_kWh": mae_k,
                "smape_kWh": sm_k,
                "window_time": t_window_ms
            })
            # params_all.append({
            #     "run": run,
            #     "window": w,
            #     "parking_time": f"d0={d0_p},δ={δ_p:.3f},δp={δp_p:.3f},α={α_p:.3f}",
            #     "kWhDelivered": f"d0={d0_k},δ={δ_k:.3f},δp={δp_k:.3f},α={α_k:.3f}"
            # })

        print(f"\nResults for run {run}:")
        print("window | MAE_pt | SMAPE_pt | MAE_kWh | SMAPE_kWh")
        for w, mpt, spt, mk, sk in run_summary:
            print(f"{w:>6} | {mpt:6.3f} | {spt:7.3f}% | {mk:7.3f} | {sk:8.3f}%")

    pd.DataFrame(metrics_all).to_csv("results\\caltech_nody\\metrics.csv", index=False)
    pd.DataFrame(params_all).to_csv("results\\caltech_nody\\params.csv", index=False)
    print("\nAll runs complete. Results saved to ./results/")

if __name__ == "__main__":
    main()
