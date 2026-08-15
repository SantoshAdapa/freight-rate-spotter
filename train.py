"""
Freight Rate Prediction - Spotter ML Assessment
=============================================================
Implementation of an ensemble LightGBM model for predicting freight rates.
Includes Bayesian target encoding, geographic feature engineering (Haversine/circuity),
and a time-based holdout validation strategy.

Usage:
    python train.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_data():
    train = pd.read_csv("train-test.csv")
    val = pd.read_csv("validation.csv")
    december = pd.read_csv("december-chart-inputs.csv")
    template = pd.read_csv("validation-predictions-template.csv")
    return train, val, december, template


# ---------------------------------------------------------------------------
# 2. TARGET ENCODING (Bayesian smoothed)
# ---------------------------------------------------------------------------

_te = {}


def build_target_encoding(train_df: pd.DataFrame):
    global _te
    train_df = train_df.copy()
    gmean = train_df["posted_rate"].mean()
    _te["global_mean"] = gmean

    def smoothed(col, k=30):
        agg = train_df.groupby(col)["posted_rate"].agg(["mean", "count"])
        agg["smoothed"] = (agg["count"] * agg["mean"] + k * gmean) / (agg["count"] + k)
        return agg["smoothed"].to_dict(), agg["count"].to_dict()

    _te["pickup_mean"], _ = smoothed("pickup")
    _te["delivery_mean"], _ = smoothed("delivery")

    train_df["_route"] = train_df["pickup"] + "|" + train_df["delivery"]
    _te["route_mean"], _te["route_count"] = smoothed("_route", k=10)

    _te["equip_mean"], _ = smoothed("equipment", k=50)

    # Rate-per-mile by equipment
    train_df["_rpm"] = train_df["posted_rate"] / train_df["distance"]
    for eq in ["Dry Van", "Reefer", "Flatbed"]:
        sub = train_df[train_df["equipment"] == eq]
        _te["rpm_" + eq] = sub["_rpm"].mean()
    _te["rpm_global"] = train_df["_rpm"].mean()


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def _haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def engineer_features(df: pd.DataFrame, fill_defaults: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    gmean = _te.get("global_mean", 2374.0)

    # --- Synthesise missing columns (December inputs) ---
    if fill_defaults:
        coords = {
            "Lexington": (36.99152, -84.99876),
            "Fort Wayne": (41.31561, -85.36206),
        }
        if "pickup_lat" not in df.columns:
            df["pickup_lat"] = df["pickup"].map(lambda c: coords.get(c, (35.65, -90.93))[0])
            df["pickup_lon"] = df["pickup"].map(lambda c: coords.get(c, (35.65, -90.93))[1])
        if "delivery_lat" not in df.columns:
            df["delivery_lat"] = df["delivery"].map(lambda c: coords.get(c, (35.65, -90.93))[0])
            df["delivery_lon"] = df["delivery"].map(lambda c: coords.get(c, (35.65, -90.93))[1])
        if "market_index" not in df.columns:
            df["market_index"] = 1.083
        if "quote_signal" not in df.columns:
            df["quote_signal"] = 2.056

    # --- Fill missing values early ---
    wt_med = df["weight"].median() if df["weight"].notna().any() else 30000.0
    mi_med = df["market_index"].median() if df["market_index"].notna().any() else 1.083
    df["weight"] = df["weight"].fillna(wt_med)
    df["market_index"] = df["market_index"].fillna(mi_med)

    # ===== DATE FEATURES =====
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    # ===== GEOGRAPHIC FEATURES (haversine, circuity) =====
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["geo_distance_approx"] = np.sqrt(df["lat_diff"]**2 + df["lon_diff"]**2)
    df["haversine"] = _haversine(
        df["pickup_lat"], df["pickup_lon"],
        df["delivery_lat"], df["delivery_lon"]
    )
    df["circuity"] = df["distance"] / (df["haversine"] + 1)

    # ===== INTERACTION FEATURES =====
    df["distance_x_market"] = df["distance"] * df["market_index"]
    df["distance_x_quote"] = df["distance"] * df["quote_signal"]
    df["weight_x_distance"] = df["weight"] * df["distance"]
    df["market_x_quote"] = df["market_index"] * df["quote_signal"]

    # ===== SIGNAL FEATURES =====
    df["quote_per_mile"] = df["quote_signal"] / (df["distance"] + 1)

    # Equipment-specific RPM estimate
    equip_col = df["equipment"] if df["equipment"].dtype == object else None
    rpm_map = {eq: _te.get("rpm_" + eq, _te.get("rpm_global", 2.2))
               for eq in ["Dry Van", "Reefer", "Flatbed"]}
    if equip_col is not None:
        df["equip_rpm"] = equip_col.map(rpm_map).fillna(_te.get("rpm_global", 2.2))
    else:
        inv_eq = {0: "Dry Van", 1: "Reefer", 2: "Flatbed"}
        df["equip_rpm"] = df["equipment"].map(inv_eq).map(rpm_map).fillna(2.2)
    df["rpm_rate_estimate"] = df["equip_rpm"] * df["distance"]

    # ===== TARGET-ENCODED FEATURES =====
    df["pickup_rate_mean"] = df["pickup"].map(_te.get("pickup_mean", {})).fillna(gmean)
    df["delivery_rate_mean"] = df["delivery"].map(_te.get("delivery_mean", {})).fillna(gmean)

    route = df["pickup"] + "|" + df["delivery"]
    df["route_rate_mean"] = route.map(_te.get("route_mean", {})).fillna(gmean)
    df["route_count"] = route.map(_te.get("route_count", {})).fillna(0)

    if equip_col is not None:
        df["equip_rate_mean"] = equip_col.map(_te.get("equip_mean", {})).fillna(gmean)
    else:
        inv_eq = {0: "Dry Van", 1: "Reefer", 2: "Flatbed"}
        df["equip_rate_mean"] = df["equipment"].map(inv_eq).map(
            _te.get("equip_mean", {})).fillna(gmean)

    # Route rate residual vs equipment-based RPM estimate
    df["route_rate_vs_estimate"] = df["route_rate_mean"] - df["rpm_rate_estimate"]

    # ===== CATEGORICAL ENCODING =====
    equipment_map = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
    if df["equipment"].dtype == object:
        df["equipment"] = df["equipment"].map(equipment_map)

    return df


# Selected features based on exploratory analysis
FEATURE_COLS = [
    # raw numeric (8)
    "distance", "weight", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "market_index", "quote_signal",
    # date (8)
    "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_month_start", "is_month_end", "day_of_year",
    # geography (5) -- added haversine & circuity
    "lat_diff", "lon_diff", "geo_distance_approx", "haversine", "circuity",
    # interactions (4)
    "distance_x_market", "distance_x_quote", "weight_x_distance", "market_x_quote",
    # signals (3) -- rpm_rate_estimate is high-gain
    "quote_per_mile", "equip_rpm", "rpm_rate_estimate",
    # target-encoded (6)
    "pickup_rate_mean", "delivery_rate_mean",
    "route_rate_mean", "route_count", "equip_rate_mean",
    "route_rate_vs_estimate",
    # categorical (1)
    "equipment",
]

CAT_FEATURES = ["equipment"]


# ---------------------------------------------------------------------------
# 4. MODEL TRAINING
# ---------------------------------------------------------------------------

def time_split(df, cutoff_month=9):
    mask = df["month"] < cutoff_month
    return df[mask].copy(), df[~mask].copy()


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print("")
    print("=" * 55)
    print("  {} Evaluation Metrics".format(label))
    print("=" * 55)
    print("  MAE  : ${:,.2f}".format(mae))
    print("  RMSE : ${:,.2f}".format(rmse))
    print("  MAPE : {:.2f}%".format(mape))
    print("  R2   : {:.6f}".format(r2))
    print("=" * 55)
    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


# Two model configs: both MAE objective, different complexity
MODEL_CONFIGS = [
    {   # M1: deeper, slower
        "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
        "learning_rate": 0.01, "num_leaves": 255,
        "min_child_samples": 20, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": -1, "seed": 42,
    },
    {   # M2: different randomization
        "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
        "learning_rate": 0.01, "num_leaves": 255,
        "min_child_samples": 20, "feature_fraction": 0.7,
        "bagging_fraction": 0.7, "bagging_freq": 5,
        "reg_alpha": 0.1, "reg_lambda": 0.1,
        "verbose": -1, "n_jobs": -1, "seed": 777,
    },
]

ENSEMBLE_WEIGHTS = [0.5, 0.5]


def train_single(X_tr, y_tr, X_te, y_te, params, label=""):
    ds_tr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_FEATURES)
    ds_te = lgb.Dataset(X_te, label=y_te, reference=ds_tr,
                        categorical_feature=CAT_FEATURES)

    model = lgb.train(
        params, ds_tr,
        num_boost_round=8000,
        valid_sets=[ds_tr, ds_te],
        valid_names=["train", "test"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=150, verbose=False),
            lgb.log_evaluation(period=500),
        ],
    )
    print("  {} best iteration: {}".format(label, model.best_iteration))
    return model


def ensemble_predict(models, weights, X):
    pred = np.zeros(len(X))
    for m, w in zip(models, weights):
        pred += w * m.predict(X, num_iteration=m.best_iteration)
    return np.maximum(pred, 1.0)


def train_ensemble(train_df, test_df):
    X_tr = train_df[FEATURE_COLS]
    y_tr = train_df["posted_rate"]
    X_te = test_df[FEATURE_COLS]
    y_te = test_df["posted_rate"]

    print("\nTraining ensemble of {} LightGBM models...".format(len(MODEL_CONFIGS)))
    print("  Train size: {:,} rows".format(len(X_tr)))
    print("  Test size:  {:,} rows".format(len(X_te)))
    print("  Features:   {}".format(len(FEATURE_COLS)))

    models = []
    for i, cfg in enumerate(MODEL_CONFIGS):
        label = "M{}".format(i + 1)
        print("\n  --- Model {} (seed={}) ---".format(i+1, cfg["seed"]))
        m = train_single(X_tr, y_tr, X_te, y_te, cfg, label)
        models.append(m)

        pred = m.predict(X_te, num_iteration=m.best_iteration)
        pred = np.maximum(pred, 1.0)
        evaluate(y_te, pred, "Model {} (individual)".format(i+1))

    # Ensemble eval
    pred_ens = ensemble_predict(models, ENSEMBLE_WEIGHTS, X_te)
    metrics = evaluate(y_te, pred_ens, "ENSEMBLE (avg)")

    # Feature importance
    imp = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": models[0].feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\nTop 15 Feature Importances (Model 1, gain):")
    for _, r in imp.head(15).iterrows():
        print("  {:30s} -> {:,.0f}".format(r["feature"], r["importance"]))

    return models, metrics


def train_final_ensemble(full_df, best_iters):
    X = full_df[FEATURE_COLS]
    y = full_df["posted_rate"]

    final_models = []
    for i, (cfg, best_iter) in enumerate(zip(MODEL_CONFIGS, best_iters)):
        print("  Retraining Model {} ({:,} rounds)...".format(i+1, best_iter))
        ds = lgb.Dataset(X, label=y, categorical_feature=CAT_FEATURES)
        m = lgb.train(cfg, ds, num_boost_round=best_iter)
        m.best_iteration = best_iter
        final_models.append(m)

    return final_models


# ---------------------------------------------------------------------------
# 5. PREDICTION & OUTPUT
# ---------------------------------------------------------------------------

def generate_validation_predictions(models, weights, val_df, template_df):
    preds = ensemble_predict(models, weights, val_df[FEATURE_COLS])

    result = template_df.copy()
    result["predicted_rate"] = result["load_id"].map(dict(zip(val_df["load_id"], preds)))

    missing = result["predicted_rate"].isna().sum()
    if missing > 0:
        print("  WARNING: {} load_ids not found!".format(missing))

    result["predicted_rate"] = result["predicted_rate"].round(2)
    result.to_csv("validation_predictions.csv", index=False)

    print("\nValidation predictions saved: validation_predictions.csv")
    print("  Rows: {:,}".format(len(result)))
    print("  Rate range: ${:,.2f} - ${:,.2f}".format(
        result["predicted_rate"].min(), result["predicted_rate"].max()))
    print("  Mean rate:  ${:,.2f}".format(result["predicted_rate"].mean()))
    return result


def generate_december_predictions(models, weights, december_df):
    dec = engineer_features(december_df, fill_defaults=True)
    preds = ensemble_predict(models, weights, dec[FEATURE_COLS])

    out = december_df.copy()
    out["predicted_rate"] = np.round(preds, 2)
    out.to_csv("december-chart-inputs.csv", index=False)

    print("\nDecember predictions saved: december-chart-inputs.csv")
    print("  Date range: {} - {}".format(out["date"].min(), out["date"].max()))
    print("  Rate range: ${:,.2f} - ${:,.2f}".format(
        out["predicted_rate"].min(), out["predicted_rate"].max()))
    print("  Mean rate:  ${:,.2f}".format(out["predicted_rate"].mean()))
    return out


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Freight Rate Prediction - Spotter ML Assessment")
    print("=" * 60)

    # 1 Load
    print("\n[1/6] Loading data...")
    train_raw, val_raw, december_raw, template = load_data()
    print("  Training:   {}".format(train_raw.shape))
    print("  Validation: {}".format(val_raw.shape))
    print("  December:   {}".format(december_raw.shape))

    # 2 Feature engineering
    print("\n[2/6] Building target encodings & engineering features...")
    build_target_encoding(train_raw)
    train_fe = engineer_features(train_raw)
    val_fe = engineer_features(val_raw)
    print("  Total features: {}".format(len(FEATURE_COLS)))

    # 3 Split
    print("\n[3/6] Time-based train/test split...")
    split_tr, split_te = time_split(train_fe, cutoff_month=9)
    print("  Train (Jan-Aug): {:,} rows".format(len(split_tr)))
    print("  Test  (Sep-Oct): {:,} rows".format(len(split_te)))

    # 4 Train
    print("\n[4/6] Training ensemble...")
    models, metrics = train_ensemble(split_tr, split_te)

    # 5 Retrain
    best_iters = [m.best_iteration for m in models]
    print("\n[5/6] Retraining ensemble on full training data...")
    final_models = train_final_ensemble(train_fe, best_iters)

    # 6 Predict
    print("\n[6/6] Generating predictions...")
    generate_validation_predictions(final_models, ENSEMBLE_WEIGHTS, val_fe, template)
    generate_december_predictions(final_models, ENSEMBLE_WEIGHTS, december_raw)

    print("\n" + "=" * 60)
    print("  DONE - Ready to run score.py")
    print("=" * 60)
    print("\nNext step:")
    print("  python score.py --predictions validation_predictions.csv "
          "--december-predictions december-chart-inputs.csv")


if __name__ == "__main__":
    main()
