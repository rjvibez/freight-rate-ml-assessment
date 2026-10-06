"""
Model Training Script for Freight Rate Prediction Challenge
Trains CatBoost regressor on the full labeled development dataset (train-test.csv).
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3958.8 * c  # miles


def preprocess_and_engineer(df, is_train=True, fit_stats=None):
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # 1. Clean weight (fix negative sign-flip artifact)
    data["weight_clean"] = data["weight"].abs()
    if is_train:
        weight_median = float(data["weight_clean"].median())
        market_index_median = float(data["market_index"].median())
        fit_stats = {
            "weight_median": weight_median,
            "market_index_median": market_index_median,
        }
    else:
        weight_median = fit_stats["weight_median"]
        market_index_median = fit_stats["market_index_median"]

    data["weight_clean"] = data["weight_clean"].fillna(weight_median)
    data["market_index_clean"] = data["market_index"].fillna(market_index_median)

    # 2. Distance and Geospatial Features
    data["haversine_dist"] = haversine_np(
        data["pickup_lon"], data["pickup_lat"],
        data["delivery_lon"], data["delivery_lat"]
    )
    data["dist_ratio"] = data["distance"] / (data["haversine_dist"] + 1.0)
    data["lat_diff"] = data["delivery_lat"] - data["pickup_lat"]
    data["lon_diff"] = data["delivery_lon"] - data["pickup_lon"]

    # 3. Domain Interaction Features
    data["est_base_quote"] = data["distance"] * data["quote_signal"]
    data["dist_x_market"] = data["distance"] * data["market_index_clean"]
    data["weight_per_mile"] = data["weight_clean"] / (data["distance"] + 1.0)

    # 4. Temporal Features
    data["dayofweek"] = data["date"].dt.dayofweek
    data["day"] = data["date"].dt.day
    data["is_weekend"] = data["dayofweek"].isin([5, 6]).astype(int)
    data["dayofyear"] = data["date"].dt.dayofyear
    data["month"] = data["date"].dt.month

    # 5. Categorical Encoding (One-Hot for equipment)
    for eq in ["Dry Van", "Reefer", "Flatbed"]:
        data[f"eq_{eq}"] = (data["equipment"] == eq).astype(int)

    feature_cols = [
        "distance", "haversine_dist", "dist_ratio", "lat_diff", "lon_diff",
        "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
        "weight_clean", "market_index_clean", "quote_signal",
        "est_base_quote", "dist_x_market", "weight_per_mile",
        "dayofweek", "day", "is_weekend", "dayofyear", "month",
        "eq_Dry Van", "eq_Reefer", "eq_Flatbed"
    ]

    return data[feature_cols], data.get("posted_rate", None), fit_stats


def main():
    print("=" * 60)
    print("FREIGHT RATE MODEL TRAINING")
    print("=" * 60)

    data_path = Path("train-test.csv")
    if not data_path.exists():
        data_path = Path("data/train_test.csv")
    print(f"Loading development data from: {data_path}")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Out-of-time validation check
    train_dev = df[df["date"] < "2025-09-01"].copy()
    val_dev = df[df["date"] >= "2025-09-01"].copy()

    print(f"Running Out-Of-Time Validation on Dev Split:")
    print(f"  Train period: Jan - Aug 2025 ({len(train_dev):,} loads)")
    print(f"  Validation period: Sep - Oct 2025 ({len(val_dev):,} loads)")

    X_tr, y_tr, dev_stats = preprocess_and_engineer(train_dev, is_train=True)
    X_va, y_va, _ = preprocess_and_engineer(val_dev, is_train=False, fit_stats=dev_stats)

    val_model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0
    )
    val_model.fit(X_tr, y_tr)
    val_preds = val_model.predict(X_va)

    mae = mean_absolute_error(y_va, val_preds)
    rmse = np.sqrt(mean_squared_error(y_va, val_preds))
    mape = np.mean(np.abs((y_va - val_preds) / y_va)) * 100
    r2 = r2_score(y_va, val_preds)

    print(f"Out-of-Time Validation Metrics:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R2:   {r2:.4f}")

    print("\nTraining Final Model on Full Development Dataset (Jan-Oct 2025)...")
    X_full, y_full, full_stats = preprocess_and_engineer(df, is_train=True)

    final_model = CatBoostRegressor(
        iterations=350,
        learning_rate=0.03,
        depth=6,
        random_seed=42,
        verbose=0
    )
    final_model.fit(X_full, y_full)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_artifact = {
        "model": final_model,
        "fit_stats": full_stats,
        "feature_names": list(X_full.columns),
        "validation_metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2
        }
    }

    with open(models_dir / "freight_model.pkl", "wb") as f:
        pickle.dump(model_artifact, f)

    print(f"Model and preprocessor saved successfully to models/freight_model.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
