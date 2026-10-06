"""
Inference Script for Freight Rate Prediction Challenge
Generates validation_predictions.csv (12,000 rows) and updates december-chart-inputs.csv.
"""

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from train import preprocess_and_engineer


def main():
    print("=" * 60)
    print("FREIGHT RATE INFERENCE & PREDICTION GENERATION")
    print("=" * 60)

    # 1. Load trained model artifact
    model_path = Path("models/freight_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}. Run train.py first.")

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    fit_stats = artifact["fit_stats"]
    feature_names = artifact["feature_names"]

    print(f"Loaded trained CatBoost model and preprocessing statistics.")

    # 2. Predict validation.csv
    val_path = Path("validation.csv")
    if not val_path.exists():
        val_path = Path("data/validation.csv")

    print(f"Reading validation data from: {val_path}")
    val_df = pd.read_csv(val_path)
    expected_rows = 12000
    if len(val_df) != expected_rows:
        print(f"Warning: validation rows = {len(val_df)}, expected {expected_rows}")

    X_val, _, _ = preprocess_and_engineer(val_df, is_train=False, fit_stats=fit_stats)
    val_preds = model.predict(X_val[feature_names])

    # Ensure all predictions are positive
    val_preds = np.clip(val_preds, a_min=1.0, a_max=None)

    submission_df = pd.DataFrame({
        "load_id": val_df["load_id"],
        "predicted_rate": np.round(val_preds, 2)
    })

    output_csv = Path("validation_predictions.csv")
    submission_df.to_csv(output_csv, index=False)
    print(f"Generated {len(submission_df):,} predictions in: {output_csv}")
    print(f"Columns: {list(submission_df.columns)}")
    print(f"Sample predictions:\n{submission_df.head(5)}")

    # 3. Predict december-chart-inputs.csv
    dec_path = Path("december-chart-inputs.csv")
    if not dec_path.exists():
        dec_path = Path("data/december_chart_inputs.csv")

    if dec_path.exists():
        print(f"\nProcessing December chart inputs from: {dec_path}")
        dec_df = pd.read_csv(dec_path)

        # Build feature representation for December chart inputs
        # Lexington -> Fort Wayne fixed lane
        dec_full = dec_df.copy()
        dec_full["pickup_lat"] = 36.99152
        dec_full["pickup_lon"] = -84.99876
        dec_full["delivery_lat"] = 41.31561
        dec_full["delivery_lon"] = -85.36206

        # Pull daily December market_index and quote_signal from validation.csv
        val_df["date_dt"] = pd.to_datetime(val_df["date"])
        dec_val = val_df[val_df["date_dt"].dt.month == 12]
        daily_macro = dec_val.groupby("date_dt").agg({
            "market_index": "mean",
            "quote_signal": "mean"
        }).reset_index()

        dec_full["date_dt"] = pd.to_datetime(dec_full["date"])
        dec_full = pd.merge(dec_full, daily_macro, on="date_dt", how="left")

        # Fallback if any date missing
        dec_full["market_index"] = dec_full["market_index"].fillna(fit_stats["market_index_median"])
        dec_full["quote_signal"] = dec_full["quote_signal"].fillna(2.05)

        X_dec, _, _ = preprocess_and_engineer(dec_full, is_train=False, fit_stats=fit_stats)
        dec_preds = model.predict(X_dec[feature_names])
        dec_preds = np.clip(dec_preds, a_min=1.0, a_max=None)

        dec_df["predicted_rate"] = np.round(dec_preds, 2)
        dec_df.to_csv(dec_path, index=False)
        print(f"Updated {len(dec_df)} December predictions in: {dec_path}")
        print(f"Sample December predictions:\n{dec_df.head(5)}")

    print("=" * 60)


if __name__ == "__main__":
    main()
