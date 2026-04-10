import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def generate_synthetic_weather_csv(path: Path, n_rows: int = 3000, seed: int = 42) -> None:
    """
    Create a beginner-friendly synthetic weather dataset so the project runs
    even when you don't have a real CSV.

    Columns are chosen to match the app + training needs:
    Date, MinTemp, Rainfall, WindSpeed, RainToday, RainTomorrow, Temp3pm
    """
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2018-01-01")
    dates = pd.date_range(start=start, periods=n_rows, freq="D")
    month = dates.month.values
    day = dates.day.values

    # Seasonality: warmer mid-year (southern hemisphere-ish), cooler start/end.
    seasonal = 7.0 * np.sin(2 * np.pi * (month - 1) / 12.0)

    wind = np.clip(rng.normal(loc=18, scale=8, size=n_rows), 0, 70)
    min_temp = np.clip(rng.normal(loc=12 + seasonal, scale=4.5, size=n_rows), -5, 30)

    # Rainfall: mostly small, sometimes heavy (gamma-like).
    rain_base = rng.gamma(shape=1.2, scale=3.0, size=n_rows)
    rain_event = rng.random(n_rows) < (0.22 + 0.05 * (wind > 25))
    rainfall = np.where(rain_event, rain_base, 0.0)
    rainfall = np.clip(rainfall + rng.normal(0, 0.6, n_rows), 0, 120)

    rain_today = (rainfall > 0.2).astype(int)

    # RainTomorrow probability (simple + explainable):
    # more rain today + more rainfall + more wind => higher chance tomorrow
    # slight seasonality component as well.
    z = (
        -1.2
        + 0.9 * rain_today
        + 0.04 * rainfall
        + 0.015 * wind
        + 0.15 * np.sin(2 * np.pi * (month - 1) / 12.0)
    )
    p_rain_tom = 1.0 / (1.0 + np.exp(-z))
    rain_tomorrow = (rng.random(n_rows) < p_rain_tom).astype(int)

    # Temperature at 3pm: correlated with min temp + seasonality, reduced a bit on rainy days.
    temp3pm = min_temp + 8.5 + 0.35 * seasonal - 0.03 * rainfall - 0.04 * wind + rng.normal(0, 2.0, n_rows)
    temp3pm = np.clip(temp3pm, -2, 45)

    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "MinTemp": np.round(min_temp, 2),
            "Rainfall": np.round(rainfall, 2),
            "WindSpeed": np.round(wind, 2),
            "RainToday": np.where(rain_today == 1, "Yes", "No"),
            "RainTomorrow": np.where(rain_tomorrow == 1, "Yes", "No"),
            "Temp3pm": np.round(temp3pm, 2),
        }
    )

    df.to_csv(path, index=False)


def pick_temperature_target(df: pd.DataFrame) -> str:
    """
    Pick a temperature target column from common weather datasets.
    Adjust this list if your CSV uses a different column name.
    """
    candidates = ["Temp3pm", "MaxTemp", "Temperature", "Temp"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not find a temperature column. Expected one of: "
        + ", ".join(candidates)
        + ". Please rename your temperature column or edit candidates in train.py."
    )


def add_simple_date_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """
    Beginner-friendly feature engineering:
    - Extract month and day from Date
    - Keep original numeric columns
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out["Month"] = out[date_col].dt.month
    out["Day"] = out[date_col].dt.day
    return out


def yes_no_to_int(series: pd.Series) -> pd.Series:
    """
    Convert Yes/No to 1/0. Unknown/missing stays as NaN (imputer will handle it).
    """
    s = series.astype(str).str.strip().str.lower()
    mapped = s.map({"yes": 1, "no": 0})
    # Turn "nan" string (from missing values) back into real NaN
    mapped = mapped.where(~s.eq("nan"), other=pd.NA)
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weather prediction models.")
    parser.add_argument(
        "--data",
        default="weather.csv",
        help="Path to CSV dataset (default: weather.csv in project folder).",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found at '{data_path}'.")
        print("Creating a synthetic dataset so you can run the project end-to-end...")
        generate_synthetic_weather_csv(data_path, n_rows=3000, seed=42)
        print(f"Saved synthetic dataset to: {data_path}")

    # Load CSV
    df = pd.read_csv(data_path)

    # Basic required columns for the app inputs + prediction targets
    required_inputs = ["Date", "MinTemp", "Rainfall", "WindSpeed", "RainToday", "RainTomorrow"]
    missing = [c for c in required_inputs if c not in df.columns]
    if missing:
        raise ValueError(
            "Your CSV is missing required columns: "
            + ", ".join(missing)
            + ".\nExpected at least: "
            + ", ".join(required_inputs)
        )

    # Choose temperature target
    temp_target = pick_temperature_target(df)

    # Feature engineering
    df = add_simple_date_features(df, date_col="Date")
    df["RainToday"] = yes_no_to_int(df["RainToday"])
    df["RainTomorrow"] = yes_no_to_int(df["RainTomorrow"])

    # Features used by BOTH models (simple + beginner-friendly)
    feature_cols = ["MinTemp", "Rainfall", "WindSpeed", "RainToday", "Month", "Day"]

    # ---------- Temperature model (Regression) ----------
    reg_df = df.dropna(subset=[temp_target]).copy()
    X_reg = reg_df[feature_cols]
    y_reg = reg_df[temp_target]

    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    temp_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]
    )
    temp_model.fit(Xr_train, yr_train)

    reg_pred = temp_model.predict(Xr_test)
    reg_mae = mean_absolute_error(yr_test, reg_pred)
    reg_r2 = r2_score(yr_test, reg_pred)

    # Save temperature model + metadata
    joblib.dump(
        {
            "model": temp_model,
            "feature_cols": feature_cols,
            "temp_target": temp_target,
        },
        "temp_model.pkl",
    )

    # ---------- RainTomorrow model (Classification) ----------
    cls_df = df.dropna(subset=["RainTomorrow"]).copy()
    X_cls = cls_df[feature_cols]
    y_cls = cls_df["RainTomorrow"].astype(int)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    rain_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    rain_model.fit(Xc_train, yc_train)

    cls_pred = rain_model.predict(Xc_test)
    cls_acc = accuracy_score(yc_test, cls_pred)

    # Save rain model + metadata
    joblib.dump(
        {
            "model": rain_model,
            "feature_cols": feature_cols,
        },
        "rain_model.pkl",
    )

    # Print simple training results so beginners see progress
    print("Training complete.")
    print(f"- Temperature target column: {temp_target}")
    print(f"- Regression MAE: {reg_mae:.2f}")
    print(f"- Regression R2:  {reg_r2:.3f}")
    print(f"- RainTomorrow Accuracy: {cls_acc:.3f}")
    print("Saved: temp_model.pkl, rain_model.pkl")


if __name__ == "__main__":
    main()
