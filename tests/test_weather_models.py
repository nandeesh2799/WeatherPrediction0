import os
import tempfile
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from train import (
    generate_synthetic_weather_csv,
    pick_temperature_target,
    add_simple_date_features,
    yes_no_to_int,
    train_weather_models,
)
from app import build_features, confidence_level, insight_message


class TestDataAndPreprocessing:
    def test_synthetic_csv_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_weather.csv"
            generate_synthetic_weather_csv(csv_path, n_rows=50, seed=123)

            assert csv_path.exists()
            df = pd.read_csv(csv_path)
            assert len(df) == 50
            expected_cols = ["Date", "MinTemp", "Rainfall", "WindSpeed", "RainToday", "RainTomorrow", "Temp3pm"]
            for col in expected_cols:
                assert col in df.columns
            assert df["RainToday"].isin(["Yes", "No"]).all()
            assert df["RainTomorrow"].isin(["Yes", "No"]).all()

    def test_pick_temperature_target(self):
        # Case 1: Temp3pm present
        df1 = pd.DataFrame({"Temp3pm": [22.5, 25.0], "MinTemp": [12.0, 14.0]})
        assert pick_temperature_target(df1) == "Temp3pm"

        # Case 2: MaxTemp fallback
        df2 = pd.DataFrame({"MaxTemp": [28.0, 30.0], "MinTemp": [15.0, 16.0]})
        assert pick_temperature_target(df2) == "MaxTemp"

        # Case 3: Missing candidate raises ValueError
        df3 = pd.DataFrame({"Pressure": [1013.0, 1015.0]})
        with pytest.raises(ValueError, match="Could not find a temperature column"):
            pick_temperature_target(df3)

    def test_add_simple_date_features(self):
        df = pd.DataFrame({"Date": ["2023-01-15", "2023-07-22", "invalid_date"]})
        result = add_simple_date_features(df, date_col="Date")

        assert "Month" in result.columns
        assert "Day" in result.columns
        assert result.loc[0, "Month"] == 1
        assert result.loc[0, "Day"] == 15
        assert result.loc[1, "Month"] == 7
        assert result.loc[1, "Day"] == 22
        assert pd.isna(result.loc[2, "Month"])

    def test_yes_no_to_int(self):
        s = pd.Series(["Yes", "No", " yes ", "NO", np.nan, "nan"])
        mapped = yes_no_to_int(s)

        assert mapped.iloc[0] == 1
        assert mapped.iloc[1] == 0
        assert mapped.iloc[2] == 1
        assert mapped.iloc[3] == 0
        assert pd.isna(mapped.iloc[4])
        assert pd.isna(mapped.iloc[5])


class TestModelTrainingAndInference:
    def test_train_and_inference_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "weather.csv"
            temp_model_path = Path(tmpdir) / "temp_model.pkl"
            rain_model_path = Path(tmpdir) / "rain_model.pkl"

            # Generate synthetic dataset
            generate_synthetic_weather_csv(csv_path, n_rows=200, seed=42)

            # Train models
            metrics = train_weather_models(
                data_path=csv_path,
                temp_model_out=temp_model_path,
                rain_model_out=rain_model_path,
                seed=42,
            )

            assert temp_model_path.exists()
            assert rain_model_path.exists()
            assert metrics["reg_mae"] > 0.0
            assert metrics["reg_r2"] > 0.50
            assert metrics["cls_acc"] > 0.50

            # Safely verify deserialization of freshly-trained models in test tempdir
            import joblib
            temp_bundle = joblib.load(temp_model_path)
            rain_bundle = joblib.load(rain_model_path)

            features = build_features(
                user_date=date(2023, 6, 15),
                mintemp=18.5,
                rainfall=0.0,
                windspeed=12.0,
                raintoday="No",
            )

            # Predict temperature
            pred_temp = float(temp_bundle["model"].predict(features[temp_bundle["feature_cols"]])[0])
            assert -10.0 <= pred_temp <= 50.0

            # Predict rain probability
            prob_rain = float(rain_bundle["model"].predict_proba(features[rain_bundle["feature_cols"]])[0][1])
            assert 0.0 <= prob_rain <= 1.0


class TestAppHelpers:
    def test_confidence_level(self):
        assert confidence_level(85.0) == "High"
        assert confidence_level(65.0) == "Medium"
        assert confidence_level(45.0) == "Low"

    def test_insight_message(self):
        msg_rain_high = insight_message("Yes", prob_rain=0.85, rainfall=10.0, windspeed=20.0)
        assert "High chance of rain" in msg_rain_high

        msg_rain_med = insight_message("Yes", prob_rain=0.60, rainfall=5.0, windspeed=15.0)
        assert "Rain is likely" in msg_rain_med

        msg_dry_stable = insight_message("No", prob_rain=0.15, rainfall=0.0, windspeed=10.0)
        assert "stable" in msg_dry_stable.lower()

        msg_dry_uncertain = insight_message("No", prob_rain=0.40, rainfall=8.0, windspeed=35.0)
        assert "noticeable rain risk" in msg_dry_uncertain.lower()
        assert "Recent rainfall" in msg_dry_uncertain
        assert "Strong winds" in msg_dry_uncertain
