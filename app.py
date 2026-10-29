from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


def apply_custom_css() -> None:
    """
    - Hide Streamlit default UI elements (Deploy button, menu, footer)
    - Dark dashboard styling like the reference screenshots
    """
    st.markdown(
        """
        <style>
          /* ---- Hide Streamlit chrome ---- */
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}

          /* "Deploy" button + top-right toolbar (Streamlit versions vary) */
          [data-testid="stDeployButton"] {display:none !important;}
          [data-testid="stToolbar"] {display:none !important;}

          /* ---- App background ---- */
          .stApp {
            background: radial-gradient(1200px 600px at 20% 0%, rgba(60, 140, 255, 0.22), transparent 60%),
                        radial-gradient(900px 500px at 80% 10%, rgba(46, 213, 255, 0.16), transparent 55%),
                        linear-gradient(180deg, #0b1220 0%, #0b172a 55%, #081120 100%);
          }

          /* ---- Main container spacing ---- */
          .block-container {
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            max-width: 1120px;
          }

          /* ---- Title ---- */
          .title-card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 1.0rem 1.2rem;
            margin-bottom: 1.0rem;
            box-shadow: 0 10px 26px rgba(0,0,0,0.35);
          }
          .weather-title {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            margin: 0;
            color: rgba(255,255,255,0.92);
          }
          .weather-subtitle {
            margin-top: 0.25rem;
            font-size: 0.95rem;
            color: rgba(255,255,255,0.65);
          }

          /* ---- Card ---- */
          .card {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 1.0rem 1.0rem;
            box-shadow: 0 10px 26px rgba(0,0,0,0.35);
          }
          .card-title {
            font-weight: 800;
            font-size: 0.98rem;
            color: rgba(255,255,255,0.88);
            margin-bottom: 0.75rem;
          }
          .muted {
            color: rgba(255,255,255,0.60);
            font-size: 0.9rem;
          }

          /* ---- Buttons ---- */
          div.stButton > button, div.stFormSubmitButton > button {
            background: linear-gradient(90deg, #2aa7ff, #1d7dff);
            color: white;
            border: 0;
            border-radius: 10px;
            padding: 0.65rem 1.1rem;
            font-weight: 700;
            width: 100%;
          }
          div.stButton > button:hover, div.stFormSubmitButton > button:hover {
            filter: brightness(1.02);
          }

          /* ---- Inputs ---- */
          .stNumberInput input, .stTextInput input, .stDateInput input {
            border-radius: 10px !important;
          }

          /* ---- Small badges ---- */
          .badge {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            background: rgba(42, 167, 255, 0.12);
            border: 1px solid rgba(42, 167, 255, 0.25);
            color: rgba(255,255,255,0.85);
            font-weight: 700;
            font-size: 0.86rem;
          }

          /* ---- Result banner ---- */
          .result-wrap { margin-top: 1.1rem; }
          .result-title {
            font-weight: 900;
            font-size: 1.05rem;
            color: rgba(255,255,255,0.92);
            margin-bottom: 0.7rem;
          }
          .banner {
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            border: 1px solid rgba(255,255,255,0.10);
            background: rgba(34, 197, 94, 0.18);
            box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.15);
            margin-bottom: 0.8rem;
          }
          .banner.danger {
            background: rgba(239, 68, 68, 0.16);
            box-shadow: inset 0 0 0 1px rgba(239, 68, 68, 0.18);
          }
          .banner .topline {
            font-weight: 800;
            color: rgba(255,255,255,0.92);
            margin-bottom: 0.12rem;
          }
          .banner .subline {
            color: rgba(255,255,255,0.70);
            font-size: 0.9rem;
          }

          /* ---- Metric cards (custom) ---- */
          .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.65rem;
            margin-bottom: 0.45rem;
          }
          .metric-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
            padding: 0.75rem 0.85rem;
          }
          .metric-label {
            color: rgba(255,255,255,0.62);
            font-size: 0.84rem;
            font-weight: 700;
          }
          .metric-value {
            margin-top: 0.25rem;
            color: rgba(255,255,255,0.92);
            font-size: 1.35rem;
            font-weight: 900;
            letter-spacing: -0.02em;
          }

          /* ---- Progress bar tweaks ---- */
          div[data-testid="stProgress"] > div {
            background: rgba(255,255,255,0.10);
            border-radius: 999px;
            overflow: hidden;
          }
          div[data-testid="stProgress"] > div > div {
            background: linear-gradient(90deg, #22c55e, #2aa7ff);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_model_bundle(path: str):
    """
    Safely load locally generated scikit-learn model bundles produced by train_weather_models.
    If the artifact is missing, triggers local model training from verified local dataset.
    """
    p = Path(path)
    if not p.exists():
        from train import train_weather_models
        train_weather_models()
    return joblib.load(p)


def build_features(user_date: date, mintemp: float, rainfall: float, windspeed: float, raintoday: str) -> pd.DataFrame:
    # Simple preprocessing matching train.py
    month = int(user_date.month)
    day = int(user_date.day)
    rain_today_val = 1 if raintoday == "Yes" else 0

    return pd.DataFrame(
        [
            {
                "MinTemp": mintemp,
                "Rainfall": rainfall,
                "WindSpeed": windspeed,
                "RainToday": rain_today_val,
                "Month": month,
                "Day": day,
            }
        ]
    )


def confidence_level(confidence_pct: float) -> str:
    if confidence_pct >= 80:
        return "High"
    if confidence_pct >= 60:
        return "Medium"
    return "Low"


def insight_message(rain_tomorrow: str, prob_rain: float, rainfall: float, windspeed: float) -> str:
    """
    Simple, human-readable interpretation for beginners.
    """
    pr = prob_rain
    if rain_tomorrow == "Yes":
        if pr >= 0.75:
            return "High chance of rain tomorrow. Consider carrying an umbrella."
        if pr >= 0.55:
            return "Rain is likely tomorrow. Keep an eye on the forecast."
        return "Some chance of rain tomorrow."
    # No rain predicted
    if pr <= 0.25:
        return "Conditions look relatively stable for the next day."
    # Edge case: model predicts No but probability not tiny
    msg = "Mostly dry expected, but there is still a noticeable rain risk."
    if rainfall > 5:
        msg += " Recent rainfall increases uncertainty."
    if windspeed > 30:
        msg += " Strong winds can also increase variability."
    return msg


def main() -> None:
    st.set_page_config(page_title="Weather Prediction App", page_icon="⛅", layout="centered")
    apply_custom_css()

    st.markdown(
        """
        <div class="title-card">
          <div class="weather-title">Weather Prediction App</div>
          <div class="weather-subtitle">Predict tomorrow’s rain probability and expected temperature using trained ML models.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Load saved models
    temp_bundle = load_model_bundle("temp_model.pkl")
    rain_bundle = load_model_bundle("rain_model.pkl")

    temp_model = temp_bundle["model"]
    temp_feature_cols = temp_bundle["feature_cols"]
    temp_target = temp_bundle.get("temp_target", "Temperature")

    rain_model = rain_bundle["model"]
    rain_feature_cols = rain_bundle["feature_cols"]

    # --- Sample presets like the screenshot ---
    presets = {
        "Dry Summer Day": {"MinTemp": 20.0, "Rainfall": 0.0, "WindSpeed": 14.0, "RainToday": "No"},
        "Storm Risk Evening": {"MinTemp": 18.0, "Rainfall": 12.0, "WindSpeed": 34.0, "RainToday": "Yes"},
        "Mild Cloudy Morning": {"MinTemp": 16.0, "Rainfall": 2.0, "WindSpeed": 18.0, "RainToday": "No"},
    }

    if "form_values" not in st.session_state:
        st.session_state.form_values = {
            "Date": date.today(),
            "MinTemp": 20.0,
            "Rainfall": 7.0,
            "WindSpeed": 15.0,
            "RainToday": "No",
        }

    left, right = st.columns([1.25, 1.0], gap="large")

    with left:
        st.markdown('<div class="card"><div class="card-title">Weather Input Form</div>', unsafe_allow_html=True)

        with st.form("weather_form", clear_on_submit=False):
            user_date = st.date_input("Date", value=st.session_state.form_values["Date"])
            mintemp = st.number_input("MinTemp (°C)", value=float(st.session_state.form_values["MinTemp"]), step=0.5)
            rainfall = st.number_input(
                "Rainfall (mm)", value=float(st.session_state.form_values["Rainfall"]), step=0.5, min_value=0.0
            )
            windspeed = st.number_input(
                "Wind Speed (km/h)", value=float(st.session_state.form_values["WindSpeed"]), step=1.0, min_value=0.0
            )
            raintoday = st.selectbox(
                "RainToday",
                options=["No", "Yes"],
                index=0 if st.session_state.form_values["RainToday"] == "No" else 1,
            )

            submitted = st.form_submit_button("Predict")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="card-title">Sample Input Reference</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Click a scenario to auto-fill the form.</div>', unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        if b1.button("Dry Summer Day"):
            st.session_state.form_values.update({"Date": date.today(), **presets["Dry Summer Day"]})
            st.rerun()
        if b2.button("Storm Risk Evening"):
            st.session_state.form_values.update({"Date": date.today(), **presets["Storm Risk Evening"]})
            st.rerun()
        if b3.button("Mild Cloudy Morning"):
            st.session_state.form_values.update({"Date": date.today(), **presets["Mild Cloudy Morning"]})
            st.rerun()

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="muted"><b>Expected feature order</b></div>
            <div class="muted">{", ".join(temp_feature_cols)}</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        X = build_features(user_date, mintemp, rainfall, windspeed, raintoday)

        # Ensure correct column order
        X_temp = X[temp_feature_cols]
        X_rain = X[rain_feature_cols]

        # Predictions
        pred_temp = float(temp_model.predict(X_temp)[0])

        # Rain prediction + probability (if available)
        rain_pred_int = int(rain_model.predict(X_rain)[0])
        rain_pred_label = "Yes" if rain_pred_int == 1 else "No"

        proba_yes = None
        if hasattr(rain_model, "predict_proba"):
            proba_yes = float(rain_model.predict_proba(X_rain)[0][1])
        else:
            proba_yes = 0.5

        prob_rain_pct = proba_yes * 100.0
        confidence_pct = max(proba_yes, 1.0 - proba_yes) * 100.0
        level = confidence_level(confidence_pct)
        message = insight_message(rain_pred_label, proba_yes, rainfall, windspeed)

        # ---- Prediction Result (like the screenshot) ----
        st.markdown('<div class="result-wrap"><div class="result-title">Prediction Result</div>', unsafe_allow_html=True)

        banner_class = "banner danger" if rain_pred_label == "Yes" else "banner"
        st.markdown(
            f"""
            <div class="{banner_class}">
              <div class="topline">RainTomorrow: {rain_pred_label}</div>
              <div class="subline">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="metric-grid">
              <div class="metric-card">
                <div class="metric-label">Predicted Temperature</div>
                <div class="metric-value">{pred_temp:.2f} °C</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Probability of Rain</div>
                <div class="metric-value">{prob_rain_pct:.2f}%</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence_pct:.2f}%</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<div class='muted'><b>Confidence Level:</b> {level}</div>", unsafe_allow_html=True)
        st.progress(int(round(confidence_pct)))
        st.markdown(
            f"<div class='muted'>Probability breakdown → Rain: {prob_rain_pct:.2f}% | No Rain: {(100.0 - prob_rain_pct):.2f}%</div>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("What inputs were used?"):
            st.write(
                {
                    "Date": str(user_date),
                    "Month": int(user_date.month),
                    "Day": int(user_date.day),
                    "MinTemp": mintemp,
                    "Rainfall": rainfall,
                    "WindSpeed": windspeed,
                    "RainToday": raintoday,
                    "Temperature target (trained on)": temp_target,
                }
            )


if __name__ == "__main__":
    main()
