# 🌦️ Weather Prediction System

[![CI Pipeline](https://github.com/nandeesh2799/WeatherPrediction0/actions/workflows/ci.yml/badge.svg)](https://github.com/nandeesh2799/WeatherPrediction0/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-7%20Passed-brightgreen?logo=pytest)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready Machine Learning system that analyzes historical meteorological data to predict next-day rainfall likelihood and afternoon temperatures. Built with Scikit-learn regression and classification pipelines, an automated synthetic dataset generator, full test suite, and an interactive dark-themed Streamlit dashboard.

---

## 🚀 Key Features

- **Dual-Model Inference Engine**:
  - **Temperature Regression**: Linear regression with median imputation to predict next-day 3 PM temperatures (`MAE: 1.74°C`, `R²: 0.933`).
  - **Rainfall Classification**: Logistic regression predicting binary rain occurrence and class probabilities.
- **Robust Feature Engineering**: Extracts temporal seasonality (`Month`, `Day`) alongside meteorological indicators (`MinTemp`, `Rainfall`, `WindSpeed`, `RainToday`).
- **Resilient Self-Bootstrapping**: If serialized model artifacts (`.pkl`) are missing on initial deployment, the application automatically initiates training or generates a balanced synthetic dataset.
- **Containerized Architecture**: Fully dockerized with multi-stage builds and Streamlit healthchecks.
- **Comprehensive Unit Testing**: End-to-end verification covering dataset generation, date parsing, pipeline training, metrics validation, and user interface helpers.

---

## 🧠 Tech Stack

* **Language**: Python 3.12
* **Data Processing**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (`Pipeline`, `SimpleImputer`, `LinearRegression`, `LogisticRegression`)
* **Serialization**: Joblib
* **Interactive Dashboard**: Streamlit
* **Containerization**: Docker, Docker Compose
* **CI/CD**: GitHub Actions

---

## 📂 Project Structure

```text
WeatherPrediction0/
├── app.py                      # Interactive Streamlit dashboard
├── train.py                    # Modular training pipeline & synthetic data generator
├── weather.csv                 # Meteorological training dataset
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration with healthchecks
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions automated test & build pipeline
└── tests/
    └── test_weather_models.py  # Comprehensive test suite (7 tests)
```

---

## 🧪 Testing & Verification

Run the test suite with pytest:

```bash
python3 -m pytest tests/ -v
```

All 7 test cases validate data generation, feature imputation, model persistence, and prediction bounds.

---

## ⚙️ Installation & Usage

### Option 1: Local Environment

```bash
# Clone the repository
git clone https://github.com/nandeesh2799/WeatherPrediction0.git
cd WeatherPrediction0

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train models (optional, app auto-trains if missing)
python3 train.py --data weather.csv

# Launch Streamlit dashboard
streamlit run app.py
```

### Option 2: Docker Containerization

Run with Docker Compose:

```bash
# Build and run container
docker compose up --build

# Access dashboard at http://localhost:8501
```

---

## 👨‍💻 Author & Engineering Showcase

**Nandeesh N K**  
Full-Stack Developer & AI Systems Engineer — Bengaluru, India  
- 🌐 **Interactive Portfolio:** [NandeeshOS (Live Demo)](https://nandeesh2799-portfolio.vercel.app)
- 💼 **LinkedIn:** [linkedin.com/in/nandeesh-n-k-729773270](https://linkedin.com/in/nandeesh-n-k-729773270)
- 🐙 **GitHub Profile:** [github.com/nandeesh2799](https://github.com/nandeesh2799)
- 📫 **Direct Email:** [nandeeshnk506@gmail.com](mailto:nandeeshnk506@gmail.com)

---

## 📜 License
This project is open-source under the **MIT License**.
