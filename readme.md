# 🏥 Healthcare AI System

> **Production First Architecture. Not Slideware.** — #ArchitectMindset

An end-to-end enterprise ML system built on real hospital data — from raw CSVs to AWS Kubernetes deployment, with full MLOps tooling, monitoring, and governance.

---

## 🎯 What This System Does

| Model | Input | Prediction | Business Value |
|---|---|---|---|
| **Visit Risk Classifier** | Patient + Visit data | Low / Medium / High risk | Helps hospital ops teams triage and allocate staff proactively |
| **Claim Outcome Predictor** | Billing + Visit data | Paid / Pending / Rejected | Helps finance teams detect rejection-prone claims before submission |

---

## 🏗️ System Architecture

```
Raw Hospital Data
(patients.csv · visits.csv · billing.csv)
        │
        ▼
  SQL Analytics Layer
     (SQLite · hospital.db)
        │
        ▼
  Exploratory Data Analysis
  (distributions · outliers · correlations)
        │
        ▼
  Feature Engineering
  (visit_frequency · avg_los · provider_rejection_rate)
        │
        ▼
  ML Models (Scikit-learn · XGBoost)
  ├── Model A — Visit Risk Classification
  └── Model B — Claim Outcome Prediction
        │
        ▼
  MLflow Experiment Tracking
  (params · metrics · model registry)
        │
        ▼
  DVC Data Versioning
  (model_table.csv · model artifacts → AWS S3)
        │
        ▼
  FastAPI Prediction Service
  ├── /health
  ├── /predict/risk
  └── /predict/claim
        │
        ▼
  Docker → AWS ECR
        │
        ▼
  AWS EKS (Kubernetes)
  (HPA · rolling updates · zero downtime)
        │
        ▼
  Monitoring & Governance
  (PSI drift detection · Model Card · Retraining Plan)
```

---

## 📁 Project Structure

```
Healthcare/
├── data/                    # Raw CSV files — never modified
│   ├── patients.csv         # 5,000 patients
│   ├── visits.csv           # 25,000 hospital visits
│   └── billing.csv          # 25,000 billing records
│
├── db/                      # SQLite database
│   └── hospital.db
│
├── notebooks/               # Phase-wise Jupyter exploration
│   ├── Phase1_SQL.ipynb     # SQL analytics layer
│   ├── Phase2_EDA.ipynb     # Exploratory data analysis
│   ├── Phase3_Modeling.ipynb # ML model development
│   └── Phase4_Evaluation.ipynb # Model evaluation & explainability
│
├── src/                     # Production Python scripts
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train_risk_model.py
│   └── train_claim_model.py
│
├── api/                     # FastAPI prediction service
│   ├── main.py
│   ├── routers/
│   │   ├── risk.py
│   │   └── claim.py
│   ├── schemas/
│   │   ├── risk_schema.py
│   │   └── claim_schema.py
│   └── prediction_logger.py
│
├── models/                  # Saved model artifacts
│   ├── risk_model.joblib
│   ├── claim_model.joblib
│   └── feature_schema.json
│
├── outputs/                 # Generated files & plots
│   ├── model_table.csv
│   └── eda_plots/
│
├── report/                  # Governance documentation
│   ├── model_card.md
│   └── monitoring_strategy.md
│
├── tests/                   # Pytest test suite
│   ├── test_api.py
│   └── test_features.py
│
├── Dockerfile               # Added in Docker section
├── docker-compose.yml       # Added in Docker section
├── .github/workflows/       # Added in CI/CD section
│   └── ci_cd.yml
├── requirements.txt
└── README.md
```

---

## 🗄️ Dataset Overview

### patients.csv — 5,000 rows

| Column | Type | Description |
|---|---|---|
| patient_id | int | Primary key |
| age | int | Patient age (1–90) |
| gender | str | M / F |
| city | str | Hyderabad, Pune, Chennai, Bangalore, Mumbai, Delhi |
| insurance_provider | str | SecureLife, HealthPlus, CareOne, MediCareX |
| chronic_flag | int | 1 = has chronic condition, 0 = none |
| registration_date | date | First registration at hospital |

### visits.csv — 25,000 rows

| Column | Type | Description |
|---|---|---|
| visit_id | int | Primary key |
| patient_id | int | Foreign key → patients |
| visit_date | date | Date of visit |
| department | str | Cardiology, Orthopedics, ICU, General, ER, Neurology |
| visit_type | str | ER, OPD, ICU |
| length_of_stay_hours | float | Duration of admission |
| **risk_score** | str | **Target A** — Low / Medium / High |
| doctor_id | int | Attending doctor (100–200) |

### billing.csv — 25,000 rows

| Column | Type | Description |
|---|---|---|
| bill_id | int | Primary key |
| visit_id | int | Foreign key → visits |
| billed_amount | float | Amount charged by hospital |
| approved_amount | float | Amount approved by insurer (nullable) |
| **claim_status** | str | **Target B** — Paid / Pending / Rejected |
| payment_days | float | Days to payment (nullable) |
| billing_date | date | Date bill was raised |

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Data & ML | Python · Pandas · NumPy · Scikit-learn · XGBoost |
| Experiment Tracking | MLflow |
| Data Versioning | DVC · AWS S3 |
| Database | SQLite · SQLAlchemy |
| API | FastAPI · Uvicorn · Pydantic |
| Monitoring | Evidently (PSI drift detection) |
| Containerisation | Docker |
| Cloud | AWS ECR · AWS EKS |
| CI/CD | GitHub Actions |
| Testing | Pytest · httpx |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [uv](https://astral.sh/uv) — fast Python package manager

### Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd Healthcare

# 2. Create virtual environment
uv venv

# 3. Activate (Windows Git Bash)
source .venv/Scripts/activate

# 4. Activate (Mac / Linux)
source .venv/bin/activate

# 5. Install all dependencies
uv pip install -r requirements.txt

# 6. Launch notebooks
jupyter notebook
```

### Run Phase by Phase

```bash
# Phase 1 — SQL Analytics
jupyter notebook notebooks/Phase1_SQL.ipynb

# Phase 2 — EDA
jupyter notebook notebooks/Phase2_EDA.ipynb

# Phase 3 — ML Modeling
jupyter notebook notebooks/Phase3_Modeling.ipynb

# Phase 4 — Evaluation
jupyter notebook notebooks/Phase4_Mlflow.ipynb
```
### Run the ML Flow
```bash
mlflow ui
```
### Run the API

```bash
uvicorn api.main:app --reload
```

API docs available at: `http://localhost:8000/docs`

---

## 📊 Model Performance

| Model | Algorithm | Test Accuracy | Weighted F1 |
|---|---|---|---|
| Visit Risk | Logistic Regression (baseline) | ~63% | 0.61 |
| Visit Risk | Random Forest | ~78% | 0.76 |
| Visit Risk | **XGBoost (final)** | **~83%** | **0.81** |
| Claim Outcome | Logistic Regression (baseline) | ~61% | 0.59 |
| Claim Outcome | **Random Forest (final)** | **~76%** | **0.74** |

> ⚠️ **Note:** This project intentionally demonstrates two data scenarios — random synthetic labels (Phase 3A) and clinically-derived labels (Phase 3B). The above numbers reflect Phase 3B (good data). This is a core teaching point of the course.

---

## 🔍 Key Teaching Points

- **Label quality over model tuning** — same pipeline, 45% → 83% accuracy by fixing the data, not the model
- **Time-based train/test split** — leakage-safe evaluation for temporal data
- **Class imbalance handling** — class_weight, balanced_subsample, SMOTE
- **Bias-variance tradeoff** — live demo of RF overfitting (97% train vs 43% test) and fix
- **Fairness analysis** — model performance broken down by gender, city, insurance provider
- **Production API design** — prediction logging, input validation, model versioning
- **Drift detection** — PSI-based early warning for model degradation

---

## 🏛️ Governance

- **Model Card** — `report/model_card.md`
- **Monitoring Strategy** — `report/monitoring_strategy.md`
- **Retraining Plan** — PSI threshold 0.2 triggers retraining pipeline

---

## 👨‍💻 Author

**Rahul Sahay**
Principal Architect · Datamatics
7× Microsoft MVP · IIT Madras AI/ML
Udemy Instructor · 47K+ Students

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rahul_Sahay-blue)](https://linkedin.com/in/rahulsahay19)
[![Udemy](https://img.shields.io/badge/Udemy-Courses-orange)](https://www.udemy.com/user/rahulsahay-2)

> *Production First Architecture. Not Slideware.* — **#ArchitectMindset**

---

## 📄 License

This project is for educational purposes as part of the Udemy course
**"AI System Design & MLOps: From Raw Data to AWS Kubernetes"**