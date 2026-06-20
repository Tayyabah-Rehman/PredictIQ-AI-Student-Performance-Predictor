# PredictIQ — AI Student Performance Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=for-the-badge&logo=chartdotjs&logoColor=white)

![Accuracy](https://img.shields.io/badge/Accuracy-94.2%25-brightgreen?style=for-the-badge)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.988-brightgreen?style=for-the-badge)
![Models](https://img.shields.io/badge/Ensemble-4%20Models-blueviolet?style=for-the-badge)
![SMOTE](https://img.shields.io/badge/Balanced-SMOTE-blue?style=for-the-badge)

<br/>

> A full-stack AI web application that predicts student academic performance with **94.2% accuracy** using a 4-model ensemble (XGBoost + Random Forest + Gradient Boosting + Extra Trees), SMOTE class balancing, and a live analytics dashboard.

<br/>

![Predictor Interface](https://raw.githubusercontent.com/Tayyabah-Rehman/PredictIQ-AI-Student-Performance-Predictor/main/Student%20Academic%20Performance%20Interface.png)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Analytics Dashboard](#analytics-dashboard)
- [Results](#results)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

**PredictIQ** takes 6 behavioral and academic inputs from a student and instantly returns:

- Predicted grade category (Excellent / Good / Average / Needs Improvement)
- Confidence score and probability distribution across all 4 classes
- Personalized, data-driven improvement tips based on the student's specific values

The system is built as a production-ready Flask web app with a REST API backend, interactive dark-theme UI, and a dedicated analytics dashboard powered by Matplotlib, Seaborn, and Chart.js.

---

## Features

- **4-Model Soft-Voting Ensemble** — XGBoost (40%) + Gradient Boosting (30%) + Random Forest (15%) + Extra Trees (15%)
- **Auto-Optimized Weights** — Grid search over hundreds of weight combinations to find the best ensemble
- **SMOTE Class Balancing** — Prevents bias toward the majority class (Excellent ~70% of raw data)
- **12 Features** — 6 raw inputs + 6 engineered (interaction terms, ratios, composite score)
- **REST API** — JSON prediction endpoint with full input validation
- **Analytics Dashboard** — 5 Matplotlib/Seaborn charts + 3 live Chart.js charts + 6 KPI cards
- **Personalized Insights** — Custom improvement tips generated per student
- **Confidence Labeling** — Very High / High / Moderate / Low per prediction
- **Quick Demo Examples** — Excellent, Average, At-Risk student presets

---

## Model Architecture

```
Input (6 features)
       │
       ▼
Feature Engineering  ──►  12 features total
       │
       ▼
StandardScaler (fitted on SMOTE training data)
       │
   ┌───┴──────────────────────────────────────────┐
   │            │               │                 │
   ▼            ▼               ▼                 ▼
XGBoost    Random Forest  Gradient Boosting  Extra Trees
(~40%)       (~15%)           (~30%)           (~15%)
   │            │               │                 │
   └────────────┴───────────────┴─────────────────┘
                              │
                    Soft Voting (weighted avg probabilities)
                              │
                              ▼
                    Grade Category + Confidence
```

### Model Hyperparameters

| Model | Estimators | Key Parameters | Individual Accuracy |
|---|---|---|---|
| XGBoost | 300 | lr=0.05, max_depth=6, subsample=0.8 | 93.7% |
| Gradient Boosting | 200 | lr=0.05, max_depth=5, subsample=0.8 | 93.3% |
| Random Forest | 300 | max_depth=14, min_leaf=2, max_features=sqrt | 92.9% |
| Extra Trees | 300 | max_depth=13, min_leaf=2, max_features=sqrt | 92.1% |
| **Ensemble** | — | Optimized soft voting | **94.2%** |

---

## Dataset

| Property | Value |
|---|---|
| Type | Synthetic — generated with realistic feature correlations |
| Size | 8,000 records |
| Raw Features | 6 |
| Engineered Features | 6 (total 12) |
| Classes | 4 (Excellent, Good, Average, Needs Improvement) |
| Balancing | SMOTE applied to training split only |
| Train / Test Split | 85% / 15% stratified |

### Input Features

| Feature | Range | Description |
|---|---|---|
| `study_hours` | 0 – 12 hrs/day | Average daily study time |
| `attendance` | 0 – 100% | Class attendance rate |
| `assignments` | 0 – 10 | Assignments submitted |
| `previous_marks` | 0 – 100 | Prior exam score |
| `sleep_hours` | 4 – 10 hrs/day | Daily sleep hours |
| `extracurricular` | 0 – 5 | Number of activities |

### Engineered Features (Internal)

| Feature | Formula |
|---|---|
| `study_attend_ratio` | `study_hours / (attendance/10)` |
| `assignment_rate` | `assignments / 10.0` |
| `marks_study_ix` | `previous_marks × study_hours / 100` |
| `attend_assign_ix` | `attendance × assignments / 100` |
| `sleep_study_ratio` | `sleep_hours / study_hours` |
| `overall_effort_score` | Weighted composite of all raw inputs |

### Class Distribution (Before SMOTE)

| Grade | Count | % |
|---|---|---|
| Excellent | ~5,600 | 70% |
| Good | ~1,440 | 18% |
| Average | ~720 | 9% |
| Needs Improvement | ~240 | 3% |

> **Why SMOTE?** Without balancing, models predict "Excellent" for everything and achieve ~70% accuracy by default — useless in practice. SMOTE oversamples minority classes to equal representation, producing ~27K balanced training samples.

---

## Project Structure

```
StudentPerformancePredictor/
│
├── app.py                        # Flask app — all routes + prediction logic
├── dashboard.py                  # Generates Matplotlib/Seaborn charts (run once)
├── requirements.txt              # All dependencies with pinned versions
├── setup.bat                     # Windows one-click setup script
├── README.md
├── .gitignore
│
├── model/
│   ├── train_model.py            # Full training pipeline
│   ├── rf_model.pkl              # Random Forest (generated)
│   ├── xgb_model.pkl             # XGBoost (generated)
│   ├── gb_model.pkl              # Gradient Boosting (generated)
│   ├── et_model.pkl              # Extra Trees (generated)
│   ├── scaler.pkl                # StandardScaler (generated)
│   ├── label_encoder.pkl         # LabelEncoder (generated)
│   ├── model_meta.json           # Metrics + weights + feature importances
│   └── dataset_stats.json        # Dashboard KPIs + correlation matrix
│
├── templates/
│   ├── index.html                # Predictor page
│   └── dashboard.html            # Analytics dashboard page
│
├── static/
│   ├── css/
│   │   ├── style.css             # Main dark theme stylesheet
│   │   └── dashboard.css         # Dashboard-specific styles
│   ├── js/
│   │   └── main.js               # All frontend logic
│   └── img/
│       ├── chart_attendance.png  # Generated by dashboard.py
│       ├── chart_marks.png
│       ├── chart_heatmap.png
│       ├── chart_categories.png
│       └── chart_scatter.png
```

---

## Installation

### Requirements

- Python 3.10
- pip
- Windows / Linux / macOS

### Step-by-Step

```bash
# 1. Clone the repository
git clone https://github.com/Tayyabah-Rehman/PredictIQ-AI-Student-Performance-Predictor.git
cd PredictIQ-AI-Student-Performance-Predictor

# 2. Create virtual environment (Python 3.10)
py -3.10 -m venv .venv          # Windows
# python3.10 -m venv .venv      # Linux/macOS

# 3. Activate virtual environment
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 4. Verify Python version
python --version                 # Must show 3.10.x

# 5. Install dependencies
pip install -r requirements.txt

# 6. Train the ML model  (~60 seconds)
python model/train_model.py

# 7. Generate dashboard charts  (~10 seconds)
python dashboard.py

# 8. Start the Flask server
python app.py
```

### Windows One-Click Setup

```cmd
setup.bat
```

Handles steps 2–7 automatically.

---

## Usage

After running `python app.py`, open your browser:

| URL | Page |
|---|---|
| `http://localhost:5000` | AI Predictor |
| `http://localhost:5000/dashboard` | Analytics Dashboard |

### Making a Prediction

1. Adjust the 6 sliders (or type values in the number fields)
2. Click **Predict Performance**
3. View grade, confidence score, probability chart, and personalized insights
4. Try the quick demo buttons: Excellent / Average / At Risk

---

## API Reference

### `POST /api/predict`

**Request:**
```json
{
  "study_hours": 7.5,
  "attendance": 90,
  "assignments": 9,
  "previous_marks": 78,
  "sleep_hours": 7.5,
  "extracurricular": 3
}
```

**Response:**
```json
{
  "grade": "Excellent",
  "confidence": 97.3,
  "confidence_lvl": "Very High",
  "class_probs": {
    "Average": 0.4,
    "Excellent": 97.3,
    "Good": 2.2,
    "Needs Improvement": 0.1
  },
  "insights": [
    {
      "icon": "✅",
      "title": "Strong Study Habit",
      "detail": "7.5 hrs/day puts you in the top academic tier."
    }
  ]
}
```

### `GET /api/model-info`
Returns model metrics, feature importances, and class distribution.

### `GET /api/dashboard-stats`
Returns all KPI values and the 6×6 Pearson correlation matrix.

### `GET /api/health`
```json
{ "status": "ok", "model": "loaded", "version": "2.0.0" }
```

---

## Analytics Dashboard

Visit `http://localhost:5000/dashboard` for:

**Dataset Insights (KPI Cards)**
- Total Records · Average Marks · Average Attendance · Avg Study Hours · Avg Sleep · Model Accuracy

**Chart.js Live Charts**
- Grade distribution donut
- Average marks by grade
- Average study hours by grade

**Matplotlib + Seaborn Charts**
- Attendance distribution by grade (histogram + KDE)
- Previous marks violin + box plot
- Feature correlation heatmap (Pearson r)
- Performance category donut + horizontal bar
- Study hours vs marks scatter (8,000 points coloured by grade)

**Correlation Table**
- Colour-coded 6×6 Pearson matrix loaded from API

---

## Results

| Metric | Score |
|---|---|
| **Ensemble Accuracy** | **94.2%** |
| Weighted F1 Score | 94.2% |
| ROC-AUC (OvR) | 0.988 |
| 5-Fold CV (RF) | ~93% |

### Per-Class Performance

| Grade | Precision | Recall | F1 |
|---|---|---|---|
| Excellent | ~0.96 | ~0.97 | ~0.96 |
| Good | ~0.90 | ~0.88 | ~0.89 |
| Average | ~0.88 | ~0.86 | ~0.87 |
| Needs Improvement | ~0.92 | ~0.90 | ~0.91 |

---

## Screenshots

### Analytics Dashboard — `http://localhost:5000/dashboard`

![Analytics Dashboard](https://raw.githubusercontent.com/Tayyabah-Rehman/PredictIQ-AI-Student-Performance-Predictor/main/Student%20Academic%20Performance%20Dashboard.png)

---

## Future Improvements

- [ ] Replace synthetic data with a real Kaggle/UCI student dataset
- [ ] Add SHAP explainability charts per prediction
- [ ] Student login + prediction history tracking (SQLite)
- [ ] Export prediction report as PDF
- [ ] Deploy to Render / Railway for a live public URL
- [ ] Hyperparameter tuning with Optuna
- [ ] API key authentication for external access
- [ ] Teacher/admin dashboard with class-wide analytics
- [ ] Urdu language support
- [ ] PostgreSQL backend for production use

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, Flask 3.0 |
| ML | XGBoost, scikit-learn (RF, GB, ET), imbalanced-learn (SMOTE) |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn, Chart.js 4.4 |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Design | Dark theme · charcoal/slate/sky-blue palette · Inter + JetBrains Mono |

---

## Author

**Tayyabah Rehman**

[![GitHub](https://img.shields.io/badge/GitHub-Tayyabah--Rehman-181717?style=flat-square&logo=github)](https://github.com/Tayyabah-Rehman)
[![Email](https://img.shields.io/badge/Email-tayyabahrehman789%40gmail.com-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:tayyabahrehman789@gmail.com)

---

<div align="center">
  <sub>If this project helped you, consider giving it a ⭐</sub>
</div>
