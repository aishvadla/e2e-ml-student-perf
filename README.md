# End-to-End Student Performance Predictor 🎓

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
![License](https://img.shields.io/badge/License-Educational-green)

A production-style, end-to-end machine learning system that predicts student exam performance based on demographic and academic background features. Built with a modular pipeline architecture covering data ingestion, preprocessing, model training, evaluation, and real-time deployment via a Flask web app.

> 📸 _Screenshot coming soon — add one of your Flask UI here for maximum impact!_

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Model Results](#model-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Docker Support](#docker-support)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

<a id="overview"></a>
## Overview

This project demonstrates a real-world ML system design with a focus on **modularity**, **scalability**, and **reproducibility**. Each stage of the pipeline (ingestion → validation → transformation → training → evaluation → deployment) is isolated into its own class, making components independently testable and replaceable.

---

<a id="dataset"></a>
## Dataset

**Source:** [Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

**Target variable:** Math score (continuous — regression problem)

**Features used:**

| Feature | Type | Description |
|---|---|---|
| Gender | Categorical | Student's gender |
| Race/Ethnicity | Categorical | Grouped ethnicity category |
| Parental Education | Categorical | Highest level of parental education |
| Lunch | Categorical | Standard vs. free/reduced lunch |
| Test Prep Course | Categorical | Whether the student completed a prep course |
| Reading Score | Numerical | Score out of 100 |
| Writing Score | Numerical | Score out of 100 |

---

<a id="ml-pipeline-architecture"></a>
## 🧠 ML Pipeline Architecture

```text
Raw Data (CSV)
      │
      ▼
Data Ingestion          ← Loads and splits into train/test sets
      │
      ▼
Data Validation         ← Schema checks, null detection
      │
      ▼
Data Transformation     ← Encoding, scaling via Scikit-Learn pipelines
      │
      ▼
Model Training          ← Trains & compares multiple regression models
      │
      ▼
Model Evaluation        ← Selects best model by R² score, saves artifact
      │
      ▼
Flask Web App           ← Serves real-time predictions from user input
```

---

<a id="model-results"></a>
## 📊 Model Results

Multiple regression algorithms were evaluated and compared on the held-out test set. Below are the actual R² scores achieved:

| Model | R² Score |
|---|---|
| **Ridge Regression** ✅ | **0.88** |
| **Linear Regression** ✅ | **0.88** |
| Gradient Boosting | 0.87 |
| AdaBoost | 0.85 |
| CatBoost | 0.85 |
| Random Forest | 0.85 |
| XGBoost | 0.85 |
| Lasso | 0.82 |
| KNN | 0.77 |
| SVR | 0.74 |
| Decision Tree | 0.68 |

**Best Performers:** Ridge and Linear Regression achieved virtually identical performance (R² = 0.8816), with the pipeline automatically selecting Ridge for deployment due to superior regularization properties. The results demonstrate that simpler models often outperform complex ensemble methods when features are properly engineered.

---

<a id="tech-stack"></a>
## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML / Modeling | Scikit-Learn, XGBoost, CatBoost |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Environment | Conda / pip + virtualenv |

---

<a id="project-structure"></a>
## 📂 Project Structure

```
e2e-ml-student-perf/
│
├── src/
│   ├── components/         # Ingestion, transformation, and training modules
│   ├── pipeline/           # Training pipeline & prediction pipeline
│   └── utils/              # Helper functions (model saving, evaluation, etc.)
│
├── artifacts/              # Saved model (.pkl) and processed datasets
├── notebooks/              # EDA and experimentation notebooks
├── templates/              # Flask HTML templates
│
├── app.py                  # Flask app — routes for UI and prediction
├── requirements.txt
└── README.md
```

---

<a id="installation--setup"></a>
## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/aishvadla/e2e-ml-student-perf.git
cd e2e-ml-student-perf
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

<a id="docker-support"></a>
## 🐳 Docker Support

This project includes Docker support for easy containerization and deployment. A pre-built Docker image is available on Docker Hub.

### Prerequisites

- Docker installed on your system

### Option A — Pull the Pre-built Image from Docker Hub

```bash
docker pull avadlamu/ml-app-student-perf:latest
```

### Run the Application with Docker

```bash
docker run -p 8080:8080 avadlamu/ml-app-student-perf:latest
```

### Option B — Build the Image Locally (Alternative)

If you prefer to build the image yourself:

```bash
docker build -t student-performance-predictor .
docker run -p 8080:8080 student-performance-predictor
```

The application will be available at `http://localhost:8080/`.

---

<a id="testing--quality-assurance"></a>
## 🧪 Testing & Quality Assurance

To ensure the reliability of the ML pipeline, this project maintains a high standard of automated testing.

- **Framework:** `pytest`
- **Coverage:** **88%** (Unit & Integration tests)
- **Components Tested:**
  - Data Ingestion (Train-Test Split logic)
  - Data Transformation (ColumnTransformer & Scaling)
  - Model Trainer (Pickle serialization & Evaluation)
  - Prediction Pipeline (Inference logic)

### Running Tests
To execute the test suite and verify the environment locally:
```bash
# Run all tests
python -m pytest tests/

# Run with coverage report
python -m pytest --cov=src tests/
```

---

<a id="usage"></a>
## 🖥️ Usage

### Option A — Use pre-trained artifacts (quick start)

The `artifacts/` folder in this repo contains a pre-trained model. You can skip training and go straight to running the app:

```bash
python application.py
```

### Option B — Retrain from scratch

```bash
# Step 1: Run the full training pipeline
python src/pipeline/train_pipeline.py

# Step 2: Start the web app
python application.py
```

Once the app is running, open your browser and navigate to:

```
http://localhost:8080/
```

Enter the student's attributes (gender, parental education, test prep course, etc.) and receive a real-time predicted math score.

---

<a id="key-design-decisions"></a>
## 💡 Key Design Decisions

**Modular pipeline components** — Each stage (ingestion, transformation, training) is a separate class. This means you can swap out the data source or add a new model without touching unrelated code.

**Scikit-Learn pipelines for preprocessing** — Categorical encoding and numerical scaling are wrapped in a single `ColumnTransformer` pipeline, preventing data leakage between train and test sets.

**Automated model selection** — All models are trained and evaluated in a single loop. The best-performing model is saved automatically to `artifacts/`, so the deployment always uses the current champion.

**Separation of training and inference** — The training pipeline (`train_pipeline.py`) and prediction pipeline (`predict_pipeline.py`) are fully decoupled. The Flask app only calls the prediction pipeline and never re-trains on user input.

---

<a id="future-improvements"></a>
## 🚀 Future Improvements

- [x] Add Docker support for containerization
- [x] Add unit tests, integration tests and validation tests (88% test coverage achieved)
- [ ] Deploy to cloud (AWS Elastic Beanstalk / GCP App Engine)
- [ ] Add CI/CD pipeline with GitHub Actions
- [ ] Add logging and experiment tracking with MLflow
- [ ] Model Monitoring, Logging and Latency
- [x] Improve Flask UI/UX

---

<a id="author"></a>
## 👤 Author

**Aishwarya Vadlamudi**
[GitHub](https://github.com/aishvadla) · [LinkedIn](https://www.linkedin.com/in/aishwaryavadlamudi/)