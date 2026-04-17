# End-to-End Student Performance Predictor 🎓

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn)
![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://hub.docker.com/r/avadlamu/ml-app-student-perf)
[![CI/CD](https://img.shields.io/badge/CI/CD-GitHub--Actions-2088FF?logo=github-actions)](https://github.com/aishvadla/e2e-ml-student-perf/actions)
[![AWS](https://img.shields.io/badge/AWS-Elastic--Beanstalk-FF9900?logo=amazon-aws)](https://github.com/aishvadla/e2e-ml-student-perf)
[![Coverage](https://img.shields.io/badge/Coverage-93%25-brightgreen)](https://github.com/aishvadla/e2e-ml-student-perf)

A production-grade, end-to-end machine learning system that predicts student exam performance from demographic and academic features. The project covers the full ML lifecycle — data ingestion, validation, preprocessing, model training, automated selection, and real-time inference — deployed as a containerized Flask web application on AWS.

## 🚀 Demo & Deployment

<p align="center">
  <img src="https://github.com/user-attachments/assets/523d91a6-4dd8-461f-b2f3-7a7dec4d7ad3" width="750" alt="Application Demo" />
</p>

---

## 📌 Table of Contents

- [Highlights](#highlights)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Model Results](#model-results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Docker Support](#docker-support)
- [CI/CD Pipeline](#ci-cd-pipeline)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [Key Design Decisions](#key-design-decisions)
- [Roadmap](#roadmap)
- [Author](#author)

---

<a id="highlights"></a>
## ⭐ Highlights

- **93% test coverage** across unit and integration tests using `pytest`
- **11 regression models** benchmarked; best model auto-selected and saved to artifacts
- **Fully containerized** — Docker image published to Docker Hub and deployed on AWS Elastic Beanstalk
- **Automated CI/CD** via GitHub Actions: test → build → push → deploy on every push to `main`
- **Zero data leakage** enforced through Scikit-Learn `ColumnTransformer` pipelines

---

<a id="ml-pipeline-architecture"></a>
## 🧠 ML Pipeline Architecture

Each stage of the pipeline is encapsulated in its own class, making components independently testable and replaceable.

```text
Raw Data (CSV)
      │
      ▼
Data Ingestion          ← Loads data; produces train/test splits
      │
      ▼
Data Validation         ← Schema checks, null detection
      │
      ▼
Data Transformation     ← Categorical encoding + numerical scaling via ColumnTransformer
      │
      ▼
Model Training          ← Trains 11 regression models in a single automated loop
      │
      ▼
Model Evaluation        ← Selects best model by R²; serializes artifact to disk
      │
      ▼
Flask Web App           ← Loads saved model; serves real-time predictions via UI
      │
      ▼
Docker + AWS            ← Containerized deployment; CI/CD managed by GitHub Actions
```

---

<a id="model-results"></a>
## 📊 Model Results

All models were evaluated on a held-out test set. The pipeline automatically selects and deploys the best performer.

| Model | R² Score |
|---|---|
| **Ridge Regression** ✅ *(deployed)* | **0.88** |
| Linear Regression | 0.88 |
| Gradient Boosting | 0.87 |
| AdaBoost | 0.85 |
| CatBoost | 0.85 |
| Random Forest | 0.85 |
| XGBoost | 0.85 |
| Lasso | 0.82 |
| K-Nearest Neighbors | 0.77 |
| Support Vector Regression | 0.74 |
| Decision Tree | 0.68 |

Ridge and Linear Regression achieved identical R² scores (0.8816). Ridge was selected for deployment due to its regularization properties, which improve generalization on unseen data. The results also highlight that well-engineered features can allow simpler models to match or outperform complex ensembles.

For detailed exploratory data analysis, see [`notebook/README.md`](./notebook/README.md).

---

<a id="tech-stack"></a>
## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| ML / Modeling | Scikit-Learn, XGBoost, CatBoost |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Cloud Deployment | AWS Elastic Beanstalk |

---

<a id="project-structure"></a>
## 📂 Project Structure

```
e2e-ml-student-perf/
│
├── src/
│   ├── components/         # Data ingestion, transformation, and model training
│   ├── pipeline/           # Training pipeline and prediction pipeline
│   └── utils/              # Shared helpers (model I/O, evaluation metrics)
│
├── tests/                  # Unit and integration test suite (93% coverage)
├── artifacts/              # Serialized model (.pkl) and processed datasets
├── notebooks/              # EDA and prototyping notebooks
├── templates/              # Flask HTML templates
│
├── app.py                  # Flask application — prediction routes and UI
├── Dockerfile
├── requirements.txt
└── README.md
```

---

<a id="getting-started"></a>
## ⚙️ Getting Started

To directly use pre-built docker images, see [Docker Support](#docker-support)

### 1. Clone the repository

```bash
git clone https://github.com/aishvadla/e2e-ml-student-perf.git
cd e2e-ml-student-perf
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

**Option A — Quick start using pre-trained artifacts**

Pre-trained model artifacts are included in the `artifacts/` directory. Skip training and launch the app directly:

```bash
python application.py
```

**Option B — Retrain from scratch**

```bash
python src/pipeline/train_pipeline.py   # Runs full pipeline; saves new model artifact
python application.py                   # Starts the web app
```

Navigate to `http://localhost:8080/` and enter student attributes to receive a predicted math score.

---

<a id="docker-support"></a>
## 🐳 Docker Support

This project includes Docker support for easy containerization and deployment. A GitHub Actions workflow now builds and pushes the Docker image to Docker Hub automatically on each push to `main`.

### Prerequisites

- Docker installed on your system

### Option A — Pull and run the pre-built image

```bash
docker pull avadlamu/ml-app-student-perf:latest
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

<a id="ci-cd-pipeline"></a>
## 🔁 CI/CD Pipeline

The project utilizes **GitHub Actions** to implement a three-stage automated pipeline:

1. **Continuous Integration (CI)**: 
   - Validates application dependencies.
   - Execution of the full test suite with `Pytest`.
   - Achievement of **93% code coverage**.
2. **Continuous Delivery (CD)**: 
   - Builds a Docker image of the application.
   - Pushes the versioned image to **DockerHub**.
3. **Continuous Deployment (CD)**: 
   - Signals **AWS Elastic Beanstalk** to pull the latest image.
   - Updates the production environment without manual intervention.

The CI/CD workflow ensures every change is validated before reaching production.

---

<a id="testing--quality-assurance"></a>
## 🧪 Testing & Quality Assurance

To ensure the reliability of the ML pipeline, this project maintains a high standard of automated testing.

- **Framework:** `pytest`
- **Coverage:** **93%** (Unit & Integration tests)
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

<a id="key-design-decisions"></a>
## 💡 Key Design Decisions

**Modular pipeline components** — Each stage (ingestion, transformation, training) is a separate class. This means you can swap out the data source or add a new model without touching unrelated code.

**Scikit-Learn pipelines for preprocessing** — Categorical encoding and numerical scaling are wrapped in a single `ColumnTransformer` pipeline, preventing data leakage between train and test sets.

**Automated model selection** — All models are trained and evaluated in a single loop. The best-performing model is saved automatically to `artifacts/`, so the deployment always uses the current champion.

**Separation of training and inference** — The training pipeline (`train_pipeline.py`) and prediction pipeline (`predict_pipeline.py`) are fully decoupled.

---

<a id="roadmap"></a>
## 🚀 Roadmap

- [x] Modular pipeline architecture
- [x] Automated model selection and artifact serialization
- [x] Unit and integration tests (93% coverage)
- [x] Docker containerization
- [x] CI/CD with GitHub Actions
- [x] Cloud deployment on AWS Elastic Beanstalk
- [ ] Experiment tracking with MLflow
- [ ] Model performance monitoring and drift detection
- [ ] Structured logging and latency instrumentation

---

<a id="author"></a>
## 👤 Author

**Aishwarya Vadlamudi**
[GitHub](https://github.com/aishvadla) · [LinkedIn](https://www.linkedin.com/in/aishwaryavadlamudi/)