# End-to-End Machine Learning Project 🚀

## Overview

This project demonstrates a production-style end-to-end machine learning pipeline for predicting student performance. It covers the full ML lifecycle—from data ingestion and preprocessing to model training, evaluation, and deployment via a web application.

The project is designed to reflect real-world ML system design, focusing on modularity, scalability, and reproducibility.

---

## 💼 Why This Project Matters

* Demonstrates **industry-relevant ML pipeline design**
* Shows ability to **structure production-ready codebases**
* Includes **model deployment using Flask**
* Highlights understanding of **data preprocessing, feature engineering, and evaluation**

---

## 🛠️ Tech Stack

* Language: Python
* Machine Learning: Scikit-Learn, XGBoost, CatBoost
* Web Framework: Flask
* Data Processing: Pandas, NumPy
* Environment Management: Conda / Pip

---

## 🧠 ML Pipeline Architecture

```text
Raw Data
   │
   ▼
Data Ingestion
   │
   ▼
Data Validation
   │
   ▼
Data Transformation
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Flask Web App (Deployment)
   │
   ▼
User Input → Prediction Output
```

---

## 📂 Project Structure

```
e2e-mlproject/
│
├── src/                # Core ML pipeline components
│   ├── components/     # Data ingestion, transformation, training modules
│   ├── pipeline/       # Training and prediction pipelines
│   └── utils/          # Helper functions
│
├── artifacts/          # Saved models, processed data
├── notebooks/          # Jupyter notebooks for EDA and experimentation
├── templates/          # HTML files (Flask frontend)
│
├── app.py              # Flask app (Training + UI + prediction)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/aishvadla/e2e-mlproject.git
cd e2e-mlproject
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🖥️ Usage

1.  **Run the training pipeline:**
    To train the model and generate the artifacts:

    ```bash
    python src/pipeline/train_pipeline.py
    ```

2.  **Start the Web App:**

    ```bash
    python app.py
    ```

    - Once started, open your browser and navigate to `http://127.0.0.1:5000/predict`.
    - Enter student attributes (e.g., gender, scores, etc.)
    - Receive real-time prediction

## 📊 Results

* Built a regression model to predict student performance
* Achieved strong predictive performance on test data
* Example metrics (may vary by run):

  * R² Score: ~0.85–0.92

---

## 📈 Key Highlights

  - **Modular Codebase:** Developed with a modular architecture for easy maintenance and scalability.
  - **Automated Data Pipeline:** Automated scripts for data preprocessing, feature engineering, and transformation using Scikit-Learn pipelines.
  - **Model Training and Evaluation:** Compares multiple models (Linear Regression, Random Forest, XGBoost, CatBoost, etc.) to select the best performing one based on R2 Score.
  - **Web Interface:** A Flask-based web application to provide real-time predictions based on user input.
---

## 🚀 Future Improvements

* Add Docker support for containerization
* Deploy to cloud (AWS / GCP / Azure)
* Add CI/CD pipeline
* Improve UI/UX of the web app

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

This project is provided as-is for educational purposes.
