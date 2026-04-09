"# E2E ML Project

This repository contains an end-to-end machine learning pipeline for a student performance prediction task.
The project demonstrates data ingestion, preprocessing, and model training scaffolding using Python, scikit-learn, and related libraries.

## Project Overview

The goal of this project is to build a ML workflow that:
- loads the raw student dataset from `notebook/data/student.csv`
- splits the dataset into train/test files under `artifacts/`
- constructs preprocessing pipelines for numeric, nominal, and ordinal features
- saves preprocessing artifacts for downstream use

## Repository Structure

- `artifacts/`
  - `raw_data.csv`, `train.csv`, `test.csv` generated during ingestion
- `notebook/`
  - exploratory notebooks and raw dataset location
- `src/`
  - `components/`
    - `data_ingestion.py` — ingestion and train/test split logic
    - `data_transformation.py` — preprocessing pipeline creation and transformation logic
    - `model_trainer.py` — model training scaffolding (to be implemented)
  - `pipeline/`
    - `train_pipeline.py` — pipeline entrypoint scaffold
    - `predict_pipeline.py` — prediction pipeline scaffold
  - `utils.py` — utility helpers for persistence and object serialization
  - `exception.py` — custom exception handling
  - `logger.py` — logging utilities

## Installation

1. Create and activate a Python virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Install the package in editable mode

```bash
python -m pip install -e .
```

## Usage

### Run data ingestion

```python
from src.components.data_ingestion import DataIngestion

ingestion = DataIngestion(test_size=0.2, random_state=42)
train_path, test_path = ingestion.initiate_data_ingestion()
print(train_path, test_path)
```

### Build and apply the preprocessing pipeline

```python
from src.components.data_transformation import DataTransformation

transformation = DataTransformation()
X_train, y_train, X_test, y_test, preprocessor = transformation.initiate_data_transformation()
```

### Notes

- The raw student dataset is expected at `notebook/data/student.csv`.
- The generated `train.csv` and `test.csv` files are saved under `artifacts/`.
- The preprocessing pipeline is saved to `artifacts/preprocessor.pkl`.
- `src/components/model_trainer.py` and `src/pipeline/predict_pipeline.py` are currently scaffolded and can be extended with model training and inference logic.

## Dependencies

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- catboost
- xgboost

## Development

- Use `black` for code formatting.
- Extend the pipeline components with training, model evaluation, and prediction support.

## Author

Aishwarya Vadlamudi
" 
