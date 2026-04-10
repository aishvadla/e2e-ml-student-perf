## End-to-End Machine Learning Project

A comprehensive machine learning pipeline for predicting student math scores. This project demonstrates professional software engineering practices with clear documentation, comprehensive exception handling, and modular component design.

## Project Overview

The goal of this project is to build a complete ML workflow that:
- Loads raw student performance data from `notebook/data/student.csv`
- Splits the dataset into train/test files under `artifacts/`
- Constructs preprocessing pipelines for numeric, nominal, and ordinal features
- Trains and evaluates multiple regression models using hyperparameter tuning
- Provides inference capabilities through a prediction pipeline
- Maintains detailed logging and error tracking throughout

## Target Variable

**math_score** — Student mathematics score (continuous, regression task)

## Features

The model uses the following student features for prediction:
- **gender** — Student gender (categorical)
- **race_ethnicity** — Race/ethnicity classification (categorical)
- **parental_level_of_education** — Parent's education level (ordinal)
- **lunch** — Lunch category (categorical)
- **test_preparation_course** — Test prep status (categorical)
- **reading_score** — Student reading score (numeric)
- **writing_score** — Student writing score (numeric)

## Repository Structure

```
e2e-mlproject/
├── artifacts/
│   ├── raw_data.csv          # Full dataset before splitting
│   ├── train.csv              # Training data
│   ├── test.csv               # Testing data
│   ├── model.pkl              # Serialized trained model
│   └── preprocessor.pkl       # Serialized feature preprocessor
├── logs/
│   └── *.log                  # Timestamped execution logs
├── notebook/
│   ├── 1.eda.ipynb            # Exploratory data analysis
│   ├── 2.model_training.ipynb # Training experimentation
│   └── data/
│       └── student.csv        # Raw student dataset
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py  # Data loading and train/test split
│   │   ├── data_transformation.py # Feature preprocessing pipelines
│   │   └── model_trainer.py   # Multi-model training and evaluation
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py  # End-to-end training orchestration
│   │   └── predict_pipeline.py # Inference and prediction
│   ├── exception.py           # Custom exception handling with traceback details
│   ├── logger.py              # Logging configuration
│   └── utils.py               # Serialization and model evaluation utilities
├── setup.py                   # Package installation metadata
├── requirements.txt           # Project dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Steps

1. Clone or download the repository

```bash
cd e2e-mlproject
```

2. Create and activate a Python virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Install the package in editable mode

```bash
python setup.py install
# or
python -m pip install -e .
```

## Usage

### Option 1: Using Flask Web Application

A Flask web application provides an easy-to-use interface for training models and making predictions.

#### Run the Application

run as a standalone script:

```bash
python app.py
```

The application will be available at `http://localhost:5000`

---

### Option 2: Using Python Code

#### 2.1: Run the Complete Training Pipeline

```python
from src.pipeline.train_pipeline import TrainPipeline

# Create and execute the training pipeline
pipeline = TrainPipeline()
pipeline.train()
```

Alternatively, run as a standalone script:

```bash
python src/pipeline/train_pipeline.py
```

#### 2.2: Run Components Individually

##### Data Ingestion

```python
from src.components.data_ingestion import DataIngestion

ingestion = DataIngestion(test_size=0.2, random_state=42)
train_path, test_path = ingestion.initiate_data_ingestion()
print(f"Train data: {train_path}")
print(f"Test data: {test_path}")
```

##### Data Transformation

```python
from src.components.data_transformation import DataTransformation

transformation = DataTransformation()
train_array, test_array = transformation.initiate_data_transformation(train_data_path, test_data_path)
```

##### Model Training

```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import numpy as np

di = DataIngestion()
train_path, test_path = di.initiate_data_ingestion()

dt = DataTransformation()
train_array, test_array = dt.initiate_data_transformation(train_path, test_path)

mt = ModelTrainer()
mt.initiate_model_trainer(train_array, test_array)
```

##### Making Predictions

```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import pandas as pd

# Create input data
custom_data = CustomData(
    gender="male",
    race_ethnicity="group A",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="none",
    reading_score=75,
    writing_score=80
)

# Convert to DataFrame
df = custom_data.get_data_as_data_frame()

# Predict
pipeline = PredictPipeline()
prediction = pipeline.predict(df)
print(f"Predicted math score: {prediction[0]:.2f}")
```


## Models Evaluated

The training pipeline compares the following regression models:

1. **Decision Tree** — Tree-based model with tuned split criteria
2. **Random Forest** — Ensemble of decision trees
3. **AdaBoost** — Adaptive boosting with learning rate tuning
4. **Gradient Boosting** — Sequential gradient optimization
5. **KNN** — K-Nearest Neighbors
6. **Linear Regression** — Baseline linear model
7. **Lasso** — L1-regularized linear regression
8. **Ridge** — L2-regularized linear regression
9. **SVR** — Support Vector Regression
10. **CatBoost** — Gradient boosting optimized for categorical features
11. **XGBoost** — Extreme Gradient Boosting

**Best model** is selected based on test set R² score (minimum threshold: 0.6).

## Documentation

All modules include comprehensive docstrings following Google/NumPy documentation style:

- **Module Level**: Overview of module purpose and contents
- **Class Level**: Description of class responsibility and usage
- **Function/Method Level**: Arguments, return values, and exceptions
- **Type Hints**: Full type annotations for better IDE support

Example:
```python
from src.utils import save_object

# Help available in IDE or Python REPL
help(save_object)
```

## Features of This Project

✅ **Modular Design** — Each component (ingestion, transformation, training) is independent  
✅ **Comprehensive Logging** — All operations logged to timestamped files under `logs/`  
✅ **Error Handling** — Custom exceptions with detailed traceback information  
✅ **Feature Preprocessing** — Separate pipelines for numeric, nominal, and ordinal features  
✅ **Hyperparameter Tuning** — GridSearchCV for optimal model parameters  
✅ **Model Persistence** — Serialize trained models and preprocessors with joblib  
✅ **Type Hints** — Full type annotations for code clarity  
✅ **Documentation** — Comprehensive docstrings for all classes and functions  
✅ **Production Ready** — Exception safety and input validation throughout  

## Dependencies

Key packages used in this project:

- **pandas** — Data manipulation and analysis
- **numpy** — Numerical computing
- **scikit-learn** — Machine learning algorithms and utilities
- **catboost** — Gradient boosting on decision trees
- **xgboost** — Extreme gradient boosting
- **seaborn** — Statistical data visualization
- **matplotlib** — Plotting library
- **joblib** — Object serialization

See `requirements.txt` for complete list with versions.

## Logging

All execution details are logged to timestamped files in the `logs/` directory:

```
logs/
├── 10_15_2024_14_30_45.log
├── 10_15_2024_15_20_12.log
└── ...
```

Each log file contains:
- Timestamp of each operation
- Logger name
- Log level (INFO, WARNING, ERROR)
- Detailed message

Example:
```
2024-10-15 14:30:45,123 - root - INFO - Entered the data ingestion component
2024-10-15 14:30:46,456 - root - INFO - Read the dataset as dataframe
```

## Exception Handling

The `CustomException` class provides detailed error information:

```python
from src.exception import CustomException

# Exceptions include file name, line number, and error message
try:
    # some operation
    pass
except Exception as e:
    raise CustomException(e, sys)
    # Output: "Error occured in python script name [file.py] line number [42] error message [details]"
```

## Development

### Adding a New Model

To add a new regression model for evaluation:

1. Import the model class in `src/components/model_trainer.py`
2. Add an entry to the `models` dictionary with name, estimator, and parameter grid
3. Run the training pipeline

### Extending the Pipeline

To add new preprocessing steps:

1. Modify `src/components/data_transformation.py`
2. Create new Pipeline steps or ColumnTransformer entries
3. Update the README with data flow changes

## Testing

To test individual components:

```bash
python -m pytest tests/  # (if test suite implemented)
```

## Author

Aishwarya Vadlamudi

## License

This project is provided as-is for educational purposes.

## Acknowledgments

- Dataset: Student performance data
- Built with scikit-learn, CatBoost, and XGBoost
- Follows Python packaging and documentation best practices
