"""
Flask Web Application for Student Performance Prediction

This module provides a web interface for training machine learning models
and making predictions on student performance data. It includes endpoints
for model training, prediction, and health checks.

Endpoints:
- /: Home page
- /predict: Prediction interface (GET for form, POST for prediction)
- /train: Train the model
- /health: Health check endpoint

Dependencies:
- Flask: Web framework
- src.pipeline.predict_pipeline: CustomData and PredictPipeline classes
- src.pipeline.train_pipeline: TrainPipeline class
"""

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline

app = Flask(__name__)

@app.route("/")
def index():
    """
    Render the home page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    """
    Handle prediction requests.

    For GET requests, render the prediction form.
    For POST requests, process the form data and return predictions.

    Returns:
        str: Rendered HTML template with prediction results for POST,
             or the form for GET.
    """
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("writing_score")),
            writing_score=float(request.form.get("reading_score")),
        )

        data_df = data.get_data_as_data_frame()
        prediction_pipeline = PredictPipeline()
        results = prediction_pipeline.predict(data_df)
        return render_template("home.html", results=results[0])


@app.route("/train", methods=["GET", "POST"])
def train_model():
    """
    Train the machine learning model.

    Initiates the training pipeline to train the model on the dataset.

    Returns:
        dict: JSON response with training status message.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.train()
        return {"message": "Model trained successfully"}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/health")
def health():
    """
    Health check endpoint.

    Returns the status of the application.

    Returns:
        dict: JSON response with health status.
    """
    return {"status": "healthy"}


@app.errorhandler(404)
def not_found(error):
    """
    Handle 404 Not Found errors.

    Args:
        error: The error object.

    Returns:
        tuple: JSON response and status code.
    """
    return {"error": "Not found"}, 404


@app.errorhandler(405)
def method_not_allowed(error):
    """
    Handle 405 Method Not Allowed errors.

    Args:
        error: The error object.

    Returns:
        tuple: JSON response and status code.
    """
    return {"error": "Method not allowed"}, 405


@app.errorhandler(500)
def internal_error(error):
    """
    Handle 500 Internal Server Error.

    Args:
        error: The error object.

    Returns:
        tuple: JSON response and status code.
    """
    return {"error": "Internal server error"}, 500


if __name__ == "__main__":
    print("👉 Open this URL in your browser:")
    print("http://127.0.0.1:5000")

    print("\n👉 To train prediction endpoint:")
    print("http://127.0.0.1:5000/train")

    print("\n👉 To test prediction endpoint:")
    print("http://127.0.0.1:5000/predict")

    app.run(host="0.0.0.0", debug=True)
