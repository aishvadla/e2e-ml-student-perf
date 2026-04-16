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

application = Flask(__name__)

@application.route("/")
def index():
    """
    Render the home page.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template("index.html")


@application.route("/predict", methods=["GET", "POST"])
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
        return render_template("home.html", results=None, features=None)
    else:
        # Capture form data
        gender = request.form.get("gender")
        ethnicity = request.form.get("ethnicity")
        parental_education = request.form.get("parental_level_of_education")
        lunch = request.form.get("lunch")
        test_prep = request.form.get("test_preparation_course")
        reading_score = float(request.form.get("writing_score"))
        writing_score = float(request.form.get("reading_score"))
        
        # Create features dictionary for display
        features = {
            "Gender": gender.capitalize(),
            "Ethnicity": ethnicity,
            "Parental Education": parental_education.title(),
            "Lunch Type": lunch.capitalize(),
            "Test Preparation": test_prep.capitalize(),
            "Reading Score": reading_score,
            "Writing Score": writing_score,
        }
        
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_education,
            lunch=lunch,
            test_preparation_course=test_prep,
            reading_score=reading_score,
            writing_score=writing_score,
        )

        data_df = data.get_data_as_data_frame()
        prediction_pipeline = PredictPipeline()
        results = prediction_pipeline.predict(data_df)
        return render_template("home.html", results=results[0], features=features)


@application.route("/health")
def health():
    """
    Health check endpoint.

    Returns the status of the application.

    Returns:
        dict: JSON response with health status.
    """
    return {"status": "healthy"}


@application.errorhandler(404)
def not_found(error):
    """
    Handle 404 Not Found errors.

    Args:
        error: The error object.

    Returns:
        tuple: JSON response and status code.
    """
    return {"error": "Not found"}, 404


@application.errorhandler(405)
def method_not_allowed(error):
    """
    Handle 405 Method Not Allowed errors.

    Args:
        error: The error object.

    Returns:
        tuple: JSON response and status code.
    """
    return {"error": "Method not allowed"}, 405


@application.errorhandler(500)
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
    print("http://localhost:8080/")

    application.run(host="0.0.0.0", port=8080, debug=False)
