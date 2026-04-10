from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for a home page


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))
        
        data_df = data.get_data_as_data_frame()
        prediction_pipeline = PredictPipeline()
        results = prediction_pipeline.predict(data_df)
        return render_template('home.html', results=results[0])


if __name__ == '__main__':
    print("👉 Open this URL in your browser:")
    print("http://127.0.0.1:5000")
    
    print("\n👉 To test prediction endpoint (POST):")
    print("http://127.0.0.1:5000/predict")

    app.run(host='0.0.0.0', debug=True)