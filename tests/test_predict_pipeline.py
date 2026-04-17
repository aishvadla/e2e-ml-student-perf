import numpy as np


def test_prediction_on_single_record():
    from src.pipeline.predict_pipeline import CustomData, PredictPipeline

    # 1. Create a mock single input
    data = CustomData(
        gender="female",
        race_ethnicity="group B",
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="none",
        reading_score=72,
        writing_score=74,
    )

    # 2. Convert to DataFrame
    df = data.get_data_as_data_frame()

    # 3. Predict
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(df)

    # 4. Check if we got a number back
    assert len(results) == 1
    assert isinstance(results[0], (int, float, np.number))
