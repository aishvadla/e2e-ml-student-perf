import numpy as np
import pytest
from unittest.mock import patch

from application import application as flask_app


@pytest.fixture
def client():
    flask_app.testing = True
    return flask_app.test_client()


def test_index_route_returns_homepage(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Student Performance Predictor" in response.data


def test_health_route_returns_healthy_status(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.is_json
    assert response.get_json() == {"status": "healthy"}


def test_404_route_returns_not_found(client):
    response = client.get("/missing-route")

    assert response.status_code == 404
    assert response.is_json
    assert response.get_json() == {"error": "Not found"}


@patch("application.PredictPipeline.predict")
def test_predict_post_invokes_pipeline(mock_predict, client):
    mock_predict.return_value = np.array([85.0])

    response = client.post(
        "/predict",
        data={
            "gender": "female",
            "ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
            "writing_score": "72",
            "reading_score": "74",
        },
    )

    assert response.status_code == 200
    assert b"85.00" in response.data
    mock_predict.assert_called_once()


def test_predict_get_returns_form(client):
    response = client.get("/predict")

    assert response.status_code == 200
    assert b"Test Preparation" in response.data
