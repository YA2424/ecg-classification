import pytest
from fastapi.testclient import TestClient
from time_series_classification.deployment.RestAPI import app, load_best_model

@pytest.fixture(autouse=True)
def mock_load_best_model(monkeypatch):
    def mock_model():
        class MockModel:
            def predict(self, data):
                return [[1, 0, 0, 0, 0]]
        return MockModel()
    monkeypatch.setattr("time_series_classification.deployment.RestAPI.load_best_model", mock_model)

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"data": [0] * 140})
    assert response.status_code == 200
    assert 'predicted_class' in response.json()

def test_predict_endpoint_invalid_json():
    response = client.post("/predict", data="invalid json")
    assert response.status_code == 422
def test_predict_endpoint_missing_data():
    response = client.post("/predict", json={})
    assert response.status_code == 422
