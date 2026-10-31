from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
