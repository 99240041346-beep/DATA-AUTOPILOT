from fastapi.testclient import TestClient
from main import app

client=TestClient(app)


def test_health_and_ready():
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200


def test_automl_regression():
    rows=[{"x":i,"y":i*2+1} for i in range(20)]
    response=client.post("/automl",json={"rows":rows,"target":"y","task":"regression"})
    assert response.status_code == 200
    data=response.json()
    assert data["target"]=="y"
    assert data["best_model"]


def test_input_row_limit():
    rows=[{"x":1}]*50001
    response=client.post("/automl",json={"rows":rows,"target":"x"})
    assert response.status_code == 413
