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


def test_invalid_task_rejected():
    rows=[{"x":i,"y":i*2+1} for i in range(20)]
    response=client.post("/automl",json={"rows":rows,"target":"y","task":"unknown"})
    assert response.status_code == 400


def test_invalid_forecast_periods_rejected():
    rows=[{"date":f"2026-01-{i:02d}","value":i} for i in range(1,11)]
    response=client.post("/forecast",json={"rows":rows,"date_column":"date","value_column":"value","periods":366})
    assert response.status_code == 400


def test_invalid_drift_threshold_rejected():
    rows=[{"x":i} for i in range(10)]
    response=client.post("/drift",json={"baseline_rows":rows,"current_rows":rows,"threshold":0})
    assert response.status_code == 400


def test_invalid_rows_rejected():
    response=client.post("/anomalies",json={"rows":[1,2,3]})
    assert response.status_code == 400
