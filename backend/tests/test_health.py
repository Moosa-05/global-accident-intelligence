from fastapi.testclient import TestClient

def test_root_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["model_loaded"] is True
    assert "version" in data
    assert "database" in data

def test_api_v1_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["model_loaded"] is True

def test_liveness_endpoints(client: TestClient):
    res_root = client.get("/health/live")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "alive"

    res_api = client.get("/api/v1/health/live")
    assert res_api.status_code == 200
    assert res_api.json()["status"] == "alive"

def test_readiness_endpoints(client: TestClient):
    res_root = client.get("/health/ready")
    assert res_root.status_code in [200, 503]
    data = res_root.json()
    assert "ready" in data
    assert "models_ready" in data
    assert "database" in data

