from fastapi.testclient import TestClient

def test_models_registry_endpoint(client: TestClient):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "platform_name" in data
    assert "global_extensibility" in data
    assert data["total_registered_models"] == 4
    assert data["commercial_cleared_count"] == 2
    assert data["restricted_research_count"] == 2
    assert len(data["models"]) == 4
    
    model_ids = [m["model_id"] for m in data["models"]]
    assert "accident_risk_v1" in model_ids
    assert "accident_severity_v2" in model_ids
    assert "human_injury_severity_v1" in model_ids
    assert "exposure_risk_v1" in model_ids

    inj_model = next(m for m in data["models"] if m["model_id"] == "human_injury_severity_v1")
    assert inj_model["status"] == "READY"
    assert inj_model["commercial_use_permitted"] is True
    assert inj_model["licensing_category"] == "COMMERCIAL_CLEARED"
    assert "COMMERCIAL SAAS CLEARED" in inj_model["licensing"]["commercial_status"]

    exp_model = next(m for m in data["models"] if m["model_id"] == "exposure_risk_v1")
    assert exp_model["status"] == "READY"
    assert exp_model["commercial_use_permitted"] is True
    assert exp_model["licensing_category"] == "COMMERCIAL_CLEARED"

    sev_model = next(m for m in data["models"] if m["model_id"] == "accident_severity_v2")
    assert sev_model["status"] == "READY"
    assert sev_model["commercial_use_permitted"] is False
    assert sev_model["licensing_category"] == "RESTRICTED_NON_COMMERCIAL"
    assert "CC BY-NC-SA 4.0" in sev_model["licensing"]["dataset_license"]
    assert "United States" in sev_model["training_geography"]

def test_models_registry_commercial_filter(client: TestClient):
    # Query commercial models only
    response = client.get("/api/v1/models?commercial_only=true")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 2
    for m in data["models"]:
        assert m["commercial_use_permitted"] is True
        assert m["licensing_category"] == "COMMERCIAL_CLEARED"
    
    model_ids = [m["model_id"] for m in data["models"]]
    assert "human_injury_severity_v1" in model_ids
    assert "exposure_risk_v1" in model_ids
    assert "accident_severity_v2" not in model_ids
    assert "accident_risk_v1" not in model_ids
