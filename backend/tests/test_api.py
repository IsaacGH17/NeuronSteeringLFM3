"""API contract and error handling with the synthetic model."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from backend.app.services.model_loader import ModelManager


def test_health(client: TestClient):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "lfm3-neuron-explorer"}


def test_diagnostics(client: TestClient):
    response = client.get("/api/diagnostics")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] in {"rocm", "cuda", "cpu"}
    assert body["model_state"] == "loaded"
    assert body["inference_device"] == "cpu"


def test_model_status(client: TestClient):
    response = client.get("/api/model/status")
    assert response.status_code == 200
    body = response.json()
    assert body["loaded"] is True
    assert body["model_id"] == "synthetic/test-model"
    assert body["dtype"] == "float32"


def test_architecture(client: TestClient):
    response = client.get("/api/model/architecture")
    assert response.status_code == 200
    body = response.json()
    assert body["tree"]["id"] == "root"
    assert len(body["layer_overview"]) == 4
    assert body["layer_overview"][1]["operator_type"] == "attention"


def test_module_detail_and_404(client: TestClient):
    response = client.get("/api/model/modules/model.layers.2.self_attn")
    assert response.status_code == 200
    assert response.json()["class_name"] == "Lfm3Attention"
    missing = client.get("/api/model/modules/model.layers.99")
    assert missing.status_code == 404


def test_baseline_generation(client: TestClient):
    response = client.post(
        "/api/experiments/baseline",
        json={"prompt": "Hello", "settings": {"max_new_tokens": 4}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["condition"] == "baseline"
    assert body["prompt_tokens"] > 0
    assert body["generated_tokens"] == 4
    assert body["output"].startswith("decoded:")
    assert body["device"] == "cpu"
    assert body["warnings"] == []


def test_baseline_clamps_tokens_and_warns(client: TestClient):
    response = client.post(
        "/api/experiments/baseline",
        json={"prompt": "Hello", "settings": {"max_new_tokens": 64}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["generated_tokens"] == 8
    assert any("clamped" in warning for warning in body["warnings"])


def test_blank_prompt_rejected(client: TestClient):
    response = client.post("/api/experiments/baseline", json={"prompt": "   "})
    assert response.status_code == 422


def test_unloaded_model_returns_conflict():
    app = create_app(ModelManager(Settings(model_id="synthetic/test-model")))
    unloaded = TestClient(app)
    assert unloaded.get("/api/model/status").json()["loaded"] is False
    assert unloaded.get("/api/model/architecture").status_code == 409
    assert unloaded.post("/api/experiments/baseline", json={"prompt": "Hello"}).status_code == 409


def test_load_rejects_unknown_device_without_network():
    app = create_app(ModelManager(Settings(model_id="synthetic/test-model")))
    response = TestClient(app).post("/api/model/load", json={"device": "quantum"})
    assert response.status_code == 502
    assert "Unsupported device" in response.json()["detail"]
    assert TestClient(app).get("/api/model/status").json()["state"] == "error"
