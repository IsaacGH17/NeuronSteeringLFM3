"""Patch sweep: grid contract, limits, and hook lifecycle."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from backend.app.services.model_loader import ModelManager


def _payload(**overrides):
    payload = {
        "recipient_prompt": "Hello",
        "donor_prompt": "World",
        "settings": {"max_new_tokens": 2},
        "layer_indices": [0, 1],
        "positions": [-1],
    }
    payload.update(overrides)
    return payload


def test_patch_sweep_returns_a_grid(client: TestClient):
    response = client.post("/api/experiments/patch-sweep", json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["layers"] == [0, 1]
    assert len(body["positions"]) == 1
    assert body["prompt_tokens"] > 0
    assert body["donor_prompt_tokens"] > 0
    assert body["baseline_output"].startswith("decoded:")
    assert body["donor_output"].startswith("decoded:")
    assert body["duration_ms"] >= 0
    assert len(body["cells"]) == 2
    for cell in body["cells"]:
        assert cell["module_id"] == f"model.layers.{cell['layer_index']}"
        assert cell["position"] == body["positions"][0]
        assert 0 <= cell["donor_position"] < body["donor_prompt_tokens"]
        assert isinstance(cell["changed"], bool)
        assert isinstance(cell["first_token_changed"], bool)
        assert 0 <= cell["common_prefix_tokens"] <= cell["generated_tokens"]
        assert cell["patched_output"].startswith("decoded:")


def test_patch_sweep_resolves_multiple_positions(client: TestClient):
    response = client.post(
        "/api/experiments/patch-sweep",
        json=_payload(layer_indices=[0], positions=[-1, 0]),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["positions"]) == 2
    assert len(body["cells"]) == 2
    assert {cell["position"] for cell in body["cells"]} == set(body["positions"])


def test_patch_sweep_cell_limit(client: TestClient, monkeypatch):
    monkeypatch.setattr("backend.app.services.patch_sweep.MAX_PATCH_SWEEP_CELLS", 1)
    response = client.post("/api/experiments/patch-sweep", json=_payload())
    assert response.status_code == 400
    assert "limited to 1 cells" in response.json()["detail"]


def test_patch_sweep_rejects_unknown_layers(client: TestClient):
    response = client.post(
        "/api/experiments/patch-sweep",
        json=_payload(layer_indices=[99]),
    )
    assert response.status_code == 400
    assert "outside the available layers" in response.json()["detail"]


def test_patch_sweep_skips_invalid_positions(client: TestClient):
    response = client.post(
        "/api/experiments/patch-sweep",
        json=_payload(layer_indices=[0], positions=[100, -1]),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["positions"]) == 1
    assert any("Position 100 is outside" in warning for warning in body["warnings"])


def test_patch_sweep_rejects_only_invalid_positions(client: TestClient):
    response = client.post(
        "/api/experiments/patch-sweep",
        json=_payload(layer_indices=[0], positions=[100]),
    )
    assert response.status_code == 400
    assert "No valid recipient positions" in response.json()["detail"]


def test_patch_sweep_removes_hooks(client: TestClient, manager):
    response = client.post("/api/experiments/patch-sweep", json=_payload())
    assert response.status_code == 200
    for module in manager.model.model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_patch_sweep_requires_a_loaded_model(experiment_store):
    unloaded = TestClient(
        create_app(ModelManager(Settings(model_id="synthetic/test-model")), experiment_store)
    )
    response = unloaded.post("/api/experiments/patch-sweep", json=_payload())
    assert response.status_code == 409
