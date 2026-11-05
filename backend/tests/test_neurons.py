"""MLP-neuron trace and group steering: contract, zero alpha, hooks."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from backend.app.services.model_loader import ModelManager


def _trace_payload(**overrides):
    payload = {
        "prompt": "Hello",
        "settings": {"max_new_tokens": 2},
        "layer_index": 0,
        "cluster_k": 4,
        "cluster_max_positions": 16,
    }
    payload.update(overrides)
    return payload


def _steer_payload(**overrides):
    payload = {
        "prompt": "Hello",
        "settings": {"max_new_tokens": 2},
        "layer_index": 0,
        "neurons": [{"neuron": 0, "weight": 1.0}],
        "alpha": 0.7,
    }
    payload.update(overrides)
    return payload


def test_neuron_trace_returns_clusters_and_records_the_run(client: TestClient):
    response = client.post("/api/experiments/neurons", json=_trace_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["layer_index"] == 0
    assert body["intermediate_size"] == 256
    assert body["hidden_size"] == 128
    assert body["residual_rms"] >= 0
    assert body["experiment_id"]
    assert body["prompt_tokens"] > 0
    assert body["output"].startswith("decoded:")
    assert len(body["clusters"]) <= 4
    assert sum(cluster["size"] for cluster in body["clusters"]) == 256
    assert len(body["top_neurons"]) <= 20
    assert len(body["assignments"]) == 256
    for cluster in body["clusters"]:
        assert cluster["top_neurons"]
        assert 0 <= cluster["mass"] <= 1
        assert -1.000001 <= cluster["cohesion"] <= 1.000001
        for top in cluster["top_neurons"]:
            assert 0 <= top["neuron"] < 256
            assert top["sigma"] >= 0
    detail = client.get(f"/api/experiments/{body['experiment_id']}")
    assert detail.status_code == 200


def test_neuron_trace_removes_hooks(client: TestClient, manager):
    response = client.post("/api/experiments/neurons", json=_trace_payload())
    assert response.status_code == 200
    for module in manager.model.model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_neuron_steering_runs_baseline_and_steered(client: TestClient, manager):
    response = client.post("/api/experiments/neuron-steering", json=_steer_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["layer_index"] == 0
    assert body["alpha"] == 0.7
    assert body["direction_norm"] > 0
    assert body["residual_rms"] >= 0
    assert len(body["deltas"]) == 1
    assert body["deltas"][0]["neuron"] == 0
    assert body["baseline"]["output"].startswith("decoded:")
    assert body["steered"]["output"].startswith("decoded:")
    assert body["duration_ms"] >= 0
    for module in manager.model.model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_neuron_steering_zero_alpha_matches_baseline(client: TestClient):
    response = client.post(
        "/api/experiments/neuron-steering",
        json=_steer_payload(alpha=0.0),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["baseline"]["output"] == body["steered"]["output"]
    assert body["deltas"][0]["delta"] == 0.0


def test_neuron_steering_rejects_unknown_neuron(client: TestClient):
    response = client.post(
        "/api/experiments/neuron-steering",
        json=_steer_payload(neurons=[{"neuron": 9999, "weight": 1.0}]),
    )
    assert response.status_code == 400
    assert "outside" in response.json()["detail"]


def test_neuron_steering_rejects_unknown_layer(client: TestClient):
    response = client.post(
        "/api/experiments/neuron-steering",
        json=_steer_payload(layer_index=99),
    )
    assert response.status_code == 400
    assert "outside the available decoder layers" in response.json()["detail"]


def test_neuron_steering_rejects_empty_group(client: TestClient):
    response = client.post(
        "/api/experiments/neuron-steering",
        json=_steer_payload(neurons=[]),
    )
    assert response.status_code == 422


def test_neuron_routes_require_a_loaded_model(experiment_store):
    unloaded = TestClient(
        create_app(ModelManager(Settings(model_id="synthetic/test-model")), experiment_store)
    )
    trace = unloaded.post("/api/experiments/neurons", json=_trace_payload())
    steer = unloaded.post("/api/experiments/neuron-steering", json=_steer_payload())
    assert trace.status_code == 409
    assert steer.status_code == 409
