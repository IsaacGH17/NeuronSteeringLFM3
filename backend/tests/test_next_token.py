"""Next-token probe: one-forward distributions, steering, and API contract."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from backend.app.services.model_loader import ModelManager


def _payload(**overrides):
    payload = {"prompt": "Hello", "top_k": 5}
    payload.update(overrides)
    return payload


def test_next_token_baseline(client: TestClient):
    response = client.post("/api/experiments/next-token", json=_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["prompt"] == "Hello"
    assert body["prompt_tokens"] > 0
    assert body["position"] == body["prompt_tokens"] - 1
    assert body["steered"] is None
    assert body["kl_divergence"] is None
    assert body["steering"] is None
    assert body["warnings"] == []
    assert body["duration_ms"] >= 0
    assert body["model_id"] == "synthetic/test-model"

    tokens = body["baseline"]["tokens"]
    assert len(tokens) == 5
    probabilities = [token["probability"] for token in tokens]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1.0 + 1e-6
    assert all(token["text"] for token in tokens)
    assert body["baseline"]["entropy"] > 0

    candidates = body["candidates"]
    assert len(candidates) == 5
    assert all(candidate["steered_probability"] is None for candidate in candidates)
    assert all(candidate["delta"] is None for candidate in candidates)
    assert candidates[0]["token_id"] == tokens[0]["token_id"]


def test_zero_strength_reproduces_the_baseline(client: TestClient):
    response = client.post(
        "/api/experiments/next-token",
        json=_payload(steering={"module_id": "model.layers.0", "feature_index": 0, "strength": 0.0}),
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["steered"] is not None
    assert body["kl_divergence"] == pytest.approx(0.0, abs=1e-6)
    baseline_ids = [token["token_id"] for token in body["baseline"]["tokens"]]
    steered_ids = [token["token_id"] for token in body["steered"]["tokens"]]
    assert baseline_ids == steered_ids
    for baseline, steered in zip(body["baseline"]["tokens"], body["steered"]["tokens"], strict=True):
        assert baseline["probability"] == pytest.approx(steered["probability"], abs=1e-6)
    for candidate in body["candidates"]:
        assert candidate["delta"] == pytest.approx(0.0, abs=1e-6)


def test_nonzero_steering_runs_the_second_forward(client: TestClient):
    response = client.post(
        "/api/experiments/next-token",
        json=_payload(steering={"module_id": "model.layers.0", "feature_index": 0, "strength": 500.0}),
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["steered"] is not None
    assert body["kl_divergence"] is not None
    assert body["kl_divergence"] >= 0
    assert len(body["steered"]["tokens"]) == 5
    assert all(candidate["steered_probability"] is not None for candidate in body["candidates"])
    assert all(candidate["delta"] is not None for candidate in body["candidates"])
    assert not any("never fired" in warning for warning in body["warnings"])


def test_next_token_removes_hooks(client: TestClient, manager):
    response = client.post(
        "/api/experiments/next-token",
        json=_payload(steering={"module_id": "model.layers.1", "feature_index": 3, "strength": 2.0}),
    )
    assert response.status_code == 200
    for module in manager.model.model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_next_token_rejects_unknown_module(client: TestClient):
    response = client.post(
        "/api/experiments/next-token",
        json=_payload(steering={"module_id": "model.nonexistent", "feature_index": 0, "strength": 1.0}),
    )
    assert response.status_code == 400
    assert "Unknown steering module" in response.json()["detail"]


def test_next_token_requires_a_loaded_model(experiment_store):
    unloaded = TestClient(
        create_app(ModelManager(Settings(model_id="synthetic/test-model")), experiment_store)
    )
    response = unloaded.post("/api/experiments/next-token", json=_payload())
    assert response.status_code == 409


@pytest.mark.parametrize("top_k", [0, 51])
def test_next_token_bounds_top_k(client: TestClient, top_k: int):
    response = client.post("/api/experiments/next-token", json=_payload(top_k=top_k))
    assert response.status_code == 422


def test_next_token_rejects_blank_prompt(client: TestClient):
    response = client.post("/api/experiments/next-token", json=_payload(prompt="   "))
    assert response.status_code == 422
