"""Activation capture, storage, and visualization payloads."""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.schemas.api import BaselineRequest, GenerationSettings
from backend.app.services.activation_capture import validate_capture_module_ids
from backend.app.services.activation_store import ActivationStore, StoredExperiment
from backend.app.services.generation import generate_baseline


def _hook_counts(model):
    return {
        name: (len(module._forward_hooks), len(module._forward_pre_hooks))
        for name, module in model.named_modules()
    }


def test_capture_returns_real_tensor_and_removes_hooks(manager):
    module_id = "model.layers.1"
    before = _hook_counts(manager.model)
    result = generate_baseline(
        manager,
        BaselineRequest(prompt="Capture me", settings=GenerationSettings(max_new_tokens=3)),
        capture_ids=[module_id],
    )
    after = _hook_counts(manager.model)
    assert before == after
    assert all(counts == (0, 0) for counts in after.values())
    assert result.response.captured_modules == [module_id]
    tensor = result.captures[module_id]
    assert tensor.shape[1] == 128
    assert tensor.dtype == torch.float32
    assert len(result.tokens) == tensor.shape[0]
    assert tensor.shape[0] <= result.response.prompt_tokens + 3


def test_capture_handles_tuple_output(manager):
    result = generate_baseline(
        manager,
        BaselineRequest(prompt="Attention", settings=GenerationSettings(max_new_tokens=2)),
        capture_ids=["model.layers.1.self_attn"],
    )
    tensor = result.captures["model.layers.1.self_attn"]
    assert tensor.shape[1] == 128
    assert len(result.tokens) == tensor.shape[0]


def test_non_captureable_module_rejected(manager):
    with pytest.raises(ValueError, match="cannot be captured"):
        validate_capture_module_ids(manager.model, ["model.embed_tokens"])


def test_unknown_capture_module_rejected(manager):
    with pytest.raises(ValueError, match="Unknown capture module"):
        validate_capture_module_ids(manager.model, ["model.layers.99"])


def test_capture_api_round_trip(client: TestClient):
    response = client.post(
        "/api/experiments/baseline",
        json={
            "prompt": "Capture this",
            "settings": {"max_new_tokens": 4},
            "capture_module_ids": ["model.layers.1.feed_forward"],
        },
    )
    assert response.status_code == 200
    run = response.json()
    assert run["captured_modules"] == ["model.layers.1.feed_forward"]
    experiment_id = run["id"]

    listing = client.get(f"/api/experiments/{experiment_id}/activations")
    assert listing.status_code == 200
    summaries = listing.json()
    assert summaries[0]["module_id"] == "model.layers.1.feed_forward"

    activation = client.get(f"/api/experiments/{experiment_id}/activations/model.layers.1.feed_forward")
    assert activation.status_code == 200
    body = activation.json()
    assert summaries[0]["shape"] == body["shape"]
    assert body["shape"][1] == 128
    assert len(body["tokens"]) == body["shape"][0]
    assert {token["phase"] for token in body["tokens"]} == {"prompt", "generated"}
    assert len(body["heatmap"]["values"]) == len(body["heatmap"]["positions"])
    assert len(body["heatmap"]["values"][0]) == len(body["heatmap"]["features"])
    assert len(body["histogram"]["counts"]) == 40
    assert len(body["top_dimensions"]) == 20
    assert body["feature_series"]["feature"] == body["top_dimensions"][0]["index"]
    assert len(body["feature_series"]["values"]) == body["shape"][0]

    with_feature = client.get(
        f"/api/experiments/{experiment_id}/activations/model.layers.1.feed_forward",
        params={"feature": 5, "position": 0},
    )
    assert with_feature.status_code == 200
    assert with_feature.json()["feature_series"]["feature"] == 5
    assert with_feature.json()["position"] == 0


def test_activation_api_errors(client: TestClient):
    response = client.post(
        "/api/experiments/baseline",
        json={
            "prompt": "Capture",
            "settings": {"max_new_tokens": 2},
            "capture_module_ids": ["model.layers.1"],
        },
    )
    experiment_id = response.json()["id"]

    assert client.get("/api/experiments/nope/activations").status_code == 404
    assert client.get(f"/api/experiments/{experiment_id}/activations/model.layers.0").status_code == 404
    assert (
        client.get(
            f"/api/experiments/{experiment_id}/activations/model.layers.1",
            params={"feature": 9999},
        ).status_code
        == 400
    )
    assert (
        client.get(
            f"/api/experiments/{experiment_id}/activations/model.layers.1",
            params={"position": 9999},
        ).status_code
        == 400
    )


def test_baseline_rejects_uncapturable_module(client: TestClient):
    response = client.post(
        "/api/experiments/baseline",
        json={"prompt": "Hello", "capture_module_ids": ["model.embed_tokens"]},
    )
    assert response.status_code == 400
    assert "cannot be captured" in response.json()["detail"]


def test_store_evicts_oldest_experiments():
    store = ActivationStore(max_experiments=2)
    for index in range(3):
        store.add(
            StoredExperiment(
                id=f"run-{index}",
                created_at=None,  # type: ignore[arg-type]
                model_id="m",
                revision="r",
                prompt="p",
                system=None,
                output="o",
                prompt_token_count=1,
                generated_tokens=1,
                tokens=[],
                captures={"layer": torch.zeros(1, 4)},
            )
        )
    with pytest.raises(KeyError):
        store.get("run-0")
    assert store.get("run-1").id == "run-1"
    assert store.get("run-2").id == "run-2"
