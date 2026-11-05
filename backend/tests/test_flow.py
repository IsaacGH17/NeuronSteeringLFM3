"""Live flow tracing: real layer metrics, descriptive clusters, and limits."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from backend.app.schemas.api import FlowRequest
from backend.app.services.activation_store import StoredExperiment
from backend.app.services.flow import cluster_profiles, compute_flow
from backend.app.services.model_loader import ModelManager


def _flow_payload(**overrides):
    payload = {
        "prompt": "Hello world",
        "settings": {"max_new_tokens": 4},
        "cluster_k": 4,
        "cluster_max_positions": 16,
    }
    payload.update(overrides)
    return payload


def test_flow_trace_returns_every_layer(client: TestClient):
    response = client.post("/api/experiments/flow", json=_flow_payload())
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["layers"]) == 4
    assert len(body["edges"]) == 3
    assert body["experiment_id"]
    assert body["output"].startswith("decoded:")
    assert body["prompt_tokens"] > 0
    assert body["global_max_norm"] > 0
    assert any("final generated token" in warning for warning in body["warnings"])
    assert all(layer["ablation_kl"] is None for layer in body["layers"])

    for index, layer in enumerate(body["layers"]):
        assert layer["layer_index"] == index
        assert layer["module_id"] == f"model.layers.{index}"
        assert layer["operator_type"] in {"attention", "conv"}
        assert layer["layer_type"]
        assert len(layer["energy"]) == len(body["tokens"])
        assert len(layer["delta"]) == len(body["tokens"])
        assert all(value >= 0 for value in layer["energy"])
        assert all(value >= 0 for value in layer["delta"])
        assert layer["mean_norm"] >= 0
        assert 0 <= layer["similarity"] <= 1.000001
        # Every hidden dimension belongs to exactly one cluster.
        assert sum(cluster["size"] for cluster in layer["clusters"]) == 128
        assert len(layer["clusters"]) <= 4
        for cluster in layer["clusters"]:
            assert cluster["components"]
            assert cluster["top_dimensions"]
            assert 0 <= cluster["mass"] <= 1
            assert -1.000001 <= cluster["cohesion"] <= 1.000001
            assert len(cluster["profile"]) == len(cluster["profile_positions"])
            for dimension in cluster["top_dimensions"]:
                assert 0 <= dimension["index"] < 128
                assert dimension["sigma"] >= 0

    for edge in body["edges"]:
        assert edge["target_layer"] == edge["source_layer"] + 1
        assert edge["write"] >= 0
        assert edge["gain"] >= 0

    # The run is recorded like any other experiment and its captures are live.
    detail = client.get(f"/api/experiments/{body['experiment_id']}")
    assert detail.status_code == 200
    assert len(detail.json()["captured_modules"]) == 4
    activations = client.get(f"/api/experiments/{body['experiment_id']}/activations")
    assert activations.status_code == 200
    assert len(activations.json()) == 4


def test_flow_ablation_importance_is_computed_on_request(client: TestClient, manager):
    response = client.post(
        "/api/experiments/flow",
        json=_flow_payload(include_ablation=True),
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert len(body["layers"]) == 4
    assert all(layer["ablation_kl"] is not None for layer in body["layers"])
    assert all(layer["ablation_kl"] >= 0 for layer in body["layers"])
    assert not any("Ablation of" in warning for warning in body["warnings"])
    for module in manager.model.model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_flow_trace_is_deterministic_for_greedy(client: TestClient):
    first = client.post("/api/experiments/flow", json=_flow_payload()).json()
    second = client.post("/api/experiments/flow", json=_flow_payload()).json()

    assert first["experiment_id"] != second["experiment_id"]
    assert first["output"] == second["output"]
    for left, right in zip(first["edges"], second["edges"], strict=True):
        assert left["write"] == pytest.approx(right["write"])
        assert left["similarity"] == pytest.approx(right["similarity"])
        assert left["gain"] == pytest.approx(right["gain"])
    for left_layer, right_layer in zip(first["layers"], second["layers"], strict=True):
        assert [cluster["size"] for cluster in left_layer["clusters"]] == [
            cluster["size"] for cluster in right_layer["clusters"]
        ]
        for left_cluster, right_cluster in zip(left_layer["clusters"], right_layer["clusters"], strict=True):
            assert left_cluster["mass"] == pytest.approx(right_cluster["mass"])
            assert [component["feature_index"] for component in left_cluster["components"]] == [
                component["feature_index"] for component in right_cluster["components"]
            ]


def test_flow_removes_hooks_after_a_trace(client: TestClient, manager):
    response = client.post("/api/experiments/flow", json=_flow_payload())
    assert response.status_code == 200
    for module in manager.model.model.modules():
        assert len(module._forward_hooks) == 0
        assert len(module._forward_pre_hooks) == 0


def test_flow_rejects_traces_over_the_position_budget(client: TestClient, monkeypatch):
    monkeypatch.setattr("backend.app.services.flow.MAX_FLOW_POSITIONS", 8)
    response = client.post(
        "/api/experiments/flow",
        json=_flow_payload(settings={"max_new_tokens": 8}),
    )
    assert response.status_code == 400
    assert "limited to 8 positions" in response.json()["detail"]


def test_flow_requires_a_loaded_model(experiment_store):
    unloaded = TestClient(
        create_app(ModelManager(Settings(model_id="synthetic/test-model")), experiment_store)
    )
    response = unloaded.post("/api/experiments/flow", json=_flow_payload())
    assert response.status_code == 409


def test_flow_rejects_experiments_without_every_layer(manager):
    stored = StoredExperiment(
        id="a" * 32,
        created_at=datetime.now(UTC),
        model_id="synthetic/test-model",
        revision="main",
        prompt="Hello",
        system=None,
        output="decoded:1",
        prompt_token_count=3,
        generated_tokens=1,
        tokens=[],
        captures={"model.layers.0": torch.zeros(3, 128)},
        warnings=[],
    )
    with pytest.raises(ValueError, match="did not capture every decoder layer"):
        compute_flow(stored, FlowRequest(prompt="Hello"), model=manager.model)


def test_cluster_profiles_separates_coherent_groups():
    positions = 64
    t = np.linspace(0, 6 * np.pi, positions, dtype=np.float32)
    wave = np.sin(t)
    rng = np.random.default_rng(0)
    group_a = wave[:, None] + 0.05 * rng.standard_normal((positions, 4)).astype(np.float32)
    group_b = -wave[:, None] + 0.05 * rng.standard_normal((positions, 4)).astype(np.float32)
    array = np.concatenate([group_a, group_b], axis=1)

    clusters, sampled = cluster_profiles(array, k=2, max_positions=positions)

    assert sampled == positions
    assert len(clusters) == 2
    assert {cluster.size for cluster in clusters} == {4}
    assert clusters[0].mass >= clusters[1].mass
    group_one = {0, 1, 2, 3}
    group_two = {4, 5, 6, 7}
    top_sets = [{dimension.index for dimension in cluster.top_dimensions} for cluster in clusters]
    assert all(top <= group_one or top <= group_two for top in top_sets)
    assert top_sets[0].isdisjoint(top_sets[1])
    for cluster in clusters:
        assert 0 <= cluster.peak_position < positions
        assert 0 <= cluster.cohesion <= 1
        assert cluster.components
