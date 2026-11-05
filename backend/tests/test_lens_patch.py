"""Logit lens, per-layer ablation, and activation patching."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient


def test_lens_returns_every_layer_and_ablation(client: TestClient):
    response = client.post(
        "/api/experiments/lens",
        json={
            "prompt": "Explain why the sky is blue.",
            "settings": {"max_new_tokens": 3},
            "top_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["layers"]) == 1 + 4  # embeddings + four decoder layers
    assert body["layers"][0]["module_id"] == "embeddings"
    assert body["layers"][0]["ablation_kl"] is None
    assert body["prompt_tokens"] > 0
    assert body["position"] == body["prompt_tokens"] - 1
    assert len(body["tokens"]) == len(body["layers"][1]["top1_ids"])
    assert body["vocab"]
    for layer in body["layers"][1:]:
        assert layer["ablation_kl"] is not None
        assert math.isfinite(layer["ablation_kl"]) and layer["ablation_kl"] >= 0
        assert layer["ablation_top1"] is not None
        assert len(layer["top_tokens"]) == 3
        assert 0.0 <= layer["top_tokens"][0]["probability"] <= 1.0
        assert str(layer["top_tokens"][0]["token_id"]) in body["vocab"]


def test_lens_selects_an_explicit_position(client: TestClient):
    response = client.post(
        "/api/experiments/lens",
        json={"prompt": "Hello", "settings": {"max_new_tokens": 1}, "position": 0},
    )
    assert response.status_code == 200, response.text
    assert response.json()["position"] == 0


def test_lens_rejects_an_out_of_range_position(client: TestClient):
    response = client.post(
        "/api/experiments/lens",
        json={"prompt": "Hello", "settings": {"max_new_tokens": 1}, "position": 9999},
    )
    assert response.status_code == 400


def test_patch_returns_both_outputs_and_metadata(client: TestClient):
    response = client.post(
        "/api/experiments/patch",
        json={
            "recipient_prompt": "Describe a rainy morning.",
            "donor_prompt": "Describe a sunny morning.",
            "module_id": "model.layers.1",
            "settings": {"max_new_tokens": 2},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["module_id"] == "model.layers.1"
    assert body["position"] == body["prompt_tokens"] - 1
    for key in ("donor_output", "baseline_output", "patched_output"):
        assert isinstance(body[key], str)
    assert isinstance(body["changed"], bool)


def test_patch_rejects_a_module_without_hidden_states(client: TestClient):
    response = client.post(
        "/api/experiments/patch",
        json={
            "recipient_prompt": "Hello",
            "donor_prompt": "Hello",
            "module_id": "model.embed_tokens",
        },
    )
    assert response.status_code == 400
    assert "cannot be captured" in response.json()["detail"]
