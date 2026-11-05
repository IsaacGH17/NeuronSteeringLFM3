"""Vector steering: weighted dimensions, sigma normalization, feature stats."""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.services.steering_engine import SteeringSpec, _steered_output, steer_modules


def test_vector_components_apply_their_weights():
    tensor = torch.zeros(1, 2, 8)
    steered, modified = _steered_output(tensor, strength=2.0, components=((1, 1.0), (4, 0.5)))
    assert modified
    assert torch.allclose(steered[..., 1], torch.full((1, 2), 2.0))
    assert torch.allclose(steered[..., 4], torch.full((1, 2), 1.0))
    untouched = [0, 2, 3, 5, 6, 7]
    assert torch.equal(steered[..., untouched], tensor[..., untouched])
    assert torch.equal(tensor, torch.zeros(1, 2, 8))


def test_sigma_normalization_scales_by_dimension_spread():
    tensor = torch.zeros(1, 4, 4)
    tensor[0, :, 2] = torch.tensor([1.0, 3.0, 5.0, 7.0])
    sigma = float(tensor[..., 2].std(unbiased=False))
    steered, _ = _steered_output(tensor, feature_index=2, strength=1.0, normalization="sigma")
    assert torch.allclose(steered[..., 2], tensor[..., 2] + sigma, atol=1e-5)
    assert torch.equal(steered[..., 3], tensor[..., 3])


def test_sigma_mode_skips_constant_dimensions():
    tensor = torch.ones(1, 3, 4)
    skipped: set[int] = set()
    steered, modified = _steered_output(
        tensor,
        feature_index=1,
        strength=2.0,
        normalization="sigma",
        skipped=skipped,
    )
    assert modified
    assert skipped == {1}
    assert torch.equal(steered, tensor)


def test_zero_strength_is_numerically_exact_for_vectors():
    torch.manual_seed(0)
    tensor = torch.randn(1, 5, 6)
    steered, _ = _steered_output(
        tensor, strength=0.0, components=((0, 1.0), (3, -2.0)), normalization="sigma"
    )
    assert torch.equal(steered, tensor)


def test_duplicate_components_are_merged():
    spec = SteeringSpec(
        module_id="model.layers.0",
        feature_index=None,
        strength=1.0,
        components=((5, 0.25), (5, 0.75), (9, 1.0)),
    )
    assert spec.components == ((5, 1.0), (9, 1.0))
    assert spec.feature_index is None


def test_spec_requires_a_direction():
    with pytest.raises(ValueError, match="feature_index or components"):
        SteeringSpec(module_id="model.layers.0", feature_index=None, strength=1.0)


def test_api_accepts_vector_direction_and_sigma(client: TestClient):
    payload = {
        "prompt": "Hello",
        "settings": {"max_new_tokens": 2},
        "steering": {
            "module_id": "model.layers.0.feed_forward",
            "strength": 1.5,
            "normalization": "sigma",
            "components": [
                {"feature_index": 5, "weight": 1.0},
                {"feature_index": 9, "weight": -0.5},
            ],
        },
    }
    response = client.post("/api/experiments/steering", json=payload)
    assert response.status_code == 200, response.text
    steering = response.json()["steered"]["steering"]
    assert steering["normalization"] == "sigma"
    assert steering["feature_index"] is None
    assert [component["feature_index"] for component in steering["components"]] == [5, 9]
    assert steering["components"][1]["weight"] == -0.5


def test_api_rejects_a_direction_with_no_dimensions(client: TestClient):
    payload = {
        "prompt": "Hello",
        "settings": {"max_new_tokens": 2},
        "steering": {"module_id": "model.layers.0.feed_forward", "strength": 1.0},
    }
    assert client.post("/api/experiments/steering", json=payload).status_code == 422


def test_feature_stats_describe_every_dimension(client: TestClient, experiment_store):
    run = client.post(
        "/api/experiments/baseline",
        json={
            "prompt": "Stats please",
            "settings": {"max_new_tokens": 2},
            "capture_module_ids": ["model.layers.0"],
        },
    ).json()
    response = client.get(f"/api/experiments/{run['id']}/activations/model.layers.0")
    assert response.status_code == 200, response.text
    stats = response.json()["feature_stats"]
    assert len(stats["mean"]) == 128
    assert len(stats["std"]) == 128
    assert all(value == value for value in stats["std"])  # no NaN

    tensor = experiment_store.activation_store.get(run["id"]).captures["model.layers.0"]
    expected_std = tensor.std(dim=0, unbiased=False)
    assert stats["std"][:5] == pytest.approx([round(float(v), 6) for v in expected_std[:5]])

def test_sigma_scales_are_calibrated_once_and_reused():
    scales: dict[int, float] = {}
    prefill = torch.zeros(1, 4, 6)
    prefill[0, :, 2] = torch.tensor([1.0, 3.0, 5.0, 7.0])
    steered_prefill, _ = _steered_output(
        prefill, feature_index=2, strength=2.0, normalization="sigma", scales=scales
    )
    sigma = float(prefill[..., 2].std(unbiased=False))
    assert scales == pytest.approx({2: sigma})
    assert torch.allclose(steered_prefill[..., 2], prefill[..., 2] + 2.0 * sigma, atol=1e-5)

    # A decode step carries one token; the cached prefill spread still applies.
    step = torch.zeros(1, 1, 6)
    steered_step, _ = _steered_output(
        step, feature_index=2, strength=2.0, normalization="sigma", scales=scales
    )
    assert steered_step[0, 0, 2].item() == pytest.approx(2.0 * sigma, abs=1e-5)


def test_sigma_mode_steers_single_token_steps_through_the_hook(manager):
    module = manager.model.model.layers[0].feed_forward
    hidden = manager.model.config.get_text_config().hidden_size
    seen: list[torch.Tensor] = []

    def capture(module, args, output):
        seen.append(output.detach().clone())

    prefill = torch.randn(1, 5, hidden)
    step = torch.randn(1, 1, hidden)
    raw_prefill = module(prefill).detach()
    raw_step = module(step).detach()
    spec = SteeringSpec(
        module_id="model.layers.0.feed_forward",
        feature_index=None,
        strength=1.5,
        components=((3, 1.0),),
        normalization="sigma",
    )
    with steer_modules(manager.model, [spec]):
        handle = module.register_forward_hook(capture)
        try:
            module(prefill)
            module(step)
        finally:
            handle.remove()

    sigma = float(raw_prefill[..., 3].std(unbiased=False))
    assert seen[0][0, :, 3] == pytest.approx(raw_prefill[0, :, 3] + 1.5 * sigma, abs=1e-5)
    assert seen[1][0, 0, 3].item() == pytest.approx(raw_step[0, 0, 3].item() + 1.5 * sigma, abs=1e-5)

