"""Steering engine: hook arithmetic, lifecycle, and API behavior."""

from __future__ import annotations

import pytest
import torch

from backend.app.schemas.api import (
    BaselineRequest,
    GenerationSettings,
)
from backend.app.schemas.api import (
    SteeringSpec as ApiSteeringSpec,
)
from backend.app.services.generation import generate_baseline, generate_steered
from backend.app.services.steering_engine import (
    SteeringSpec,
    _steered_output,
    steer_modules,
    validate_steering_spec,
)


def _no_hooks(model):
    return all(
        len(module._forward_hooks) == 0 and len(module._forward_pre_hooks) == 0
        for module in model.model.modules()
    )


def test_steer_output_adds_strength_only_to_target_dimension():
    tensor = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    steered, modified = _steered_output(tensor, feature_index=3, strength=2.5)
    assert modified
    # Out-of-place: input untouched.
    assert tensor[0, 1, 3].item() == 9.0
    assert steered[0, 1, 3].item() == 11.5
    for index in range(6):
        if index != 3:
            assert torch.equal(steered[..., index], tensor[..., index])


def test_steer_output_handles_tuple_return():
    tensor = torch.ones(1, 1, 4)
    output = (tensor, tensor.clone())
    steered, modified = _steered_output(output, feature_index=0, strength=5.0)
    assert modified
    assert isinstance(steered, tuple) and len(steered) == 2
    assert steered[0][0, 0, 0].item() == 6.0
    assert torch.equal(steered[1], output[1])
    assert tensor[0, 0, 0].item() == 1.0


def test_steer_output_ignores_non_hidden_tensors():
    assert _steered_output([1, 2, 3], feature_index=0, strength=1.0)[1] is False
    matrix = torch.ones(2, 2)
    assert _steered_output(matrix, feature_index=0, strength=1.0)[1] is False


def test_validate_steering_spec_bounds(manager):
    with pytest.raises(ValueError, match="Unknown steering module"):
        validate_steering_spec(
            manager.model,
            SteeringSpec("model.nonexistent", 0, 1.0),
            strength_limit=50.0,
        )
    with pytest.raises(ValueError, match="cannot be steered"):
        validate_steering_spec(manager.model, SteeringSpec("model.embed_tokens", 0, 1.0), strength_limit=50.0)
    with pytest.raises(ValueError, match="feature_index 128 is outside"):
        validate_steering_spec(
            manager.model,
            SteeringSpec("model.layers.0", 128, 1.0),
            strength_limit=50.0,
        )
    with pytest.raises(ValueError, match="safety limit"):
        validate_steering_spec(
            manager.model,
            SteeringSpec("model.layers.0", 0, 51.0),
            strength_limit=50.0,
        )
    validated = validate_steering_spec(
        manager.model, SteeringSpec("model.layers.0", 127, -3.5), strength_limit=50.0
    )
    assert validated == SteeringSpec("model.layers.0", 127, -3.5)


def test_steering_context_removes_hooks_on_success(manager):
    assert _no_hooks(manager.model)
    with steer_modules(manager.model, [SteeringSpec("model.layers.0", 5, 1.0)]):
        assert not _no_hooks(manager.model)
    assert _no_hooks(manager.model)


def test_steering_context_removes_hooks_after_exception(manager):
    with pytest.raises(RuntimeError, match="boom"):
        with steer_modules(manager.model, [SteeringSpec("model.layers.1", 0, 1.0)]):
            assert not _no_hooks(manager.model)
            raise RuntimeError("boom")
    assert _no_hooks(manager.model)


def test_steering_changes_module_output(manager):
    model = manager.model
    layer = model.model.layers[0]
    hidden_size = model.config.get_text_config().hidden_size
    x = torch.randn(1, 3, hidden_size)

    with steer_modules(model, [SteeringSpec("model.layers.0.feed_forward", 3, 7.0)]):
        steered = layer.feed_forward(x)
    unsteered = layer.feed_forward(x)

    difference = (steered - unsteered).detach()
    assert torch.allclose(difference[..., 3], torch.full_like(difference[..., 3], 7.0))
    assert difference.abs().sum() == pytest.approx(7.0 * difference.shape[0] * difference.shape[1])


def test_zero_strength_matches_baseline(manager):
    request = BaselineRequest(prompt="Hello", settings=GenerationSettings(max_new_tokens=4))
    baseline = generate_baseline(manager, request).response
    steered = generate_steered(manager, request, SteeringSpec("model.layers.0", 7, strength=0.0)).response
    assert steered.output == baseline.output


def test_negative_strength_differs_from_positive(manager):
    request = BaselineRequest(prompt="Hello", settings=GenerationSettings(max_new_tokens=4))
    positive = generate_steered(manager, request, SteeringSpec("model.layers.0", 7, strength=6.0)).response
    negative = generate_steered(manager, request, SteeringSpec("model.layers.0", 7, strength=-6.0)).response
    assert positive.output != negative.output


def test_steering_does_not_change_weights(manager):
    before = {name: param.detach().clone() for name, param in manager.model.named_parameters()}
    generate_steered(
        manager,
        BaselineRequest(prompt="Hello", settings=GenerationSettings(max_new_tokens=3)),
        SteeringSpec("model.layers.1.feed_forward", 10, strength=4.0),
    )
    for name, param in manager.model.named_parameters():
        assert torch.equal(before[name], param), f"parameter {name} changed"


def test_steering_api_pair_and_validation(client):
    payload = {
        "prompt": "Hello",
        "settings": {"max_new_tokens": 3},
        "steering": {"module_id": "model.layers.0.feed_forward", "feature_index": 5, "strength": 2.0},
        "capture_module_ids": ["model.layers.0.feed_forward"],
    }
    response = client.post("/api/experiments/steering", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["baseline"]["condition"] == "baseline"
    assert body["steered"]["condition"] == "steered"
    assert body["steered"]["steering"]["module_id"] == "model.layers.0.feed_forward"

    experiment_id = body["steered"]["id"]
    baseline_id = body["baseline"]["id"]
    diff = client.get(
        f"/api/experiments/{experiment_id}/activations/model.layers.0.feed_forward"
        f"?compare_with={baseline_id}&position=0"
    )
    assert diff.status_code == 200, diff.text
    diff_body = diff.json()
    assert diff_body["comparison"].startswith("difference:")
    # At prompt position 0 the trajectories have not diverged yet, so the
    # difference is exactly the steered dimension.
    top = diff_body["top_dimensions"][0]
    assert top["index"] == 5
    assert top["value"] == pytest.approx(2.0, abs=1e-4)


def test_steering_api_rejects_invalid_specs(client):
    for spec, status in [
        ({"module_id": "model.no", "feature_index": 0, "strength": 1.0}, 400),
        ({"module_id": "model.layers.0", "feature_index": 999, "strength": 1.0}, 400),
        ({"module_id": "model.layers.0", "feature_index": 0, "strength": 2000.0}, 400),
    ]:
        payload = {
            "prompt": "Hello",
            "settings": {"max_new_tokens": 3},
            "steering": spec,
        }
        response = client.post("/api/experiments/steering", json=payload)
        assert response.status_code == status, f"{spec} -> {response.status_code}"


def test_spec_from_api_schema_roundtrip():
    api_spec = ApiSteeringSpec(module_id="model.layers.2", feature_index=3, strength=1.5)
    engine_spec = SteeringSpec(
        module_id=api_spec.module_id, feature_index=api_spec.feature_index, strength=api_spec.strength
    )
    assert engine_spec.feature_index == 3
