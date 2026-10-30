"""Generation behavior and the hook-free baseline guarantee."""

from __future__ import annotations

import torch

from backend.app.schemas.api import BaselineRequest, GenerationSettings
from backend.app.services.generation import generate_baseline


def _hook_snapshot(model):
    return {
        name: (len(module._forward_hooks), len(module._forward_pre_hooks), len(module._backward_hooks))
        for name, module in model.named_modules()
    }


def test_baseline_installs_no_hooks(manager):
    before = _hook_snapshot(manager.model)
    response = generate_baseline(
        manager,
        BaselineRequest(prompt="Hello", settings=GenerationSettings(max_new_tokens=3)),
    )
    after = _hook_snapshot(manager.model)
    assert response.generated_tokens == 3
    assert before == after
    assert all(counts == (0, 0, 0) for counts in after.values())


def test_baseline_does_not_change_weights(manager):
    before = {name: param.detach().clone() for name, param in manager.model.named_parameters()}
    generate_baseline(
        manager,
        BaselineRequest(prompt="Hello", settings=GenerationSettings(max_new_tokens=2)),
    )
    for name, param in manager.model.named_parameters():
        assert torch.equal(before[name], param), f"parameter {name} changed"


def test_greedy_is_deterministic(manager):
    request = BaselineRequest(prompt="Determinism", settings=GenerationSettings(max_new_tokens=5))
    first = generate_baseline(manager, request)
    second = generate_baseline(manager, request)
    assert first.output == second.output


def test_system_prompt_reaches_template(manager):
    response = generate_baseline(
        manager,
        BaselineRequest(
            prompt="Question",
            system="Be terse",
            settings=GenerationSettings(max_new_tokens=1),
        ),
    )
    assert response.system == "Be terse"
    assert response.prompt_tokens > 0
