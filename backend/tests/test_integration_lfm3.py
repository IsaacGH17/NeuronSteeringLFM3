"""Optional integration test against the real LiquidAI/LFM3-1.7B checkpoint.

Skipped unless ``LFM3_RUN_INTEGRATION=1``; the weights must already be cached
(or the machine must have network access and ~3.5 GB free disk space).
"""

from __future__ import annotations

import os

import pytest

from backend.app.core.settings import Settings
from backend.app.schemas.api import BaselineRequest, GenerationSettings
from backend.app.services.architecture_inspector import build_architecture
from backend.app.services.generation import generate_baseline
from backend.app.services.model_loader import ModelManager

pytestmark = pytest.mark.skipif(
    os.environ.get("LFM3_RUN_INTEGRATION") != "1",
    reason="set LFM3_RUN_INTEGRATION=1 to load the real 1.7B checkpoint",
)


@pytest.fixture(scope="module")
def real_manager() -> ModelManager:
    manager = ModelManager(Settings(device=os.environ.get("LFM3_DEVICE", "auto"), max_new_tokens=16))
    manager.load()
    return manager


def test_real_architecture(real_manager: ModelManager):
    status = real_manager.status()
    architecture = build_architecture(
        real_manager.model,
        model_id=status.model_id,
        revision=status.revision,
    )
    assert architecture.model_class == "Lfm3ForCausalLM"
    assert architecture.num_hidden_layers == 24
    assert architecture.hidden_size == 2048
    assert architecture.vocab_size == 128_000
    assert len(architecture.layer_overview) == 24
    assert architecture.layer_types[2] == "full_attention"
    assert architecture.layer_types[4] == "sliding_attention"
    assert architecture.layer_types[0] == "conv"
    assert architecture.layer_overview[0].operator_class == "Lfm3ShortConv"
    assert architecture.layer_overview[2].operator_class == "Lfm3Attention"


def test_real_baseline_generation(real_manager: ModelManager):
    response = generate_baseline(
        real_manager,
        BaselineRequest(
            prompt="Reply with exactly the word: ready",
            settings=GenerationSettings(max_new_tokens=8),
        ),
    )
    assert response.prompt_tokens > 0
    assert response.generated_tokens > 0
    assert response.output.strip() != ""
    assert response.device == real_manager.device
