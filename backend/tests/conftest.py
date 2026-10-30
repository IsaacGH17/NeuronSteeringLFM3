"""Shared fixtures: a tiny synthetic LFM3 and an offline tokenizer."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app
from backend.app.services.model_loader import ModelManager
from backend.app.vendor.configuration_lfm3 import Lfm3Config
from backend.app.vendor.modeling_lfm3 import Lfm3ForCausalLM

TINY_CONFIG = dict(
    pad_token_id=0,
    vocab_size=256,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
    sliding_window=8,
    conv_L_cache=7,
    layer_types=["conv", "full_attention", "sliding_attention", "conv"],
)


class FakeTokenizer:
    """Minimal tokenizer surface used by the generation service."""

    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
        enable_thinking: bool | None = None,
    ) -> str:
        rendered = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        ids = [1 + (ord(char) % 200) for char in text][:32] or [1]
        input_ids = torch.tensor([ids])
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "decoded:" + ",".join(str(value) for value in values)


@pytest.fixture
def tiny_model() -> Lfm3ForCausalLM:
    torch.manual_seed(0)
    return Lfm3ForCausalLM(Lfm3Config(**TINY_CONFIG)).eval()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def manager(tiny_model: Lfm3ForCausalLM, fake_tokenizer: FakeTokenizer) -> ModelManager:
    instance = ModelManager(Settings(model_id="synthetic/test-model", max_new_tokens=8))
    instance.inject(tiny_model, fake_tokenizer)
    return instance


@pytest.fixture
def client(manager: ModelManager) -> TestClient:
    return TestClient(create_app(manager))
