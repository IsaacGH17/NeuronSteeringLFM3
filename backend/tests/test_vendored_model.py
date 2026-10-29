"""Offline smoke tests against the vendored LFM3 code — no model download required."""

import torch
from transformers.utils import is_torch_available

from backend.app.vendor.configuration_lfm3 import Lfm3Config
from backend.app.vendor.modeling_lfm3 import Lfm3ForCausalLM

TINY_CONFIG = dict(
    pad_token_id=None,
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


def tiny_model() -> Lfm3ForCausalLM:
    config = Lfm3Config(**TINY_CONFIG)
    torch.manual_seed(0)
    return Lfm3ForCausalLM(config).eval()


def test_transformers_torch_available():
    assert is_torch_available()


def test_forward_shapes():
    model = tiny_model()
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 12))
    with torch.no_grad():
        out = model(input_ids=ids)
    assert out.logits.shape == (1, 12, TINY_CONFIG["vocab_size"])


def test_generate_runs():
    model = tiny_model()
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 6))
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False, use_cache=True)
    assert out.shape == (1, 11)


def test_steering_hook_changes_output():
    """Steering a decoder-layer output dimension outside prefill hooks affects logits."""
    model = tiny_model()
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 8))
    target = model.model.layers[2]

    def add_alpha(module, args, output):
        h = output.clone()
        h[..., 0] = h[..., 0] + 50.0
        return h

    with torch.no_grad():
        base = model(input_ids=ids).logits
        handle = target.register_forward_hook(add_alpha)
        steered = model(input_ids=ids).logits
        handle.remove()
        restored = model(input_ids=ids).logits

    assert not torch.allclose(base, steered)
    assert torch.allclose(base, restored)


def test_weights_unchanged_by_steering():
    """Steering must not mutate persistent parameters."""

    model = tiny_model()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    ids = torch.randint(0, TINY_CONFIG["vocab_size"], (1, 8))
    handle = model.model.layers[1].register_forward_hook(lambda m, a, o: o.clone() + 1.0)
    with torch.no_grad():
        model(input_ids=ids)
    handle.remove()
    for name, p in model.named_parameters():
        assert torch.equal(before[name], p), f"parameter {name} was mutated"
