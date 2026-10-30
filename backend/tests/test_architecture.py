"""Architecture discovery comes from the real model object, not a hardcoded list."""

from __future__ import annotations

import pytest

from backend.app.schemas.api import ModuleNode
from backend.app.services.architecture_inspector import (
    build_architecture,
    build_layer_overview,
    build_tree,
    describe_module,
    get_module,
)
from backend.tests.conftest import TINY_CONFIG


def flatten(node: ModuleNode) -> dict[str, ModuleNode]:
    result = {node.id: node}
    for child in node.children:
        result.update(flatten(child))
    return result


def test_tree_root_and_layer_nodes(tiny_model):
    tree = build_tree(tiny_model)
    nodes = flatten(tree)
    assert tree.id == "root"
    assert "model" in nodes
    assert "model.layers" in nodes
    for index, layer_type in enumerate(TINY_CONFIG["layer_types"]):
        node = nodes[f"model.layers.{index}"]
        assert node.class_name == "Lfm3DecoderLayer"
        assert node.layer_type == layer_type
        assert node.supports_steering is True


def test_tree_reflects_actual_children(tiny_model):
    nodes = flatten(build_tree(tiny_model))
    layer0 = nodes["model.layers.0"]
    child_ids = {child.id for child in layer0.children}
    assert "model.layers.0.conv" in child_ids
    assert "model.layers.0.feed_forward" in child_ids
    assert "model.layers.0.operator_norm" in child_ids
    assert nodes["model.layers.0.conv"].supports_steering is True
    assert nodes["model.embed_tokens"].supports_steering is False


def test_layer_overview_matches_config(tiny_model):
    blocks = build_layer_overview(tiny_model)
    assert [block.layer_type for block in blocks] == TINY_CONFIG["layer_types"]
    assert [block.operator_type for block in blocks] == ["conv", "attention", "attention", "conv"]
    assert blocks[1].operator_class == "Lfm3Attention"
    assert blocks[0].operator_class == "Lfm3ShortConv"
    assert all(block.parameter_count > 0 for block in blocks)


def test_describe_module_reports_parameters(tiny_model):
    detail = describe_module(tiny_model, "model.layers.0.feed_forward")
    assert detail.class_name == "Lfm3MLP"
    assert detail.supports_steering is True
    assert detail.direct_parameter_count == 0
    assert detail.parameter_count == 3 * TINY_CONFIG["hidden_size"] * TINY_CONFIG["intermediate_size"]
    assert set(detail.child_ids) == {
        "model.layers.0.feed_forward.w1",
        "model.layers.0.feed_forward.w2",
        "model.layers.0.feed_forward.w3",
    }
    assert detail.parent_id == "model.layers.0"

    projection = describe_module(tiny_model, "model.layers.0.feed_forward.w2")
    assert [param.name for param in projection.parameters] == ["weight"]
    assert projection.direct_parameter_count == projection.parameter_count


def test_unknown_module_raises(tiny_model):
    with pytest.raises(KeyError):
        get_module(tiny_model, "model.layers.99")
    with pytest.raises(KeyError):
        describe_module(tiny_model, "model.layers.99")


def test_architecture_metadata(tiny_model):
    architecture = build_architecture(tiny_model, model_id="synthetic/test-model", revision="test")
    assert architecture.model_id == "synthetic/test-model"
    assert architecture.num_hidden_layers == TINY_CONFIG["num_hidden_layers"]
    assert architecture.hidden_size == TINY_CONFIG["hidden_size"]
    assert architecture.layer_types == TINY_CONFIG["layer_types"]
    assert len(architecture.layer_overview) == TINY_CONFIG["num_hidden_layers"]
