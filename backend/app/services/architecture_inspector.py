"""Discover the real module hierarchy of the loaded model.

Nothing here is hardcoded to a layer count: every node comes from
``model.named_modules()`` and every block from the model's own ``config``.
"""

from __future__ import annotations

import re
from typing import Any

from backend.app.schemas.api import (
    ArchitectureResponse,
    LayerBlock,
    ModuleDetail,
    ModuleNode,
    ParameterInfo,
)

ROOT_ID = "root"
_LAYER_ID = re.compile(r"^model\.layers\.(\d+)$")

STEERABLE_CLASSES = {
    "Lfm3DecoderLayer": "Decoder-layer output: [batch, seq, hidden_size].",
    "Lfm3Attention": "Attention output: [batch, seq, hidden_size].",
    "Lfm3ShortConv": "Short-conv output: [batch, seq, hidden_size].",
    "Lfm3MLP": "Feed-forward output: [batch, seq, hidden_size].",
}


def _module_id(name: str) -> str:
    return ROOT_ID if name == "" else name


def _parent_id(module_id: str) -> str | None:
    if module_id == ROOT_ID:
        return None
    if "." not in module_id:
        return ROOT_ID
    return module_id.rsplit(".", 1)[0]


def _text_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "get_text_config"):
        return config.get_text_config()
    return config


def _parameter_count(module: Any, *, recurse: bool = True) -> int:
    return sum(p.numel() for p in module.parameters(recurse=recurse))


def _layer_type_for(module_id: str, layer_types: list[str]) -> str | None:
    match = _LAYER_ID.match(module_id)
    if match is None:
        return None
    index = int(match.group(1))
    if 0 <= index < len(layer_types):
        return layer_types[index]
    return None


def _steering_support(class_name: str) -> tuple[bool, str | None]:
    reason = STEERABLE_CLASSES.get(class_name)
    if reason is not None:
        return True, reason
    return (
        False,
        "Only modules whose forward output is the hidden-state tensor [batch, seq, hidden_size] are exposed.",
    )


def get_module(model: Any, module_id: str) -> Any:
    if module_id == ROOT_ID:
        return model
    try:
        return model.get_submodule(module_id)
    except AttributeError as exc:
        raise KeyError(f"Unknown module id {module_id!r}") from exc


def build_tree(model: Any) -> ModuleNode:
    layer_types = list(getattr(_text_config(model), "layer_types", []) or [])
    nodes: dict[str, ModuleNode] = {}
    children: dict[str, list[str]] = {}

    for name, module in model.named_modules():
        node_id = _module_id(name)
        class_name = type(module).__name__
        supports_steering, reason = _steering_support(class_name)
        nodes[node_id] = ModuleNode(
            id=node_id,
            name=class_name if node_id == ROOT_ID else node_id.rsplit(".", 1)[-1],
            class_name=class_name,
            parameter_count=_parameter_count(module),
            child_count=sum(1 for _ in module.children()),
            layer_type=_layer_type_for(node_id, layer_types),
            supports_steering=supports_steering,
            steering_reason=reason,
        )
        parent = _parent_id(node_id)
        if parent is not None:
            children.setdefault(parent, []).append(node_id)

    def attach(node_id: str) -> ModuleNode:
        node = nodes[node_id]
        node.children = [attach(child_id) for child_id in children.get(node_id, [])]
        return node

    return attach(ROOT_ID)


def build_layer_overview(model: Any) -> list[LayerBlock]:
    config = _text_config(model)
    layer_types = list(getattr(config, "layer_types", []) or [])
    try:
        layers = model.model.layers
    except AttributeError:
        return []

    blocks: list[LayerBlock] = []
    for index, layer in enumerate(layers):
        layer_id = f"model.layers.{index}"
        layer_type = layer_types[index] if index < len(layer_types) else "unknown"
        if hasattr(layer, "self_attn"):
            operator = layer.self_attn
            operator_type = "attention"
            operator_name = "self_attn"
        elif hasattr(layer, "conv"):
            operator = layer.conv
            operator_type = "conv"
            operator_name = "conv"
        else:
            continue
        blocks.append(
            LayerBlock(
                layer_index=index,
                module_id=layer_id,
                layer_type=layer_type,
                operator_type=operator_type,
                operator_module_id=f"{layer_id}.{operator_name}",
                operator_class=type(operator).__name__,
                ffn_module_id=f"{layer_id}.feed_forward",
                parameter_count=_parameter_count(layer),
            )
        )
    return blocks


def describe_module(model: Any, module_id: str) -> ModuleDetail:
    module = get_module(model, module_id)
    config = _text_config(model)
    layer_types = list(getattr(config, "layer_types", []) or [])
    class_name = type(module).__name__
    supports_steering, reason = _steering_support(class_name)
    parameters = [
        ParameterInfo(
            name=name,
            shape=list(param.shape),
            dtype=str(param.dtype).replace("torch.", ""),
            num_parameters=param.numel(),
        )
        for name, param in module.named_parameters(recurse=False)
    ]
    return ModuleDetail(
        id=module_id,
        name=class_name if module_id == ROOT_ID else module_id.rsplit(".", 1)[-1],
        class_name=class_name,
        parent_id=_parent_id(module_id),
        parameter_count=_parameter_count(module),
        direct_parameter_count=_parameter_count(module, recurse=False),
        layer_type=_layer_type_for(module_id, layer_types),
        supports_steering=supports_steering,
        steering_reason=reason,
        child_ids=[
            f"{module_id}.{name}" if module_id != ROOT_ID else name for name, _ in module.named_children()
        ],
        parameters=parameters,
    )


def build_architecture(
    model: Any,
    *,
    model_id: str,
    revision: str,
) -> ArchitectureResponse:
    config = _text_config(model)
    layer_types = list(getattr(config, "layer_types", []) or [])
    return ArchitectureResponse(
        model_id=model_id,
        revision=revision,
        model_class=type(model).__name__,
        num_hidden_layers=int(getattr(config, "num_hidden_layers", len(layer_types))),
        hidden_size=int(getattr(config, "hidden_size", 0)),
        vocab_size=int(getattr(config, "vocab_size", 0)),
        layer_types=layer_types,
        tree=build_tree(model),
        layer_overview=build_layer_overview(model),
    )
