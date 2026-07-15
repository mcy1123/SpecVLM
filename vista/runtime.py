from __future__ import annotations

from typing import Iterable
import math

import torch


def gather_visual_kv(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    selection,
):
    """Gather selected visual KV positions and preserve every non-visual KV."""
    if selection is None:
        return key_states, value_states, attention_mask, None
    cache_length = key_states.shape[-2]
    prefix_end = min(selection["prefix_end"], cache_length)
    prefix_keep = selection["prefix_keep_positions"].to(key_states.device)
    prefix_keep = prefix_keep[prefix_keep < prefix_end]
    suffix_keep = torch.arange(prefix_end, cache_length, device=key_states.device)
    keep_indices = torch.cat([prefix_keep, suffix_keep])
    key_states = key_states.index_select(2, keep_indices)
    value_states = value_states.index_select(2, keep_indices)
    if attention_mask is not None:
        attention_mask = attention_mask.index_select(
            -1, keep_indices.to(attention_mask.device)
        )
    return key_states, value_states, attention_mask, keep_indices


def set_visual_selection(
    model,
    visual_positions: torch.Tensor,
    selected_visual_ordinals: Iterable[int] | torch.Tensor,
) -> None:
    """Enable sparse visual KV reads on every language attention layer."""
    visual_positions = torch.as_tensor(visual_positions, dtype=torch.long).cpu()
    selected_ordinals = torch.as_tensor(
        list(selected_visual_ordinals)
        if not isinstance(selected_visual_ordinals, torch.Tensor)
        else selected_visual_ordinals.detach().cpu(),
        dtype=torch.long,
    )
    if selected_ordinals.numel():
        if selected_ordinals.min() < 0 or selected_ordinals.max() >= visual_positions.numel():
            raise IndexError("Selected visual ordinal is outside the visual-token list")
        selected_positions = visual_positions[selected_ordinals]
    else:
        selected_positions = torch.empty(0, dtype=torch.long)

    prefix_end = int(visual_positions.max().item()) + 1
    prefix_mask = torch.ones(prefix_end, dtype=torch.bool)
    prefix_mask[visual_positions] = False
    prefix_mask[selected_positions] = True
    prefix_keep_positions = torch.where(prefix_mask)[0]

    for layer in model.model.layers:
        layer.self_attn.vista_visual_selection = {
            "visual_positions": visual_positions,
            "selected_positions": selected_positions,
            "prefix_end": prefix_end,
            "prefix_keep_positions": prefix_keep_positions,
        }


def clear_visual_selection(model) -> None:
    for layer in model.model.layers:
        layer.self_attn.vista_visual_selection = None


def start_score_collection(model, visual_positions: torch.Tensor, last_n_layers: int) -> None:
    if last_n_layers <= 0:
        raise ValueError("last_n_layers must be positive")
    layers = model.model.layers
    first_collected_layer = max(0, len(layers) - last_n_layers)
    positions = torch.as_tensor(visual_positions, dtype=torch.long).cpu()
    for layer_index, layer in enumerate(layers):
        attention = layer.self_attn
        attention.vista_collector_state = None
        attention.vista_collect_visual_positions = (
            positions if layer_index >= first_collected_layer else None
        )


def clear_score_collection(model) -> None:
    for layer in model.model.layers:
        attention = layer.self_attn
        attention.vista_collect_visual_positions = None
        attention.vista_collector_state = None


def finish_score_collection(model, boundary_index: int) -> torch.Tensor:
    """Compute reduced visual relevance for one verified tree boundary."""
    layer_scores = []
    try:
        for layer in model.model.layers:
            state = getattr(layer.self_attn, "vista_collector_state", None)
            if state is None:
                continue
            query_states = state["query_states"]
            key_states = state["key_states"]
            visual_positions = state["visual_positions"].to(key_states.device)
            if boundary_index < 0 or boundary_index >= query_states.shape[-2]:
                raise IndexError(
                    f"Tree boundary {boundary_index} exceeds query length "
                    f"{query_states.shape[-2]}"
                )
            query = query_states[:, :, boundary_index:boundary_index + 1, :]
            visual_keys = key_states.index_select(2, visual_positions)
            groups = query.shape[1] // visual_keys.shape[1]
            visual_keys = visual_keys.repeat_interleave(groups, dim=1)
            scores = torch.matmul(query, visual_keys.transpose(-2, -1))
            scores = torch.softmax(scores.float() / math.sqrt(query.shape[-1]), dim=-1)
            scores = scores.mean(dim=1).squeeze(0).squeeze(0).cpu()
            layer_scores.append(scores)
        if not layer_scores:
            raise RuntimeError("No verifier layers produced VISTA collector state")
        return torch.stack(layer_scores).mean(dim=0)
    finally:
        clear_score_collection(model)
