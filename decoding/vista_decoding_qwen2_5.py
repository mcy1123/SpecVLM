"""VISTA-SD decoding for Qwen2.5-VL.

The target verifier remains dense. The draft model performs a full multimodal
prefill, then gathers only the visual KV positions selected by an AnchorBank
during tree drafting.
"""

import time

import torch

from decoding.tree_decoding_qwen2_5 import (
    _move_dict_to_device,
    evaluate_posterior,
    generate_candidates,
    generate_tree_buffers,
    reset_tree_mode,
    tree_decoding,
    tree_draft,
    update_inference_inputs,
)
from kv_cache.kv_cache import initialize_past_key_values
from tree_choices.choices import mc_sim_7b_63
from utils.utils import convert_attention_to_score, get_last_video_idx
from utils.utils_c import generate_tree_buffers_draft
from vista import (
    AnchorBank,
    clear_score_collection,
    clear_visual_selection,
    finish_score_collection,
    set_visual_selection,
    start_score_collection,
)


def _clone_inputs(inputs):
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
    }


@torch.no_grad()
def initialize_vista(
    inputs,
    model,
    draft_model,
    past_key_values,
    draft_past_key_values,
    video_token_id,
):
    full_inputs = _clone_inputs(inputs)
    full_input_ids = full_inputs["input_ids"].clone()
    visual_positions = torch.where(full_input_ids[0] == video_token_id)[0].cpu()
    if visual_positions.numel() == 0:
        raise ValueError("VISTA-SD requires at least one visual token")

    last_video_idx = get_last_video_idx(full_input_ids[0], video_token_id)
    text_input_ids = full_input_ids[:, last_video_idx + 1:].clone()

    target_video_inputs = _clone_inputs(full_inputs)
    target_video_inputs["input_ids"] = full_input_ids[:, :last_video_idx + 1]
    target_video_inputs["attention_mask"] = full_inputs["attention_mask"][:, :last_video_idx + 1]

    model(**target_video_inputs, past_key_values=past_key_values)
    target_text_output = model(
        input_ids=text_input_ids,
        past_key_values=past_key_values,
        output_attentions=True,
    )
    sample_token = torch.argmax(target_text_output.logits[:, -1], dim=-1)[:, None]
    initial_scores = torch.tensor(
        convert_attention_to_score(
            target_text_output.attentions,
            full_input_ids,
            video_token_id,
        ),
        dtype=torch.float32,
    )

    draft_device = draft_model.model.embed_tokens.weight.device
    draft_inputs = _move_dict_to_device(full_inputs, draft_device)
    draft_model(**draft_inputs, past_key_values=draft_past_key_values)

    return sample_token, full_input_ids, visual_positions, initial_scores


def _route_indices(route_mode, anchor_bank):
    if route_mode == "implicit_only":
        return torch.empty(0, dtype=torch.long), "implicit"
    if route_mode in {"anchor_only", "gated"}:
        return anchor_bank.indices, "anchor"
    raise ValueError(f"Unsupported route_mode: {route_mode}")


def _next_route(
    route_mode,
    anchor_bank,
    accept_length,
    sample_p,
    failure_accept_threshold,
    gate_margin_threshold,
):
    if route_mode != "gated":
        indices, route = _route_indices(route_mode, anchor_bank)
        return indices, route, None
    top_two = torch.topk(sample_p.float(), k=2).values
    margin = float((top_two[0] - top_two[1]).item())
    use_anchor = (
        accept_length <= failure_accept_threshold
        or margin < gate_margin_threshold
    )
    if use_anchor:
        return anchor_bank.indices, "anchor", margin
    return torch.empty(0, dtype=torch.long), "implicit", margin


@torch.no_grad()
def VISTA_generate(
    inputs,
    model,
    draft_model,
    processor,
    max_new_tokens=512,
    video_token_id=151656,
    tree_choices=mc_sim_7b_63,
    anchor_ratio=0.1,
    anchor_ema=0.8,
    route_mode="anchor_only",
    anchor_mode="static",
    refresh_interval=4,
    gate_margin_threshold=0.0,
    failure_accept_threshold=1,
    anchor_attn_layers=4,
):
    if anchor_mode not in {"static", "dynamic"}:
        raise ValueError(f"Unsupported anchor_mode: {anchor_mode}")
    torch.cuda.synchronize()
    inference_start = time.time()
    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    clear_visual_selection(draft_model)

    tree_buffers = generate_tree_buffers(
        tree_choices,
        device=model.model.layers[-1].self_attn.q_proj.weight.device,
    )
    tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
        model.lm_head.weight.device
    )
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices
    draft_model.tree_buffer = generate_tree_buffers_draft(
        tree_choices,
        device=draft_model.model.layers[-1].self_attn.q_proj.weight.device,
    )

    past_key_values, past_data, current_length = initialize_past_key_values(model)
    model.model.past_key_values = past_key_values
    model.model.past_key_values_data = past_data
    model.model.current_length_data = current_length

    draft_past, draft_data, draft_length = initialize_past_key_values(draft_model)
    reset_tree_mode(model)
    reset_tree_mode(draft_model)

    input_ids = inputs["input_ids"].clone()
    original_input_len = input_ids.shape[1]
    sample_token, input_ids, visual_positions, initial_scores = initialize_vista(
        inputs,
        model,
        draft_model,
        past_key_values,
        draft_past,
        video_token_id,
    )
    anchor_bank = AnchorBank(
        initial_scores,
        anchor_ratio=anchor_ratio,
        ema=anchor_ema,
    )
    # A gated run starts conservatively with visual anchors.
    selected_ordinals, initial_route = _route_indices(route_mode, anchor_bank)
    set_visual_selection(draft_model, visual_positions, selected_ordinals)

    torch.cuda.synchronize()
    decode_start = time.time()
    prefill_time = decode_start - inference_start
    route_counts = {"implicit": 0, "anchor": 0}
    route_counts[initial_route] += 1
    refresh_time = 0.0
    gate_margins = []
    draft_time = 0.0
    verify_time = 0.0

    try:
        first_id = sample_token.to(input_ids.device)
        torch.cuda.synchronize()
        draft_start = time.time()
        tree_logits = tree_draft(
            first_id,
            draft_model,
            draft_past,
            input_ids.shape[1] + 1,
        )
        torch.cuda.synchronize()
        draft_time += time.time() - draft_start
        model.model.tree_mask = tree_buffers["tree_attn_mask"]

        new_token = 0
        accept_lengths = []
        round_index = 0
        while new_token < max_new_tokens:
            candidates, tree_candidates = generate_candidates(
                tree_logits,
                tree_buffers["tree_indices"],
                tree_buffers["retrieve_indices"],
                sample_token,
                processor,
            )
            if anchor_mode == "dynamic":
                start_score_collection(model, visual_positions, anchor_attn_layers)
            torch.cuda.synchronize()
            verify_start = time.time()
            logits, _ = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                tree_buffers["tree_position_ids"],
                input_ids,
                tree_buffers["retrieve_indices"],
            )
            torch.cuda.synchronize()
            verify_time += time.time() - verify_start
            best_candidate, accept_length, sample_p = evaluate_posterior(logits, candidates)
            accept_lengths.append(accept_length)

            refresh_due = anchor_mode == "dynamic" and (
                (round_index + 1) % refresh_interval == 0
                or accept_length <= failure_accept_threshold
            )
            if refresh_due:
                boundary_index = int(
                    tree_buffers["retrieve_indices"][best_candidate, accept_length].item()
                )
                torch.cuda.synchronize()
                refresh_start = time.time()
                verify_scores = finish_score_collection(model, boundary_index)
                anchor_bank.update(verify_scores)
                torch.cuda.synchronize()
                refresh_time += time.time() - refresh_start
            elif anchor_mode == "dynamic":
                clear_score_collection(model)

            selected_ordinals, next_route, margin = _next_route(
                route_mode,
                anchor_bank,
                accept_length,
                sample_p,
                failure_accept_threshold,
                gate_margin_threshold,
            )
            if margin is not None:
                gate_margins.append(margin)
            set_visual_selection(draft_model, visual_positions, selected_ordinals)

            torch.cuda.synchronize()
            draft_start = time.time()
            input_ids, tree_logits, new_token, _, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                tree_buffers["retrieve_indices"],
                logits,
                tree_logits,
                new_token,
                past_data,
                current_length,
                draft_model,
                draft_past,
                draft_data,
                draft_length,
                sample_p,
            )
            torch.cuda.synchronize()
            draft_time += time.time() - draft_start
            route_counts[next_route] += 1
            round_index += 1

            generated = input_ids[0, original_input_len:]
            if processor.tokenizer.eos_token_id in generated.tolist():
                eos_offset = (generated == processor.tokenizer.eos_token_id).nonzero(
                    as_tuple=True
                )[0][0].item()
                input_ids = input_ids[:, :original_input_len + eos_offset + 1]
                new_token = eos_offset + 1
                break

        if new_token > max_new_tokens:
            input_ids = input_ids[:, :original_input_len + max_new_tokens]
            new_token = max_new_tokens

        torch.cuda.synchronize()
        end = time.time()
        metrics = anchor_bank.metrics()
        peak_memory = max(
            torch.cuda.max_memory_allocated(device_index)
            for device_index in range(torch.cuda.device_count())
        )
        return {
            "output_ids": input_ids,
            "inference_time": end - inference_start,
            "decoding_time": end - decode_start,
            "prefill_time": prefill_time,
            "draft_time": draft_time,
            "verification_time": verify_time,
            "mean_accept_length": (
                sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0
            ),
            "generate_len": new_token,
            "route_counts": route_counts,
            "anchor_refresh_time": refresh_time,
            "mean_gate_margin": (
                sum(gate_margins) / len(gate_margins) if gate_margins else None
            ),
            "gate_margins": gate_margins,
            "peak_memory_bytes": peak_memory,
            **metrics,
        }
    finally:
        clear_score_collection(model)
        clear_visual_selection(draft_model)
        reset_tree_mode(model)
        reset_tree_mode(draft_model)
