"""Evaluation entry point for VISTA-SD and its correctness baseline."""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from utils.utils import clip_input_video, load_data, load_model


def parse_gpu_ids(value, visible_count):
    result = [item.strip() for item in value.split(",") if item.strip()]
    if any(int(item) < 0 or int(item) >= visible_count for item in result):
        raise ValueError(f"Logical GPU list {value!r} exceeds {visible_count} visible GPUs")
    return result


def scalar(value):
    return value.item() if torch.is_tensor(value) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--draft_model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--task", default="VideoDetailCaption")
    parser.add_argument("--frame_num", type=int, default=8)
    parser.add_argument("--evaluation_num", type=int, default=1)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--data_num", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--gpu_ids", default="0,1")
    parser.add_argument("--target_gpu_ids", default="0")
    parser.add_argument("--draft_gpu_ids", default="1")
    parser.add_argument("--anchor_ratio", type=float, default=0.1)
    parser.add_argument("--anchor_ema", type=float, default=0.8)
    parser.add_argument(
        "--route_mode",
        choices=["implicit_only", "anchor_only", "gated"],
        default="anchor_only",
    )
    parser.add_argument(
        "--anchor_mode", choices=["static", "dynamic"], default="static"
    )
    parser.add_argument("--refresh_interval", type=int, default=4)
    parser.add_argument("--anchor_attn_layers", type=int, default=4)
    parser.add_argument("--gate_margin_threshold", type=float, default=0.0)
    parser.add_argument("--failure_accept_threshold", type=int, default=1)
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--skip_ar", action="store_true")
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    visible_count = len(args.gpu_ids.split(","))
    target_gpus = parse_gpu_ids(args.target_gpu_ids, visible_count)
    draft_gpus = parse_gpu_ids(args.draft_gpu_ids, visible_count)

    from decoding.tree_decoding_qwen2_5 import AR_generate
    from decoding.vista_decoding_qwen2_5 import VISTA_generate

    model, draft_model, processor, video_token_id = load_model(
        "qwen2_5_vl",
        args.base_model_path,
        args.draft_model_path,
        target_gpus=target_gpus,
        draft_gpus=draft_gpus,
    )
    model.eval()
    draft_model.eval()
    dataset = load_data(args.task, args.data_num, args.data_path)
    sample_count = min(
        args.evaluation_num,
        max(0, len(dataset) - args.sample_offset),
    )
    records = []

    for sample_index in tqdm(
        range(args.sample_offset, args.sample_offset + sample_count)
    ):
        instance = dataset[sample_index]
        ar_output = None
        prompt_length = None
        if not args.skip_ar:
            ar_inputs = clip_input_video(
                processor,
                args.task,
                instance,
                frame_num=args.frame_num,
                model_type="qwen2_5_vl",
                data_path=args.data_path,
                min_pixels=args.min_pixels,
                max_pixels=args.max_pixels,
            )
            prompt_length = ar_inputs["input_ids"].shape[1]
            ar_output = AR_generate(
                ar_inputs,
                model,
                max_new_tokens=args.max_new_tokens,
                video_token_id=video_token_id,
                processor=processor,
            )
            records.append({
                "sample_index": sample_index,
                "method": "ar",
                "decoding_time": float(scalar(ar_output["decoding_time"])),
                "generate_len": int(scalar(ar_output["generate_len"])),
            })

        vista_inputs = clip_input_video(
            processor,
            args.task,
            instance,
            frame_num=args.frame_num,
            model_type="qwen2_5_vl",
            data_path=args.data_path,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
        if prompt_length is None:
            prompt_length = vista_inputs["input_ids"].shape[1]
        vista_output = VISTA_generate(
            vista_inputs,
            model,
            draft_model,
            processor,
            max_new_tokens=args.max_new_tokens,
            video_token_id=video_token_id,
            anchor_ratio=args.anchor_ratio,
            anchor_ema=args.anchor_ema,
            route_mode=args.route_mode,
            anchor_mode=args.anchor_mode,
            refresh_interval=args.refresh_interval,
            gate_margin_threshold=args.gate_margin_threshold,
            failure_accept_threshold=args.failure_accept_threshold,
            anchor_attn_layers=args.anchor_attn_layers,
        )
        vista_record = {
            "sample_index": sample_index,
            "method": "vista_sd",
            "decoding_time": float(scalar(vista_output["decoding_time"])),
            "inference_time": float(scalar(vista_output["inference_time"])),
            "prefill_time": float(scalar(vista_output["prefill_time"])),
            "draft_time": float(scalar(vista_output["draft_time"])),
            "verification_time": float(scalar(vista_output["verification_time"])),
            "generate_len": int(scalar(vista_output["generate_len"])),
            "mean_accept_length": float(scalar(vista_output["mean_accept_length"])),
            "route_counts": vista_output["route_counts"],
            "anchor_refresh_time": vista_output["anchor_refresh_time"],
            "mean_gate_margin": vista_output["mean_gate_margin"],
            "gate_margins": vista_output["gate_margins"],
            "visual_token_count": vista_output["visual_token_count"],
            "anchor_budget": vista_output["anchor_budget"],
            "anchor_refresh_count": vista_output["anchor_refresh_count"],
            "anchor_mean_jaccard": vista_output["anchor_mean_jaccard"],
            "peak_memory_bytes": int(vista_output["peak_memory_bytes"]),
        }
        if ar_output is not None:
            ar_tokens = ar_output["output_ids"][0, prompt_length:prompt_length + args.max_new_tokens]
            vista_tokens = vista_output["output_ids"][0, prompt_length:prompt_length + args.max_new_tokens]
            vista_record["exact_match_ar"] = bool(
                ar_tokens.shape == vista_tokens.shape
                and torch.equal(ar_tokens.cpu(), vista_tokens.cpu())
            )
        records.append(vista_record)
        print(json.dumps(vista_record, ensure_ascii=False, indent=2))

    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
