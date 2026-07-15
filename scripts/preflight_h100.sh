#!/usr/bin/env bash
set -euo pipefail

: "${TARGET_MODEL_PATH:?Set TARGET_MODEL_PATH to Qwen2.5-VL-32B-Instruct}"
: "${DRAFT_MODEL_PATH:?Set DRAFT_MODEL_PATH to Qwen2.5-VL-7B-Instruct}"
: "${VIDEO_DATA_PATH:?Set VIDEO_DATA_PATH to VideoDetailCaption}"

PYTHON_BIN="${PYTHON_BIN:-python}"

for path in "$TARGET_MODEL_PATH" "$DRAFT_MODEL_PATH" "$VIDEO_DATA_PATH"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
done

"$PYTHON_BIN" - <<'PY'
import importlib
import os
import torch

required = ["transformers", "datasets", "accelerate", "av", "qwen_vl_utils"]
for package in required:
    importlib.import_module(package)

print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"visible_gpus={torch.cuda.device_count()}")
if torch.cuda.device_count() < 4:
    raise SystemExit("VISTA H100 preset requires at least four visible GPUs")
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"gpu[{index}]={props.name} memory={props.total_memory / 2**30:.1f}GiB")
    if "H100" not in props.name and os.environ.get("ALLOW_NON_H100") != "1":
        raise SystemExit(
            "Non-H100 GPU detected. Set ALLOW_NON_H100=1 only for script validation."
        )
PY

echo "H100 preflight passed."
