#!/usr/bin/env bash
set -euo pipefail

for frames in 64 128; do
    FRAME_NUM="$frames" bash scripts/run_baselines_h100.sh
    FRAME_NUM="$frames" bash scripts/run_vista_h100.sh
done

"${PYTHON_BIN:-python}" scripts/summarize_results.py "${RESULT_ROOT:-results/h100}"
