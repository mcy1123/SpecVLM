#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Calculate a frozen verifier-margin threshold from VISTA JSONL"
    )
    parser.add_argument("results", type=Path)
    parser.add_argument("--percentile", type=float, default=30.0)
    parser.add_argument("--value-only", action="store_true")
    args = parser.parse_args()

    margins = []
    with args.results.open(encoding="utf-8") as input_file:
        for line in input_file:
            record = json.loads(line)
            if record.get("method") == "vista_sd":
                margins.extend(record.get("gate_margins", []))
    if not margins:
        raise SystemExit("No gate_margins found in the supplied results")
    threshold = float(np.percentile(np.asarray(margins), args.percentile))
    payload = {
        "samples": len(margins),
        "percentile": args.percentile,
        "gate_margin_threshold": threshold,
    }
    print(threshold if args.value_only else json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
