#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in args.root.rglob("results.jsonl"):
        with path.open(encoding="utf-8") as input_file:
            for line in input_file:
                record = json.loads(line)
                record["source"] = str(path.parent.relative_to(args.root))
                grouped[(record["source"], record["method"])].append(record)

    summary = []
    for (source, method), records in sorted(grouped.items()):
        times = [float(record["decoding_time"]) for record in records]
        lengths = [int(record["generate_len"]) for record in records]
        total_time = sum(times)
        row = {
            "source": source,
            "method": method,
            "samples": len(records),
            "mean_decoding_time": statistics.mean(times),
            "median_decoding_time": statistics.median(times),
            "tokens_per_second": sum(lengths) / total_time if total_time else None,
        }
        accept = [record["mean_accept_length"] for record in records if "mean_accept_length" in record]
        exact = [record["exact_match_ar"] for record in records if "exact_match_ar" in record]
        if accept:
            row["mean_accept_length"] = statistics.mean(accept)
        if exact:
            row["exact_match_rate"] = sum(exact) / len(exact)
        summary.append(row)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    output_path = args.root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
