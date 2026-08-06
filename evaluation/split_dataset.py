"""Create reproducible development and held-out splits from evaluation JSONL."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evaluation" / "datasets"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_key(case: dict[str, Any]) -> str:
    """Keep HaluEval grounded/hallucinated answer pairs together."""
    identifier = str(case["id"])
    for suffix in ("-grounded", "-hallucinated"):
        if identifier.endswith(suffix):
            return identifier.removesuffix(suffix)
    return identifier


def write_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Split benchmark cases into development and held-out test sets.")
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL benchmark")
    parser.add_argument("--dev-output", type=Path, default=DATASETS / "halueval_dev.jsonl")
    parser.add_argument("--test-output", type=Path, default=DATASETS / "halueval_test.jsonl")
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.dev_ratio < 1:
        raise ValueError("--dev-ratio must be between 0 and 1.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in load_jsonl(args.input):
        groups[group_key(case)].append(case)
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    split_index = round(len(keys) * args.dev_ratio)
    dev_cases = [case for key in keys[:split_index] for case in groups[key]]
    test_cases = [case for key in keys[split_index:] for case in groups[key]]
    write_jsonl(args.dev_output, dev_cases)
    write_jsonl(args.test_output, test_cases)
    print(f"Development: {len(dev_cases)} cases ({split_index} source groups) -> {args.dev_output}")
    print(f"Held-out test: {len(test_cases)} cases ({len(keys) - split_index} source groups) -> {args.test_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
