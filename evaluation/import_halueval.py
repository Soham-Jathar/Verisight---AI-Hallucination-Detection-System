"""Convert local HaluEval QA data into VeriSight's reproducible JSONL format.

Download the official HaluEval repository yourself, then pass the path to its
``data/qa_data.json`` file. The converter never calls an LLM or search API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "datasets" / "halueval_qa.jsonl"


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := flatten_text(item)))
    if isinstance(value, dict):
        return "\n".join(part for item in value.values() if (part := flatten_text(item)))
    return ""


def build_cases(samples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, sample in enumerate(samples[:limit]):
        knowledge = flatten_text(sample.get("knowledge"))
        question = flatten_text(sample.get("question"))
        right_answer = flatten_text(sample.get("right_answer"))
        hallucinated_answer = flatten_text(sample.get("hallucinated_answer"))
        if not knowledge or not right_answer or not hallucinated_answer:
            continue

        evidence = [{
            "title": "HaluEval QA knowledge",
            "url": f"benchmark://halueval/qa/{index}",
            "snippet": knowledge[:12000],
            "credibility": 1.0,
            "source_quality": "Benchmark evidence",
        }]
        base = {
            "category": "halueval_qa",
            "question": question,
            "evaluation_level": "answer",
            "evidence": evidence,
        }
        cases.extend([
            {**base, "id": f"halueval-qa-{index}-grounded", "answer": right_answer, "expected_status": "supported"},
            {**base, "id": f"halueval-qa-{index}-hallucinated", "answer": hallucinated_answer, "expected_status": "unsupported"},
        ])
    if not cases:
        raise ValueError("No usable QA examples were found in the input file.")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="Import the official HaluEval QA benchmark.")
    parser.add_argument("--input", type=Path, required=True, help="Path to HaluEval data/qa_data.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Converted JSONL output path")
    parser.add_argument("--limit", type=int, default=200, help="Number of source QA samples to convert")
    args = parser.parse_args()

    samples = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        raise ValueError("HaluEval QA input must be a JSON list.")
    cases = build_cases(samples, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    print(f"Converted {len(cases)} answer-level cases to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
