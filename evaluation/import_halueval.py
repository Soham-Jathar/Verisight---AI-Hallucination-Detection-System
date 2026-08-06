"""Convert local HaluEval task data into VeriSight's reproducible JSONL format.

Download the official HaluEval repository yourself, then pass the path to its
``data/qa_data.json``, ``data/dialogue_data.json``, or
``data/summarization_data.json`` file. The converter never calls an LLM or
search API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "evaluation" / "datasets"


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(part for item in value if (part := flatten_text(item)))
    if isinstance(value, dict):
        return "\n".join(part for item in value.values() if (part := flatten_text(item)))
    return ""


def build_cases(
    samples: list[dict[str, Any]],
    task: str,
    limit: int,
    offset: int = 0,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, sample in enumerate(samples[offset:offset + limit], start=offset):
        if task == "qa":
            knowledge = flatten_text(sample.get("knowledge"))
            context = flatten_text(sample.get("question"))
            grounded_output = flatten_text(sample.get("right_answer"))
            hallucinated_output = flatten_text(sample.get("hallucinated_answer"))
            context_field = {"question": context}
            evidence_title = "HaluEval QA knowledge"
            snippet_limit = 12_000
        elif task == "dialogue":
            # Preserve the full turn history. It is useful when manually
            # reviewing any dialogue-verification failure in predictions.json.
            knowledge = flatten_text(sample.get("knowledge"))
            context = flatten_text(sample.get("dialogue_history"))
            grounded_output = flatten_text(sample.get("right_response"))
            hallucinated_output = flatten_text(sample.get("hallucinated_response"))
            context_field = {
                "question": context,
                "conversation_history": context,
            }
            evidence_title = "HaluEval dialogue knowledge"
            snippet_limit = 12_000
        else:
            # Summaries should be checked against the source document itself,
            # not a web search result. Preserve enough of the document for the
            # claim-focused NLI excerpt selector to find relevant sentences.
            knowledge = flatten_text(sample.get("document"))
            context = "Summarize the supplied document faithfully."
            grounded_output = flatten_text(sample.get("right_summary"))
            hallucinated_output = flatten_text(sample.get("hallucinated_summary"))
            context_field = {"question": context, "document_context": knowledge[:40_000]}
            evidence_title = "HaluEval summarization document"
            snippet_limit = 40_000

        if not knowledge or not context or not grounded_output or not hallucinated_output:
            continue

        evidence = [{
            "title": evidence_title,
            "url": f"benchmark://halueval/{task}/{index}",
            "snippet": knowledge[:snippet_limit],
            "credibility": 1.0,
            "source_quality": "Benchmark evidence",
        }]
        base = {
            "category": f"halueval_{task}",
            "evaluation_level": "answer",
            "evidence": evidence,
            **context_field,
        }
        cases.extend([
            {**base, "id": f"halueval-{task}-{index}-grounded", "answer": grounded_output, "expected_status": "supported"},
            {**base, "id": f"halueval-{task}-{index}-hallucinated", "answer": hallucinated_output, "expected_status": "unsupported"},
        ])
    if not cases:
        raise ValueError(f"No usable {task} examples were found in the input file.")
    return cases


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Accept both historical JSONL releases and JSON-list mirrors."""
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(parsed, list):
        raise ValueError("HaluEval input must be a JSON list or JSONL file.")
    if not all(isinstance(sample, dict) for sample in parsed):
        raise ValueError("Every HaluEval QA example must be a JSON object.")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Import an official HaluEval benchmark task.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a HaluEval data JSON or JSONL file")
    parser.add_argument("--task", choices=("qa", "dialogue", "summarization"), default="qa", help="Structure of the supplied HaluEval file")
    parser.add_argument("--output", type=Path, help="Converted JSONL output path")
    parser.add_argument("--limit", type=int, default=200, help="Number of source task examples to convert")
    parser.add_argument("--offset", type=int, default=0, help="Number of source task samples to skip before conversion")
    args = parser.parse_args()

    samples = load_samples(args.input)
    if args.offset < 0:
        raise ValueError("--offset cannot be negative.")
    output = args.output or DATASETS / f"halueval_{args.task}.jsonl"
    cases = build_cases(samples, args.task, args.limit, args.offset)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    print(f"Converted {len(cases)} {args.task} answer-level cases to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
