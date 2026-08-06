"""Run a reproducible, local benchmark of VeriSight claim verification.

This script deliberately evaluates supplied claim/evidence pairs. It does not
make LLM, search, or Tavily requests, so runs are repeatable and cost-free.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.schemas import EvidenceSource  # noqa: E402
from app.services.verifier import verify_claims  # noqa: E402
from metrics import answer_risk_metrics, classification_metrics  # noqa: E402


DEFAULT_DATASET = ROOT / "evaluation" / "datasets" / "starter_claims.jsonl"
DEFAULT_RESULTS = ROOT / "evaluation" / "results"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        required = {"id", "expected_status", "evidence"}
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Dataset line {line_number} is missing: {', '.join(sorted(missing))}")
        if not item.get("claim") and not item.get("answer"):
            raise ValueError(f"Dataset line {line_number} needs either a claim or an answer.")
        cases.append(item)
    if not cases:
        raise ValueError("The evaluation dataset has no cases.")
    return cases


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    evidence = [EvidenceSource.model_validate(source) for source in case["evidence"]]
    answer = case.get("answer") or case["claim"]
    started = time.perf_counter()
    assessments = verify_claims(answer, evidence, question=case.get("question", ""))
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    if case.get("evaluation_level") == "answer":
        # A generated answer is hallucinated if even one extracted factual claim
        # is contradicted. Otherwise retain an uncertain result when evidence is
        # incomplete rather than calling the whole answer supported.
        if not assessments:
            # Keep the three-class benchmark metric conservative: an answer
            # with no factual assertion is not accepted as evidence-grounded.
            # The product itself reports this as "not applicable", without a
            # reliability score or hallucination label.
            predicted_status = "uncertain"
            confidence = 0.0
            rationale = "No externally verifiable factual claim was extracted from this response."
            evidence_quality = 0.0
            source_agreement = 0.0
        else:
            statuses = [assessment.status for assessment in assessments]
            predicted_status = (
                "unsupported" if "unsupported" in statuses
                else "uncertain" if "uncertain" in statuses
                else "supported"
            )
            confidence = round(sum(assessment.confidence for assessment in assessments) / len(assessments), 2)
            rationale = f"Answer-level result aggregated from {len(assessments)} extracted claim(s)."
            evidence_quality = round(sum((assessment.evidence_quality or 0) for assessment in assessments) / len(assessments), 2)
            source_agreement = round(sum((assessment.source_agreement or 0) for assessment in assessments) / len(assessments), 2)
    else:
        assessment = assessments[0]
        predicted_status = assessment.status
        confidence = assessment.confidence
        rationale = assessment.rationale
        evidence_quality = assessment.evidence_quality
        source_agreement = assessment.source_agreement
    return {
        "id": case["id"],
        "category": case.get("category", "general"),
        "evaluation_level": case.get("evaluation_level", "claim"),
        "claim": case.get("claim") or answer,
        "expected_status": case["expected_status"],
        "predicted_status": predicted_status,
        "confidence": confidence,
        "evidence_quality": evidence_quality,
        "source_agreement": source_agreement,
        "latency_ms": latency_ms,
        "correct": predicted_status == case["expected_status"],
        "rationale": rationale,
        "source_count": len(evidence),
        "claim_count": len(assessments),
    }


def write_reports(results: list[dict[str, Any]], output_dir: Path, dataset: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / run_id
    output_path.mkdir()
    metrics = classification_metrics(
        (item["expected_status"] for item in results),
        (item["predicted_status"] for item in results),
    )
    metrics["average_latency_ms"] = round(sum(item["latency_ms"] for item in results) / len(results), 2)
    if any(item["evaluation_level"] == "answer" for item in results):
        metrics["answer_risk_detection"] = answer_risk_metrics(
            (item["expected_status"] for item in results),
            (item["predicted_status"] for item in results),
        )
    metrics["dataset"] = str(dataset.relative_to(ROOT)) if dataset.is_relative_to(ROOT) else str(dataset)
    metrics["generated_at"] = datetime.now(UTC).isoformat()

    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_path / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (output_path / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        fields = list(results[0].keys())
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the VeriSight claim verifier.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSONL benchmark file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS, help="Directory for JSON and CSV reports.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases (0 means all).")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[: args.limit]
    results = [evaluate_case(case) for case in cases]
    report_path = write_reports(results, args.output_dir, args.dataset)
    metrics = classification_metrics(
        (item["expected_status"] for item in results),
        (item["predicted_status"] for item in results),
    )
    if any(item["evaluation_level"] == "answer" for item in results):
        metrics["answer_risk_detection"] = answer_risk_metrics(
            (item["expected_status"] for item in results),
            (item["predicted_status"] for item in results),
        )
    average_latency = sum(item["latency_ms"] for item in results) / len(results)
    print(f"Evaluated {len(results)} claims")
    print(f"Accuracy: {metrics['accuracy']:.2%} | Macro F1: {metrics['macro_f1']:.2%}")
    if risk_metrics := metrics.get("answer_risk_detection"):
        print(
            "Hallucination-risk detection: "
            f"precision {risk_metrics['precision']:.2%} | "
            f"recall {risk_metrics['recall']:.2%} | F1 {risk_metrics['f1']:.2%}"
        )
    print(f"Average verification latency: {average_latency:.2f} ms")
    print(f"Reports written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
