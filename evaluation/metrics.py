"""Metrics used by the reproducible verification benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


LABELS = ("supported", "unsupported", "uncertain")


def safe_divide(numerator: float, denominator: float) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def classification_metrics(
    expected: Iterable[str],
    predicted: Iterable[str],
) -> dict[str, object]:
    """Return multi-class metrics without requiring scikit-learn."""
    actual = list(expected)
    guessed = list(predicted)
    if len(actual) != len(guessed):
        raise ValueError("Expected and predicted label counts must match.")

    confusion = {
        label: {prediction: 0 for prediction in LABELS}
        for label in LABELS
    }
    for truth, guess in zip(actual, guessed, strict=True):
        if truth not in LABELS or guess not in LABELS:
            raise ValueError(f"Unknown evaluation label: {truth!r} / {guess!r}")
        confusion[truth][guess] += 1

    per_label: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other][label] for other in LABELS if other != label)
        false_negative = sum(confusion[label][other] for other in LABELS if other != label)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_label[label] = {
            "support": sum(confusion[label].values()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    evaluated_labels = [label for label in LABELS if per_label[label]["support"] > 0]
    return {
        "cases": len(actual),
        "accuracy": safe_divide(sum(truth == guess for truth, guess in zip(actual, guessed)), len(actual)),
        "macro_f1": round(
            sum(per_label[label]["f1"] for label in evaluated_labels) / len(evaluated_labels),
            4,
        ),
        "evaluated_labels": evaluated_labels,
        "per_label": per_label,
        "confusion_matrix": confusion,
        "expected_distribution": dict(Counter(actual)),
        "predicted_distribution": dict(Counter(guessed)),
    }


def answer_risk_metrics(
    expected: Iterable[str],
    predicted: Iterable[str],
) -> dict[str, float | int | dict[str, int]]:
    """Score hallucination-risk detection for binary answer benchmarks.

    HaluEval labels a whole answer as grounded or hallucinated. VeriSight has a
    useful third outcome, ``uncertain``: it means the answer is not accepted as
    supported. For safety-oriented answer checking, both uncertain and
    unsupported answers are therefore treated as a risk flag.
    """
    actual = list(expected)
    guessed = list(predicted)
    if len(actual) != len(guessed):
        raise ValueError("Expected and predicted label counts must match.")

    true_positive = false_positive = true_negative = false_negative = 0
    for truth, guess in zip(actual, guessed, strict=True):
        expected_risk = truth == "unsupported"
        predicted_risk = guess != "supported"
        if expected_risk and predicted_risk:
            true_positive += 1
        elif not expected_risk and predicted_risk:
            false_positive += 1
        elif not expected_risk and not predicted_risk:
            true_negative += 1
        else:
            false_negative += 1

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    return {
        "accuracy": safe_divide(true_positive + true_negative, len(actual)),
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
    }
