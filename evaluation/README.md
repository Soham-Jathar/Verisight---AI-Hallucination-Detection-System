# Evaluation module

This module measures the **claim-verification layer**, not whether Gemini or
Groq produces fluent answers. It runs fixed, labelled claim/evidence pairs so
it does not consume LLM, Tavily, or search API quota.

## What it measures

- **Accuracy** — total correct claim verdicts.
- **Precision, recall, F1** for `supported`, `unsupported`, and `uncertain`.
- **Macro F1** — treats all three verdict classes equally.
- **Confusion matrix** — shows which verdicts are being confused.
- **Average latency** — local verification time per claim.

## Run locally

From the project root, with the backend environment active:

```powershell
.\backend\.venv\Scripts\python.exe evaluation\run.py
```

For a quick smoke test:

```powershell
.\backend\.venv\Scripts\python.exe evaluation\run.py --limit 3
```

Reports are written under `evaluation/results/<UTC timestamp>/`:

- `metrics.json` — aggregate scores and confusion matrix.
- `predictions.json` — full prediction data for the report/dashboard later.
- `predictions.csv` — spreadsheet-friendly output.

## Dataset format

Each line in `datasets/starter_claims.jsonl` is one labelled case:

```json
{
  "id": "python-supported",
  "claim": "Guido van Rossum created the Python programming language.",
  "expected_status": "supported",
  "evidence": [{"title": "...", "url": "...", "snippet": "..."}]
}
```

The starter set is only a smoke-test dataset. For the final report, add a
held-out dataset built from public benchmarks such as FEVER, HaluEval, or
RAGTruth, document the source and split, and never tune thresholds on the same
test cases used for the final score.
