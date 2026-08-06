# Evaluation module

This module measures the **claim-verification layer**, not whether Gemini or
Groq produces fluent answers. It runs fixed, labelled claim/evidence pairs so
it does not consume LLM, Tavily, or search API quota.

The verifier intentionally excludes purely conversational content such as
greetings, acknowledgements, questions, and personal preferences. In the app,
such a response is shown as **not requiring factual verification** rather than
receiving a misleading reliability score. In the offline three-class benchmark,
it remains a conservative `uncertain` result because it cannot be confirmed
from evidence.

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

## HaluEval QA benchmark

HaluEval QA includes source knowledge, a grounded answer, and a deliberately
hallucinated answer. The importer produces one supported and one unsupported
answer-level case per source example. It uses only the provided benchmark
evidence, so no LLM or web-search API calls are made during import or scoring.

Download the official repository once, then convert a bounded held-out subset:

```powershell
git clone --depth 1 https://github.com/RUCAIBox/HaluEval.git evaluation\vendor\HaluEval

.\backend\.venv\Scripts\python.exe evaluation\import_halueval.py `
  --input evaluation\vendor\HaluEval\data\qa_data.json `
  --limit 200

.\backend\.venv\Scripts\python.exe evaluation\run.py `
  --dataset evaluation\datasets\halueval_qa.jsonl
```

The 200 source examples become 400 answer-level cases. Keep the downloaded and
converted public data out of Git; the project `.gitignore` already does this.

## HaluEval dialogue benchmark

Dialogue uses the same evidence-grounded verdict pipeline, but each example
also includes the preceding conversation. The importer preserves that history
in every converted case, allowing failures to be reviewed in context. It still
uses only fixed HaluEval evidence and makes no external requests.

```powershell
.\backend\.venv\Scripts\python.exe evaluation\import_halueval.py `
  --task dialogue `
  --input evaluation\vendor\HaluEval\data\dialogue_data.json `
  --limit 200

.\backend\.venv\Scripts\python.exe evaluation\run.py `
  --dataset evaluation\datasets\halueval_dialogue.jsonl
```

As with QA, use one dialogue subset for development and reserve a fresh,
non-overlapping `--offset` range for the final reported result.

Create separate dialogue development and held-out files. Do not use the
splitter defaults here: those default filenames belong to the QA experiment.

```powershell
.\backend\.venv\Scripts\python.exe evaluation\split_dataset.py `
  --input evaluation\datasets\halueval_dialogue.jsonl `
  --dev-output evaluation\datasets\halueval_dialogue_dev.jsonl `
  --test-output evaluation\datasets\halueval_dialogue_test.jsonl

# Use this while investigating dialogue-verification improvements.
.\backend\.venv\Scripts\python.exe evaluation\run.py `
  --dataset evaluation\datasets\halueval_dialogue_dev.jsonl

# Run only after dialogue improvements are final.
.\backend\.venv\Scripts\python.exe evaluation\run.py `
  --dataset evaluation\datasets\halueval_dialogue_test.jsonl
```

For HaluEval answer-level runs, reports include two valid views:

- **Strict verdict metrics** — `supported`, `unsupported`, and `uncertain` are
  treated as separate outcomes.
- **Hallucination-risk metrics** — both `unsupported` and `uncertain` count as
  a risk flag, because the system did not accept the answer as supported.

Use the strict score to evaluate verdict quality and the risk score to evaluate
the detector’s ability to prevent potentially hallucinated answers from being
accepted as factual.

Before improving any verification thresholds, create a reproducible split:

```powershell
.\backend\.venv\Scripts\python.exe evaluation\split_dataset.py `
  --input evaluation\datasets\halueval_qa.jsonl

# Run this while making improvements.
.\backend\.venv\Scripts\python.exe evaluation\run.py `
  --dataset evaluation\datasets\halueval_dev.jsonl

# Run this only after the improvements are final.
.\backend\.venv\Scripts\python.exe evaluation\run.py `
  --dataset evaluation\datasets\halueval_test.jsonl
```

The importer keeps each grounded/hallucinated HaluEval answer pair in the same
split. This prevents evidence and answer pairs from leaking into the held-out
result.

Because the first 400-case result has already been viewed, use a fresh,
non-overlapping source range for the final report result after tuning:

```powershell
.\backend\.venv\Scripts\python.exe evaluation\import_halueval.py `
  --input evaluation\vendor\HaluEval\data\qa_data.json `
  --offset 200 --limit 200 `
  --output evaluation\datasets\halueval_final_holdout.jsonl

.\backend\.venv\Scripts\python.exe evaluation\run.py `
  --dataset evaluation\datasets\halueval_final_holdout.jsonl
```

## Optional local dashboard

Normal chats deliberately do not expose research metrics. To inspect the latest
local report, set these values before restarting both services:

```text
# backend/.env
EVALUATION_DASHBOARD_ENABLED=true

# frontend/.env
VITE_ENABLE_EVALUATION_DASHBOARD=true
```

The sidebar will then show **Research metrics**. Keep both values false for the
normal user-facing deployment.
