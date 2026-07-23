# Backend

FastAPI service for the VeriSight hallucination detection frontend.

## Setup

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (`{"status":"ok"}`) |
| GET | `/` | Service banner |
| POST | `/api/analyze` | Run web verification pipeline |

### `POST /api/analyze`

Request:

```json
{
  "question": "Who created Python and when was it first released?",
  "mode": "web"
}
```

Response (frontend uses `message`; additional fields are ready for later UI work):

```json
{
  "question": "...",
  "mode": "web",
  "stage": "complete",
  "message": "Analyzed 2 claim(s) using 3 evidence source(s)...",
  "answer": "...",
  "evidence": [{ "title": "...", "url": "...", "snippet": "..." }],
  "claims": [{ "claim": "...", "status": "supported", "confidence": 0.42, "rationale": "..." }],
  "reliability_score": 0.61
}
```

Supported modes today:

- `web` — retrieves evidence from Wikipedia and DuckDuckGo, verifies claim overlap
- `document`, `hybrid` — return `501 Not Implemented` (frontend options are disabled)

## Environment variables

See `.env.example`. All are optional for local development.

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | unset | Enables LLM answer generation |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Compatible API base URL |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated frontend origins |
| `REQUEST_TIMEOUT_SECONDS` | `30` | External HTTP timeout |

Without `OPENAI_API_KEY`, the service still works by synthesizing an answer from retrieved web evidence.

## Tests

```bash
pytest
```

## Frontend integration

The React app expects the backend at `http://localhost:8000` by default. Override with `VITE_API_URL` in the frontend `.env` if needed.

Run both services locally:

```bash
# terminal 1
cd backend && uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend && npm run dev
```
