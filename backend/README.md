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

Supported modes:

- `web` — retrieves and filters web evidence, then verifies generated factual claims
- `document` — verifies against uploaded PDF evidence
- `hybrid` — combines uploaded-document and web evidence

## Environment variables

See `.env.example`. All are optional for local development.

| Variable | Default | Purpose |
|----------|---------|---------|
| `GEMINI_API_KEY` | unset | Enables Gemini answer generation |
| `GEMINI_MODEL` | configured model | Gemini model name |
| `GROQ_API_KEY` | unset | Enables Groq answer generation |
| `GROQ_MODEL` | configured model | Groq model name |
| `TAVILY_API_KEY` | unset | Enables Tavily web retrieval |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated frontend origins |
| `REQUEST_TIMEOUT_SECONDS` | `30` | External HTTP timeout |

Without an LLM provider key, the API returns a clear provider-configuration error rather than inventing an answer.

## Tests

```bash
pytest
```

## Frontend integration

The React app expects the backend at `http://localhost:8000` by default. Override with `VITE_API_URL` in the frontend `.env` if needed. For production, use the public HTTPS backend URL and set the matching frontend URL in `CORS_ORIGINS`.

Run both services locally:

```bash
# terminal 1
cd backend && uvicorn app.main:app --reload --port 8000

# terminal 2
cd frontend && npm run dev
```
