# Evidence-Grounded AI Hallucination & Uncertainty Detection System

This project will generate an LLM answer, retrieve evidence, verify factual claims, and present explainable hallucination, uncertainty, and reliability signals.

## Implemented features

- React chat interface with Supabase authentication and persistent chat history
- Gemini and Groq answer generation with optional model comparison
- Web, uploaded-document, and hybrid evidence retrieval
- Claim extraction and claim-level NLI verification with source-quality checks
- Evidence-linked citations, reliability, confidence, source-agreement, and uncertainty signals
- Deterministic handling for supported mathematical questions
- PDF upload, speech-to-text input, evidence-grounded corrections, and follow-up context
- HaluEval and custom regression evaluation suites

## Quick start

### Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

See `backend/README.md` for environment variables and API details.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://localhost:8000` for API calls. Override with `VITE_API_URL` if needed.

## Deployment

The included `render.yaml` configures the FastAPI service. Before deploying, set the backend provider keys in the host dashboard; do not commit a real `.env` file. Set `CORS_ORIGINS` to the deployed frontend domain and set the frontend's `VITE_API_URL` to the deployed backend URL.
