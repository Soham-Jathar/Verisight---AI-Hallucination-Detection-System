# Evidence-Grounded AI Hallucination & Uncertainty Detection System

This project will generate an LLM answer, retrieve evidence, verify factual claims, and present explainable hallucination, uncertainty, and reliability signals.

## Milestone 1

- React frontend
- FastAPI backend
- Health endpoint
- Web evidence retrieval and claim verification pipeline
- Local development workflow

Later milestones add document ingestion, hybrid retrieval, richer verification methods, storage, evaluation, and deployment.

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
