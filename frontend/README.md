# VeriSight Frontend

React and Vite frontend for the VeriSight evidence-grounded AI reliability platform.

It provides the conversational interface for asking questions, uploading PDFs, selecting an LLM provider, viewing claim-level verification, opening evidence citations, comparing models, and managing saved conversations.

## User-facing capabilities

- Chat-style question and follow-up flow
- Web, document, and hybrid verification modes
- PDF upload and document-backed questioning
- Speech-to-text input where supported by the browser
- Gemini, Groq, and provider comparison controls
- Supported / needs-review / unsupported claim cards
- Reliability, confidence, uncertainty, evidence-quality, and source-agreement signals
- Evidence and citation links for factual verification
- Supabase sign-up, sign-in, password recovery, and persistent chat history
- Rename and delete controls for saved signed-in conversations

## Prerequisites

- Node.js 20 or later
- The FastAPI backend running locally or deployed
- Optional Supabase project values for sign-in and persistent history

## Configuration

Copy `.env.example` to `.env`:

```text
VITE_API_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=your_publishable_key
```

`VITE_API_URL` must point to the FastAPI backend. Do not put Gemini, Groq, Tavily, or other private provider keys in this frontend environment file.

## Run locally

```powershell
npm install
npm run dev
```

Vite is configured to use `http://localhost:5173` in normal local development.

## Build for deployment

```powershell
npm run build
```

The generated `dist/` folder is static output suitable for a host such as Render Static Sites. Before deployment, set `VITE_API_URL` to the public HTTPS URL of the deployed backend and configure the backend to allow the frontend origin through CORS.

## Related documentation

- [Project overview](../README.md)
- [Backend API documentation](../backend/README.md)
- [Evaluation methodology](../evaluation/README.md)
