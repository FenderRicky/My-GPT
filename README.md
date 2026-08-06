# Ask Vineeth — Personal GPT

A small FastAPI app that answers questions about Vineeth Thadigotla ("Ricky Fender") —
full-stack dev, UI/UX designer, and AI engineer. Deploys as a serverless function on
Vercel. No database, no vector search — the knowledge base is small enough to inline
directly into the prompt, so it's a single Gemini API call per question.

## Live demo

Visit the deployed URL → type a question → get an answer.
Try: "what's his tech stack?", "tell me about ARIA", "what roles is he open to?"

## Project structure

```
.
├── api/
│   └── index.py        # FastAPI app — Vercel auto-detects this as a serverless function
├── requirements.txt    # Python dependencies
└── README.md
```

Vercel's Python runtime automatically treats any file under `api/` as a serverless
function, as long as it exports a top-level `app` (FastAPI instance). No `vercel.json`
is needed for this structure.

## How it works

- `GET /` — serves a small HTML chat UI
- `POST /api/ask` — takes `{ "question": "..." }`, returns `{ "answer": "..." }`
- The system prompt (in `api/index.py`) contains Vineeth's bio, skills, projects, and
  approved contact links. Every request sends this + the question to Gemini in one call.
- A soft in-memory rate limiter caps requests per warm instance (20/hour) as a basic
  safety net against runaway usage.

## Setup

### 1. Get a Gemini API key
Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create a key.

**Strongly recommended:** set a hard daily/monthly quota on that key in Google AI Studio
or Google Cloud Console. This is enforced by Google server-side and is the real
protection against exceeding your free tier — the in-app rate limiter alone isn't
bulletproof on serverless (Vercel can spin up multiple instances, each with its own
counter).

### 2. Deploy to Vercel
1. Push this repo to GitHub.
2. Import the repo in [vercel.com](https://vercel.com).
3. In **Project Settings → Environment Variables**, add:
   - Key: `GEMINI_API_KEY`
   - Value: your Gemini API key
   - Environment: Production (and Preview, if you want)
4. Deploy. Vercel auto-detects `api/index.py` as a Python serverless function.

### 3. Redeploy after any env var change
Environment variable changes don't apply to existing deployments — after adding/editing
one, go to **Deployments → (latest) → ⋯ → Redeploy**.

## Updating what the bot knows

Edit the `SYSTEM_PROMPT` string in `api/index.py`. It's plain text — add, remove, or
correct facts directly. No re-training or embeddings step needed; changes take effect
on the next deploy.

**Privacy note:** the prompt is explicitly scoped to only share the approved public
links (portfolio, GitHub, LinkedIn, Instagram, freelance email) and is instructed to
never share a phone number or personal email. Keep that scoping in mind if you add more
personal info later.

## Local testing (optional)

```bash
pip install -r requirements.txt
pip install uvicorn
GEMINI_API_KEY=your-key-here uvicorn api.index:app --reload
```

Then visit `http://localhost:8000`.

## Tech

FastAPI · Gemini API (`google-genai`) · deployed on Vercel serverless functions
