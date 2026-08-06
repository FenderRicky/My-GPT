# Ask Vineeth — Personal GPT

A small FastAPI app that answers questions about Vineeth Thadigotla ("Ricky Fender") —
full-stack dev, UI/UX designer, and AI engineer. Deploys as a serverless function on
Vercel. No database, no vector search — the knowledge base is small enough to inline
directly into the prompt, so it's a single API call per question. Runs on Groq's free
tier (no credit card, ~1,000 requests/day, fast inference).

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
  approved contact links. Every request sends this + the question to Groq in one call.
- A soft in-memory rate limiter caps requests per warm instance (20/hour) as a basic
  safety net against runaway usage, on top of Groq's own account-level free-tier limit.

## Setup

### 1. Get a Groq API key
Go to [console.groq.com/keys](https://console.groq.com/keys), sign up with just an
email (no credit card), and create a key.

Groq's free tier is enforced server-side per account — roughly 30 requests/min and
~1,000 requests/day for `llama-3.3-70b-versatile` (check console.groq.com for your
account's current limits, these can change). Once you hit it, Groq returns a 429 —
that's your real backstop, independent of how many serverless instances Vercel spins up.

### 2. Deploy to Vercel
1. Push this repo to GitHub.
2. Import the repo in [vercel.com](https://vercel.com).
3. In **Project Settings → Environment Variables**, add:
   - Key: `GROQ_API_KEY`
   - Value: your Groq API key
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
GROQ_API_KEY=your-key-here uvicorn api.index:app --reload
```

Then visit `http://localhost:8000`.

## Tech

FastAPI · Groq API (`groq` SDK, Llama 3.3 70B) · deployed on Vercel serverless functions
