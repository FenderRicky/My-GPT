"""
Personal GPT API for Vineeth Thadigotla — Vercel FastAPI entrypoint.
Runs on Groq's free tier (OpenAI-compatible API) — no credit card, generous
daily limit, fast inference.

Why no RAG / embeddings here:
The knowledge base is a single resume — a few hundred words. Retrieval only
earns its cost when the knowledge base is too big to fit in one prompt.
Here, it's cheaper and simpler to just inline the whole resume as context
on every request: one API call per question.

Rate limiting:
This adds a soft per-instance limit as a safety net, but Groq's own free-tier
quota (enforced server-side, per API key/account) is the real backstop — see
the note at the bottom of this file.
"""

import os
import time
from collections import deque

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GENERATION_MODEL = "llama-3.3-70b-versatile"

# Soft rate limit: max requests per rolling window, per warm instance.
MAX_REQUESTS = 20
WINDOW_SECONDS = 3600  # 1 hour

app = FastAPI(title="Ask Vineeth")

_client = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY environment variable is not set.",
            )
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Soft rate limiter (per warm instance — see note at bottom of file)
# ---------------------------------------------------------------------------

_request_times: deque = deque()


def check_rate_limit():
    now = time.time()
    while _request_times and now - _request_times[0] > WINDOW_SECONDS:
        _request_times.popleft()
    if len(_request_times) >= MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="This demo is rate-limited to keep API costs predictable. Try again later.",
        )
    _request_times.append(now)


# ---------------------------------------------------------------------------
# Knowledge (inlined directly — no retrieval needed for a doc this small)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the personal AI assistant of Vineeth Thadigotla (also known professionally as "Ricky Fender"). You answer questions about him accurately and helpfully, speaking about him in the third person unless asked to write something as if you were him. Stay grounded only in the facts below — if something isn't covered, say you don't have that detail rather than guessing. Keep answers concise and conversational.

ABOUT VINEETH:
- Full-stack developer, UI/UX designer, and AI engineer.
- B.Tech CSE (Data Science), MLR Institute of Technology (MLRIT), Hyderabad. 3rd year, batch 2023-2027.
- Based in Hyderabad. Open to full-stack development and UI/UX design roles.

APPROVED CONTACT / LINKS (only share these — never any other personal info like phone number or personal email):
- Portfolio: https://rickyfender.netlify.app
- GitHub: https://github.com/FenderRicky
- LinkedIn: https://www.linkedin.com/in/vineeth-thadigotla-0569381b9/
- Instagram: https://www.instagram.com/fender_ricky
- Email: rickyfender00@gmail.com

TECH STACK: Next.js, React, FastAPI, Node.js, MongoDB, PostgreSQL, Tailwind, TypeScript, Ollama, ChromaDB.

UI/UX SKILLS: Figma design systems, responsive design, design thinking, brand strategy. Tools: Figma, Adobe XD, Photoshop, Illustrator, Canva.

FLAGSHIP PROJECTS:
1. InsightFlow — AI-Powered Intelligence Platform. Full-Stack + UI/UX Lead. Built with Next.js, Express, MongoDB, Figma. Enterprise-grade UI handling 1000+ profile analyses with <2s response time. Built an intelligent scoring algorithm evaluating 15+ skill dimensions with 94% accuracy. Architected a RESTful API layer with Express and MongoDB for real-time data ingestion. Created a comprehensive design system in Figma (40+ components) with dark/light modes. Structured modular front-end architecture in Next.js for rapid iteration across dashboard and reporting views.

2. ARIA — Local AI OS Layer. Architect + Lead Developer. Built with Python, FastAPI, Next.js, Ollama, ChromaDB. A fully local AI system with persistent memory using vector embeddings (ChromaDB). Multi-modal context awareness integrating screen capture, voice, and semantic search. Integrated Tesseract OCR for real-time screen-context capture so the assistant can reference on-screen content mid-conversation. Serves language models locally via Ollama, removing cloud API latency and keeping all user data on-device. Deployed without external APIs on 16GB systems — zero cloud dependencies.

3. Brand Identity Design — Designer & Brand Strategist, 2024-present. Delivered 10+ brand redesigns including full visual identities, guidelines, and mockups. Example: Creatiwise.com redesign — 4 logo concepts, 8 variations, brand guide, competitive analysis. Defines cohesive color and typography systems tailored to each client's brand positioning (e.g. navy #1A1F5E and magenta #E8186D for one client). Manages the full design lifecycle from concept sketching through client-ready guideline documentation and revisions.

KEY DIFFERENTIATORS: Full-stack expertise spanning design, code, and deployment. AI/ML integration specialist. Figma design systems architect. 0-to-1 product builder. 15+ successful client projects.

If asked for contact details or links, share from the APPROVED CONTACT / LINKS list above only. Never share a phone number or personal email — those are not public."""


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


DEMO_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Ask Vineeth</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #0a0c10;
    color: #e9eaf0;
    max-width: 720px;
    margin: 0 auto;
    padding: 48px 20px;
  }
  h1 { font-size: 1.5rem; margin-bottom: 4px; }
  p.sub { color: #7d8296; margin-top: 0; margin-bottom: 32px; }
  textarea {
    width: 100%;
    min-height: 80px;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #232838;
    background: #12151c;
    color: #e9eaf0;
    font-size: 1rem;
    resize: vertical;
  }
  button {
    margin-top: 12px;
    padding: 10px 20px;
    border: none;
    border-radius: 8px;
    background: #e8186d;
    color: white;
    font-size: 0.95rem;
    cursor: pointer;
  }
  button:disabled { background: #232838; cursor: not-allowed; }
  .result {
    margin-top: 24px;
    padding: 16px;
    border-radius: 8px;
    background: #12151c;
    border: 1px solid #232838;
    white-space: pre-wrap;
    line-height: 1.55;
    display: none;
  }
  .result.visible { display: block; }
  .error { color: #ff6b6b; }
</style>
</head>
<body>
  <h1>Ask about Vineeth</h1>
  <p class="sub">Full-stack dev · UI/UX designer · AI engineer</p>

  <textarea id="question" placeholder="e.g. What's his tech stack?"></textarea>
  <br />
  <button id="askBtn">Ask</button>

  <div id="result" class="result"></div>

<script>
  const btn = document.getElementById('askBtn');
  const questionEl = document.getElementById('question');
  const resultEl = document.getElementById('result');

  async function ask() {
    const question = questionEl.value.trim();
    if (!question) return;

    btn.disabled = true;
    btn.textContent = 'Thinking...';
    resultEl.classList.remove('visible', 'error');

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      resultEl.textContent = data.answer;
      resultEl.classList.add('visible');
    } catch (err) {
      resultEl.textContent = 'Error: ' + err.message;
      resultEl.classList.add('visible', 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Ask';
    }
  }

  btn.addEventListener('click', ask);
  questionEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask();
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def demo_page():
    return DEMO_HTML


@app.get("/api")
@app.get("/api/")
def root():
    return {"status": "ok", "message": "Ask Vineeth API is running"}


@app.post("/api/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be empty.")

    check_rate_limit()

    client = get_client()
    completion = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.question},
        ],
        temperature=0.6,
    )

    return QueryResponse(answer=completion.choices[0].message.content)


# ---------------------------------------------------------------------------
# IMPORTANT — the real usage cap
# ---------------------------------------------------------------------------
# The in-memory rate limiter above only protects a single warm serverless
# instance. Vercel can spin up multiple instances under load, each with its
# own counter, so it's a soft speed bump, not a hard guarantee.
#
# Groq's free tier is enforced server-side per API key/account (roughly
# 30 requests/min and ~1,000 requests/day for llama-3.3-70b-versatile as of
# 2026 — check console.groq.com for your account's current limits). That's
# the real backstop: once you hit it, Groq itself returns a 429, regardless
# of how many serverless instances Vercel spins up.
