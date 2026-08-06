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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --paper:#f7f6f2;
    --ink:#15151a;
    --muted:#8a8a8f;
    --line:#e3e1da;
    --navy:#1a1f5e;
    --magenta:#e8186d;
    --serif:'Fraunces', serif;
    --sans:'Inter', -apple-system, sans-serif;
  }
  *{box-sizing:border-box;}
  html,body{margin:0; padding:0;}
  body{
    font-family:var(--sans);
    color:var(--ink);
    background:var(--paper);
    min-height:100vh;
    position:relative;
    overflow-x:hidden;
  }

  /* subtle responsive background — two soft blurs + faint grain, low opacity throughout */
  .bg{
    position:fixed;
    inset:0;
    z-index:0;
    pointer-events:none;
  }
  .bg::before, .bg::after{
    content:'';
    position:absolute;
    width:60vmax;
    height:60vmax;
    border-radius:50%;
    filter:blur(90px);
    opacity:0.10;
  }
  .bg::before{ background:var(--navy); top:-20vmax; left:-15vmax; }
  .bg::after{ background:var(--magenta); bottom:-25vmax; right:-18vmax; opacity:0.08; }
  .grain{
    position:fixed; inset:0; z-index:0; pointer-events:none;
    opacity:0.035; mix-blend-mode:multiply;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  }

  .wrap{
    position:relative;
    z-index:1;
    max-width:620px;
    margin:0 auto;
    padding:72px 24px 60px;
  }

  header{ margin-bottom:44px; }
  .eyebrow{
    font-size:12px;
    letter-spacing:0.12em;
    text-transform:uppercase;
    color:var(--navy);
    font-weight:600;
    margin-bottom:14px;
  }
  h1{
    font-family:var(--serif);
    font-weight:500;
    font-size:clamp(32px, 6vw, 46px);
    line-height:1.08;
    letter-spacing:-0.01em;
    margin:0 0 12px;
  }
  h1 em{
    font-style:italic;
    color:var(--magenta);
  }
  p.sub{
    color:var(--muted);
    font-size:15.5px;
    line-height:1.6;
    max-width:440px;
    margin:0;
  }

  /* input — hairline, no boxy card */
  .composer{
    margin-top:38px;
    border-bottom:1.5px solid var(--ink);
    padding-bottom:10px;
    display:flex;
    align-items:flex-end;
    gap:14px;
    transition:border-color 0.2s ease;
  }
  .composer:focus-within{ border-color:var(--magenta); }
  textarea{
    flex:1;
    border:none;
    background:transparent;
    resize:none;
    font-family:var(--sans);
    font-size:17px;
    color:var(--ink);
    padding:6px 0;
    min-height:28px;
    max-height:120px;
  }
  textarea::placeholder{ color:var(--muted); }
  textarea:focus{ outline:none; }
  button.send{
    font-family:var(--sans);
    font-weight:600;
    font-size:13px;
    letter-spacing:0.02em;
    background:none;
    border:none;
    color:var(--ink);
    cursor:pointer;
    padding:6px 2px;
    white-space:nowrap;
    display:flex;
    align-items:center;
    gap:6px;
  }
  button.send:hover{ color:var(--magenta); }
  button.send:disabled{ color:var(--muted); cursor:default; }
  button.send svg{ width:14px; height:14px; }

  .suggestions{
    display:flex;
    flex-wrap:wrap;
    gap:8px 18px;
    margin-top:16px;
  }
  .sug{
    font-size:13px;
    color:var(--muted);
    cursor:pointer;
    border-bottom:1px solid transparent;
    transition:all 0.15s ease;
  }
  .sug:hover{ color:var(--navy); border-color:var(--navy); }

  /* Q&A transcript */
  .transcript{ margin-top:52px; }
  .entry{
    padding:28px 0;
    border-top:1px solid var(--line);
    animation:rise 0.4s ease both;
  }
  .entry:first-child{ border-top:1px solid var(--line); }
  @keyframes rise{
    from{ opacity:0; transform:translateY(6px); }
    to{ opacity:1; transform:translateY(0); }
  }
  .q-label{
    font-size:11px;
    font-weight:600;
    letter-spacing:0.1em;
    color:var(--magenta);
    margin-bottom:8px;
  }
  .q-text{
    font-family:var(--serif);
    font-size:19px;
    font-weight:500;
    line-height:1.35;
    margin-bottom:18px;
  }
  .a-label{
    font-size:11px;
    font-weight:600;
    letter-spacing:0.1em;
    color:var(--muted);
    margin-bottom:8px;
  }
  .a-text{
    font-size:16px;
    line-height:1.7;
    color:var(--ink);
    white-space:pre-wrap;
  }
  .a-text.error{ color:#b3261e; }
  .a-text.loading{ color:var(--muted); }

  .empty{
    color:var(--muted);
    font-size:14px;
    padding:36px 0;
    border-top:1px solid var(--line);
  }

  footer{
    margin-top:56px;
    padding-top:20px;
    border-top:1px solid var(--line);
    display:flex;
    flex-wrap:wrap;
    gap:18px;
    font-size:13px;
  }
  footer a{
    color:var(--muted);
    text-decoration:none;
    border-bottom:1px solid transparent;
  }
  footer a:hover{ color:var(--navy); border-color:var(--navy); }

  @media (max-width:480px){
    .wrap{ padding:52px 18px 40px; }
    .q-text{ font-size:17px; }
  }
</style>
</head>
<body>
  <div class="bg"></div>
  <div class="grain"></div>

  <div class="wrap">
    <header>
      <div class="eyebrow">Ask, on the record</div>
      <h1>Everything about<br><em>Vineeth</em>, answered.</h1>
      <p class="sub">Full-stack developer, UI/UX designer, and AI engineer. Ask about his projects, stack, or what he's looking for next.</p>

      <div class="composer">
        <textarea id="question" placeholder="What's his tech stack?" rows="1"></textarea>
        <button class="send" id="sendBtn">
          Ask
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 12h14M13 6l6 6-6 6"/></svg>
        </button>
      </div>

      <div class="suggestions">
        <span class="sug" data-q="What's his tech stack?">tech stack</span>
        <span class="sug" data-q="Tell me about ARIA.">ARIA</span>
        <span class="sug" data-q="What design work has he done?">design work</span>
        <span class="sug" data-q="What roles is he open to?">open roles</span>
      </div>
    </header>

    <div class="transcript" id="transcript">
      <div class="empty" id="emptyState">Nothing asked yet — start above, or pick a prompt.</div>
    </div>

    <footer>
      <a href="https://rickyfender.netlify.app" target="_blank">Portfolio</a>
      <a href="https://github.com/FenderRicky" target="_blank">GitHub</a>
      <a href="https://www.linkedin.com/in/vineeth-thadigotla-0569381b9/" target="_blank">LinkedIn</a>
      <a href="https://www.instagram.com/fender_ricky" target="_blank">Instagram</a>
    </footer>
  </div>

<script>
  const input = document.getElementById('question');
  const sendBtn = document.getElementById('sendBtn');
  const transcript = document.getElementById('transcript');
  const emptyState = document.getElementById('emptyState');

  function autoresize(){
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 120) + 'px';
  }

  function addEntry(question){
    emptyState.style.display = 'none';
    const entry = document.createElement('div');
    entry.className = 'entry';
    entry.innerHTML = `
      <div class="q-label">QUESTION</div>
      <div class="q-text"></div>
      <div class="a-label">ANSWER</div>
      <div class="a-text loading">Thinking…</div>
    `;
    entry.querySelector('.q-text').textContent = question;
    transcript.prepend(entry);
    return entry.querySelector('.a-text');
  }

  async function ask(question){
    const q = (question || input.value).trim();
    if (!q) return;

    input.value = '';
    autoresize();
    sendBtn.disabled = true;

    const answerEl = addEntry(q);

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Request failed');
      answerEl.textContent = data.answer;
      answerEl.classList.remove('loading');
    } catch (err) {
      answerEl.textContent = 'Something went wrong: ' + err.message;
      answerEl.classList.remove('loading');
      answerEl.classList.add('error');
    } finally {
      sendBtn.disabled = false;
    }
  }

  sendBtn.addEventListener('click', () => ask());
  input.addEventListener('input', autoresize);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      ask();
    }
  });
  document.querySelectorAll('.sug').forEach(el => {
    el.addEventListener('click', () => ask(el.dataset.q));
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
