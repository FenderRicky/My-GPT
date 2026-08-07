# Ask Vineeth — Personal GPT

A lightweight AI-powered Personal GPT built with **FastAPI** and deployed as a **Vercel Serverless Function**.

The assistant answers questions about **Vineeth Thadigotla ("Ricky Fender")**, including his experience, projects, skills, tech stack, and professional background. Instead of using a database or vector search, the knowledge base is embedded directly into the system prompt, allowing each request to be handled with a single API call.

Powered by **Groq's Llama 3.3 70B** for fast, low-latency responses.

---

## ✨ Features

- FastAPI backend
- Serverless deployment on Vercel
- Groq API integration
- No database required
- No embeddings or vector search
- Simple single-request architecture
- Lightweight and easy to customize
- Basic in-memory rate limiting

---

## Project Structure

```text
.
├── api/
│   └── index.py
├── requirements.txt
└── README.md
```

---

## How It Works

### GET /

Serves the web-based chat interface.

### POST /api/ask

Accepts:

```json
{
  "question": "Your question"
}
```

Returns:

```json
{
  "answer": "Generated response"
}
```

Each request includes:

- System Prompt
- User Question

The prompt contains approved public information about Vineeth including:

- Skills
- Projects
- Experience
- Contact links

Everything is sent to Groq in a single API request.

---

## Setup

### 1. Create a Groq API Key

Create an API key from:

https://console.groq.com/keys

No credit card is required.

---

### 2. Deploy to Vercel

1. Push this repository to GitHub.
2. Import it into Vercel.
3. Add the following Environment Variable:

```
GROQ_API_KEY=your_api_key
```

4. Deploy.

Vercel automatically detects `api/index.py` as a Python Serverless Function.

---

### 3. Redeploy

Whenever an environment variable changes:

Deployments → Latest Deployment → Redeploy

---

## Updating Knowledge

Edit the `SYSTEM_PROMPT` inside:

```text
api/index.py
```

Update the information directly.

No retraining, indexing, or embedding generation is required.

---

## Privacy

The assistant is intentionally restricted to sharing only approved public information such as:

- Portfolio
- GitHub
- LinkedIn
- Instagram
- Freelance Email

Personal contact details such as phone numbers or private email addresses should not be added.

---

## Local Development

Install dependencies:

```bash
pip install -r requirements.txt
pip install uvicorn
```

Run:

```bash
GROQ_API_KEY=your-key-here uvicorn api.index:app --reload
```

Open:

```
http://localhost:8000
```

---

## Tech Stack

- FastAPI
- Groq SDK
- Llama 3.3 70B
- Python
- Vercel Serverless Functions

---

## License

This project is intended for personal portfolio and educational purposes.
