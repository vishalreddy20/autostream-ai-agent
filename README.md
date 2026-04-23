# AutoStream AI Agent

> A production-quality conversational AI agent built with LangGraph + Gemini 1.5 Flash.
> Classifies user intent, answers product questions via RAG, and captures qualified sales leads.

---

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishalreddy20/autostream-ai-agent.git
   cd autostream-ai-agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the environment**
   - macOS / Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Get your Google Gemini API key** (free — no billing required)
   - Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - Sign in with your Google account
   - Click **Create API key**
   - Copy the key immediately

6. **Create your `.env` file**
   ```bash
   GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxx
   ```
   > ⚠️ Never commit `.env` to GitHub. It is already excluded by `.gitignore`.

7. **Run the agent**
   ```bash
   python main.py
   ```

---

## Architecture

### Why LangGraph over AutoGen?

LangGraph is chosen because it provides **explicit, deterministic control** over state transitions. Each node is a pure Python function that receives the current `AgentState` and returns a partial state update — LangGraph merges updates automatically. This makes debugging trivial: you can inspect the exact state at every conversation turn.

AutoGen is designed for multi-agent debate systems where agents negotiate with each other. This project is a single-agent, linear workflow (classify → route → respond → collect lead). LangGraph's `conditional_edges` feature maps intent classifications to the correct handler with zero ambiguity, whereas AutoGen's approach would add unnecessary orchestration overhead.

### How MemorySaver Works

`MemorySaver` is LangGraph's built-in in-memory checkpointer. Each conversation session is identified by a unique `thread_id` (UUID). On every `.invoke()` call, LangGraph:
1. Loads the full `AgentState` from memory using `thread_id`
2. Runs the graph starting from the entry node
3. Saves the updated state back to memory

This means multi-turn context (message history, lead fields, `awaiting_field`) is persisted across turns with zero manual state management. Each unique `thread_id` represents an isolated conversation — multiple users can run concurrently without state collision.

### Why RAG Uses JSON, Not a Vector Database

The knowledge base is small (2 plans, 3 policies, 3 FAQs) and fully deterministic. ChromaDB or FAISS would add ~300MB of dependencies, require embedding model calls, and introduce hard-to-debug semantic drift where similar-but-wrong chunks get retrieved. Keyword-based JSON search is:
- **Faster**: no embedding computation
- **Transparent**: you know exactly which chunks matched
- **Accurate**: no false positives from cosine similarity on short domain-specific text

### Node Functions

| Node | Purpose |
|------|---------|
| `classify_intent` | Uses Gemini 1.5 Flash at `temperature=0` to output exactly one of: `greeting`, `inquiry`, `high_intent` |
| `handle_greeting` | Generates a warm, brand-consistent 2-sentence reply |
| `handle_inquiry` | Retrieves grounding context from `knowledge.json` and answers strictly from that context |
| `handle_high_intent` | Sequential field collection: name → email → platform, with per-field validation |
| `execute_lead_capture` | Calls `mock_lead_capture()` only when all three fields are confirmed; sets `lead_captured=True` |
| `handle_unknown` | Graceful fallback that redirects the user back to AutoStream topics |

### Conditional Edge Routing

```
START
  └─► classify_intent
        ├─► [greeting]      → handle_greeting → END
        ├─► [inquiry]       → handle_inquiry  → END
        ├─► [high_intent]   → handle_high_intent
        │                         ├─► [all fields collected] → execute_lead_capture → END
        │                         └─► [fields missing]       → END (await next turn)
        └─► [unknown]       → handle_unknown  → END
```

**Special bypass**: If `awaiting_field` is set (mid-collection), `route_by_intent` skips classification entirely and sends the message directly to `handle_high_intent`. This prevents the classifier from misinterpreting a raw email address as a `greeting`.

### LLM Choice: Gemini 1.5 Flash

1. **Free Tier Generosity**: Gemini 1.5 Flash provides 15 requests per minute and 1,500 requests per day entirely for free without requiring a credit card, making it the most robust choice for development and demonstrations.
2. **Speed**: Flash is Google's lightweight, extremely fast model — critical for multi-turn conversational latency.
3. **Instruction adherence**: Accurately outputs single-word classifications for intent detection.
4. **JSON discipline**: Strong structured output capabilities without needing extra parsing libraries.

---

## WhatsApp Integration via Webhooks

### Deployment Architecture

To deploy this agent on WhatsApp, wrap it in a **FastAPI HTTP server** and register the webhook with the **WhatsApp Business API**:

1. **Deploy as FastAPI server** — wrap `build_graph()` and expose a REST endpoint
2. **Expose `POST /webhook`** — this is the URL Meta will send messages to
3. **Register webhook** on [Meta Developer Console](https://developers.facebook.com) under your WhatsApp Business App → Webhooks → Subscribe to `messages`
4. **Verify webhook** — Meta sends a `GET` with a `hub.challenge` token; your server must echo it back
5. **Parse incoming JSON** — extract `sender_phone_number` and `message_body` from the payload
6. **Use phone number as `thread_id`** — this gives each WhatsApp user their own isolated LangGraph state
7. **Invoke the agent graph** with `HumanMessage(content=message_body)`
8. **Send reply via WhatsApp Cloud API** — POST to:
   `https://graph.facebook.com/v18.0/{phone-number-id}/messages`
9. **State persists per phone number** — `MemorySaver` keeps the full conversation alive between turns

### FastAPI Webhook Snippet

```python
from fastapi import FastAPI, Request
from langchain_core.messages import HumanMessage
from agent.graph import build_graph
import httpx, os

app = FastAPI()
graph = build_graph()
WHATSAPP_TOKEN = os.environ["WHATSAPP_TOKEN"]
PHONE_NUMBER_ID = os.environ["PHONE_NUMBER_ID"]
VERIFY_TOKEN = os.environ["VERIFY_TOKEN"]

@app.get("/webhook")
async def verify(request: Request):
    params = request.query_params
    if params.get("hub.verify_token") == VERIFY_TOKEN:
        return int(params.get("hub.challenge"))
    return {"error": "Invalid verify token"}

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    entry = body["entry"][0]["changes"][0]["value"]
    message = entry["messages"][0]
    sender = message["from"]           # phone number → use as thread_id
    text = message["text"]["body"]
    config = {"configurable": {"thread_id": sender}}
    turn_input = {"messages": [HumanMessage(content=text)]}
    result = graph.invoke(turn_input, config=config)
    ai_reply = [m for m in result["messages"] if m.type == "ai"][-1].content
    async with httpx.AsyncClient() as client:
        await client.post(
            f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages",
            headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"},
            json={"messaging_product": "whatsapp", "to": sender,
                  "type": "text", "text": {"body": ai_reply}}
        )
    return {"status": "ok"}
```

---

## Project Structure

```
autostream-agent/
├── main.py                  ← CLI entry point (conversation loop)
├── agent/
│   ├── __init__.py
│   ├── graph.py             ← LangGraph StateGraph definition
│   ├── nodes.py             ← All node functions (intent, rag, lead, tool)
│   ├── state.py             ← AgentState TypedDict
│   └── tools.py             ← mock_lead_capture function
├── rag/
│   ├── __init__.py
│   ├── knowledge.json       ← Local knowledge base (pricing + policies)
│   └── retriever.py         ← RAG retrieval logic (JSON search, no vector DB)
├── .env                     ← API key storage (never commit this)
├── .env.example             ← Template with placeholder values
├── .gitignore               ← Excludes .env and __pycache__
├── requirements.txt         ← All dependencies with pinned versions
└── README.md                ← This file
```

---

## Edge Cases Handled

| # | Scenario | Handling |
|---|----------|----------|
| 1 | User provides name + email + platform in one message | Regex extracts all three; `awaiting_field` set to `None` immediately |
| 2 | User provides invalid email format | `validate_email()` returns `False`; re-asks without losing `lead_name` |
| 3 | User asks a pricing question mid-collection | Detects `?` in message; pauses collection, answers briefly, re-asks field |
| 4 | User tries to skip platform field | Rejects "skip"/"doesn't matter"; explains platform is required |
| 5 | API rate limit / network error | Wraps LLM calls; retries once after 5s on rate limit; graceful fallback on connection errors |
| 6 | Empty or whitespace-only input | Skipped with `if not user_input: continue` in `main.py` |
| 7 | User re-engages after lead is captured | `lead_captured=True` guard returns "already captured" without re-triggering the tool |

---

## Demo Conversation Flow

```
You: Hi there!
Assistant: Hey there! 👋 Welcome to AutoStream...

You: What's the difference between Basic and Pro?
Assistant: Great question! Basic is $29/month for 10 videos at 720p...

You: That sounds great. I want to sign up for Pro.
Assistant: Excellent! Let me collect a few quick details. What's your full name?

You: Vishal Reddy
Assistant: Great to meet you, Vishal! What's your email address?

You: vishal@example.com
Assistant: Perfect! Which creator platform do you primarily use?

You: YouTube
[LEAD CAPTURED]
  Name:     Vishal Reddy
  Email:    vishal@example.com
  Platform: YouTube
Assistant: You're all set! 🎉 Our team will reach out to vishal@example.com within 24 hours...
```
