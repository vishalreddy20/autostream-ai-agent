# agent/nodes.py

import os
import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from agent.state import AgentState
from agent.tools import mock_lead_capture, validate_email
from rag.retriever import retrieve_context

# ── Lazy LLM initialisation ──────────────────────────────────────────────────
# Uses Gemini 1.5 Flash via Google AI Studio (completely free — no billing).
# We defer creation until the first node call so load_dotenv() in main.py
# always runs before we read GOOGLE_API_KEY.

_llm_deterministic = None
_llm_generative = None

def _get_llm_deterministic() -> ChatGoogleGenerativeAI:
    global _llm_deterministic
    if _llm_deterministic is None:
        _llm_deterministic = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
    return _llm_deterministic

def _get_llm_generative() -> ChatGoogleGenerativeAI:
    global _llm_generative
    if _llm_generative is None:
        _llm_generative = ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            temperature=0.3,
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
    return _llm_generative


def _safe_llm_invoke(llm, messages, fallback_text="I'm having trouble connecting right now. Please try again in a moment."):
    """
    Wraps LLM invocations with error handling for:
    - Rate limit errors → sleep 5 seconds and retry once
    - Connection errors → return fallback message
    - Any other exception → log and return generic error
    """
    import anthropic
    try:
        return llm.invoke(messages)
    except anthropic.RateLimitError:
        print("[WARN] Rate limit hit. Retrying in 5 seconds...")
        time.sleep(5)
        try:
            return llm.invoke(messages)
        except Exception as e:
            print(f"[ERROR] Retry failed: {e}")
            class _FakeResp:
                content = fallback_text
            return _FakeResp()
    except anthropic.APIConnectionError as e:
        print(f"[ERROR] Connection error: {e}")
        class _FakeResp:
            content = fallback_text
        return _FakeResp()
    except Exception as e:
        print(f"[ERROR] Unexpected LLM error: {e}")
        class _FakeResp:
            content = "An unexpected error occurred. Please try again."
        return _FakeResp()

def _extract_text(response) -> str:
    """Safely extracts text from LLM response which may be a string or a list of dicts."""
    content = response.content
    if isinstance(content, list):
        return ''.join([item.get('text', '') for item in content if isinstance(item, dict) and 'text' in item]).strip()
    return str(content).strip()


# ─────────────────────────────────────────────
# NODE 1: Intent Classifier
# ─────────────────────────────────────────────
def classify_intent(state: AgentState) -> dict:
    """
    Determines the user's intent from their latest message.

    This node uses a strict system prompt that forces the LLM to output
    ONLY a single word. This is critical — any other output format would
    break the conditional routing in the graph.

    Edge cases handled:
    - If the user switches from inquiry to high-intent mid-conversation,
      the classifier catches it immediately.
    - If the user mentions price AND interest ("I want the Pro plan"),
      it correctly classifies as high_intent, not inquiry.
    - Ambiguous messages default to "inquiry" not "unknown".
    """
    last_message = state["messages"][-1].content

    system_prompt = """You are an intent classifier for AutoStream, a SaaS video editing tool.

Classify the user's message into EXACTLY ONE of these categories:
- greeting: Any casual greeting, hello, hi, thanks, or small talk with no product intent
- inquiry: Questions about product features, pricing, plans, policies, or comparisons
- high_intent: Clear signals of wanting to sign up, start a trial, purchase, or try the product

Rules:
1. Output ONLY the category word. Nothing else. No punctuation. No explanation.
2. If the message contains BOTH a question AND buying intent, classify as high_intent.
3. "How much does it cost?" is inquiry. "I want to try the Pro plan" is high_intent.
4. "That sounds good, I'll sign up" is high_intent.
5. "Tell me about your pricing" is inquiry.
6. "Hi" or "Hello" is greeting.
7. If genuinely unclear, output: inquiry

Output one word only."""

    response = _safe_llm_invoke(_get_llm_deterministic(), [
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_message)
    ])

    raw_intent = _extract_text(response).lower()

    # Validate the output — LLMs can occasionally output unexpected text
    valid_intents = {"greeting", "inquiry", "high_intent"}
    intent = raw_intent if raw_intent in valid_intents else "inquiry"

    return {"intent": intent}


# ─────────────────────────────────────────────
# NODE 2: Greeting Handler
# ─────────────────────────────────────────────
def handle_greeting(state: AgentState) -> dict:
    """
    Generates a friendly, brand-aware greeting response.
    Keeps the conversation open by inviting product questions.
    """
    system_prompt = """You are a friendly sales assistant for AutoStream,
an AI-powered video editing SaaS for content creators.
Respond warmly to greetings. Keep it to 2 sentences maximum.
Invite them to ask about our plans or features. Do not make up any prices or features."""

    response = _safe_llm_invoke(_get_llm_generative(), [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])

    return {"messages": [AIMessage(content=_extract_text(response))]}


# ─────────────────────────────────────────────
# NODE 3: RAG Retrieval + Response
# ─────────────────────────────────────────────
def handle_inquiry(state: AgentState) -> dict:
    """
    Retrieves relevant context from the knowledge base and generates
    a grounded, accurate response.

    The system prompt explicitly prohibits the LLM from making up
    any information not present in the provided context. This is the
    most critical safety rule in the entire agent — without it, the LLM
    will hallucinate pricing details confidently and incorrectly.

    The retrieved context is appended to the system prompt, not to the
    conversation history. This keeps the conversation history clean and
    prevents context from accumulating across turns.
    """
    last_message = state["messages"][-1].content
    context = retrieve_context(last_message)

    system_prompt = f"""You are a knowledgeable sales assistant for AutoStream.

CONTEXT (use ONLY this information to answer — do not invent any details):
{context}

Rules:
1. Answer only using information from the CONTEXT above.
2. If the answer is not in the context, say: "I don't have that information right now.
   Would you like to speak with our sales team?"
3. Format pricing responses clearly — use the plan name, price, and key features.
4. Keep responses concise — 3-5 sentences maximum unless the question requires more.
5. End responses with a gentle engagement question when natural."""

    response = _safe_llm_invoke(_get_llm_generative(), [
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])

    return {
        "messages": [AIMessage(content=_extract_text(response))],
        "rag_context": context
    }


# ─────────────────────────────────────────────
# NODE 4: High-Intent Detector + Lead Collection
# ─────────────────────────────────────────────
def handle_high_intent(state: AgentState) -> dict:
    """
    This node runs when the user shows high intent to sign up.

    It implements a sequential field collection pattern:
    1. If lead already captured → send confirmation message, stop.
    2. Inspect awaiting_field to know what to ask for next.
    3. If awaiting_field is set → the user's last message IS the answer
       to that field. Extract and validate it, then ask for the next field.
    4. If no awaiting_field → first time entering this node. Start by
       acknowledging intent and asking for name.

    Edge cases:
    - User provides all three fields in one message: handled by extracting
      all three before proceeding.
    - User provides invalid email: re-ask with a friendly error message.
    - User tries to skip a field: agent gently re-asks for the same field.
    - User asks a question mid-collection (contains "?"): pause and answer briefly,
      then re-ask the current field.
    """

    # Guard: do not run if lead already captured
    if state.get("lead_captured"):
        response = "Your information has already been captured! Our team will reach out to you shortly."
        return {"messages": [AIMessage(content=response)]}

    awaiting = state.get("awaiting_field")
    last_message = state["messages"][-1].content.strip()

    updates = {}
    response_text = ""

    # Edge Case 3: user asks a question mid-collection — detect "?" and pause
    if awaiting is not None and "?" in last_message:
        context = retrieve_context(last_message)
        field_prompt_map = {
            "name": "What's your full name?",
            "email": "What's your email address?",
            "platform": "Which creator platform do you primarily use? (e.g. YouTube, Instagram, TikTok, X)"
        }
        system_prompt = f"""You are a helpful assistant for AutoStream.
The user is in the middle of signing up but asked a question. Answer briefly using only this context:
{context}
After answering, remind them you still need their {awaiting} to complete sign-up."""

        resp = _safe_llm_invoke(_get_llm_generative(), [
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_message)
        ])
        return {"messages": [AIMessage(content=_extract_text(resp))]}

    if awaiting is None:
        # Edge Case 1: check if the user provided all three fields in one message
        # Look for email pattern
        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', last_message)
        platforms = ["youtube", "instagram", "tiktok", "twitter", "x", "facebook", "linkedin"]
        platform_found = next((p for p in platforms if p in last_message.lower()), None)

        if email_match and platform_found:
            # Extract all three from the single message
            email = email_match.group(0)
            # Name is whatever is before the email (rough heuristic)
            name_part = last_message[:last_message.lower().find(email_match.group(0).lower())].strip().rstrip(",").strip()
            if len(name_part) >= 2 and not name_part.isdigit():
                updates["lead_name"] = name_part
                updates["lead_email"] = email
                updates["lead_platform"] = platform_found.capitalize()
                updates["awaiting_field"] = None
                response_text = f"Great, {name_part.split()[0]}! Let me confirm your details and get you set up. Just a moment..."
                updates["messages"] = [AIMessage(content=response_text)]
                return updates

        # First time in high_intent — greet and ask for name
        response_text = (
            "Excellent! I'd love to get you set up with AutoStream Pro. "
            "Let me collect a few quick details.\n\n"
            "What's your full name?"
        )
        updates["awaiting_field"] = "name"

    elif awaiting == "name":
        # Validate: name should be at least 2 characters, no numbers
        if len(last_message) >= 2 and not last_message.isdigit():
            updates["lead_name"] = last_message
            updates["awaiting_field"] = "email"
            response_text = f"Great to meet you, {last_message.split()[0]}! What's your email address?"
        else:
            response_text = "That doesn't look like a name. Could you share your full name?"

    elif awaiting == "email":
        # Validate email format
        if validate_email(last_message):
            updates["lead_email"] = last_message
            updates["awaiting_field"] = "platform"
            response_text = (
                "Perfect! Last question — which creator platform do you primarily use? "
                "(e.g. YouTube, Instagram, TikTok, X)"
            )
        else:
            response_text = (
                "That email address doesn't look quite right. "
                "Could you double-check and enter it again?"
            )

    elif awaiting == "platform":
        # Edge Case 4: user tries to skip the platform field
        skip_phrases = ["skip", "doesn't matter", "dont matter", "no matter", "n/a", "none"]
        if any(phrase in last_message.lower() for phrase in skip_phrases):
            response_text = (
                "We need your primary platform to set up your account correctly. "
                "Which platform do you create content for? (e.g. YouTube, Instagram, TikTok, X)"
            )
        elif len(last_message) >= 2:
            updates["lead_platform"] = last_message
            updates["awaiting_field"] = None
            # All three fields are now collected — graph will route to execute_lead_capture
        else:
            response_text = "Which platform do you mainly create content for?"

    if response_text:
        updates["messages"] = [AIMessage(content=response_text)]

    return updates


# ─────────────────────────────────────────────
# NODE 5: Lead Capture Tool Execution
# ─────────────────────────────────────────────
def execute_lead_capture(state: AgentState) -> dict:
    """
    Called only when all three lead fields are confirmed non-None
    and lead_captured is False.

    This is the ONLY node that calls mock_lead_capture().
    The graph's conditional edge to this node ensures it is NEVER
    triggered prematurely — the condition checks all three fields
    plus the lead_captured guard.

    On success: sets lead_captured=True and sends confirmation.
    On failure: catches ValueError from the tool and asks the user
    to correct the invalid data (should not happen due to prior
    validation, but handled defensively).
    """
    try:
        result = mock_lead_capture(
            name=state["lead_name"],
            email=state["lead_email"],
            platform=state["lead_platform"]
        )

        confirmation = (
            f"You're all set! We've captured your details and our team will "
            f"reach out to {state['lead_email']} within 24 hours to get your "
            f"AutoStream Pro trial activated. Welcome aboard!"
        )

        return {
            "messages": [AIMessage(content=confirmation)],
            "lead_captured": True
        }

    except ValueError as e:
        error_response = (
            f"There was an issue with your information: {str(e)}. "
            f"Could you provide it again?"
        )
        return {
            "messages": [AIMessage(content=error_response)],
            "lead_captured": False
        }


# ─────────────────────────────────────────────
# NODE 6: Fallback Handler
# ─────────────────────────────────────────────
def handle_unknown(state: AgentState) -> dict:
    """
    Catches any intent that doesn't match the three main categories.
    Also handles nonsensical input, gibberish, or out-of-scope questions.
    Redirects the user gently back to AutoStream topics.
    """
    response = (
        "I'm AutoStream's assistant, so I'm best equipped to help you with "
        "our video editing plans, pricing, and features. Is there something "
        "specific about AutoStream I can help you with?"
    )
    return {"messages": [AIMessage(content=response)]}
