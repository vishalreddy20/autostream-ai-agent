# rag/retriever.py

import json
import os

KNOWLEDGE_PATH = os.path.join(os.path.dirname(__file__), "knowledge.json")

def load_knowledge() -> dict:
    """Load the full knowledge base from disk."""
    with open(KNOWLEDGE_PATH, "r") as f:
        return json.load(f)

def retrieve_context(query: str) -> str:
    """
    Search the knowledge base for content relevant to the user's query.
    Returns a formatted string that the LLM will use as grounding context.

    Strategy:
    - Always include product name and tagline.
    - If query mentions price/cost/plan/basic/pro → include plans.
    - If query mentions refund/cancel/support/trial → include policies.
    - If no keyword match → return full knowledge base as context.

    This keyword routing prevents the LLM from hallucinating details
    about plans or policies that aren't in the knowledge base.
    """
    kb = load_knowledge()
    query_lower = query.lower()

    context_parts = [
        f"Product: {kb['product']}",
        f"Description: {kb['tagline']}"
    ]

    pricing_keywords = ["price", "cost", "plan", "basic", "pro",
                        "month", "4k", "resolution", "video", "caption",
                        "unlimited", "how much", "pricing", "tier"]

    policy_keywords = ["refund", "cancel", "support", "trial", "free",
                       "policy", "guarantee", "return", "upgrade"]

    include_plans = any(kw in query_lower for kw in pricing_keywords)
    include_policies = any(kw in query_lower for kw in policy_keywords)

    # If no keyword matches, include everything (safe fallback)
    if not include_plans and not include_policies:
        include_plans = True
        include_policies = True

    if include_plans:
        context_parts.append("\n--- PRICING PLANS ---")
        for plan in kb["plans"]:
            captions = "Yes" if plan["ai_captions"] else "No"
            context_parts.append(
                f"{plan['name']} Plan: ${plan['price_monthly']}/month | "
                f"{plan['videos_per_month']} videos/month | "
                f"{plan['resolution']} resolution | AI Captions: {captions} | "
                f"Support: {plan['support']}"
            )

    if include_policies:
        context_parts.append("\n--- POLICIES ---")
        for policy in kb["policies"]:
            context_parts.append(f"- {policy['detail']}")

        context_parts.append("\n--- FAQs ---")
        for faq in kb["faqs"]:
            if any(kw in query_lower for kw in faq["q"].lower().split()):
                context_parts.append(f"Q: {faq['q']}\nA: {faq['a']}")

    return "\n".join(context_parts)
