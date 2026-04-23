# agent/graph.py

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent.state import AgentState
from agent.nodes import (
    classify_intent,
    handle_greeting,
    handle_inquiry,
    handle_high_intent,
    execute_lead_capture,
    handle_unknown
)


def route_by_intent(state: AgentState) -> str:
    """
    Routes to the correct handler node after intent classification.

    This function is called by LangGraph's conditional_edges.
    The string it returns must exactly match a node name in the graph.

    Special case: If we're in the middle of collecting lead fields
    (awaiting_field is set), skip re-classification and go directly
    to handle_high_intent. This prevents the classifier from
    misinterpreting field values (like an email address) as a greeting.
    """
    # If collecting lead info, bypass intent routing
    if state.get("awaiting_field") is not None:
        return "handle_high_intent"

    intent = state.get("intent", "unknown")

    routing = {
        "greeting": "handle_greeting",
        "inquiry": "handle_inquiry",
        "high_intent": "handle_high_intent",
        "unknown": "handle_unknown"
    }

    return routing.get(intent, "handle_unknown")


def should_capture_lead(state: AgentState) -> str:
    """
    After handle_high_intent runs, determine the next step.

    Routes to execute_lead_capture ONLY when:
    1. All three fields are collected (not None, not empty).
    2. awaiting_field is None (collection is complete).
    3. lead_captured is False (not already done).

    Otherwise routes to END (the agent already sent a response asking
    for the next field — the conversation continues next turn).
    """
    all_collected = (
        state.get("lead_name") is not None and
        state.get("lead_email") is not None and
        state.get("lead_platform") is not None and
        state.get("awaiting_field") is None and
        not state.get("lead_captured", False)
    )

    return "execute_lead_capture" if all_collected else END


def build_graph():
    """
    Constructs the full LangGraph StateGraph.

    Flow:
    START → classify_intent → [route_by_intent] → handler node

    handle_high_intent → [should_capture_lead] →
        execute_lead_capture → END
        OR
        END (waiting for next user input)

    All other handlers → END

    Memory: MemorySaver stores the full AgentState between turns.
    thread_id: Each conversation gets a unique thread_id so multiple
    users can run concurrently without state collision.
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_greeting", handle_greeting)
    graph.add_node("handle_inquiry", handle_inquiry)
    graph.add_node("handle_high_intent", handle_high_intent)
    graph.add_node("execute_lead_capture", execute_lead_capture)
    graph.add_node("handle_unknown", handle_unknown)

    # Entry point: always classify intent first
    graph.set_entry_point("classify_intent")

    # After classification, route to the appropriate handler
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_greeting": "handle_greeting",
            "handle_inquiry": "handle_inquiry",
            "handle_high_intent": "handle_high_intent",
            "handle_unknown": "handle_unknown"
        }
    )

    # After lead collection attempt, decide whether to capture or wait
    graph.add_conditional_edges(
        "handle_high_intent",
        should_capture_lead,
        {
            "execute_lead_capture": "execute_lead_capture",
            END: END
        }
    )

    # Terminal nodes: all go to END
    graph.add_edge("handle_greeting", END)
    graph.add_edge("handle_inquiry", END)
    graph.add_edge("execute_lead_capture", END)
    graph.add_edge("handle_unknown", END)

    # Compile with persistent memory
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
