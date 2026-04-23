# main.py

import os
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from agent.graph import build_graph

load_dotenv()  # Load ANTHROPIC_API_KEY from .env

def run_agent():
    """
    CLI conversation loop for the AutoStream agent.

    Each session gets a unique thread_id. LangGraph's MemorySaver
    uses this ID to store and retrieve state between turns.

    The initial state is set on the first invocation. On subsequent
    invocations, LangGraph loads the full state from memory using
    the thread_id — no manual state passing required.
    """
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state — all fields start as None/False/empty
    initial_state = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "awaiting_field": None,
        "rag_context": None
    }

    print("\n" + "="*60)
    print("  AutoStream AI Assistant")
    print("  Type 'quit' or 'exit' to end the conversation")
    print("="*60 + "\n")

    # First run: initialize state
    first_run = True

    while True:
        user_input = input("You: ").strip()

        # Edge Case 6: skip empty or whitespace-only input
        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\nAssistant: Thanks for chatting! Have a great day.\n")
            break

        # Build the input for this turn
        turn_input = {"messages": [HumanMessage(content=user_input)]}

        if first_run:
            # Merge initial state with first message
            turn_input = {**initial_state, **turn_input}
            first_run = False

        try:
            result = graph.invoke(turn_input, config=config)

            # Extract and print the last AI message
            ai_messages = [
                m for m in result["messages"]
                if hasattr(m, 'type') and m.type == "ai"
            ]

            if ai_messages:
                print(f"\nAssistant: {ai_messages[-1].content}\n")
            else:
                print("\nAssistant: (no response generated)\n")

        except Exception as e:
            print(f"\n[ERROR] An unexpected error occurred: {e}")
            print("Please try again or type 'quit' to exit.\n")

if __name__ == "__main__":
    run_agent()
