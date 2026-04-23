from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from graph.nodes import (
    planner_node,
    researcher_node,
    critic_node,
    synthesizer_node,
    writer_node
)
from graph.edges import should_continue
from memory.checkpointer import get_checkpointer
from memory.session import generate_thread_id, list_all_sessions, get_session_query
from utils import save_report
import sys

load_dotenv()


def build_graph(checkpointer):
    """Builds and returns the full research agent graph with memory."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")

    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "passed": "synthesizer",
            "needs_improvement": "researcher"
        }
    )

    graph.add_edge("synthesizer", "writer")
    graph.add_edge("writer", END)

    return graph.compile(checkpointer=checkpointer)


def run_research(query: str, thread_id: str = None):
    """
    Runs the full research graph on a given query.
    Optionally accepts a thread_id to resume a previous session.
    """
    print(f"\n{'='*60}")
    print(f"Research Query: {query}")
    print(f"{'='*60}")

    checkpointer = get_checkpointer()

    if thread_id is None:
        thread_id = generate_thread_id()

    print(f"[MEMORY] Session ID: {thread_id}")

    config = {"configurable": {"thread_id": thread_id}}

    app = build_graph(checkpointer)

    initial_state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "research_results": [],
        "critique": "",
        "critique_score": 0.0,
        "critique_passed": False,
        "follow_up_questions": [],
        "iteration_count": 0,
        "max_iterations": 3,
        "synthesis": "",
        "final_report": "",
        "messages": []
    }

    try:
        final_state = app.invoke(initial_state, config=config)
        return final_state, thread_id

    except Exception as e:
        print(f"\n[ERROR] Graph execution failed: {e}")
        print(f"[MEMORY] Session saved as: {thread_id}")
        print(f"[MEMORY] You can resume this session later using thread_id: {thread_id}")
        raise


def print_results(result, session_id):
    """Prints the final report and run statistics."""
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(result["final_report"])

    print(f"\n{'='*60}")
    print("RUN STATISTICS")
    print(f"{'='*60}")
    print(f"Session ID:                {session_id}")
    print(f"Total questions researched: {len(result['research_results'])}")
    print(f"Critic score:              {result['critique_score']:.2f}/1.00")
    print(f"Iterations completed:      {result['iteration_count']}")
    print(f"Report length:             {len(result['final_report'])} characters")

    save_report(result["final_report"], result.get("query", "unknown"))


if __name__ == "__main__":

    while True:
        print("\n" + "="*60)
        print("RESEARCH AGENT")
        print("="*60)
        print("What would you like to research?")
        print("(e.g. What are the latest advancements in large language models?)")
        print("\nEnter 0 to resume a previous session.")
        print("Enter 1 to quit.")
        print("="*60)

        user_input = input("\n> ").strip()

        # ── Quit ──
        if user_input == "1":
            print("\nGoodbye!")
            sys.exit(0)

        # ── Resume Mode ──
        elif user_input == "0":
            while True:
                print("\n" + "="*60)
                print("RESUME A PREVIOUS SESSION")
                print("="*60)
                print("Enter your Session ID to resume.")#example: research_20260423_175327_d379b8b0
                print("Enter 0 to go back to the research menu.")
                print("="*60)

                session_input = input("\nSession ID > ").strip()

                # Back to main menu
                if session_input == "0":
                    break

                if not session_input:
                    print("Please enter a valid session ID.")
                    continue

                # Look up the query automatically from the checkpoint
                checkpointer = get_checkpointer()
                saved_query = get_session_query(checkpointer, session_input)

                if not saved_query:
                    print(f"\n[ERROR] Could not find session '{session_input}'.")
                    print("Please double check your session ID and try again.")
                    continue

                print(f"\n[MEMORY] Found session with query: \"{saved_query}\"")
                print("[MEMORY] Resuming session...")

                try:
                    result, session_id = run_research(saved_query, thread_id=session_input)
                    print_results(result, session_id)
                except Exception as e:
                    print(f"\n[ERROR] Could not resume session: {e}")

                break  # Return to main menu after resume

        # ── New Research Mode ──
        elif user_input:
            try:
                result, session_id = run_research(user_input)
                print_results(result, session_id)
            except Exception as e:
                print(f"\n[ERROR] Research failed: {e}")

        else:
            print("Please enter a query, 0 to resume a session, or 1 to quit.")

    # Save report to file
    save_report(result["final_report"], query)