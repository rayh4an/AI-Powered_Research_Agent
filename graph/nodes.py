from graph.state import ResearchState
from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.critic import run_critic
from agents.synthesizer import run_synthesizer
from agents.writer import run_writer


def planner_node(state: ResearchState) -> ResearchState:
    """Node 1: Takes the user query and breaks it into sub-questions."""
    print("\n[PLANNER] Breaking query into sub-questions...")

    query = state["query"]
    sub_questions = run_planner(query)

    print(f"[PLANNER] Generated {len(sub_questions)} sub-questions:")
    for i, q in enumerate(sub_questions, 1):
        print(f"  {i}. {q}")

    return {
        **state,
        "sub_questions": sub_questions,
        "iteration_count": 0,
        "follow_up_questions": []
    }


def researcher_node(state: ResearchState) -> ResearchState:
    """Node 2: Takes sub-questions and searches the web for each one."""
    print("\n[RESEARCHER] Searching the web for each sub-question...")

    iteration = state.get("iteration_count", 0)

    if iteration > 0 and state.get("follow_up_questions"):
        questions = state["follow_up_questions"]
        print(f"[RESEARCHER] Running follow-up searches (iteration {iteration})...")
    else:
        questions = state["sub_questions"]

    new_results = run_researcher(questions)

    existing_results = state.get("research_results", [])
    combined_results = existing_results + new_results

    print(f"[RESEARCHER] Completed {len(new_results)} searches.")

    return {
        **state,
        "research_results": combined_results
    }


def critic_node(state: ResearchState) -> ResearchState:
    """Node 3: Evaluates research quality and decides whether to continue."""
    print("\n[CRITIC] Evaluating research quality...")

    query = state["query"]
    research_results = state["research_results"]
    iteration_count = state.get("iteration_count", 0)

    critique_result = run_critic(query, research_results)

    score = critique_result["score"]
    passed = critique_result["passed"]
    critique = critique_result["critique"]
    follow_up_questions = critique_result["follow_up_questions"]

    print(f"[CRITIC] Score: {score:.2f}/1.00")
    print(f"[CRITIC] Passed: {passed}")
    print(f"[CRITIC] Critique: {critique}")

    if not passed and follow_up_questions:
        print(f"[CRITIC] Follow-up questions needed:")
        for i, q in enumerate(follow_up_questions, 1):
            print(f"  {i}. {q}")

    return {
        **state,
        "critique": critique,
        "critique_score": score,
        "critique_passed": passed,
        "follow_up_questions": follow_up_questions,
        "iteration_count": iteration_count + 1
    }


def synthesizer_node(state: ResearchState) -> ResearchState:
    """Node 4: Merges all research findings into a coherent synthesis."""
    print("\n[SYNTHESIZER] Merging all research findings...")

    query = state["query"]
    research_results = state["research_results"]
    critique = state.get("critique", "")

    synthesis = run_synthesizer(query, research_results, critique)

    print(f"[SYNTHESIZER] Synthesis complete ({len(synthesis)} characters)")

    return {
        **state,
        "synthesis": synthesis
    }


def writer_node(state: ResearchState) -> ResearchState:
    """Node 5: Turns the synthesis into a polished final report."""
    print("\n[WRITER] Generating final report...")

    query = state["query"]
    synthesis = state["synthesis"]
    research_results = state["research_results"]

    final_report = run_writer(query, synthesis, research_results)

    print(f"[WRITER] Report complete ({len(final_report)} characters)")

    return {
        **state,
        "final_report": final_report
    }