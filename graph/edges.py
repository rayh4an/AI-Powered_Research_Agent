from graph.state import ResearchState


def should_continue(state: ResearchState) -> str:
    """
    Routes the graph after the critic node.
    - If critique passed or max iterations reached: move to synthesizer
    - If critique failed: loop back to researcher
    """
    critique_passed = state.get("critique_passed", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)

    if critique_passed:
        print(f"\n[ROUTER] Research passed! Moving to synthesis...")
        return "passed"

    if iteration_count >= max_iterations:
        print(f"\n[ROUTER] Max iterations ({max_iterations}) reached. Moving to synthesis...")
        return "passed"

    print(f"\n[ROUTER] Research needs improvement. Looping back (iteration {iteration_count})...")
    return "needs_improvement"