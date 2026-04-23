from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    # The original user query
    query: str

    # Planner breaks query into sub-questions
    sub_questions: List[str]

    # Raw research results from sub-agents
    research_results: List[dict]

    # Critic's evaluation of the research
    critique: str
    critique_score: float          # 0.0 - 1.0
    critique_passed: bool

    # Questions the critic flagged as needing more research
    follow_up_questions: List[str]

    # How many reflection loops have run
    iteration_count: int
    max_iterations: int            # safety limit to prevent infinite loops

    # Final outputs
    synthesis: str
    final_report: str

    # Conversation/message history
    messages: Annotated[list, add_messages]