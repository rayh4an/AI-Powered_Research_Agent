from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import time

load_dotenv()


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")


CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research critic. Your job is to evaluate research results 
and determine if they sufficiently answer the original query.

You must respond ONLY with a JSON object in this exact format:
{{
    "score": 0.0,
    "passed": false,
    "critique": "Your critique here",
    "follow_up_questions": []
}}

Scoring guide:
- 0.0 to 0.4: Poor — major gaps, missing key information
- 0.5 to 0.6: Adequate — covers basics but missing depth
- 0.7 to 0.8: Good — mostly complete with minor gaps
- 0.9 to 1.0: Excellent — comprehensive and thorough

Rules:
- Set "passed" to true if score >= 0.7
- If passed is false, provide 2-3 specific follow_up_questions to fill the gaps
- If passed is true, follow_up_questions should be an empty list
- Be strict but fair in your scoring
"""),
    ("human", """Original query: {query}

Research results:
{research_summary}

Critic feedback to address:
{critique}

Evaluate the research and respond with the JSON object.""")
])


def format_research_summary(research_results: list[dict]) -> str:
    summary = ""
    for i, item in enumerate(research_results, 1):
        summary += f"\n{i}. Question: {item['question']}\n"
        summary += f"   Answer: {item['answer']}\n"
        summary += f"   Sources: {len(item['sources'])} found\n"
    return summary


def run_critic(
    query: str,
    research_results: list[dict],
    max_retries: int = 3
) -> dict:
    """
    Evaluates research results and returns a critique with score.
    Retries up to max_retries times on failure.
    """
    llm = get_llm()
    chain = CRITIC_PROMPT | llm

    research_summary = format_research_summary(research_results)

    for attempt in range(max_retries):
        try:
            response = chain.invoke({
                "query": query,
                "research_summary": research_summary,
                "critique": ""
            })

            content = response.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)

            return {
                "score": float(result.get("score", 0.0)),
                "passed": bool(result.get("passed", False)),
                "critique": result.get("critique", ""),
                "follow_up_questions": result.get("follow_up_questions", [])
            }

        except json.JSONDecodeError as e:
            print(f"  [CRITIC] JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

        except Exception as e:
            print(f"  [CRITIC] Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    # Fallback to avoid infinite loops
    print("  [CRITIC] All retries failed, defaulting to pass")
    return {
        "score": 0.5,
        "passed": True,
        "critique": "Critic evaluation failed, proceeding with available research.",
        "follow_up_questions": []
    }