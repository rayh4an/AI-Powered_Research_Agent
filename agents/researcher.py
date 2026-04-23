from tools.search import get_search_tool
from dotenv import load_dotenv
import time

load_dotenv()


def run_researcher(
    sub_questions: list[str],
    max_retries: int = 3
) -> list[dict]:
    """
    Takes a list of sub-questions and searches the web for each one.
    Retries failed searches up to max_retries times.
    """
    search_tool = get_search_tool(max_results=3)
    all_results = []

    for question in sub_questions:
        print(f"  Searching: {question}")

        success = False
        for attempt in range(max_retries):
            try:
                raw = search_tool.invoke(question)

                answer = raw.get("answer", "")
                sources = raw.get("results", [])

                all_results.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })

                success = True
                break

            except Exception as e:
                print(f"  [RESEARCHER] Search failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        if not success:
            print(f"  [RESEARCHER] All retries failed for: {question}")
            all_results.append({
                "question": question,
                "answer": "Search failed after multiple attempts.",
                "sources": []
            })

    return all_results