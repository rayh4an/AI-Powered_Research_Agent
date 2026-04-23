from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import json
import time

load_dotenv()


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research planner. Your job is to take a user's research 
query and break it down into EXACTLY 5 specific sub-questions that together will 
fully answer the original query.

Rules:
- You MUST generate EXACTLY 5 sub-questions, no more, no less
- Each sub-question should be specific and searchable
- Sub-questions should cover different aspects of the topic
- Return ONLY a JSON array of exactly 5 strings, nothing else

Example output:
["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]
"""),
    ("human", "Research query: {query}")
])


def run_planner(query: str, max_retries: int = 3) -> list[str]:
    """
    Takes a query and returns exactly 5 sub-questions.
    Retries up to max_retries times on failure.
    """
    llm = get_llm()
    chain = PLANNER_PROMPT | llm

    for attempt in range(max_retries):
        try:
            response = chain.invoke({"query": query})

            content = response.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            sub_questions = json.loads(content)

            if isinstance(sub_questions, list):
                # Enforce exactly 5
                if len(sub_questions) > 5:
                    sub_questions = sub_questions[:5]
                while len(sub_questions) < 5:
                    sub_questions.append(f"What else should be known about {query}?")
                return sub_questions

        except json.JSONDecodeError as e:
            print(f"  [PLANNER] JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

        except Exception as e:
            print(f"  [PLANNER] Error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    # Fallback: 5 variations of the original query
    print("  [PLANNER] All retries failed, using fallback sub-questions")
    return [
        f"What is {query}?",
        f"What are the latest developments in {query}?",
        f"What are the key challenges related to {query}?",
        f"What are the main applications of {query}?",
        f"What is the future outlook for {query}?"
    ]