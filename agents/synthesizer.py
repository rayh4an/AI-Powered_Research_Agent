from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")

    # --- PAID ROUTE A: OpenAI ---
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(model="gpt-4o-mini")

    # --- PAID ROUTE B: Anthropic ---
    # from langchain_anthropic import ChatAnthropic
    # return ChatAnthropic(model="claude-haiku-4-5-20251001")


SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a research synthesizer. Your job is to take multiple 
research findings and merge them into a single coherent, well-organized synthesis.

Rules:
- Combine related findings together
- Remove any duplicate information
- Resolve any contradictions by noting different perspectives
- Preserve all important facts and insights
- Organize findings into clear themes
- Keep track of which sources support which claims
- Write in clear, academic but readable prose
"""),
    ("human", """Original research query: {query}

Research findings to synthesize:
{research_findings}

Critic feedback to address:
{critique}

Produce a thorough synthesis of all the findings.""")
])


def format_research_for_synthesis(research_results: list[dict]) -> str:
    """Formats research results into a detailed string for the synthesizer."""
    formatted = ""
    for i, item in enumerate(research_results, 1):
        formatted += f"\n{'='*40}\n"
        formatted += f"Finding {i}\n"
        formatted += f"Question: {item['question']}\n"
        formatted += f"Answer: {item['answer']}\n"

        if item.get('sources'):
            formatted += "Sources:\n"
            for source in item['sources'][:3]:
                if isinstance(source, dict):
                    url = source.get('url', 'N/A')
                    content = source.get('content', '')[:200]
                    formatted += f"  - {url}\n"
                    if content:
                        formatted += f"    Preview: {content}...\n"

    return formatted


def run_synthesizer(
    query: str,
    research_results: list[dict],
    critique: str
) -> str:
    """Synthesizes all research results into a coherent summary."""
    llm = get_llm()
    chain = SYNTHESIZER_PROMPT | llm

    research_findings = format_research_for_synthesis(research_results)

    response = chain.invoke({
        "query": query,
        "research_findings": research_findings,
        "critique": critique
    })

    return response.content