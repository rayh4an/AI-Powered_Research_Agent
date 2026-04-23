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


WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research report writer. Your job is to take 
a research synthesis and turn it into a polished, well-structured report.

The report MUST follow this exact structure ONCE and ONLY ONCE:

# [Report Title]

## Executive Summary
A 3-4 sentence overview of the key findings.

## Introduction
Brief background on the topic and why it matters.

## Key Findings

### [Theme 1 Title]
Detailed discussion of first major theme.

### [Theme 2 Title]
Detailed discussion of second major theme.

### [Theme 3 Title]
Detailed discussion of third major theme.

## Challenges and Limitations
Discussion of gaps, challenges, or limitations identified.

## Conclusion
Summary of findings and implications.

## Sources
List all sources cited in the report.

CRITICAL RULES:
- Each section must appear EXACTLY ONCE in the report
- Do NOT repeat the Conclusion section
- Do NOT repeat the Sources section
- Do NOT add any content after the Sources section
- The report ends after the final source is listed
- Write in clear, professional prose
- Cite sources inline using [Source: URL] format
"""),
    ("human", """Research query: {query}

Synthesis to turn into a report:
{synthesis}

Source URLs available:
{sources}

Write the full research report now. Remember: each section appears exactly once.""")
])


def collect_sources(research_results: list[dict]) -> str:
    """Collects all unique source URLs from research results."""
    sources = []
    seen_urls = set()

    for item in research_results:
        for source in item.get('sources', []):
            if isinstance(source, dict):
                url = source.get('url', '')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    title = source.get('title', url)
                    sources.append(f"- {title}: {url}")

    return "\n".join(sources) if sources else "No sources available"


def run_writer(
    query: str,
    synthesis: str,
    research_results: list[dict]
) -> str:
    """Takes the synthesis and produces a final polished report."""
    llm = get_llm()
    chain = WRITER_PROMPT | llm

    sources = collect_sources(research_results)

    response = chain.invoke({
        "query": query,
        "synthesis": synthesis,
        "sources": sources
    })

    return response.content