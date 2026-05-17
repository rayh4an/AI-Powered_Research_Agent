# This project is an AI-powered Research Agent system built with LangGraph and LangChain. 
Given any research topic, the Research Agent autonomously breaks the query into 5 targeted sub-questions, searches the live web for each one, and then runs the results through an independent critic agent that scores the research quality from 0.0 to 1.0. If the score falls below 0.7, the system automatically generates follow-up questions and loops back to search again, repeating until the quality threshold is met. Once the research passes, a synthesizer merges all findings, and a writer agent produces a clean, fully cited report. Every session is saved to a local database so you can resume any previous research without starting over.

# Tech Stack
- LangGraph — orchestrates the multi-agent pipeline
- LangChain — LLM framework and tool wrappers
- Groq (Llama 3.3 70B) — LLM powering all agents
- Tavily API — live web search for each sub-question
- SQLite — persistent memory and session management
- Streamlit — web UI with real-time pipeline updates

# Setup
# 1. Add a .env file with the following keys:
- GROQ_API_KEY=your_key
- TAVILY_API_KEY=your_key
- LANGSMITH_API_KEY=your_key

# 2. Create and activate a virtual environment:
- python -m venv venv
- venv\Scripts\activate

# 3. Install dependencies:
- pip install -r requirements.txt
- pip install aiosqlite langgraph-checkpoint-sqlite langchain-tavily

# 4. To run the project, run this command in your virtual environment:
- streamlit run frontend/app.py
