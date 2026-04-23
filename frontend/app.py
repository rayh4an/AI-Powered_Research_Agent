import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from memory.session import generate_thread_id, get_session_query
from utils import save_report

load_dotenv()

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    .app-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        color: #f1f5f9;
        line-height: 1.1;
        margin: 0 0 0.15rem 0;
    }
    .app-title span { color: #60a5fa; font-style: italic; }

    .app-subtitle {
        font-size: 0.9rem;
        color: #94a3b8;
        margin: 0 0 1.25rem 0;
        font-weight: 300;
    }

    /* Pipeline strip — transparent bg, white text */
    .pipeline-strip {
        display: flex;
        gap: 0;
        margin: 0 0 1.5rem 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #334155;
    }
    .pipeline-step {
        flex: 1;
        padding: 0.9rem 0.6rem;
        text-align: center;
        background: transparent;
        border-right: 1px solid #334155;
        transition: background 0.2s;
    }
    .pipeline-step:last-child { border-right: none; }
    .pipeline-step:hover { background: rgba(96,165,250,0.07); }
    .pipeline-icon { font-size: 1.4rem; display: block; margin-bottom: 0.25rem; }
    .pipeline-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        display: block;
    }
    .pipeline-desc {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 0.2rem;
        display: block;
        line-height: 1.35;
    }

    hr { border-color: #1e293b; margin: 1rem 0; }

    /* Input labels */
    .input-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }

    /* Status pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.55rem 1rem;
        border-radius: 100px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.2rem 0;
    }
    .status-running { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
    .status-done    { background:#dcfce7; color:#14532d; border:1px solid #86efac; }
    .status-error   { background:#fee2e2; color:#7f1d1d; border:1px solid #fca5a5; }

    /* Session tag */
    .session-tag {
        display: inline-block;
        background: transparent;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.35rem 0.7rem;
        font-family: monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 0.6rem;
    }

    /* Metric cards */
    .metric-row { display:flex; gap:0.75rem; margin:0.75rem 0; }
    .metric-card {
        flex:1; background:transparent;
        border:1px solid #334155;
        border-radius:10px; padding:0.9rem;
        text-align:center;
    }
    .metric-value {
        font-family:'DM Serif Display',serif;
        font-size:1.5rem; color:#f1f5f9;
    }
    .metric-label {
        font-size:0.68rem; color:#64748b;
        text-transform:uppercase;
        letter-spacing:0.05em; margin-top:0.15rem;
    }

    /* Example label */
    .example-label {
        font-size:0.68rem; font-weight:600;
        color:#64748b; text-transform:uppercase;
        letter-spacing:0.07em; margin-bottom:0.4rem;
    }

    /* Suppress Streamlit's form-submit-on-enter for textareas */
    textarea { resize: vertical; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Session State Init
# ─────────────────────────────────────────
for key, default in {
    "running": False,
    "cancelled": False,
    "final_state": None,
    "session_id": None,
    "mode": "research",
    "active_query": "",
    "trigger_query": None,   # used to fire quick-example runs
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────
def build_graph(checkpointer):
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
        "critic", should_continue,
        {"passed": "synthesizer", "needs_improvement": "researcher"}
    )
    graph.add_edge("synthesizer", "writer")
    graph.add_edge("writer", END)
    return graph.compile(checkpointer=checkpointer)


# ─────────────────────────────────────────
# Research Runner
# ─────────────────────────────────────────
def run_research_streaming(query: str, thread_id: str, status_placeholder):
    checkpointer = get_checkpointer()
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

    def show(msg, kind="running"):
        status_placeholder.markdown(
            f'<div class="status-pill status-{kind}">{msg}</div>',
            unsafe_allow_html=True
        )

    show("⏳ Planning sub-questions...", "running")

    final_state = None
    prev_sub_q = []
    prev_results_count = 0
    prev_critique = ""
    prev_synthesis = ""

    for step in app.stream(initial_state, config=config, stream_mode="values"):
        if st.session_state.cancelled:
            show("🛑 Cancelled.", "error")
            return None

        final_state = step
        sub_q     = step.get("sub_questions", [])
        results   = step.get("research_results", [])
        critique  = step.get("critique", "")
        synthesis = step.get("synthesis", "")
        report    = step.get("final_report", "")
        score     = step.get("critique_score", 0.0)
        passed    = step.get("critique_passed", False)
        iteration = step.get("iteration_count", 0)

        if sub_q and sub_q != prev_sub_q and not results:
            show(f"⏳ Researching {len(sub_q)} sub-questions...", "running")
            prev_sub_q = sub_q
        elif results and len(results) != prev_results_count and not critique:
            show(f"⏳ Critic evaluating {len(results)} results...", "running")
            prev_results_count = len(results)
        elif critique and critique != prev_critique and not synthesis:
            if passed:
                show(f"⏳ Synthesizing findings... (score {score:.2f} ✓)", "running")
            else:
                show(f"⏳ Score {score:.2f} — follow-up search (iteration {iteration})...", "running")
                prev_results_count = 0
            prev_critique = critique
        elif synthesis and synthesis != prev_synthesis and not report:
            show("⏳ Writing final report...", "running")
            prev_synthesis = synthesis
        elif report:
            show(f"✅ Done! Report ready ({len(report)} chars)", "done")

    return final_state


# ─────────────────────────────────────────
# Run helper (shared by button + examples)
# ─────────────────────────────────────────
def execute_research(query: str, thread_id: str = None):
    """Renders the pipeline status inline and returns final state."""
    if thread_id is None:
        thread_id = generate_thread_id()

    st.session_state.session_id = thread_id
    st.session_state.running = True
    st.session_state.cancelled = False

    st.markdown(
        f'<div class="session-tag">🔑 Session: {thread_id}</div>',
        unsafe_allow_html=True
    )
    st.markdown("**⚡ Agent Pipeline**")
    status_placeholder = st.empty()

    cancel_col, _ = st.columns([1, 5])
    with cancel_col:
        if st.button("🔴 Cancel", key=f"cancel_{thread_id}"):
            st.session_state.cancelled = True

    with st.spinner("Research in progress..."):
        try:
            final_state = run_research_streaming(query, thread_id, status_placeholder)
        except Exception as e:
            st.error(f"Research failed: {e}")
            st.session_state.running = False
            return None

    st.session_state.running = False
    return final_state


# ─────────────────────────────────────────
# Results Renderer
# ─────────────────────────────────────────
def show_results(state):
    st.markdown("---")
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-value">{len(state.get('research_results', []))}</div>
            <div class="metric-label">Questions Researched</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{state.get('critique_score', 0):.2f}</div>
            <div class="metric-label">Critic Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{state.get('iteration_count', 0)}</div>
            <div class="metric-label">Iterations</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{len(state.get('final_report', ''))}</div>
            <div class="metric-label">Report Chars</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📄 Final Report")
    st.markdown(state.get("final_report", "No report available."))
    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "⬇️ Download Markdown",
            data=state.get("final_report", ""),
            file_name=f"report_{st.session_state.session_id}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_b:
        if st.button("💾 Save to Reports Folder", use_container_width=True):
            fp = save_report(state.get("final_report", ""), state.get("query", "unknown"))
            st.success(f"Saved: {fp}")
    with col_c:
        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            for k in ["final_state", "session_id", "active_query", "cancelled", "trigger_query"]:
                st.session_state[k] = None if k != "cancelled" else False
            st.session_state.mode = "research"
            st.rerun()

    with st.expander("🔍 Critic Feedback"):
        st.write(state.get("critique", "—"))
    with st.expander("❓ Sub-questions Generated"):
        for i, q in enumerate(state.get("sub_questions", []), 1):
            st.write(f"{i}. {q}")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():

    # ── Header ──
    st.markdown('<p class="app-title">🔬 Research <span>Agent</span></p>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Multi-agent · Self-correcting · Powered by LangGraph</p>', unsafe_allow_html=True)

    # ── Pipeline strip ──
    st.markdown("""
    <div class="pipeline-strip">
        <div class="pipeline-step">
            <span class="pipeline-icon">🧠</span>
            <span class="pipeline-label">Plan</span>
            <span class="pipeline-desc">Breaks your query into exactly 5 targeted sub-questions covering different angles</span>
        </div>
        <div class="pipeline-step">
            <span class="pipeline-icon">🔍</span>
            <span class="pipeline-label">Research</span>
            <span class="pipeline-desc">Searches the live web for each sub-question, pulling answers and sources</span>
        </div>
        <div class="pipeline-step">
            <span class="pipeline-icon">⚖️</span>
            <span class="pipeline-label">Critique</span>
            <span class="pipeline-desc">An independent critic scores research quality 0–1 and identifies gaps</span>
        </div>
        <div class="pipeline-step">
            <span class="pipeline-icon">🔄</span>
            <span class="pipeline-label">Correct</span>
            <span class="pipeline-desc">If score is below 0.7, loops back and runs follow-up searches to fill gaps</span>
        </div>
        <div class="pipeline-step">
            <span class="pipeline-icon">📝</span>
            <span class="pipeline-label">Report</span>
            <span class="pipeline-desc">Synthesizes all findings into a structured, cited report with sources</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Show results if already loaded ──
    if st.session_state.final_state and not st.session_state.running:
        if st.session_state.active_query:
            st.markdown(
                f'<div class="session-tag">🔑 {st.session_state.session_id} &nbsp;|&nbsp; {st.session_state.active_query[:80]}</div>',
                unsafe_allow_html=True
            )
        show_results(st.session_state.final_state)
        return

    # ── Fire a queued quick-example run ──
    if st.session_state.trigger_query:
        query = st.session_state.trigger_query
        st.session_state.trigger_query = None
        st.session_state.active_query = query
        thread_id = generate_thread_id()
        final_state = execute_research(query, thread_id)
        if final_state and not st.session_state.cancelled:
            st.session_state.final_state = final_state
            st.rerun()
        return

    # ── RESEARCH MODE ──
    if st.session_state.mode == "research":

        col_input, col_examples = st.columns([3, 1])

        with col_input:
            st.markdown('<div class="input-label">Research Query</div>', unsafe_allow_html=True)
            query = st.text_area(
                label="query_hidden",
                label_visibility="collapsed",
                value=st.session_state.active_query,
                placeholder="e.g. What are the latest advancements in large language models?",
                height=120,
                key="query_input"
            )

            btn_col1, btn_col2 = st.columns([3, 2])
            with btn_col1:
                run_btn = st.button(
                    "🚀 Run Research",
                    type="primary",
                    use_container_width=True,
                    disabled=not query or st.session_state.running
                )
            with btn_col2:
                if st.button("📂 Resume Previous Session", use_container_width=True):
                    st.session_state.mode = "resume"
                    st.rerun()

            # Pipeline fires here — directly below the buttons, no gap
            if run_btn and query:
                st.session_state.active_query = query
                thread_id = generate_thread_id()
                final_state = execute_research(query, thread_id)
                if final_state and not st.session_state.cancelled:
                    st.session_state.final_state = final_state
                    st.rerun()

        with col_examples:
            st.markdown('<div class="example-label">Quick Examples</div>', unsafe_allow_html=True)
            examples = [
                "Latest advancements in quantum computing",
                "Impact of AI on healthcare in 2025",
                "Current state of renewable energy",
                "Recent breakthroughs in cancer research",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                    st.session_state.trigger_query = ex
                    st.rerun()

    # ── RESUME MODE ──
    elif st.session_state.mode == "resume":

        st.markdown('<div class="input-label">Enter your Session ID to resume</div>', unsafe_allow_html=True)

        session_id_input = st.text_area(
            label="sid_hidden",
            label_visibility="collapsed",
            placeholder="e.g. research_20260420_231244_d6c97b9c",
            height=80,
            key="session_id_field"
        )

        load_col, back_col = st.columns([2, 2])
        with load_col:
            load_btn = st.button(
                "📂 Load Session",
                type="primary",
                use_container_width=True,
                disabled=not session_id_input
            )
        with back_col:
            if st.button("📝 Research Query", use_container_width=True):
                st.session_state.mode = "research"
                st.rerun()

        if load_btn and session_id_input:
            checkpointer = get_checkpointer()
            saved_query = get_session_query(checkpointer, session_id_input.strip())

            if not saved_query:
                st.error(f"No session found with ID: `{session_id_input}`. Please check and try again.")
            else:
                st.info(f"Found session — Query: **\"{saved_query}\"**")
                st.session_state.active_query = saved_query
                final_state = execute_research(saved_query, session_id_input.strip())
                if final_state and not st.session_state.cancelled:
                    st.session_state.final_state = final_state
                    st.session_state.session_id = session_id_input.strip()
                    st.session_state.mode = "research"
                    st.rerun()


if __name__ == "__main__":
    main()
#streamlit run frontend/app.py