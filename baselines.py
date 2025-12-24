# baselines.py
from typing import Dict, Any, List
import time

from app import vector_search, is_file_related_question
from main import app as langgraph_app, llm, researcher_node, writer_node

def _run_graph(task: str) -> Dict[str, Any]:
    """Run your existing LangGraph workflow and capture trace + outputs."""
    state = {
        "task": task,
        "research_data": [],
        "critique_feedback": "",
        "status": "starting",
        "revision_number": 0
    }

    trace: List[Dict[str, Any]] = []
    final_answer = ""
    critic_status = ""
    critic_feedback = ""
    researcher_out = ""

    for event in langgraph_app.stream(state):
        agent = list(event.keys())[0]
        data = event[agent]
        trace.append({"agent": agent, "data": data})

        if agent == "researcher":
            researcher_out = (data.get("research_data") or [""])[-1] if data.get("research_data") else ""
        if agent == "critic":
            critic_status = data.get("status", "")
            critic_feedback = data.get("critique_feedback", "")
        if agent == "writer":
            final_answer = data.get("final_report", "")
            break

    return {
        "final_answer": final_answer,
        "critic_status": critic_status,
        "critic_feedback": critic_feedback,
        "researcher_out": researcher_out,
        "trace": trace
    }


def baseline_standard_rag(query: str, force_document: bool) -> Dict[str, Any]:
    """
    Baseline 1: Standard RAG (no agents, no critic).
    - If force_document: retrieve + single LLM answer constrained to context.
    - Else: single LLM answer WITHOUT web tools (we keep it conservative).
    """
    t0 = time.time()

    if force_document:
        ctx = vector_search(query)
        prompt = (
            "Answer using ONLY the CONTEXT.\n"
            "If the context is insufficient, say: 'I don't have enough information in the provided document.'\n\n"
            f"QUESTION:\n{query}\n\nCONTEXT:\n{ctx}"
        )
        answer = llm.invoke(prompt).content
        mode = "document"
    else:
        ctx = ""
        prompt = (
            "Answer only if you are confident. If you are not sure, say you don't know.\n\n"
            f"QUESTION:\n{query}"
        )
        answer = llm.invoke(prompt).content
        mode = "webless_llm"

    return {
        "system": "baseline_standard_rag",
        "mode": mode,
        "query": query,
        "context": ctx,
        "answer": answer,
        "latency_ms": int((time.time() - t0) * 1000),
    }


def baseline_agent_no_critic(query: str, force_document: bool) -> Dict[str, Any]:
    """
    Baseline 2: Agentic without critic.
    FIXED: Now truly bypasses the graph to avoid using the critic's routing logic.
    """
    t0 = time.time()

    # --- DOCUMENT MODE (Unchanged) ---
    if force_document:
        ctx = vector_search(query)
        task = (
            "You are a document-grounded assistant.\n"
            "Use ONLY the provided document context. If missing, say you don't have enough information.\n\n"
            f"QUESTION:\n{query}\n\nDOCUMENT CONTEXT:\n{ctx}"
        )
        answer = llm.invoke(task).content
        return {
            "system": "baseline_agent_no_critic",
            "mode": "document",
            "query": query,
            "context": ctx,
            "answer": answer,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # --- WEB MODE (Fixed) ---
    # OLD ERROR: out = _run_graph(query) <-- This used the critic!
    
    # NEW LOGIC: Manual Chain (Researcher -> Writer)
    print(f"--- Running Baseline 2 (No Critic) for: {query} ---")
    
    # 1. Initialize State
    state = {
        "task": query,
        "research_data": [],
        "revision_number": 0,
        "critique_feedback": "", # Empty, because no critic exists
    }

    # 2. Run Researcher ONLY ONCE (No loops)
    research_result = researcher_node(state)
    
    # 3. Update state with research results
    state["research_data"] = research_result["research_data"]
    
    # 4. Run Writer immediately (Blind trust)
    writer_result = writer_node(state)
    final_answer = writer_result["final_report"]

    return {
        "system": "baseline_agent_no_critic",
        "mode": "web",
        "query": query,
        "context": (state["research_data"] or [""])[-1],
        "answer": final_answer,
        "latency_ms": int((time.time() - t0) * 1000),
    }

def proposed_full_system(query: str, force_document: bool) -> Dict[str, Any]:
    """
    Proposed: Your Researcher->Critic->Writer.
    For Track B we FORCE the mode so results are controlled.
    """
    t0 = time.time()

    if force_document:
        ctx = vector_search(query)
        task = (
            "You are running in DOCUMENT MODE.\n"
            "Rules:\n"
            "1) Use ONLY the DOCUMENT CONTEXT.\n"
            "2) Do NOT use web search.\n"
            "3) If context is insufficient, the Critic must REJECT and the Writer must refuse.\n\n"
            f"QUESTION:\n{query}\n\nDOCUMENT CONTEXT:\n{ctx}"
        )
        out = _run_graph(task)
        mode = "document"
        context_used = ctx
    else:
        out = _run_graph(query)
        mode = "web"
        context_used = out["researcher_out"]

    return {
        "system": "proposed_full_system",
        "mode": mode,
        "query": query,
        "context": context_used,
        "critic_status": out["critic_status"],
        "critic_feedback": out["critic_feedback"],
        "answer": out["final_answer"],
        "latency_ms": int((time.time() - t0) * 1000),
        "trace": out["trace"],
    }
