from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from tavily import TavilyClient
import os
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

# 2. Setup Tools
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.environ.get("GROQ_API_KEY"))
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# 3. Define State
class AgentState(TypedDict, total=False):
    task: str
    research_data: List[str]
    critique_feedback: str
    status: str
    revision_number: int
    final_report: str


def researcher_node(state: AgentState):
    print("\n- RESEARCHER AGENT -")

    task = state.get("task", "")
    revision = state.get("revision_number", 0)
    last_feedback = state.get("critique_feedback", "")

    # --- A. INFINITE LOOP PROTECTION ---
    if revision >= 3:
        print("    ⚠️ Max retries reached. Forcing finish.")
        existing = state.get("research_data", [])
        if not existing:
            existing = ["Note: Maximum search attempts reached. No usable web results were found."]
        return {
            "research_data": existing,
            "status": "force_finish",
            "revision_number": revision + 1,
        }

    # --- B. PINECONE / RAG CHECK ---
    # If app.py attached vector context, skip web searching completely.
    if "RELEVANT CONTEXT FROM PINECONE" in task:
        print("    🧠 Using Vector Context (document mode).")
        try:
            context_data = task.split("RELEVANT CONTEXT FROM PINECONE:")[-1].strip()
        except Exception:
            context_data = task

        # If pinecone context is empty, don't pretend it worked
        # if not context_data or len(context_data.strip()) < 30:
        #     return {
        #         "research_data": ["Error: Pinecone context was empty or too short."],
        #         "status": "searching",
        #         "revision_number": revision + 1,
        #     }

        return {
            "research_data": [context_data],
            "status": "searching",
            "revision_number": revision + 1,
        }

    # --- C. GENERATE CLEAN WEB SEARCH QUERY ---
    # IMPORTANT: if Critic rejected previously, force a DIFFERENT query
    query_generator_prompt = f"""
You are a Search Expert.

User Input:
{task}

Critic Feedback (if any):
{last_feedback}

Task:
- Extract the core question for a Google search.
- Remove chat history and filler words.
- Keep it under 10 words.
- If the critic rejected before, produce a NEW/DIFFERENT query (use synonyms or a more specific term).
Output ONLY the search query.
""".strip()

    try:
        search_query = llm.invoke(query_generator_prompt).content.strip().replace('"', "")
    except Exception:
        search_query = task[:60].strip()

    if not search_query:
        search_query = task[:60].strip()

    print(f"    🔎 Searching Web for: '{search_query}'")

    # --- D. EXECUTE SEARCH ---
    try:
        results = tavily.search(query=search_query, search_depth="advanced", max_results=5)
        content = []
        for result in results.get("results", []):
            url = result.get("url", "")
            snippet = result.get("content", "")
            if url or snippet:
                content.append(f"Source: {url}\nContent: {snippet}\n")

        new_data = "\n".join(content).strip()
        if not new_data:
            new_data = "Error: No results found."

    except Exception as e:
        new_data = f"Error during search: {e}"

    return {
        "research_data": [new_data],
        "status": "searching",
        "revision_number": revision + 1,
    }


def critic_node(state: AgentState):
    print("\n--- 🧐 CRITIC AGENT ---")
    task = state["task"]
    data = state["research_data"][-1]
    revision = state.get("revision_number", 0)

    is_doc_mode = "RELEVANT CONTEXT FROM PINECONE" in task

    # If empty data, reject for web; for doc allow writer to refuse cleanly
    if not data or len(data.strip()) < 30:
        if is_doc_mode:
            return {
                "critique_feedback": "STATUS: APPROVE_WITH_CAVEAT\nDocument context is too short/empty; writer must refuse or ask for more context.",
                "status": "approve_with_caveat"
            }
        return {
            "critique_feedback": "STATUS: REJECT\nSearch returned no usable results.",
            "status": "rejected"
        }

    # Web-specific error handling
    if (not is_doc_mode) and ("Error" in data or "No results" in data):
        return {
            "critique_feedback": "STATUS: REJECT\nWeb search returned no usable results.",
            "status": "rejected"
        }

    prompt = f"""
You are the Critic in a reliability-first agent system.

USER QUERY:
{task}

EVIDENCE (web snippets or document context):
{data}

Decide one of the following:

1) STATUS: APPROVE
- Evidence directly answers the user query with sufficient support.

2) STATUS: APPROVE_WITH_CAVEAT
- Evidence partially answers the query OR answers it but misses details.
- In this case, the Writer must: (a) answer only what is supported, (b) explicitly state what's missing.

3) STATUS: REJECT
- Evidence is irrelevant, contradictory, or unusable.

Return EXACTLY in this format:
STATUS: <APPROVE|APPROVE_WITH_CAVEAT|REJECT>
REASON: <one short reason>
MISSING: <what is missing, if any>
"""
    response = llm.invoke(prompt).content.strip()

    up = response.upper()
    if "STATUS: APPROVE_WITH_CAVEAT" in up:
        status = "approve_with_caveat"
    elif "STATUS: APPROVE" in up:
        status = "approve"
    else:
        status = "rejected"
    # 🔒 STRICT MODE: downgrade weak approvals for factual queries
    if status == "approve_with_caveat":
        factual_triggers = [
        "what year", "how many", "exact", "accuracy", "authors",
        "parameter", "score", "architecture", "compare"
        ]
        task_lower = task.lower()
        if any(t in task_lower for t in factual_triggers):
            status = "rejected"
            response += "\n\nNOTE: Partial evidence is insufficient for factual claims. Treated as REJECT."

    print(f"    Critic Decision: {status.upper()}")
    return {
        "critique_feedback": response,
        "status": status
    }



def writer_node(state: AgentState):
    print("\n--- ✍️ WRITER AGENT ---")
    task = state.get("task", "")
    data = (state.get("research_data") or [""])[-1]
    critic = state.get("critique_feedback", "")

    # If data looks wrong/empty, refuse safely (still ok)
    if not data or "Error:" in data or "No results" in data:
        return {
            "status": "finished",
            "final_report": "I couldn’t find reliable information to answer this question without guessing."
        }

    prompt = f"""
You are the Writer in a no-hallucination system.

RULES (must follow):
1) Use ONLY the EVIDENCE provided below. Do not add external facts.
2) If evidence is incomplete, answer partially and explicitly state limitations.
3) If evidence does not contain the answer, refuse clearly and say what is missing.
4) Every key claim must be explicitly grounded using phrases like:
   - "From the provided document context: ..."
   - "From the retrieved web evidence: ..."

CRITIC FEEDBACK:
{critic}

USER REQUEST:
{task}

EVIDENCE:
{data}

Write a concise answer. If you refuse, make it helpful (what you would need to answer).
""".strip()

    final_report = llm.invoke(prompt).content
    return {"status": "finished", "final_report": final_report}



# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("critic", critic_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("researcher")

# ✅ Correct routing logic:
# researcher -> critic (web mode)
# critic -> researcher (if rejected)
# critic -> writer (if approved)
# researcher -> writer (if finished / pinecone)
def route_after_researcher(state: AgentState):
    status = state.get("status", "")
    if status in ["finished", "force_finish"]:
        return "writer"
    return "critic"


def route_after_critic(state: AgentState):
    status = state.get("status", "")
    if status in ["approve", "approve_with_caveat"]:
        return "writer"
    if status == "rejected":
        return "researcher"
    return "researcher"


workflow.add_conditional_edges("researcher", route_after_researcher)
workflow.add_conditional_edges("critic", route_after_critic)
workflow.add_edge("writer", END)

app = workflow.compile()
