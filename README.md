# Researcher–Critic–Writer: Agentic RAG System for Hallucination Prevention

## Overview

This project implements an **agent-based Retrieval-Augmented Generation (RAG) system** designed to improve the reliability of Large Language Models (LLMs) by **preventing hallucinated answers**.  
Unlike standard RAG pipelines that generate responses immediately after retrieval, this system enforces **explicit validation through a multi-agent architecture**.

The system is composed of three cooperating agents:
- **Researcher** – retrieves information from the web or documents
- **Critic** – evaluates whether the retrieved evidence is sufficient and reliable
- **Writer** – generates a final answer only after approval

By introducing a critic-driven validation loop, the system ensures that answers are produced **only when supported by evidence**, otherwise retrying retrieval or refusing safely.

---

## Key Features

- Multi-agent architecture using LangGraph
- Hybrid RAG (web search + document-based retrieval)
- Explicit hallucination prevention via critic validation
- Retry mechanism for weak or insufficient evidence
- Safe refusal when reliable information is unavailable
- End-to-end and component-level evaluation framework
- Designed for reliability over speed

---

## System Architecture

The system follows a **Research → Validate → Generate** pipeline:

1. User query is received through the Chainlit interface
2. The **Researcher agent** retrieves information:
   - Web search (Tavily API), or
   - Vector search (Pinecone) if a document is uploaded
3. The **Critic agent** evaluates the retrieved evidence:
   - Approves if evidence is sufficient
   - Rejects and triggers a retry if evidence is weak
4. The **Writer agent** generates the final response using only approved evidence
5. If evidence is insufficient after retries, the system refuses safely

This design makes hallucination prevention a **structural property**, not a prompt-level heuristic.

---

## Project Structure

```

Researcher_Multi_Agent/
│
├── README.md                # Project overview, setup, and usage instructions
├── app.py                   # Chainlit application entry point (UI + mode routing)
├── auth.py                  # Authentication and session management logic
├── baselines.py             # Baseline systems (no-critic agent, standard RAG)
├── chainlit.md              # Chainlit UI configuration and instructions
├── eval_suite.py            # Track-B evaluation runner (JSONL logging)
├── history.db               # SQLite database for chat and interaction history
├── main.py                  # Core multi-agent graph (Researcher, Critic, Writer)
├── plot_results.py          # Visualization of evaluation metrics (graphs)
├── queries.json             # Evaluation query set (system-level testing)
├── requirements.txt         # Python dependencies
├── score_results.py         # Scoring heuristics for Track-B metrics
├── setup_db.py              # Initializes the SQLite database schema
├── trackB_results.jsonl     # Raw evaluation logs (Track B)
├── trackB_scored.csv        # Scored evaluation results (CSV format)
└── trackB_summary.txt       # Aggregated evaluation summary statistics
````

---

## Setup Instructions

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
INDEX_NAME=your_pinecone_index
```

---

## Running the Application

### Initialize Database (run once)

```bash
python setup_db.py
```

### Start the Chat Interface

```bash
chainlit run app.py -w
```

Open the browser at:
`http://localhost:8000` #run locally

---

## Using the System

* Ask general questions → system uses **web-based RAG**
* Upload a PDF → system switches to **document-based RAG**
* If evidence is insufficient → system retries or refuses safely
* The system never guesses or fabricates unsupported facts

---
## Evaluation

### Compared Systems
- **Baseline Agent (No Critic)**  
  Single-agent LLM without validation or refusal logic.

- **Baseline Standard RAG**  
  Retrieval-Augmented Generation without agent-level critique.

- **Proposed Multi-Agent RAG System**  
  Researcher–Critic–Writer architecture with enforced validation.

---

### Evaluation Metrics
- Grounded Answers (%)
- Hallucination Risk (%)
- Refusal Rate (%)
- Correct Refusal Rate (%)
- Average Latency (ms)

---

### Key Results
- Higher grounded answer rate compared to baselines.
- Significant reduction in hallucination risk.
- Non-Conservative refusal behavior.
- Increased latency due to multi-agent validation.
---

## Design Philosophy

* Reliability over speed
* Architectural enforcement instead of prompt-based safety
* Explicit validation before generation
* Conservative refusal policy
* Transparent evaluation and error analysis

---

## Limitations

* Higher latency due to multi-agent coordination
* Conservative refusal behavior may reduce answer coverage
* Evaluation depends on the difficulty of the query set
* Web retrieval quality affects overall grounding

---

## Future Work

* Adaptive critic strictness based on query type
* Confidence-aware refusal thresholds
* Larger adversarial evaluation sets
* Automated evidence scoring
* Latency optimization

---

## References

* Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
* Singh, *Agentic Retrieval-Augmented Generation: A Survey*, arXiv 2025
* Xu et al., *ActiveRAG*, arXiv 2024
* Nguyen et al., *MA-RAG*, arXiv 2025

---

## License

This project is developed for academic purposes as part of LLM Applications using Agentic AI course.
