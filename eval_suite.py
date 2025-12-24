# eval_suite.py
import json
from baselines import (
    baseline_standard_rag,
    baseline_agent_no_critic,
    proposed_full_system,
)

def load_queries(path="queries.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_all(out_path="trackB_results.jsonl"):
    queries = load_queries()

    with open(out_path, "w", encoding="utf-8") as f:
        for item in queries:
            query = item["query"]
            qtype = item.get("type", "web")

            force_document = (qtype == "doc")

            r1 = baseline_standard_rag(query, force_document=force_document)
            r1["type"] = qtype
            f.write(json.dumps(r1, ensure_ascii=False) + "\n")

            r2 = baseline_agent_no_critic(query, force_document=force_document)
            r2["type"] = qtype
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")

            r3 = proposed_full_system(query, force_document=force_document)
            r3["type"] = qtype
            f.write(json.dumps(r3, ensure_ascii=False) + "\n")

    print(f"Saved results to: {out_path}")

if __name__ == "__main__":
    run_all()
