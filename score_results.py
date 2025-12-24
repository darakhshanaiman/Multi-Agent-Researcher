import json
import re
import csv
from collections import defaultdict
from statistics import mean, median

INPUT_JSONL = "trackB_results.jsonl"
OUT_CSV = "trackB_scored.csv"
OUT_SUMMARY = "trackB_summary.txt"

# --- Heuristic patterns ---
REFUSAL_PATTERNS = [
    r"don't have enough information",
    r"do not have enough information",
    r"can't answer",
    r"cannot answer",
    r"i don't know",
    r"not enough context",
    r"insufficient context",
    r"missing from the provided",
]

# A very light hallucination proxy:
# If answer contains numbers but context is empty, flag as risky.
NUMBER_PATTERN = re.compile(r"\b\d+(\.\d+)?\b")

def looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in REFUSAL_PATTERNS)

def count_numbers(text: str) -> int:
    if not text:
        return 0
    return len(NUMBER_PATTERN.findall(text))

def tokenish_len(text: str) -> int:
    # simple approx for length comparisons (no tokenizer dependency)
    return len((text or "").split())

def safe_div(a, b):
    return (a / b) if b else 0.0

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def score_row(row: dict) -> dict:
    """
    Adds heuristic metrics to a single run.
    These are proxies, useful for quick Track-B tables.
    """
    ans = row.get("answer", "") or ""
    ctx = row.get("context", "") or ""
    qtype = row.get("type", "unknown")
    mode = row.get("mode", "unknown")

    refusal = looks_like_refusal(ans)
    ans_words = tokenish_len(ans)
    ctx_words = tokenish_len(ctx)
    nums = count_numbers(ans)

    # "Grounding proxy": if document mode, require non-empty context and not refusal
    grounded_proxy = (mode == "document" and ctx_words > 0 and not refusal)

    # "Risky hallucination proxy": numbers without context (only when not refusing)
    risky_hallucination_proxy = (nums > 0 and ctx_words == 0 and not refusal)

    # --- NEW: fairness metrics for safety-first systems ---
    ctx_lower = ctx.lower()

    # Correct refusal proxy:
    # If it refused AND context is tiny/empty OR contains errors => likely a correct refusal
    correct_refusal_proxy = int(
        refusal and (
            ctx_words < 40 or
            "error" in ctx_lower or
            "no results" in ctx_lower
        )
    )

    # Over-refusal proxy:
    # If it refused BUT context is large => likely over-refusal (it probably could answer)
    over_refusal_proxy = int(refusal and ctx_words > 120)

    # Routing proxy: doc queries should be document mode, web queries should not be doc mode
    routing_correct = None
    if qtype in ("doc", "web", "adv"):
        if qtype == "doc":
            routing_correct = (mode == "document")
        elif qtype == "web":
            routing_correct = (mode != "document")
        else:
            routing_correct = None  # don't score routing on adversarial by default

    return {
        **row,
        "answer_words": ans_words,
        "context_words": ctx_words,
        "is_refusal": int(refusal),
        "grounded_proxy": int(grounded_proxy),
        "risky_hallucination_proxy": int(risky_hallucination_proxy),
        "correct_refusal_proxy": correct_refusal_proxy,
        "over_refusal_proxy": over_refusal_proxy,
        "routing_correct": ("" if routing_correct is None else int(routing_correct)),
    }


def write_csv(rows, path):
    # Collect all fieldnames (stable superset)
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    # Avoid huge trace field in CSV by default
    if "trace" in fieldnames:
        fieldnames.remove("trace")
        fieldnames.append("trace")  # keep at end if you want; or comment next line to exclude
    # If you prefer to exclude trace entirely, uncomment:
    # fieldnames.remove("trace")

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # shrink trace in CSV (optional)
            rr = dict(r)
            if "trace" in rr and isinstance(rr["trace"], list):
                rr["trace"] = f"[{len(rr['trace'])} events]"
            w.writerow(rr)

def summarize(rows, out_path):
    """
    Produces Track-B style aggregate stats:
    - refusal rate
    - grounded proxy rate
    - risky hallucination proxy rate
    - correct refusal proxy rate (NEW)
    - over-refusal proxy rate (NEW)
    - routing correctness (for doc/web only)
    - latency stats
    """
    by_system = defaultdict(list)
    for r in rows:
        by_system[r.get("system", "unknown")].append(r)

    lines = []
    lines.append("TRACK B EVALUATION SUMMARY (Heuristic + Logged Metrics)")
    lines.append(f"Input file: {INPUT_JSONL}")
    lines.append("")

    def stat_block(system_name, rs):
        lat = [r.get("latency_ms", 0) for r in rs if isinstance(r.get("latency_ms", None), int)]
        refusal = sum(r.get("is_refusal", 0) for r in rs)
        grounded = sum(r.get("grounded_proxy", 0) for r in rs)
        risky = sum(r.get("risky_hallucination_proxy", 0) for r in rs)

        # NEW fairness metrics
        correct_refusal = sum(r.get("correct_refusal_proxy", 0) for r in rs)
        over_refusal = sum(r.get("over_refusal_proxy", 0) for r in rs)

        # routing correctness only where not blank
        routing_vals = [int(r["routing_correct"]) for r in rs if str(r.get("routing_correct", "")).strip() != ""]
        routing_correct = sum(routing_vals)
        routing_total = len(routing_vals)

        n = len(rs)
        return [
            f"System: {system_name}",
            f"Runs: {n}",
            f"Refusal rate: {safe_div(refusal, n):.2%}",
            f"Correct refusal proxy rate: {safe_div(correct_refusal, n):.2%}",
            f"Over-refusal proxy rate: {safe_div(over_refusal, n):.2%}",
            f"Grounded proxy rate (doc-mode + non-empty context + not refusal): {safe_div(grounded, n):.2%}",
            f"Risky hallucination proxy rate (numbers w/ empty context + not refusal): {safe_div(risky, n):.2%}",
            (f"Routing accuracy (doc/web only): {safe_div(routing_correct, routing_total):.2%} "
             f"({routing_correct}/{routing_total})" if routing_total else "Routing accuracy: N/A"),
            (f"Latency ms: mean={mean(lat):.0f}, median={median(lat):.0f}, min={min(lat)}, max={max(lat)}" if lat else "Latency ms: N/A"),
            ""
        ]

    # Overall block
    lines.append("=== Overall ===")
    overall_lat = [r.get("latency_ms", 0) for r in rows if isinstance(r.get("latency_ms", None), int)]
    lines.append(f"Total runs: {len(rows)}")
    if overall_lat:
        lines.append(
            f"Latency ms overall: mean={mean(overall_lat):.0f}, median={median(overall_lat):.0f}, "
            f"min={min(overall_lat)}, max={max(overall_lat)}"
        )
    lines.append("")

    # Per-system blocks
    lines.append("=== By System ===")
    for sys_name in sorted(by_system.keys()):
        lines.extend(stat_block(sys_name, by_system[sys_name]))

    # By type blocks (doc/web/adv)
    lines.append("=== By Query Type & System ===")
    by_type_system = defaultdict(list)
    for r in rows:
        by_type_system[(r.get("type", "unknown"), r.get("system", "unknown"))].append(r)

    for (qtype, sys_name) in sorted(by_type_system.keys(), key=lambda x: (x[0], x[1])):
        rs = by_type_system[(qtype, sys_name)]
        lat = [r.get("latency_ms", 0) for r in rs if isinstance(r.get("latency_ms", None), int)]
        refusal = sum(r.get("is_refusal", 0) for r in rs)
        grounded = sum(r.get("grounded_proxy", 0) for r in rs)
        risky = sum(r.get("risky_hallucination_proxy", 0) for r in rs)
        correct_refusal = sum(r.get("correct_refusal_proxy", 0) for r in rs)
        over_refusal = sum(r.get("over_refusal_proxy", 0) for r in rs)
        n = len(rs)

        line = (
            f"{qtype} | {sys_name} | runs={n} "
            f"| refusal={safe_div(refusal,n):.2%} "
            f"| correct_refusal={safe_div(correct_refusal,n):.2%} "
            f"| over_refusal={safe_div(over_refusal,n):.2%} "
            f"| grounded={safe_div(grounded,n):.2%} "
            f"| risky={safe_div(risky,n):.2%}"
        )
        if lat:
            line += f" | latency_mean={mean(lat):.0f}ms"

        lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    rows = load_jsonl(INPUT_JSONL)
    scored = [score_row(r) for r in rows]

    write_csv(scored, OUT_CSV)
    summarize(scored, OUT_SUMMARY)

    print(f"✅ Wrote: {OUT_CSV}")
    print(f"✅ Wrote: {OUT_SUMMARY}")
    print("Tip: Add manual labels by editing the CSV (columns like human_correctness, human_groundedness).")

if __name__ == "__main__":
    main()
