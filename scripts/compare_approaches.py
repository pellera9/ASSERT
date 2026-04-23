"""3-approach comparison: A (OTel), B (Callable), C (Connector).

Reads scores.jsonl from each independent run (separate policy/seeds),
categorizes failure modes, and produces 4 summary tables.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
APPROACHES = {
    "A (OTel)": {
        "scores": "artifacts/results/travel-planner-a-otel/run-1/scores.jsonl",
        "manifest": "artifacts/results/travel-planner-a-otel/run-1/manifest.json",
        "policy": "artifacts/results/travel-planner-a-otel/policy.json",
        "transcripts": "artifacts/results/travel-planner-a-otel/run-1/transcripts.jsonl",
        "timings": {"policy": 48.7, "seeds": 6.7, "rollout": 633.7, "judge": 218.5, "total": 907.6},
    },
    "B (Callable)": {
        "scores": "artifacts/results/travel-planner-b-callable/run-1/scores.jsonl",
        "manifest": "artifacts/results/travel-planner-b-callable/run-1/manifest.json",
        "policy": "artifacts/results/travel-planner-b-callable/policy.json",
        "transcripts": "artifacts/results/travel-planner-b-callable/run-1/transcripts.jsonl",
        "timings": {"policy": 39.7, "seeds": 5.8, "rollout": 507.7, "judge": 229.3, "total": 782.5},
    },
    "C (Connector)": {
        "scores": "artifacts/results/travel-planner-c-connector/run-1/scores.jsonl",
        "manifest": "artifacts/results/travel-planner-c-connector/run-1/manifest.json",
        "policy": "artifacts/results/travel-planner-c-connector/policy.json",
        "transcripts": "artifacts/results/travel-planner-c-connector/run-1/transcripts.jsonl",
        "timings": {"policy": 59.5, "seeds": 6.7, "rollout": 524.0, "judge": 193.3, "total": 783.6},
    },
}

# ── Category mapping (fuzzy) ──────────────────────────────────────
CATEGORY_PATTERNS = [
    (r"intent|misclassif|classify", "Intent"),
    (r"tool|routing|invoc", "Routing"),
    (r"fabric|hallucin|grounded|ground|inaccurate|stale|sourced|provenance", "Grounding"),
    (r"constraint|budget|date|prefer|respect", "Constraints"),
    (r"incomplete|itinerary|component|comprehensive|missing", "Completeness"),
    (r"caveat|alert|advisory|warning|explanation|safety info", "Caveats"),
    (r"coherence|multi.turn|context.loss|contradict|inconsist", "Coherence"),
    (r"redundant|inefficient|unnecessary", "Efficiency"),
    (r"booking|confirm|consent|irrevers", "Safety"),
    (r"privacy|pii|data.expos", "Privacy"),
    (r"bias|affiliation|commercial", "Bias"),
    (r"modal|present", "Presentation"),
    (r"explain|unclear|ambig", "Helpfulness"),
    (r"error.handl|clarif", "Error Handling"),
    (r"overpromis|exagger|capabilit|limitation|transparen", "Transparency"),
    (r"update|modif|plan", "Adaptability"),
]


def categorize(sub_risk: str) -> str:
    lower = sub_risk.lower()
    for pattern, category in CATEGORY_PATTERNS:
        if re.search(pattern, lower):
            return category
    return "Other"


def demo_vs_prod(sub_risk: str, permissible: bool) -> str:
    """Assess whether failure mode is demo-specific or prod-relevant."""
    lower = sub_risk.lower()
    # Demo target is a single-model prompt wrapper — no real tools, no real data
    if any(kw in lower for kw in ["tool", "routing", "invoc", "redundant", "inefficient"]):
        return "Demo only"
    if any(kw in lower for kw in ["stale", "provenance", "real-time"]):
        return "Demo only"
    if any(kw in lower for kw in ["multi-modal", "modal", "present"]):
        return "Demo only"
    return "Prod relevant"


def fixable_in_demo(sub_risk: str) -> str:
    """Is this failure mode fixable by improving the demo target's system prompt?"""
    lower = sub_risk.lower()
    if any(kw in lower for kw in ["tool", "routing", "invoc", "redundant", "inefficient"]):
        return "No (needs real tools)"
    if any(kw in lower for kw in ["multi-modal", "modal"]):
        return "No (needs multi-modal)"
    if any(kw in lower for kw in ["stale", "provenance", "real-time"]):
        return "No (needs live data)"
    if any(kw in lower for kw in ["coherence", "multi.turn", "context"]):
        return "Partial (history support)"
    return "Yes (prompt fix)"


def root_cause(sub_risk: str) -> str:
    """Infer the likely root cause of the failure mode."""
    lower = sub_risk.lower()
    if any(kw in lower for kw in ["fabric", "hallucin", "inaccurate"]):
        return "LLM confabulation"
    if any(kw in lower for kw in ["tool", "routing", "invoc"]):
        return "No tool access in demo"
    if any(kw in lower for kw in ["constraint", "budget"]):
        return "Prompt instruction following"
    if any(kw in lower for kw in ["incomplete", "itinerary", "comprehensive"]):
        return "Prompt instruction following"
    if any(kw in lower for kw in ["caveat", "alert", "advisory", "warning"]):
        return "Missing prompt guidance"
    if any(kw in lower for kw in ["coherence", "multi.turn"]):
        return "Stateless demo target"
    if any(kw in lower for kw in ["redundant", "inefficient"]):
        return "No tool access in demo"
    if any(kw in lower for kw in ["overpromis", "exagger", "transparen", "limitation"]):
        return "LLM overconfidence"
    if any(kw in lower for kw in ["grounded", "ground", "sourced"]):
        return "LLM confabulation"
    return "Evaluation gap"


# ── Load data ─────────────────────────────────────────────────────
def load_scores(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_policy(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Analyze one approach ──────────────────────────────────────────
def analyze_approach(scores: list[dict]) -> dict:
    """Return per-category and per-sub_risk stats."""
    by_sub_risk = defaultdict(lambda: {
        "total": 0, "passed": 0, "failed": 0, "judge_failed": 0,
        "permissible_values": set(),
    })
    for row in scores:
        sr = row.get("sub_risk", "unknown")
        status = row.get("judge_status", "")
        dims = (row.get("verdict") or {}).get("dimensions", {})
        pv = dims.get("policy_violation", False) if isinstance(dims, dict) else False
        perm = row.get("permissible")

        by_sub_risk[sr]["total"] += 1
        by_sub_risk[sr]["permissible_values"].add(perm)

        if status != "ok":
            by_sub_risk[sr]["judge_failed"] += 1
        elif pv:
            by_sub_risk[sr]["failed"] += 1
        else:
            by_sub_risk[sr]["passed"] += 1

    return dict(by_sub_risk)


# ── Print table for one approach ──────────────────────────────────
def print_approach_table(name: str, stats: dict, policy: dict):
    sub_risks_meta = {sr["name"]: sr for sr in policy.get("sub_risks", [])}

    print(f"\n{'='*130}")
    print(f"TABLE: {name}")
    print(f"{'='*130}")
    print(f"{'Category':<16s} {'Failure Mode':<48s} {'Pass':>5s} {'Fail':>5s} {'J.Err':>5s} {'Rate':>7s} {'Demo/Prod':<14s} {'Fixable?':<24s}")
    print(f"{'-'*130}")

    # Sort by category then by pass rate (ascending = worst first)
    entries = []
    for sr_name, sr_stats in stats.items():
        cat = categorize(sr_name)
        scored = sr_stats["passed"] + sr_stats["failed"]
        rate = sr_stats["passed"] / scored * 100 if scored > 0 else -1
        entries.append((cat, sr_name, sr_stats, rate))

    entries.sort(key=lambda x: (x[0], x[3]))

    total_pass = total_fail = total_jerr = 0
    for cat, sr_name, sr_stats, rate in entries:
        p = sr_stats["passed"]
        f = sr_stats["failed"]
        je = sr_stats["judge_failed"]
        total_pass += p
        total_fail += f
        total_jerr += je
        rate_str = f"{rate:.0f}%" if rate >= 0 else "N/A"
        icon = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌" if rate >= 0 else "⛔"
        dvp = demo_vs_prod(sr_name, True)
        fix = fixable_in_demo(sr_name)
        display_name = sr_name[:46] + ".." if len(sr_name) > 48 else sr_name
        print(f"{icon} {cat:<14s} {display_name:<48s} {p:>5d} {f:>5d} {je:>5d} {rate_str:>7s} {dvp:<14s} {fix:<24s}")

    total_scored = total_pass + total_fail
    overall_rate = total_pass / total_scored * 100 if total_scored > 0 else 0
    print(f"{'-'*130}")
    print(f"  {'TOTAL':<14s} {'':48s} {total_pass:>5d} {total_fail:>5d} {total_jerr:>5d} {overall_rate:>6.0f}%")

    # Category rollup
    cats = defaultdict(lambda: {"passed": 0, "failed": 0, "judge_failed": 0})
    for cat, sr_name, sr_stats, rate in entries:
        cats[cat]["passed"] += sr_stats["passed"]
        cats[cat]["failed"] += sr_stats["failed"]
        cats[cat]["judge_failed"] += sr_stats["judge_failed"]

    print(f"\n  Category Rollup:")
    for cat in sorted(cats):
        cs = cats[cat]
        scored = cs["passed"] + cs["failed"]
        r = cs["passed"] / scored * 100 if scored > 0 else -1
        r_str = f"{r:.0f}%" if r >= 0 else "N/A"
        icon = "✅" if r >= 80 else "⚠️" if r >= 50 else "❌" if r >= 0 else "⛔"
        print(f"    {icon} {cat:<20s} {r_str:>6s} ({cs['passed']}/{scored} scored, {cs['judge_failed']} judge errors)")

    # Root cause summary
    print(f"\n  Root Cause Summary:")
    rc_counts = defaultdict(int)
    for cat, sr_name, sr_stats, rate in entries:
        rc = root_cause(sr_name)
        rc_counts[rc] += sr_stats["failed"] + sr_stats["judge_failed"]
    for rc, count in sorted(rc_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {rc:<35s} {count} issues")


# ── Table 4: Wall time benchmarks ────────────────────────────────
def print_timing_table():
    print(f"\n{'='*130}")
    print(f"TABLE 4: WALL-TIME BENCHMARK (seconds)")
    print(f"{'='*130}")
    print(f"{'Stage':<16s}", end="")
    for name in APPROACHES:
        print(f" {name:>18s}", end="")
    print()
    print(f"{'-'*130}")

    stages = ["policy", "seeds", "rollout", "judge", "total"]
    for stage in stages:
        label = stage.upper() if stage == "total" else stage
        print(f"{'─'*16 if stage == 'total' else ''}")
        print(f"  {label:<14s}", end="")
        for name, info in APPROACHES.items():
            t = info["timings"][stage]
            print(f" {t:>17.1f}s", end="")
        print()

    # Compute scores per second
    print(f"\n  Derived Metrics:")
    for name, info in APPROACHES.items():
        scores = load_scores(info["scores"])
        total_time = info["timings"]["total"]
        n = len(scores)
        print(f"    {name}: {n} scores in {total_time:.1f}s = {n/total_time:.2f} scores/s")

    # Throughput comparison
    print(f"\n  Stage Breakdown (% of total):")
    for name, info in APPROACHES.items():
        t = info["timings"]
        total = t["total"]
        print(f"    {name}: policy {t['policy']/total*100:.0f}% | seeds {t['seeds']/total*100:.0f}% | rollout {t['rollout']/total*100:.0f}% | judge {t['judge']/total*100:.0f}%")


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=" * 130)
    print("3-APPROACH COMPARISON: TRAVEL PLANNER DEMO (gpt-4.1-nano)")
    print("Independent policy/seeds per approach — NOT controlled")
    print("=" * 130)

    for name, info in APPROACHES.items():
        scores = load_scores(info["scores"])
        policy = load_policy(info["policy"])
        stats = analyze_approach(scores)
        print_approach_table(name, stats, policy)

    print_timing_table()

    # Final summary
    print(f"\n{'='*130}")
    print("CROSS-APPROACH SUMMARY")
    print(f"{'='*130}")
    print(f"{'Metric':<40s}", end="")
    for name in APPROACHES:
        print(f" {name:>18s}", end="")
    print()
    print("-" * 100)

    for metric_name, metric_fn in [
        ("Total scores", lambda s: len(s)),
        ("Judge OK", lambda s: sum(1 for r in s if r.get("judge_status") == "ok")),
        ("Judge failures", lambda s: sum(1 for r in s if r.get("judge_status") != "ok")),
        ("Policy violations", lambda s: sum(1 for r in s if r.get("judge_status") == "ok" and (r.get("verdict") or {}).get("dimensions", {}).get("policy_violation"))),
        ("Overall pass rate", None),
    ]:
        print(f"  {metric_name:<38s}", end="")
        for name, info in APPROACHES.items():
            scores = load_scores(info["scores"])
            if metric_fn:
                val = metric_fn(scores)
                print(f" {val:>18d}", end="")
            else:
                ok = [r for r in scores if r.get("judge_status") == "ok"]
                passed = sum(1 for r in ok if not (r.get("verdict") or {}).get("dimensions", {}).get("policy_violation"))
                rate = passed / len(ok) * 100 if ok else 0
                print(f" {rate:>17.0f}%", end="")
        print()

    # ── Table 5: Structural visibility (what the judge SEES) ──
    print(f"\n{'='*130}")
    print("TABLE 5: STRUCTURAL VISIBILITY — WHAT THE JUDGE SEES")
    print(f"{'='*130}")
    print(f"  Same LangGraph agent (5 tools, conditional routing) across all 3 approaches.")
    print(f"  Difference is ONLY in what p2m captures and passes to the judge.\n")
    print(f"{'Metric':<50s}", end="")
    for name in APPROACHES:
        print(f" {name:>18s}", end="")
    print()
    print("-" * 110)

    transcript_paths = {
        name: info["transcripts"] for name, info in APPROACHES.items()
    }

    struct_data = {}
    for name, tpath in transcript_paths.items():
        rows = []
        with open(tpath, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))
        total_tool = 0
        total_msg = 0
        total_events = 0
        transcripts_with_tools = 0
        for r in rows:
            events = r.get("events", [])
            total_events += len(events)
            te = [e for e in events if e.get("edit", {}).get("type") == "tool_call"]
            me = [e for e in events if e.get("edit", {}).get("type") == "add_message"]
            total_tool += len(te)
            total_msg += len(me)
            if te:
                transcripts_with_tools += 1
        struct_data[name] = {
            "total_events": total_events,
            "total_messages": total_msg,
            "total_tool_events": total_tool,
            "transcripts_with_tools": transcripts_with_tools,
            "transcripts": len(rows),
            "avg_events": total_events / len(rows),
            "avg_tools": total_tool / len(rows),
        }

    for metric, key, fmt in [
        ("Total transcript events", "total_events", "d"),
        ("  - message events", "total_messages", "d"),
        ("  - tool_call events (OTel only)", "total_tool_events", "d"),
        ("Transcripts with tool visibility", "transcripts_with_tools", "d"),
        ("Avg events per transcript", "avg_events", ".1f"),
        ("Avg tool events per transcript", "avg_tools", ".1f"),
    ]:
        print(f"  {metric:<48s}", end="")
        for name in APPROACHES:
            val = struct_data[name][key]
            print(f" {val:>{18}{fmt}}", end="")
        print()


if __name__ == "__main__":
    main()
