#!/usr/bin/env python3
"""Study 3: Native End-to-End Comparison — analysis of existing results.

Parses results from three pipelines (changliu2/LangGraph, Bloom/SimEnv,
origin/main) each running in its native mode, computes unified metrics,
and writes study3_analysis.json + study3_report.md.
"""

import json
import pathlib
import datetime
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ── helpers ──────────────────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_changliu2_scores(path: pathlib.Path, label: str) -> dict:
    """Analyze a changliu2 framework-eval scores.jsonl file."""
    records = load_jsonl(path)
    total = len(records)

    judge_ok = [r for r in records if r.get("judge_status") == "ok"]
    judge_errors = [r for r in records if r.get("judge_status") != "ok"]

    violations = []
    non_violations = []
    overrefusals = []
    sub_risk_violations = Counter()

    for r in judge_ok:
        dims = r.get("verdict", {}).get("dimensions", {})
        pv = dims.get("policy_violation", False)
        ov = dims.get("overrefusal", False)
        if pv:
            violations.append(r)
            sr = r.get("sub_risk", "unknown")
            sub_risk_violations[sr] += 1
        else:
            non_violations.append(r)
        if ov:
            overrefusals.append(r)

    # Count node-level violations across all seeds
    node_violation_counts = Counter()
    for r in judge_ok:
        for nj in r.get("verdict", {}).get("node_judgments", []):
            if nj.get("violated"):
                node_violation_counts[nj["node_name"]] += 1

    violation_rate = len(violations) / total * 100 if total else 0
    judge_fail_rate = len(judge_errors) / total * 100 if total else 0

    return {
        "label": label,
        "source": str(path.relative_to(ROOT)),
        "seed_count": total,
        "judge_ok": len(judge_ok),
        "judge_errors": len(judge_errors),
        "judge_failure_rate_pct": round(judge_fail_rate, 1),
        "violations": len(violations),
        "violation_rate_pct": round(violation_rate, 1),
        "overrefusals": len(overrefusals),
        "overrefusal_rate_pct": round(len(overrefusals) / total * 100, 1) if total else 0,
        "failure_modes_triggered": len(sub_risk_violations),
        "sub_risk_violations": dict(sub_risk_violations.most_common()),
        "node_violation_counts": dict(node_violation_counts.most_common()),
        "risk_spec": records[0].get("risk", "unknown") if records else "unknown",
        "judge_model": records[0].get("judge_model", "unknown") if records else "unknown",
        "target": records[0].get("target", "unknown") if records else "unknown",
    }


def analyze_bloom_judgment(path: pathlib.Path, label: str) -> dict:
    """Analyze a Bloom judgment.json file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    stats = data.get("summary_statistics", {})
    judgments = data.get("judgments", [])
    failed = data.get("failed_judgments", [])

    scores = [j["behavior_presence"] for j in judgments if "behavior_presence" in j]
    unrealism_scores = [j["unrealism"] for j in judgments if "unrealism" in j]
    eval_aware_scores = [j["evaluation_awareness"] for j in judgments if "evaluation_awareness" in j]

    return {
        "label": label,
        "source": str(path.relative_to(ROOT)),
        "behavior_name": data.get("behavior_name", "unknown"),
        "model": data.get("model", "unknown"),
        "seed_count": data.get("total_conversations", len(judgments)),
        "successful_judgments": data.get("successful_count", len(judgments)),
        "failed_judgments": data.get("failed_count", len(failed)),
        "avg_behavior_presence": stats.get("average_behavior_presence_score",
                                            round(sum(scores) / len(scores), 1) if scores else None),
        "min_behavior_presence": stats.get("min_behavior_presence_score", min(scores) if scores else None),
        "max_behavior_presence": stats.get("max_behavior_presence_score", max(scores) if scores else None),
        "elicitation_rate": stats.get("elicitation_rate"),
        "elicitation_rate_pct": round(stats.get("elicitation_rate", 0) * 100, 1),
        "avg_unrealism": stats.get("average_unrealism",
                                    round(sum(unrealism_scores) / len(unrealism_scores), 1) if unrealism_scores else None),
        "avg_eval_awareness": stats.get("average_evaluation_awareness",
                                         round(sum(eval_aware_scores) / len(eval_aware_scores), 1) if eval_aware_scores else None),
        "metajudgment_scores": data.get("metajudgment_scores"),
    }


def analyze_main_scores(path: pathlib.Path, label: str) -> dict:
    """Analyze an origin/main scores.jsonl file."""
    records = load_jsonl(path)
    total = len(records)

    judge_ok = [r for r in records if r.get("judge_status") == "ok"]
    judge_errors = [r for r in records if r.get("judge_status") != "ok"]

    violations = []
    overrefusals = []
    node_violation_counts = Counter()
    factor_counts = Counter()

    for r in judge_ok:
        dims = r.get("verdict", {}).get("dimensions", {})
        pv = dims.get("policy_violation", False)
        ov = dims.get("overrefusal", False)
        if pv:
            violations.append(r)
        if ov:
            overrefusals.append(r)
        # Count node-level violations
        for nj in r.get("verdict", {}).get("node_judgments", []):
            if nj.get("violated"):
                node_violation_counts[nj["node_name"]] += 1
        # Count factors
        factor = r.get("factors", {}).get("behavior", "")
        if factor:
            factor_counts[factor] += 1

    violation_rate = len(violations) / total * 100 if total else 0
    judge_fail_rate = len(judge_errors) / total * 100 if total else 0

    return {
        "label": label,
        "source": str(path.relative_to(ROOT)),
        "seed_count": total,
        "judge_ok": len(judge_ok),
        "judge_errors": len(judge_errors),
        "judge_failure_rate_pct": round(judge_fail_rate, 1),
        "violations": len(violations),
        "violation_rate_pct": round(violation_rate, 1),
        "overrefusals": len(overrefusals),
        "overrefusal_rate_pct": round(len(overrefusals) / total * 100, 1) if total else 0,
        "node_violation_counts": dict(node_violation_counts.most_common()),
        "factor_distribution": dict(factor_counts.most_common()),
        "concept": records[0].get("concept", "unknown") if records else "unknown",
        "judge_model": records[0].get("judge_model", "unknown") if records else "unknown",
        "target": records[0].get("target", "unknown") if records else "unknown",
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # changliu2 native: LangGraph bulk
    lg_single = analyze_changliu2_scores(
        ROOT / "artifacts" / "results" / "framework-eval-bulk" / "langgraph" / "scores.jsonl",
        "changliu2 — LangGraph single-agent (100 seeds, quality risk)",
    )
    lg_multi = analyze_changliu2_scores(
        ROOT / "artifacts" / "results" / "framework-eval-bulk" / "langgraph-multi" / "scores.jsonl",
        "changliu2 — LangGraph multi-agent (100 seeds, quality risk)",
    )

    # Bloom native
    bloom_safety = analyze_bloom_judgment(
        ROOT / "artifacts" / "comparison" / "study1" / "bloom" / "travel_planner_safety" / "judgment.json",
        "Bloom — SimEnv safety (5 seeds)",
    )
    bloom_quality = analyze_bloom_judgment(
        ROOT / "artifacts" / "comparison" / "study1" / "bloom" / "travel_planner_quality" / "judgment.json",
        "Bloom — SimEnv quality (5 seeds)",
    )

    # origin/main native
    main_safety = analyze_main_scores(
        ROOT / "artifacts" / "comparison" / "study1" / "main" / "study1-main-safety" / "baseline" / "scores.jsonl",
        "origin/main — safety (8 seeds)",
    )
    main_quality = analyze_main_scores(
        ROOT / "artifacts" / "comparison" / "study1" / "main" / "study1-main-quality" / "baseline" / "scores.jsonl",
        "origin/main — quality (8 seeds)",
    )

    # ── assemble output ──
    analysis = {
        "study": "Study 3: Native End-to-End Comparison",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "caveat": (
            "NOT an apples-to-apples comparison. Each pipeline ran in its intended "
            "native mode with different target agents, seed counts, tool implementations, "
            "and risk specs. This shows real-world operational performance."
        ),
        "pipelines": {
            "changliu2": {
                "mode": "LangGraph multi-agent with real MCP tool servers, OTel traced",
                "langgraph_single": lg_single,
                "langgraph_multi": lg_multi,
            },
            "bloom": {
                "mode": "SimEnv with evaluator-simulated tools, natural conversation style",
                "safety": bloom_safety,
                "quality": bloom_quality,
            },
            "origin_main": {
                "mode": "Inline system prompt with simulated tools, new config format",
                "safety": main_safety,
                "quality": main_quality,
            },
        },
    }

    # ── write JSON ──
    out_dir = ROOT / "artifacts" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "study3_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"✓ Wrote {json_path}")

    # ── write Markdown report ──
    md = generate_report(analysis, lg_single, lg_multi, bloom_safety, bloom_quality,
                         main_safety, main_quality)
    md_path = out_dir / "study3_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✓ Wrote {md_path}")


def generate_report(analysis, lg_single, lg_multi, bloom_safety, bloom_quality,
                    main_safety, main_quality) -> str:
    lines = []
    w = lines.append

    w("# Study 3: Native End-to-End Performance\n")
    w(f"> Generated: {analysis['generated_at']}\n")
    w("⚠️ **Caveat**: Not an apples-to-apples comparison. Each pipeline ran in its")
    w("intended native mode with different target agents, seed counts, and tool")
    w("implementations. This shows real-world operational performance, not a")
    w("controlled comparison (see Study 1 and Study 2 for those).\n")

    # ── Summary table ──
    w("## Summary Table\n")
    w("| Pipeline | Risk Spec | Seeds | Violation Rate | Judge Failures | Key Metric |")
    w("|----------|-----------|-------|---------------|----------------|------------|")
    w(f"| changliu2 LangGraph single | Quality | {lg_single['seed_count']} | "
      f"{lg_single['violation_rate_pct']}% ({lg_single['violations']}/{lg_single['seed_count']}) | "
      f"{lg_single['judge_errors']} | overrefusal: {lg_single['overrefusal_rate_pct']}% |")
    w(f"| changliu2 LangGraph multi | Quality | {lg_multi['seed_count']} | "
      f"{lg_multi['violation_rate_pct']}% ({lg_multi['violations']}/{lg_multi['seed_count']}) | "
      f"{lg_multi['judge_errors']} | overrefusal: {lg_multi['overrefusal_rate_pct']}% |")
    w(f"| Bloom SimEnv | Safety | {bloom_safety['seed_count']} | "
      f"elicit {bloom_safety['elicitation_rate_pct']}% | "
      f"{bloom_safety['failed_judgments']} | behavior: {bloom_safety['avg_behavior_presence']}/10 |")
    w(f"| Bloom SimEnv | Quality | {bloom_quality['seed_count']} | "
      f"elicit {bloom_quality['elicitation_rate_pct']}% | "
      f"{bloom_quality['failed_judgments']} | behavior: {bloom_quality['avg_behavior_presence']}/10 |")
    w(f"| origin/main | Safety | {main_safety['seed_count']} | "
      f"{main_safety['violation_rate_pct']}% ({main_safety['violations']}/{main_safety['seed_count']}) | "
      f"{main_safety['judge_errors']} | overrefusal: {main_safety['overrefusal_rate_pct']}% |")
    w(f"| origin/main | Quality | {main_quality['seed_count']} | "
      f"{main_quality['violation_rate_pct']}% ({main_quality['violations']}/{main_quality['seed_count']}) | "
      f"{main_quality['judge_errors']} | overrefusal: {main_quality['overrefusal_rate_pct']}% |")
    w("")

    # ── changliu2 detail ──
    w("---\n")
    w("## changliu2 — LangGraph (100 seeds each, quality risk)\n")
    w("Real MCP tool servers, multi-node graph, OpenTelemetry traced.\n")

    w("### Single-Agent Architecture")
    w(f"- **Source**: `{lg_single['source']}`")
    w(f"- **Target**: `{lg_single['target']}`")
    w(f"- **Judge**: {lg_single['judge_model']}")
    w(f"- **Seeds**: {lg_single['seed_count']}")
    w(f"- **Violations**: {lg_single['violations']} ({lg_single['violation_rate_pct']}%)")
    w(f"- **Overrefusals**: {lg_single['overrefusals']} ({lg_single['overrefusal_rate_pct']}%)")
    w(f"- **Judge failures**: {lg_single['judge_errors']} ({lg_single['judge_failure_rate_pct']}%)")
    if lg_single["node_violation_counts"]:
        w(f"- **Node violations**: {len(lg_single['node_violation_counts'])} distinct failure modes")
        for name, count in list(lg_single["node_violation_counts"].items())[:5]:
            w(f"  - {name}: {count}")
    w("")

    w("### Multi-Agent Architecture")
    w(f"- **Source**: `{lg_multi['source']}`")
    w(f"- **Target**: `{lg_multi['target']}`")
    w(f"- **Judge**: {lg_multi['judge_model']}")
    w(f"- **Seeds**: {lg_multi['seed_count']}")
    w(f"- **Violations**: {lg_multi['violations']} ({lg_multi['violation_rate_pct']}%)")
    w(f"- **Overrefusals**: {lg_multi['overrefusals']} ({lg_multi['overrefusal_rate_pct']}%)")
    w(f"- **Judge failures**: {lg_multi['judge_errors']} ({lg_multi['judge_failure_rate_pct']}%)")
    if lg_multi["node_violation_counts"]:
        w(f"- **Node violations**: {len(lg_multi['node_violation_counts'])} distinct failure modes")
        for name, count in list(lg_multi["node_violation_counts"].items())[:5]:
            w(f"  - {name}: {count}")
    w("")

    # ── Bloom detail ──
    w("---\n")
    w("## Bloom — SimEnv (5 seeds × 2 risk specs)\n")
    w("Evaluator-simulated tools, natural conversation style.\n")

    for label, b in [("Safety", bloom_safety), ("Quality", bloom_quality)]:
        w(f"### {label} — `{b['behavior_name']}`")
        w(f"- **Source**: `{b['source']}`")
        w(f"- **Model**: {b['model']}")
        w(f"- **Seeds**: {b['seed_count']}")
        w(f"- **Avg behavior presence**: {b['avg_behavior_presence']}/10 "
          f"(range {b['min_behavior_presence']}–{b['max_behavior_presence']})")
        w(f"- **Elicitation rate**: {b['elicitation_rate_pct']}%")
        w(f"- **Avg unrealism**: {b['avg_unrealism']}/10")
        w(f"- **Avg eval-awareness**: {b['avg_eval_awareness']}/10")
        w(f"- **Failed judgments**: {b['failed_judgments']}")
        if b.get("metajudgment_scores"):
            w(f"- **Metajudgment**: {b['metajudgment_scores']}")
        w("")

    # ── origin/main detail ──
    w("---\n")
    w("## origin/main — New Science (8 seeds × 2 risk specs)\n")
    w("New config format, inline system prompt with simulated tools, design stage.\n")

    for label, m in [("Safety", main_safety), ("Quality", main_quality)]:
        w(f"### {label} — `{m['concept']}`")
        w(f"- **Source**: `{m['source']}`")
        w(f"- **Target**: `{m['target']}`")
        w(f"- **Judge**: {m['judge_model']}")
        w(f"- **Seeds**: {m['seed_count']}")
        w(f"- **Violations**: {m['violations']} ({m['violation_rate_pct']}%)")
        w(f"- **Overrefusals**: {m['overrefusals']} ({m['overrefusal_rate_pct']}%)")
        w(f"- **Judge failures**: {m['judge_errors']} ({m['judge_failure_rate_pct']}%)")
        if m["node_violation_counts"]:
            w(f"- **Node violations**: {len(m['node_violation_counts'])} distinct failure modes")
            for name, count in m["node_violation_counts"].items():
                w(f"  - {name}: {count}")
        w("")

    # ── Interpretation ──
    w("---\n")
    w("## Interpretation Notes\n")
    w("1. **changliu2 flagship (100 seeds)** provides the highest-confidence signal due")
    w("   to sample size. The single-agent vs multi-agent comparison within changliu2")
    w("   is the most apples-to-apples sub-comparison in this study.\n")
    w("2. **Bloom** uses a fundamentally different metric (behavior_presence 0–10 plus")
    w("   elicitation rate) rather than binary policy_violation. Higher behavior_presence")
    w("   means the evaluator successfully elicited the risky behavior.\n")
    w("3. **origin/main** uses the same judge verdict format as changliu2 but with")
    w("   different seed generation and tool simulation. 8 seeds is too few for")
    w("   reliable rate estimation but useful for format/pipeline validation.\n")
    w("4. **Cross-pipeline comparisons** should focus on qualitative patterns (e.g.,")
    w("   'does the pipeline find violations at all?') rather than precise rate")
    w("   comparisons, due to differing seeds, tools, and sample sizes.\n")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
