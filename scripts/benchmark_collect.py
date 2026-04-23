"""Collect benchmark results across frameworks for report generation."""
import json
from pathlib import Path

base = Path("artifacts/results")

def analyze(suite, run):
    p = base / suite / run / "scores.jsonl"
    if not p.exists():
        return None
    scores = [json.loads(l) for l in p.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    violations = sum(1 for s in scores if s.get("verdict", {}).get("dimensions", {}).get("policy_violation"))
    overrefusals = sum(1 for s in scores if s.get("verdict", {}).get("dimensions", {}).get("overrefusal"))
    jf = sum(1 for s in scores if not s.get("verdict") or not s.get("verdict", {}).get("dimensions"))
    modes = set(s.get("sub_risk", "") for s in scores if s.get("verdict", {}).get("dimensions", {}).get("policy_violation"))
    return {"total": len(scores), "violations": violations, "overrefusals": overrefusals, "judge_fails": jf, "modes": modes}

mini = {
    "openai": ("framework-eval-mini", "openai"),
    "litellm": ("framework-eval-mini", "litellm"),
    "langchain": ("framework-eval-mini", "langchain"),
    "langgraph": ("framework-eval-mini", "langgraph"),
    "dspy": ("framework-eval-mini", "dspy"),
    "crewai": ("framework-eval-mini", "crewai"),
    "neurosan": ("travel-planner-neurosan-v1", "custom-otel"),
}
full = {
    "openai": ("framework-eval-mini", "openai-54judge"),
    "litellm": ("framework-eval-mini", "litellm-54judge"),
    "langchain": ("framework-eval-mini", "langchain-54judge"),
    "langgraph": ("framework-eval-mini", "langgraph-54judge"),
    "dspy": ("framework-eval-mini", "dspy-54judge"),
    "crewai": ("framework-eval-mini", "crewai-54judge"),
    "neurosan": ("travel-planner-neurosan-v1", "neurosan-54judge"),
}
labels = {
    "openai": "OpenAI SDK", "litellm": "LiteLLM", "langchain": "LangChain/LangGraph",
    "langgraph": "LangGraph (mock tools)", "dspy": "DSPy", "crewai": "CrewAI",
    "neurosan": "Custom (NeurOSan)",
}
types = {k: "auto (Phoenix)" for k in mini}
types["neurosan"] = "manual (OTel API)"

print("=== GPT-5.4 JUDGE RESULTS ===")
print("| Framework | Instrumentation | Seeds | Policy Violations | Overrefusals | Judge Failures | Unique Failure Modes |")
print("|-----------|-----------------|------:|-----------------:|-------------:|---------------:|--------------------:|")
all_modes_54 = {}
for name in full:
    s, r = full[name]
    d = analyze(s, r)
    all_modes_54[name] = d["modes"]
    print(f"| {labels[name]} | {types[name]} | {d['total']} | {d['violations']} | {d['overrefusals']} | {d['judge_fails']} | {len(d['modes'])} |")

print()
print("=== FAILURE MODE MATRIX (gpt-5.4 judge) ===")
all_fm = set()
for m in all_modes_54.values():
    all_fm |= m
keys = list(full.keys())
print("| Failure Mode |", " | ".join(keys), "|")
print("|---|" + "|".join([":---:"] * len(keys)) + "|")
for fm in sorted(all_fm):
    row = [("Y" if fm in all_modes_54.get(n, set()) else "") for n in keys]
    print(f"| {fm} |", " | ".join(row), "|")
print(f"\nTotal unique failure modes (gpt-5.4 judge): {len(all_fm)}")

print()
print("=== COMPARISON: gpt-5.4-mini vs gpt-5.4 judge ===")
print("| Framework | Judge | Violations | Judge Failures | Failure Modes |")
print("|-----------|-------|----------:|---------------:|--------------:|")
for name in keys:
    s_m, r_m = mini[name]
    dm = analyze(s_m, r_m)
    s_f, r_f = full[name]
    df = analyze(s_f, r_f)
    print(f"| {labels[name]} | gpt-5.4-mini | {dm['violations']} | {dm['judge_fails']} | {len(dm['modes'])} |")
    print(f"| | **gpt-5.4** | **{df['violations']}** | **{df['judge_fails']}** | **{len(df['modes'])}** |")
