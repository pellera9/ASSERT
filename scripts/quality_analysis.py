"""Per-behavior quality analysis with pass rates and improvement recommendations."""
import json
from collections import defaultdict

SUITES = {
    "A (OTel)": "artifacts/results/travel-planner-otel-v1/approach-a/scores.jsonl",
    "B (Callable)": "artifacts/results/travel-planner-callable-v1/approach-b/scores.jsonl",
    "C (Connector)": "artifacts/results/travel-planner-adapter-v1/approach-c/scores.jsonl",
}

# Collect all data
all_records = []
for approach, path in SUITES.items():
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            dims = s.get("verdict", {}).get("dimensions", {})
            just = s.get("verdict", {}).get("justification", "")
            all_records.append({
                "approach": approach,
                "seed_id": s.get("seed_id", ""),
                "kind": s.get("kind", ""),
                "sub_risk": s.get("sub_risk", "unknown"),
                "permissible": s.get("permissible", True),
                "pv": dims.get("policy_violation", False),
                "ov": dims.get("overrefusal", False),
                "ha": dims.get("harm_actionability", False),
                "justification": just,
            })

# Group by behavior pattern (normalize similar names)
NORMALIZE = {
    "Explicit clarification of ambiguous intent": "Intent: Clarify ambiguous requests",
    "Clarifying questions when intent ambiguous": "Intent: Clarify ambiguous requests",
    "Failing to ask required clarifying questions": "Intent: Clarify ambiguous requests",
    "Appropriate clarification and confirmation": "Intent: Clarify ambiguous requests",
    "Misclassifying user intent": "Intent: Classify correctly",
    "Advisory-only reply when action expected (no booking performed)": "Intent: Classify correctly",
    
    "Calling or routing to the wrong tool/action": "Routing: Correct tool selection",
    "Calling the wrong tool or incorrect routing": "Routing: Correct tool selection",
    "Wrong tool or connector selection": "Routing: Correct tool selection",
    "Skipping required tool checks (availability/pricing/eligibility)": "Routing: Required tool checks",
    "Skipping required availability checks before booking": "Routing: Required tool checks",
    "Skipping required tool or verification step": "Routing: Required tool checks",
    "Unnecessary or redundant tool calls": "Routing: Efficient tool usage",
    "Inefficient or redundant tool usage": "Routing: Efficient tool usage",
    
    "Fabricated or invented booking details": "Grounding: No fabricated details",
    "Fabricated factual details (invented flights, hotels, confirmations, prices)": "Grounding: No fabricated details",
    "Fabricated identifiers, prices, or confirmations": "Grounding: No fabricated details",
    "Overly specific unsourced claims": "Grounding: No fabricated details",
    "Asserting availability or guarantees without verification": "Grounding: Sourced claims only",
    "Quoted prices or availability without provenance or validity period": "Grounding: Sourced claims only",
    "Incomplete sourcing or missing provenance for dynamic facts": "Grounding: Sourced claims only",
    "Clear, tool-attributed results and sourcing": "Grounding: Sourced claims only",
    "Providing stale, outdated, or unverifiable information": "Grounding: Sourced claims only",
    "Price, currency, or fee inconsistencies and stale data": "Grounding: Sourced claims only",
    
    "Ignoring or violating explicit user constraints": "Constraints: Respect user limits",
    "Violation of explicit user constraints": "Constraints: Respect user limits",
    
    "Producing incomplete plans missing required components": "Completeness: Full itinerary",
    "Missing required trip components": "Completeness: Full itinerary",
    "Omitting required itinerary components": "Completeness: Full itinerary",
    "Incomplete itinerary or missing required components": "Completeness: Full itinerary",
    "Incomplete or ambiguous itinerary elements": "Completeness: Full itinerary",
    "Complete itinerary respecting stated constraints": "Completeness: Full itinerary",
    
    "Failing to surface important caveats, advisories, or entry requirements when available": "Caveats: Surface warnings",
    "Failure to surface relevant caveats or travel requirements": "Caveats: Surface warnings",
    "Proactive safety, regulatory, and timing caveats": "Caveats: Surface warnings",
    
    "Multi-turn context loss, contradictions, or repeated questioning": "Coherence: Multi-turn consistency",
    "Multi-turn contradiction or context loss": "Coherence: Multi-turn consistency",
    "Poor multi-turn coherence (losing or contradicting prior context)": "Coherence: Multi-turn consistency",
    "Internal contradictions within a single response": "Coherence: Multi-turn consistency",
    
    "Informational travel guidance": "Helpfulness: Informational guidance",
    "High-level informational planning or advice": "Helpfulness: Informational guidance",
    "Basic itinerary construction and recommendation": "Helpfulness: Informational guidance",
    "Unclear or missing booking action steps and next actions": "Helpfulness: Clear next steps",
    
    "Booking or taking irreversible actions without explicit user consent": "Safety: Confirm before booking",
    "Privacy-sensitive data exposure": "Safety: Data privacy",
    "Unnecessary or risky personal data (PII) requests and exposures": "Safety: Data privacy",
    "Commercial bias or hidden affiliation in recommendations": "Safety: Unbiased recommendations",
}

by_behavior = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0, "justifications": []})

for r in all_records:
    normalized = NORMALIZE.get(r["sub_risk"], f"Other: {r['sub_risk']}")
    by_behavior[normalized]["total"] += 1
    if r["pv"]:
        by_behavior[normalized]["failed"] += 1
        by_behavior[normalized]["justifications"].append(r["justification"][:200])
    else:
        by_behavior[normalized]["passed"] += 1

# Sort by failure rate (worst first)
sorted_behaviors = sorted(by_behavior.items(), key=lambda x: x[1]["failed"]/max(x[1]["total"],1), reverse=True)

print("=" * 110)
print("TRAVEL PLANNER QUALITY — PER-BEHAVIOR PASS RATE ANALYSIS")
print(f"(Aggregated across all 3 approaches, {len(all_records)} total evaluations)")
print("=" * 110)
print()
print(f"{'Behavior':<50s} {'Pass Rate':>10s} {'Passed':>8s} {'Failed':>8s} {'Total':>7s}")
print("-" * 110)

for behavior, stats in sorted_behaviors:
    total = stats["total"]
    passed = stats["passed"]
    failed = stats["failed"]
    rate = passed / total * 100 if total > 0 else 0
    icon = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
    print(f"{icon} {behavior:<48s} {rate:>8.0f}%  {passed:>6d}  {failed:>6d}  {total:>5d}")

# Category rollup
print()
print("=" * 110)
print("CATEGORY ROLLUP")
print("=" * 110)
print()

categories = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
for behavior, stats in sorted_behaviors:
    cat = behavior.split(":")[0]
    categories[cat]["total"] += stats["total"]
    categories[cat]["passed"] += stats["passed"]
    categories[cat]["failed"] += stats["failed"]

sorted_cats = sorted(categories.items(), key=lambda x: x[1]["failed"]/max(x[1]["total"],1), reverse=True)
for cat, stats in sorted_cats:
    rate = stats["passed"] / stats["total"] * 100
    icon = "✅" if rate >= 80 else "⚠️" if rate >= 50 else "❌"
    print(f"{icon} {cat:<30s} {rate:>6.0f}% pass  ({stats['passed']}/{stats['total']})")

# Improvement recommendations
print()
print("=" * 110)
print("IMPROVEMENT RECOMMENDATIONS FOR DEMO TARGET")
print("=" * 110)

failing_behaviors = [(b, s) for b, s in sorted_behaviors if s["failed"] / max(s["total"], 1) > 0.3]
for behavior, stats in failing_behaviors:
    rate = stats["passed"] / stats["total"] * 100
    print(f"\n{'─'*80}")
    print(f"❌ {behavior} — {rate:.0f}% pass ({stats['failed']}/{stats['total']} failures)")
    # Show sample justifications
    for j in stats["justifications"][:2]:
        print(f"   Sample: {j[:150]}...")
    
    # Recommend fix
    cat = behavior.split(":")[0]
    if cat == "Grounding":
        print(f"   FIX: Add disclaimer to system prompt: 'Always state that prices and availability")
        print(f"        are estimates and must be verified. Never invent flight numbers or hotel names.'")
    elif cat == "Completeness":
        print(f"   FIX: Add to system prompt: 'Every itinerary must include: transport, accommodation,")
        print(f"        estimated total cost, and a checklist of items the user should verify.'")
    elif cat == "Intent":
        print(f"   FIX: Add to system prompt: 'When the user's request is ambiguous or complex,")
        print(f"        ask 1-2 clarifying questions before generating an itinerary.'")
    elif cat == "Routing":
        print(f"   FIX: Add to system prompt: 'Always explicitly verify availability and pricing")
        print(f"        before making recommendations. State what you checked and what you could not.'")
    elif cat == "Caveats":
        print(f"   FIX: Add to system prompt: 'Always mention visa requirements, travel advisories,")
        print(f"        seasonal considerations, and health precautions for international travel.'")
    elif cat == "Constraints":
        print(f"   FIX: Add to system prompt: 'Restate the user's budget and date constraints at the")
        print(f"        start of your response. Ensure every recommendation fits within them.'")
    elif cat == "Coherence":
        print(f"   FIX: The mock target is stateless (single litellm.completion call). Add conversation")
        print(f"        history to the callable — currently each turn is independent.")
    elif cat == "Safety":
        print(f"   FIX: Add to system prompt: 'Never request sensitive personal information.")
        print(f"        Be transparent about any affiliations or biases in recommendations.'")
    elif cat == "Helpfulness":
        print(f"   FIX: Add to system prompt: 'Always provide actionable next steps the user can take.")
        print(f"        Include specific links, phone numbers, or booking platforms when relevant.'")
    else:
        print(f"   FIX: Review the system prompt for coverage of this behavior category.")
