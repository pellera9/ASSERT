"""Verify judge schema has no strict-mode violations."""
import json
import re
from p2m.core.judge import build_judge_schema, resolve_dimensions

dims = resolve_dimensions(["harm_actionability", "overrefusal", "policy_violation"])
schema = build_judge_schema(dims, include_citations=True, citation_style="xml", node_count=5)
s = json.dumps(schema)
bad = re.findall(r'"(minLength|maxLength|minimum|maximum|maxItems|minItems|pattern|format)"', s)
if bad:
    print(f"VIOLATIONS: {bad}")
else:
    print("Schema is strict-mode clean")
print(f"Schema size: {len(s)} chars")
