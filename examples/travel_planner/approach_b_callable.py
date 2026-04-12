"""Approach B: Callable target (black-box).

USER-SIDE CODE: 3 lines — a thin wrapper function.
P2M-SIDE CODE: Already shipped in p2m/core/session.py (CallableSession).

This is the simplest approach. The developer writes a function, p2m calls it.
No instrumentation, no framework adapters, no config plumbing.
Visibility: input/output only (black box).

Line count comparison:
  User writes ............. 3 lines (this file)
  p2m provides ............ CallableSession in p2m/core/session.py (already shipped)
"""

from __future__ import annotations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USER-SIDE: What the developer writes (3 lines total)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from examples.travel_planner.agent import chat_sync


def target(message: str) -> str:
    """Thin wrapper — p2m calls this with a seed prompt, gets a string back."""
    return chat_sync(message)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P2M-SIDE: Already exists — nothing to write
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# p2m/core/session.py ships CallableSession which:
#   1. Imports the callable from "module:function" format
#   2. Detects sync/async and history support automatically
#   3. Invokes the callable and wraps the response in TurnResult
#
# The YAML config is all that connects the two:
#
#   rollout:
#     target:
#       callable: examples.travel_planner.approach_b_callable:target
#
# That's it. No adapter code, no instrumentation, no session classes.
# The trade-off: p2m sees only the final string response. No visibility
# into internal tool calls, LLM chains, or execution traces.
