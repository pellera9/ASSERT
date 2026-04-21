"""Approach B: Callable target — two return types, two visibility levels.

USER-SIDE CODE: A thin wrapper function returning either str or ModelResponse.
P2M-SIDE CODE: Already shipped in p2m/core/session.py (CallableSession).

CallableSession auto-detects the return type:
  - fn(str) -> str                 → black-box (1/8 behaviors visible)
  - fn(str) -> litellm.ModelResponse → tool calls + usage visible (4/8 behaviors)

Both variants below wrap the SAME LangGraph agent.
"""

from examples.travel_planner.agent import chat_sync

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant 1: str return (black-box, 1/8 visibility)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def target_str(message: str) -> str:
    """Returns final text only. Judge sees input/output, nothing else."""
    return chat_sync(message)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Variant 2: ModelResponse return (4/8 visibility)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def target(message: str):
    """Returns litellm-style response. Judge sees tool calls + usage.

    CallableSession._normalize_callable_result detects the .choices
    attribute and extracts tool_calls, model name, and token usage.
    """
    import litellm

    # Same agent — but we call litellm directly so we can return the
    # raw response object with tool call metadata intact.
    response = litellm.completion(
        model="azure/gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a travel planner with tools."},
            {"role": "user", "content": message},
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search flights to a destination.",
                "parameters": {
                    "type": "object",
                    "properties": {"destination": {"type": "string"}},
                    "required": ["destination"],
                },
            },
        }],
        temperature=0.2,
        max_tokens=4000,
    )
    return response  # raw litellm.ModelResponse — CallableSession extracts tool_calls


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# YAML config for either variant:
#
#   # Black-box (str):
#   target:
#     callable: examples.travel_planner.approach_b_callable:target_str
#
#   # Rich (ModelResponse):
#   target:
#     callable: examples.travel_planner.approach_b_callable:target
#
# The actual experiment in approach_b_config.yaml uses
# p2m.travel_target:target which returns ModelResponse with tool
# calls extracted from the full LangGraph execution.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
