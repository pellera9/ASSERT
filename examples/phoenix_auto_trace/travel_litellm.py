"""Travel planner — LiteLLM (provider-agnostic).

Instrumentation: 2 lines. Agent code: LiteLLM completion with tools.
Traces captured: LLM calls, tool calls, token counts, latency, model name.
"""

# pip install openinference-instrumentation-litellm arize-phoenix-otel
from phoenix.otel import register
register(auto_instrument=True)

import json
import litellm
from examples.phoenix_auto_trace._tools import simulate_tool, SYSTEM_PROMPT, OPENAI_TOOLS


def chat(message: str) -> str:
    """Multi-round travel planner via LiteLLM — swap model= to change provider."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    response = litellm.completion(
        model="azure/gpt-5.4-mini",
        messages=messages,
        tools=OPENAI_TOOLS,
        tool_choice="auto",
    )
    msg = response.choices[0].message

    while msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            result = simulate_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({
                "tool_call_id": tc.id,
                "role": "tool",
                "name": tc.function.name,
                "content": result,
            })

        response = litellm.completion(
            model="azure/gpt-5.4-mini",
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

    return msg.content or ""


if __name__ == "__main__":
    print(chat("Plan a week in Tokyo for under $3000"))
