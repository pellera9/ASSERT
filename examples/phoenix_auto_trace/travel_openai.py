"""Travel planner — OpenAI direct (function calling).

Instrumentation: 2 lines. Agent code: standard OpenAI SDK.
Traces captured: LLM calls, tool calls with args/results, token counts, latency.
"""

# pip install openinference-instrumentation-openai arize-phoenix-otel
from phoenix.otel import register
register(auto_instrument=True)

import json
from openai import OpenAI
from examples.phoenix_auto_trace._tools import simulate_tool, SYSTEM_PROMPT, OPENAI_TOOLS

client = OpenAI()


def chat(message: str) -> str:
    """Multi-round travel planner using OpenAI function calling."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
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
                "content": result,
            })

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

    return msg.content or ""


if __name__ == "__main__":
    print(chat("Plan a week in Tokyo for under $3000"))
