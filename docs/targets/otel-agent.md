# OTel Agent Target

Use the OTel agent target when you have a real framework agent and want Adaptive Eval to inspect the execution trajectory, not only the final answer.

## When to use it

Use this path for agents built with frameworks such as:

- LangGraph
- LangChain
- OpenAI Agents SDK
- CrewAI
- LlamaIndex
- AutoGen / Microsoft Agent Framework
- DSPy

Exact trace fidelity depends on what the framework and instrumentation emit. The best-supported paths capture model calls, tool calls, arguments, routing, latency, and intermediate messages.

## How it works

1. Add Phoenix/OpenInference auto-instrumentation to your agent wrapper.
2. Expose a Python callable that accepts the latest user message.
3. Point `target.callable` at that function.
4. Configure `target.trace`.

Example wrapper:

```python
from phoenix.otel import register

register(auto_instrument=True)

from examples.travel_planner_langgraph.agent import chat_sync
```

Example YAML:

```yaml
pipeline:
  rollout:
    target:
      callable: examples.travel_planner_langgraph.auto_trace:chat_sync
      trace:
        backend: phoenix
        group_by: session.id
    auditor:
      model: { name: azure/gpt-5.4-mini, temperature: 0.0 }
    max_turns: 6
```

## Why this is the primary agent path

For multi-agent and tool-using systems, final text is often insufficient. Trace capture helps answer:

- Which tool did the agent call?
- What arguments did it send?
- Did it skip a required safety or validation step?
- Did it route to the wrong node?
- Did a tool return data that the final answer ignored or fabricated around?

## Flagship example

Run:

```powershell
uv run p2m run --config examples\travel_planner_langgraph\eval_config.yaml
```

Then inspect:

- `transcripts.jsonl`
- `scores.jsonl`
- Phoenix traces, if Phoenix is running

## Caveat

`target.callable` is the integration boundary. If your callable does not accept conversation history, each target invocation is a fresh call from the agent's perspective while Adaptive Eval maintains the outer transcript.
