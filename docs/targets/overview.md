# Target Support Overview

Adaptive Eval can evaluate any agent or multi-agent system. Pick the simplest path that matches how your agent is built.

## Decision tree

```text
Do you have a Python entry function for your agent or multi-agent system?
  yes -> use the callable agent target
         (optionally add OTel/Phoenix trace capture for richer judge evidence)
  no
    Is it a hosted model with a system prompt or tools?
      yes -> use model + tools target
```

## Target paths

| Path | Best for | Config anchor |
|---|---|---|
| Callable agent target (any agent) | Any agent or multi-agent system you can invoke from Python — frameworks, custom orchestration, REST clients, thin model wrappers | `target.callable` (optionally `target.trace` for OTel/Phoenix capture) |
| Model + tools target | Hosted prompt agents or simple model/tool setups | `target.model`, `target.system_prompt`, `target.tools` |

## Recommended default

For real agents and multi-agent systems, start with the callable agent target. It is the universal integration boundary: any agent that runs in Python can be evaluated this way. When the judge would benefit from seeing tool calls, routing, or intermediate decisions, add `target.trace` and Phoenix/OpenInference instrumentation as an opt-in upgrade.

> New to OpenTelemetry? You can ignore it on the first run. The callable target only requires a Python function; trace capture is an optional enhancement.

## What is not a preview-first path

The customer preview docs do not lead with an external connector path. For most agents, `target.callable` is simpler, easier to debug, and closer to how developers already run local code.
