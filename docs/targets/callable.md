# Callable Target

Use the callable target when your app or agent has a Python entry function.

## Shape

The callable can be synchronous or asynchronous. The simplest shape is:

```python
def chat_sync(message: str) -> str:
    return "assistant response"
```

Then configure:

```yaml
pipeline:
  rollout:
    target:
      callable: package.module:chat_sync
```

## Conversation history

If your callable accepts a `history` parameter, Adaptive Eval can pass prior user/assistant turns:

```python
def chat_sync(message: str, history: list[dict[str, str]]) -> str:
    ...
```

Use this shape for agents that need multi-turn state during scenario rollout.

## Return values

The callable can return:

- a plain string
- a structured model response supported by the runtime
- a dictionary with text/content fields

For the richest observability, combine callable target with OTel instrumentation when your callable runs a framework agent.

## When to choose callable without OTel

Use plain callable when:

- you want a quick integration
- final text is enough for the first eval
- the target does not yet emit useful spans
- you are evaluating a small wrapper around an existing system

Upgrade to OTel when tool calls, routing, or intermediate decisions matter.
