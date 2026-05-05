# Target Support Overview

Adaptive Eval can evaluate several target shapes. Pick the simplest path that matches how your agent is built.

## Decision tree

```text
Do you have a framework agent?
  yes -> use OTel agent target
  no
    Do you have a Python entry function?
      yes -> use callable target
      no
        Is it a hosted model with a system prompt or tools?
          yes -> use model + tools target
```

## Target paths

| Path | Best for | Config anchor |
|---|---|---|
| OTel agent target | Real framework agents where you want trace-grounded evaluation | `target.callable` + `target.trace` |
| Callable target | Python functions that accept a message and return text | `target.callable` |
| Model + tools target | Hosted prompt agents or simple model/tool setups | `target.model`, `target.system_prompt`, `target.tools` |

## Recommended default

If you are evaluating a real agent built with a framework, start with the OTel agent target. It gives the judge richer evidence than final text alone.

## What is not a preview-first path

The customer preview docs do not lead with an external connector path. For most agents, `target.callable` plus OTel trace capture is simpler, easier to debug, and closer to how developers already run local code.
