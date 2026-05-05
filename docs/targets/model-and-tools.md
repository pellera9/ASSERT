# Model and Tools Target

Use the model and tools target for simple prompt agents: a hosted model, a system prompt, and optionally tool definitions.

## Hosted model

```yaml
pipeline:
  rollout:
    target:
      model:
        name: azure/gpt-5.4-mini
        temperature: 0.0
        max_tokens: 8000
      system_prompt: |
        You are a helpful assistant. Follow the product policy and ask clarifying
        questions when user constraints are missing.
```

This is the fastest way to smoke-test a single model target.

## Hosted model with Python tools

```yaml
pipeline:
  rollout:
    target:
      model:
        name: azure/gpt-5.4-mini
      tools:
        module: examples.agents.health_assistant
```

Use this when the tool implementation exists and can run locally.

## Hosted model with simulated tools

```yaml
pipeline:
  rollout:
    target:
      model:
        name: azure/gpt-5.4-mini
      tools:
        toolset: examples/agents/health_assistant_tools.yaml
        simulator: azure/gpt-5.4-mini
```

Simulated tools are useful when:

- your prompt agent has a planned tool schema
- real backends are not available yet
- you want to test whether the model calls the right tool and uses plausible results

They are not a replacement for tracing a real framework agent. If you already have a LangGraph, CrewAI, LlamaIndex, OpenAI Agents SDK, AutoGen/MAF, or similar agent, prefer the OTel agent target.
