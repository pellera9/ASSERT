# Pipeline Configs

Run any config with:

```bash
uv run p2m run --config examples/pipes/<name>.yaml
```

## Target shapes

Each config evaluates a health assistant for harmful medical advice. They share the same risk, system prompt, and pipeline structure — only the target block differs. Pick the one that matches your setup.

| Config | Target | What it demonstrates |
|---|---|---|
| `health_assistant.yaml` | hosted, no tools | Plain chat model. Simplest full pipeline. |
| `health_assistant_sandbox.yaml` | hosted + sandbox-backed module | Real Python tools via `examples.agents.health_assistant`, with one Docker container per conversation. Requires Docker and may pull `python:3.11-bookworm` on first use. |
| `health_assistant_simulated_tools.yaml` | hosted + fixed toolset | Tool schemas from a YAML file, simulator model generates responses. |
| `health_assistant_generated_tools.yaml` | hosted + per-seed tools | Each seed carries its own tool definitions. |
| `health_assistant_external.yaml` | external connector | External agent owns the conversation; p2m records and scores it. Requires local Docker Compose because `examples.agents.openclaw` starts one containerized agent session per conversation, and it expects `AZURE_API_KEY` plus `AZURE_API_BASE` in the host environment. |
