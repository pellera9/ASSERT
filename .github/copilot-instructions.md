# Copilot Instructions — Adaptive Eval (P2M)

## Commands

```bash
# Setup
uv venv && uv sync

# Run the canonical evaluation pipeline
uv run p2m run --config examples/pipes/health_assistant.yaml

# Tests
uv run pytest -q                        # full suite
uv run pytest tests/test_cli.py -q      # single file
uv run pytest tests/test_cli.py::CliTest::test_removed_commands_are_unavailable -q  # single test

# Lint & format
ruff check --fix && ruff format
pyright                                  # type checking

# Viewer (SvelteKit results browser)
cd viewer && npm install && npm run dev  # dev server
cd viewer && npm run check && npm run build  # type-check + build
```

## Architecture

Four-stage YAML-driven evaluation pipeline: **policy → seeds → rollout → judge**.

```
CLI (p2m/cli.py) → config.py → runner.py → stages/{policy,seeds,rollout,judge}.py
```

- **policy** — LLM expands a risk definition (markdown in `examples/risks/`) into a structured taxonomy (`policy.json`).
- **seeds** — LLM generates test cases (prompts + multi-turn scenarios) from the taxonomy → `seeds.jsonl`.
- **rollout** — Executes seeds against the target (hosted model, tool-using agent, or external connector) → `transcripts.jsonl`.
- **judge** — LLM scores each conversation against the policy → `scores.jsonl` + `metrics.json`.

Artifacts land in `artifacts/results/<suite>/<run>/`. The viewer reads them directly from disk — no database.

### Key modules

| Module | Role |
|---|---|
| `p2m/config.py` + `p2m/core/config_model.py` | YAML loading, validation, typed dataclasses |
| `p2m/runner.py` | Stage orchestration, manifest management |
| `p2m/core/session.py` | `HostedSession`, `CallableSession`, `HTTPEndpointSession`, `ExternalSession` |
| `p2m/core/model_client.py` | LiteLLM-backed model calls, normalized tool-call types |
| `p2m/core/transcript.py` | Event-based transcript model shared by rollout, judge, and viewer |
| `p2m/core/judge.py` | Verdict aggregation, citation resolution |

### Target shapes (rollout)

The `pipeline.rollout.target` block determines execution mode:

- Plain hosted model: `target.model.name`
- Hosted + Python tools: `target.model` + `target.tools.module`
- Hosted + simulated tools: `target.model` + `target.tools.toolset` + `target.tools.simulator`
- External agent: `target.connector`

### Systematization stages

Optional stages (`systematization`, `systematization_convert`) produce `systematization.json` → `policy.json`. These are separate from the standard 4-stage pipeline.

## Conventions

- **Python 3.11+**. Type hints on public helpers. `snake_case` everywhere in Python/YAML. `PascalCase` in Svelte.
- **Conventional Commits**: `fix:`, `feat:`, `refactor:`, `docs:`.
- **No hardcoding** unless necessary. No fallbacks unless explicitly requested. No unnecessary abstractions.
- **Model names** use LiteLLM's `provider/model-name` format (e.g., `azure/gpt-5.4`, `openai/gpt-4o`). Credentials come from env vars or `.env`.
- **`model`** in YAML config is always a mapping (`name`, `temperature`, `max_tokens`), never a bare string.
- **Config validation**: `max_turns` lives on `pipeline.rollout`, not inside `auditor`.
- **Tests**: pytest with `unittest.TestCase` classes. Name tests after behavior. Focus on pipeline I/O, artifact schemas, judgment aggregation. Run `uv run pytest -q` before PRs.
- **Cleanup on every touched file**: inline single-use helpers, remove dead code/config keys/wrappers/fallbacks. One parsing path, one prompt path, one config path. Constants at module top.
- **Docs**: two audiences (PM: what it does / contributor: how to change). Code examples must be real and runnable. Update `docs/architecture-map.md` if control flow changes.

## Terminology

Use new terminology — do not frame adaptive evaluation as safety-only:

| Old term | Current term |
|---|---|
| Risk | Failure Modes |
| Risk Definitions | Failure Mode Definitions |
| Suite | Dataset |
| Pipeline | Generation |
| Taxonomy | Requirement Map |
| Seeds | Test Cases |
| Audit | Conversations |
