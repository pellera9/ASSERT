# p2m

**p2m** (policy-to-measurements) turns a concept description into an evaluation run:

1. **policy** — systematize the concept and generate a behavioral taxonomy
2. **design** — combine behaviors and contextual factors into a covering array
3. **seeds** — generate prompts and scenarios for each covering-array row
4. **rollout** — run the target model on the generated test cases
5. **judge** — score transcripts against the behavioral taxonomy

## Quickstart

```bash
uv venv && uv sync
cp .env.example .env
source .env
uv run p2m run --config examples/pipes/health_assistant.yaml
```

Results go to `artifacts/results/<suite>/<run>/`. Open the viewer:

```bash
cd viewer && npm install && npm run dev
```

## Minimal config

Put `concept.md` next to your YAML with the risk or behavioral description, then run `uv run p2m run --config my_eval.yaml`.

```yaml
concept:
  name: my_concept

context: |
  What the target does and who uses it.

default_model:
  name: azure/gpt-5.4-mini

pipeline:
  policy: {}
  seeds:
    prompt:
      sample_size: 10
  rollout:
    target:
      system_prompt: |
        You are a helpful assistant.
  judge: {}
```

Add contextual factors to test behaviors across different conditions:

```yaml
factors:
  - name: user_type
    levels:
      - name: novice
        definition: A user with no domain expertise.
      - name: expert
        definition: A user with deep domain knowledge.
```

When factors are present, add `design` to the pipeline:

```yaml
pipeline:
  policy: {}
  design: {}
  seeds:
    prompt:
      sample_size: 10
  rollout:
    target:
      system_prompt: |
        You are a helpful assistant.
  judge: {}
```

After the design stage writes `design.json`, the `behavior` block is yours to edit: subset it, rewrite definitions for the suite, or add custom probes with `"permissible": true|false`. See [Behavior factor and permissible resolution](CONFIG_REFERENCE.md#behavior-factor-and-permissible-resolution) for details.

Full config schema: [`CONFIG_REFERENCE.md`](CONFIG_REFERENCE.md).

## CLI

```bash
p2m run --config eval.yaml           # run full pipeline
p2m run --config eval.yaml --from design  # re-run from design stage
```

Run `p2m --help` for all options.
