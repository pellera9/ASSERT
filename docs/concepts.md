# Concepts

Adaptive Eval is a spec-driven evaluation pipeline for AI agents.

The mental model is:

```text
spec -> behavior categories -> test cases -> execute -> judge -> artifacts
```

## Spec

The spec is the plain-English behavior definition you author before running the pipeline. In current YAML, this is configured with `concept.name`, which points to a `.md` file next to the config.

Example:

```yaml
concept:
  name: travel_planner_eval
```

This loads `travel_planner_eval.md`.

## Behavior categories

The `policy` stage reads the spec and target context, then creates a structured taxonomy of behaviors or failure modes to test.

```yaml
pipeline:
  policy:
    behavior_count: 6
```

Output: `policy.json`.

## Variations

Variations are coverage axes. In current YAML, these are `factors`.

```yaml
factors:
  - name: traveler_type
    description: The type of traveler using the planner.
  - name: trip_type
    description: The kind of trip being planned.
```

The generator uses these to avoid a narrow test set. For example, the same budget constraint can be tested across family travel, business travel, urgent travel, and accessibility-sensitive travel.

## Test cases

The `seeds` stage generates:

- `prompt` seeds for single-turn inputs.
- `scenario` seeds for multi-turn adversarial conversations.

Output: `seeds.jsonl`.

## Execute

The `rollout` stage executes each generated test case against your target.

For real framework agents, prefer a callable entrypoint plus OTel trace capture:

```yaml
pipeline:
  rollout:
    target:
      callable: examples.travel_planner_langgraph.auto_trace:chat_sync
      trace:
        backend: phoenix
        group_by: session.id
```

Output: `transcripts.jsonl`.

## Judge

The `judge` stage scores each transcript against your dimensions and rubrics.

```yaml
pipeline:
  judge:
    dimensions:
      policy_violation:
        description: Did the agent violate the eval spec?
        rubric: |
          true = the agent fabricated, ignored constraints, or used tools incorrectly
          false = the agent stayed grounded and respected the requirements
```

Outputs: `scores.jsonl` and `metrics.json`.

## Artifacts are the product surface

Adaptive Eval does not require a hosted service to inspect results. The artifacts are local JSON/JSONL files that can be reviewed, diffed, archived, and used in CI.
