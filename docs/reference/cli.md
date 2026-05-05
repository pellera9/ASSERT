# CLI Reference

Adaptive Eval is CLI-first.

## Run a config

```powershell
uv run p2m run --config examples\travel_planner_langgraph\eval_config.yaml
```

## Re-run one stage

```powershell
uv run p2m run --config examples\travel_planner_langgraph\eval_config.yaml --force-stage seeds
```

Use this when you intentionally changed a stage input and want to regenerate downstream artifacts.

## List runs

```powershell
uv run p2m results list
```

## Show run status

```powershell
uv run p2m results status travel-planner-langgraph-v1 demo-1
```

## Compare runs

```powershell
uv run p2m results compare <suite> <run-a> <run-b>
```

## Analyze generated test cases

```powershell
uv run p2m analysis seed-metrics --policy artifacts\results\<suite>\policy.json --seeds artifacts\results\<suite>\seeds.jsonl
```

## Where outputs go

All outputs are written under:

```text
artifacts/results/<suite>/<run>/
```

The CLI does not require a hosted service.
