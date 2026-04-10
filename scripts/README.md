# Scripts

Run scripts in this directory with `uv run python ...` from the repo root so they see the project package and pinned dependencies.

## `policy_scenario_sampling.py`

This script reads `artifacts/results/<suite>/policy.json` and writes reusable scenario-sampling artifacts under `artifacts/tmp/<suite>/scenario_sampling/`. `design` writes `scenario_design.json` and always runs with web search enabled plus `high` reasoning, so the design model must support that path. `generate` reads `policy.json` plus `scenario_design.json` and writes method-specific seed artifacts under `artifacts/tmp/<suite>/scenario_sampling/<method>/`, including `seeds.jsonl`, `observed_labels.jsonl`, `summary.json`, `supplementary_metrics.json`, and `report.md`. It also refreshes a root-level `scenario_sampling_report.md`. `tuple_sampled` also writes `sampled_tuples.json`.

```bash
uv run python scripts/policy_scenario_sampling.py design \
  --policy artifacts/results/<suite>/policy.json \
  --model azure/gpt-5.4-mini

uv run python scripts/policy_scenario_sampling.py generate \
  --policy artifacts/results/<suite>/policy.json \
  --design artifacts/tmp/<suite>/scenario_sampling/scenario_design.json \
  --model azure/gpt-5.4-mini \
  --method tuple_sampled \
  --sample-size 24
```

## `turn_checkpoint_judge.py`

This script re-judges transcript prefixes from one completed run at fixed rollout-turn checkpoints. It reads `<run>/transcripts.jsonl`, `<run>/config.yaml`, and the suite `policy.json`, then writes checkpoint artifacts under `<run>/checkpoint_judge/` by default: `checkpoint_scores.jsonl`, `checkpoint_metrics.json`, and `<dimension>_by_turn.svg`.

```bash
uv run python scripts/turn_checkpoint_judge.py \
  --run-dir artifacts/results/<suite>/<run> \
  --checkpoint-step 5 \
  --dimension policy_violation
```

## `export_suite_results.py`

This script consolidates all runs in one suite into flat review tables under `artifacts/results/<suite>/exports/`. It reads the suite artifacts plus each run's `manifest.json`, `transcripts.jsonl`, `scores.jsonl`, and `metrics.json`. `--format csv` writes `runs.csv`, `seeds.csv`, `conversations.csv`, `scores.csv`, and `relevant_nodes.csv`. `--format excel` writes `suite_results.xlsx`. `--format html` writes `suite_results.html`. If you omit `--format`, the script writes CSV only. Excel export requires `uv sync --extra analysis`. Runs with `scores.jsonl` must already have current `metrics.json` with `by_relevant_node`.

```bash
uv run python scripts/export_suite_results.py \
  --suite relationship-entanglement-v1 \
  --format csv \
  --format excel \
  --format html
```

## `auditor_pairwise_eval.py`

This script compares two auditor rollout runs on the same suite by matching scenario `seed_id`s and judging two separate axes for each matched pair. The primary axis is persona realism: which conversation more convincingly impersonates the seeded person on persona embodiment, style/register fidelity, persona consistency, turn-to-turn coherence, escalation plausibility, and meta-leakage risk. The second axis is auditor effectiveness: which conversation uses its turns more effectively to surface policy-relevant evidence through stronger elicitation, adaptivity, evidence yield, pressure calibration, and turn efficiency. Both judge prompts are blinded to run IDs and rollout metadata, and each axis is judged twice per pair: once as A/B and once as B/A. Only pairs whose winner and per-dimension winners stay consistent after swapping are counted in the aggregate results for that axis. The script writes `pairwise_scores.jsonl`, `pairwise_metrics.json`, `pairwise_summary.md`, and a judge-written `pairwise_eval.md` under `artifacts/tmp/<suite>/pairwise/<run-a>_vs_<run-b>/` by default. In `pairwise_scores.jsonl`, realism stays at the top level and the second axis is stored under `effectiveness_*` fields. In `pairwise_metrics.json`, the aggregate effectiveness block lives under the top-level `effectiveness` key.

```bash
uv run python scripts/auditor_pairwise_eval.py \
  --run-a artifacts/results/<suite>/<run-a> \
  --run-b artifacts/results/<suite>/<run-b> \
  --judge-model azure/gpt-5.4
```

## `scenario_failure_prediction.py`

This script predicts policy violations from scenario metadata before running conversations. It runs four stages: (0) per-seed failure-rate distribution, (1) baselines (global rate, sub-risk rate, embedding nearest neighbor, logistic regression on embeddings), (2) zero-shot LLM forecaster with field ablations, (3) retrieval-augmented LLM forecaster. It also runs two robustness checks: within-sub-risk discrimination and auditor transfer. The script auto-detects the primary auditor (most common across runs) and uses the remaining runs for the transfer check. Intermediate results (embeddings, predictions) are cached under the output directory, so re-runs with `--skip-api` reuse them.

```bash
uv run python scripts/scenario_failure_prediction.py \
  --suite relationship-entanglement-v1 \
  --model gpt-5.4-mini
```

## `run_pairwise_expansion.sh`

Shell script that runs `auditor_pairwise_eval.py` for new auditor comparisons (GPT-5.4-mini and GPT-5.4-nano) against GPT-5.4 and GPT-5-railfree, across all 9 judge models. Skips comparisons whose output directory already exists. Requires all four auditor runs to be complete (checks for `metrics.json`).

```bash
bash scripts/run_pairwise_expansion.sh
```

## `aggregate_pairwise_results.py`

Scans all `pairwise_metrics.json` files under `artifacts/tmp/relationship-entanglement-v1/pairwise/`, deduplicates by (comparison, judge), excludes GPT-5-nano, and writes a CSV table to stdout. The output has one row per (comparison, judge) with realism and effectiveness consistency rates and win counts.

```bash
uv run python scripts/aggregate_pairwise_results.py > pairwise_table.csv
```
