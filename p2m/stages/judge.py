"""Score unified transcript rollout artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from p2m.config import resolve_stage_paths
from p2m.core.async_utils import gather_limited
from p2m.core.io import METRICS_FILE, SCORES_FILE, TRANSCRIPTS_FILE
from p2m.core.io import get_permissible_flag
from p2m.core.io import load_jsonl, load_prompt_text, resolve_path, write_json, write_jsonl
from p2m.core.judge import (
    build_judge_contract,
    get_verdict_dimension,
    infer_judge_status,
    is_valid_event_flag,
    resolve_dimensions,
    run_transcript_judge as run_llm_judge,
)
from p2m.core.transcript import Transcript, TranscriptEvent, TranscriptMetadata

SCOPE = "run"
SUITE_OUTPUT = None

JUDGE_SYSTEM_PROMPT = load_prompt_text("judge_system.md")


def write_metrics(
    records: list[dict[str, Any]],
    metrics_path: Path,
    *,
    policy_raw: dict[str, Any],
) -> None:
    """Aggregate score rows into the metrics artifact."""
    if not records:
        return
    sub_risks = policy_raw.get("sub_risks")
    if not isinstance(sub_risks, list):
        raise ValueError("policy_raw must contain sub_risks list")

    permissible_records = [row for row in records if get_permissible_flag(row, default=True)]
    not_permissible_records = [row for row in records if not get_permissible_flag(row, default=True)]

    all_dims = sorted(
        {
            key
            for row in records
            if isinstance(row.get("verdict"), dict)
            for key, value in ((row["verdict"].get("dimensions")) or {}).items()
            if is_valid_event_flag(value)
        }
    )

    def compute_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate judge stats for one slice of score rows."""
        scored_items = [item for item in items if infer_judge_status(item) == "ok"]
        total = len(items)
        failures = total - len(scored_items)
        result: dict[str, Any] = {
            "count": total,
            "scored_count": len(scored_items),
            "judge_failures": failures,
            "judge_failure_rate": failures / total if total else 0.0,
            "dimensions": {},
        }
        for dim in all_dims:
            raw_values = []
            for item in scored_items:
                value = get_verdict_dimension(item.get("verdict"), dim)
                if is_valid_event_flag(value):
                    raw_values.append(bool(value))
            flagged_count = sum(1 for value in raw_values if value)
            clear_count = len(raw_values) - flagged_count
            rate = flagged_count / len(raw_values) if raw_values else 0.0
            result["dimensions"][dim] = {
                "count": len(raw_values),
                "flagged_count": flagged_count,
                "clear_count": clear_count,
                "rate": rate,
            }
            result[f"{dim}_rate"] = rate
        return result

    by_sub_risk: dict[str, dict[str, Any]] = {}
    for row in records:
        sub_risk = str(row.get("sub_risk", ""))
        payload = by_sub_risk.setdefault(
            sub_risk,
            {"permissible": get_permissible_flag(row, default=True), "items": []},
        )
        payload["items"].append(row)

    scored_records = [row for row in records if infer_judge_status(row) == "ok"]
    by_relevant_node = []
    for node_index, node_payload in enumerate(sub_risks):
        node_name = str(node_payload.get("name") or "") if isinstance(node_payload, dict) else ""
        relevant_rows: list[dict[str, Any]] = []
        node_violated_count = 0
        for row in scored_records:
            verdict = row.get("verdict")
            if not isinstance(verdict, dict):
                continue
            node_judgments = verdict.get("node_judgments")
            if not isinstance(node_judgments, list) or node_index >= len(node_judgments):
                continue
            node = node_judgments[node_index]
            if not isinstance(node, dict) or node.get("relevant") is not True:
                continue
            relevant_rows.append(row)
            if node.get("violated") is True:
                node_violated_count += 1

        support = len(relevant_rows)
        node_dimensions: dict[str, dict[str, Any]] = {}
        for dim in all_dims:
            flagged_count = 0
            for row in relevant_rows:
                value = get_verdict_dimension(row.get("verdict"), dim)
                if not is_valid_event_flag(value):
                    raise ValueError(f"missing or invalid dimension '{dim}' in scored row")
                flagged_count += int(value)
            clear_count = support - flagged_count
            node_dimensions[dim] = {
                "count": support,
                "flagged_count": flagged_count,
                "clear_count": clear_count,
                "rate": flagged_count / support if support else 0.0,
            }

        by_relevant_node.append(
            {
                "node_index": node_index,
                "node_name": node_name,
                "permissible": (
                    get_permissible_flag(node_payload, default=True)
                    if isinstance(node_payload, dict)
                    else True
                ),
                "support": support,
                "node_violated_count": node_violated_count,
                "node_violated_rate": node_violated_count / support if support else 0.0,
                "dimensions": node_dimensions,
            }
        )

    write_json(
        metrics_path,
        {
            "overall_permissible": compute_stats(permissible_records),
            "overall_not_permissible": compute_stats(not_permissible_records),
            "by_sub_risk": [
                {
                    "sub_risk": sub_risk,
                    "permissible": payload["permissible"],
                    **compute_stats(payload["items"]),
                }
                for sub_risk, payload in sorted(by_sub_risk.items())
            ],
            "by_relevant_node": by_relevant_node,
        },
    )


async def run_judge(
    *,
    transcripts_path: str,
    policy_path: str | None = None,
    save_dir: str | None = None,
    evaluation: Any,
) -> dict[str, Any]:
    """Score transcript rows and write score and metric artifacts."""
    judge_model = str(evaluation.judge.model.name)
    judge_temperature = evaluation.judge.model.temperature
    judge_max_tokens = evaluation.judge.model.max_tokens
    judge_n = evaluation.judge.n
    judge_dimensions = evaluation.judge.dimensions
    resolved_transcripts_path = resolve_path(transcripts_path)
    rows = load_jsonl(resolved_transcripts_path)
    if not rows:
        raise ValueError(f"No transcripts found in {transcripts_path}")

    out_dir = resolve_path(save_dir or str(resolved_transcripts_path.parent))
    out_dir.mkdir(parents=True, exist_ok=True)
    if not policy_path:
        raise ValueError("judge stage requires policy_path")
    resolved_policy_path = resolve_path(policy_path)
    if not resolved_policy_path.exists():
        raise ValueError(f"Policy file not found: {policy_path}")
    policy_raw = json.loads(resolved_policy_path.read_text(encoding="utf-8"))
    judge_contract = build_judge_contract(
        template=JUDGE_SYSTEM_PROMPT,
        policy_raw=policy_raw,
        judge_dimensions=judge_dimensions or [],
        citation_style="xml",
        schema_name="transcript_judgment",
    )

    async def score_row(output_index: int, row: dict[str, Any]) -> dict[str, Any]:
        """Score a single transcript row with the judge model."""
        transcript_metadata = TranscriptMetadata(
            kind=str(row.get("kind") or ""),
            seed_id=str(row.get("seed_id") or ""),
            risk=str(row.get("risk") or ""),
            sub_risk=str(row.get("sub_risk") or ""),
            permissible=get_permissible_flag(row, default=True),
            target=str(row.get("target") or ""),
            auditor_model=str(row.get("auditor_model") or ""),
        )
        transcript = Transcript(
            metadata=transcript_metadata,
            events=[
                TranscriptEvent.model_validate(event)
                for event in row.get("events", [])
                if isinstance(event, dict)
            ],
            stop_reason=row.get("stop_reason"),
        )
        transcript_xml, index_to_message_id = transcript.format_transcript_xml(
            "target",
            skip_system=False,
        )
        judge_result = await run_llm_judge(
            judge_model=judge_model,
            system_prompt=judge_contract["system_prompt"],
            user_message=f"# Transcript\n{transcript_xml}",
            transcript=transcript,
            index_to_message_id=index_to_message_id,
            score_keys=judge_contract["score_keys"],
            policy_raw=policy_raw,
            judge_n=judge_n,
            judge_temperature=judge_temperature,
            judge_max_tokens=judge_max_tokens,
            response_schema=judge_contract["response_schema"],
        )

        score_row = {
            "kind": row.get("kind", ""),
            "seed_id": row.get("seed_id", ""),
            "risk": row.get("risk", ""),
            "sub_risk": row.get("sub_risk", ""),
            "permissible": get_permissible_flag(row, default=True),
            "judge_model": judge_model,
            "target": row.get("target", ""),
            "auditor_model": row.get("auditor_model", ""),
            "judge_status": judge_result["judge_status"],
            "judge_error": judge_result["judge_error"],
            "verdict": judge_result["verdict"],
        }
        if judge_result.get("multi_judge") is not None:
            score_row["multi_judge"] = judge_result["multi_judge"]
        return {
            "output_index": output_index,
            "judge_status": judge_result["judge_status"],
            "score_row": score_row,
            "raw": judge_result["raw"],
        }

    _judge_done = 0
    _judge_total = len(rows)

    async def worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
        """Wrap row scoring so concurrent execution returns structured errors."""
        nonlocal _judge_done
        output_index, row = item
        try:
            result = await score_row(output_index, row)
        except Exception as exc:
            result = {
                "output_index": output_index,
                "error": exc,
            }
        _judge_done += 1
        kind = row.get("kind", "")
        label = row.get("sub_risk") or row.get("risk") or row.get("seed_id", "")
        kind_tag = f"[{kind}] " if kind else ""
        status = "✓" if result.get("error") is None else f"✗ {type(result['error']).__name__}"
        click.echo(f"  judge [{_judge_done}/{_judge_total}] {status} {kind_tag}{label}", err=True)
        return result

    results = sorted(
        await gather_limited(
            list(enumerate(rows)),
            limit=evaluation.rollout.concurrency,
            worker=worker,
        ),
        key=lambda result: result["output_index"],
    )

    # Separate successes from errors. Content filter and LLM errors are
    # recorded as judge failures rather than crashing the entire stage,
    # because the eval tool generates adversarial prompts by design and
    # Azure content filters will routinely trigger on them.
    successful_results = [result for result in results if result.get("error") is None]
    error_results = [result for result in results if result.get("error") is not None]

    for result in error_results:
        error = result["error"]
        idx = result["output_index"]
        error_row = {
            "kind": rows[idx].get("kind", ""),
            "seed_id": rows[idx].get("seed_id", ""),
            "risk": rows[idx].get("risk", ""),
            "sub_risk": rows[idx].get("sub_risk", ""),
            "permissible": get_permissible_flag(rows[idx], default=True),
            "judge_model": judge_model,
            "target": rows[idx].get("target", ""),
            "auditor_model": rows[idx].get("auditor_model", ""),
            "judge_status": "error",
            "judge_error": str(error),
            "verdict": "error",
        }
        successful_results.append({"score_row": error_row, "output_index": idx})
        click.echo(f"  ⚠ Row {idx} judge error (skipped): {type(error).__name__}: {error}", err=True)

    successful_results.sort(key=lambda r: r["output_index"])
    score_rows = [result["score_row"] for result in successful_results]
    scores_path = out_dir / SCORES_FILE
    write_jsonl(scores_path, score_rows)
    metrics_path = out_dir / METRICS_FILE
    write_metrics(score_rows, metrics_path, policy_raw=policy_raw)
    return {
        "scores_path": str(scores_path),
        "metrics_path": str(metrics_path),
        "count": len(score_rows),
        "judge_failures": sum(1 for r in score_rows if r.get("judge_status") != "ok"),
        "judge_errors": len(error_results),
    }


async def run(ctx: dict[str, Any], raw_cfg: dict[str, Any]) -> dict[str, str]:
    """Validate config and run the scoring workflow."""
    evaluation = ctx.get("evaluation")
    if evaluation is None or not evaluation.judge.model:
        raise ValueError("judge stage requires evaluation.judge.model")
    if evaluation.judge.dimensions:
        resolve_dimensions(list(evaluation.judge.dimensions))
    cfg = resolve_stage_paths(
        {
            "transcripts_path": raw_cfg.get("transcripts_path") or str(Path(ctx["run_root"]) / TRANSCRIPTS_FILE),
            "policy_path": raw_cfg.get("policy_path") or str(Path(ctx["suite_root"]) / "policy.json"),
            "save_dir": raw_cfg.get("save_dir") or str(ctx["run_root"]),
        },
        cfg_path=ctx["config_path"],
        artifacts_root=ctx["artifacts_root"],
    )
    result = await run_judge(
        transcripts_path=cfg["transcripts_path"],
        policy_path=cfg.get("policy_path"),
        save_dir=cfg.get("save_dir"),
        evaluation=ctx["evaluation"],
    )
    return {
        "scores_path": result["scores_path"],
        "metrics_path": result["metrics_path"],
        "_summary": {
            "count": result.get("count", 0),
            "failures": result.get("judge_failures", 0),
            "errors": result.get("judge_errors", 0),
            "judge_model": ctx["evaluation"].judge.model if ctx.get("evaluation") else "",
        },
    }
