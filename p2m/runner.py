"""Minimal sequential runner for the p2m stage pipeline."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from p2m.config import ConfigError, load_config, load_runtime_context
from p2m.core.config_model import RunManifest, SuiteMetadata
from p2m.core.io import write_json
from p2m.stages import STAGES

load_dotenv()


def _load_context(
    *,
    config: str,
) -> dict[str, Any]:
    """Load one config file into runtime context."""
    cfg_path = Path(config).resolve()
    raw = load_config(cfg_path)
    return load_runtime_context(raw, cfg_path, stage_modules=STAGES)


def _write_suite_metadata(ctx: dict[str, Any]) -> None:
    """Write the minimal suite metadata payload."""
    suite_root = Path(ctx["suite_root"])
    suite_path = suite_root / "suite.json"
    existing: dict[str, Any] = {}
    if suite_path.exists():
        try:
            existing = json.loads(suite_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    meta = SuiteMetadata(
        created_at=existing.get("created_at", datetime.now(timezone.utc).isoformat()),
    )
    write_json(suite_path, meta.to_dict())


def _build_manifest(ctx: dict[str, Any]) -> RunManifest:
    """Build the initial run manifest."""
    return RunManifest(
        started_at=datetime.now(timezone.utc).isoformat(),
    )


def _write_manifest(manifest: RunManifest, run_root: Path) -> None:
    """Persist manifest to disk."""
    manifest_path = run_root / "manifest.json"
    write_json(manifest_path, manifest.to_dict())


def run_pipeline(
    *,
    config: str,
    force_stages: list[str] | None = None,
    strict: bool = False,
) -> int:
    """Execute the configured stages sequentially and persist suite/run metadata."""

    try:
        ctx = _load_context(config=config)
        ctx["strict"] = strict
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 1

    requested_force_stages = set(force_stages or [])
    configured_stage_names = {stage_name for stage_name, _ in ctx["stages"]}
    invalid_forced = sorted(requested_force_stages.difference(configured_stage_names))
    if invalid_forced:
        joined = ", ".join(invalid_forced)
        print(f"[config error] --force-stage stage(s) not present in config: {joined}", file=sys.stderr)
        return 1

    stages_to_run: list[tuple[str, Any, dict[str, Any]]] = []
    for stage_name, raw_cfg in ctx["stages"]:
        if not raw_cfg.get("enabled", True):
            continue

        module = STAGES[stage_name]

        if module.SCOPE == "suite" and module.SUITE_OUTPUT and stage_name not in requested_force_stages:
            output_path = Path(ctx["suite_root"]) / module.SUITE_OUTPUT
            if output_path.exists():
                continue

        stages_to_run.append((stage_name, module, raw_cfg))

    suite_root = Path(ctx["suite_root"])
    suite_root.mkdir(parents=True, exist_ok=True)
    _write_suite_metadata(ctx)
    run_root = Path(ctx["run_root"]) if ctx.get("run_root") else None
    selected_run_stage = any(module.SCOPE == "run" for _, module, _ in stages_to_run)
    manifest = None
    if selected_run_stage and run_root is not None:
        run_root.mkdir(parents=True, exist_ok=True)
        manifest = _build_manifest(ctx)
        config_path = ctx.get("config_path")
        if config_path is not None and Path(config_path).is_file():
            shutil.copy2(config_path, run_root / "config.yaml")
    failed_stage: str | None = None
    pipeline_start = time.monotonic()

    for stage_name, module, raw_cfg in stages_to_run:
        if manifest is not None and module.SCOPE == "run":
            manifest.stages[stage_name] = "running"
            _write_manifest(manifest, run_root)
        print(f"  {stage_name} ...", file=sys.stderr, flush=True)
        stage_start = time.monotonic()
        try:
            asyncio.run(module.run(ctx, raw_cfg))
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
            traceback.print_exc(file=sys.stderr)

        elapsed = time.monotonic() - stage_start
        if ok:
            print(f"  {stage_name} done ({elapsed:.1f}s)", file=sys.stderr, flush=True)
        else:
            print(f"  {stage_name} failed ({elapsed:.1f}s)", file=sys.stderr, flush=True)

        if manifest is not None and module.SCOPE == "run":
            manifest.stages[stage_name] = "completed" if ok else "failed"
            manifest.status = "running" if ok else "failed"
            _write_manifest(manifest, run_root)

        if ok and module.SCOPE == "suite":
            _write_suite_metadata(ctx)

        if not ok:
            failed_stage = stage_name
            break

    total_elapsed = time.monotonic() - pipeline_start
    if failed_stage is None:
        print(f"  pipeline completed ({total_elapsed:.1f}s)", file=sys.stderr, flush=True)
    else:
        print(f"  pipeline failed at {failed_stage} ({total_elapsed:.1f}s)", file=sys.stderr, flush=True)

    if manifest is None:
        return 0 if failed_stage is None else 1

    manifest.ended_at = datetime.now(timezone.utc).isoformat()
    manifest.status = "completed" if failed_stage is None else "failed"
    _write_manifest(manifest, run_root)
    return 0 if failed_stage is None else 1
