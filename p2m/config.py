"""Load YAML configs and build the minimal runtime context for p2m."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from p2m.core.config_model import (
    DEFAULT_AUDITOR_MAX_TURNS,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_JUDGE_TEMPERATURE,
    DEFAULT_ROLLOUT_CONCURRENCY,
    DEFAULT_ROLLOUT_MAX_TOOL_CALLS,
    DEFAULT_ROLLOUT_MAX_TOKENS,
    DEFAULT_ROLLOUT_TEMPERATURE,
    AuditorConfig,
    EvaluationConfig,
    JudgeConfig,
    ModelConfig,
    PipelineConfig,
    RolloutConfig,
    TargetConfig,
    ToolsConfig,
)

ROOT = Path(__file__).resolve().parent.parent
RISKS_DIR = ROOT / "examples" / "risks"
OUTPUT_PATH_KEYS = {"save_dir", "save_path"}
PIPELINE_STAGE_ORDER = (
    "policy",
    "seeds",
    "rollout",
    "judge",
    "systematization",
    "systematization_convert",
)
STANDARD_PIPELINE_STAGES = {"policy", "seeds", "rollout", "judge"}
SYSTEMATIZATION_PIPELINE_STAGES = {"systematization", "systematization_convert"}


class ConfigError(Exception):
    pass


def require(condition: bool, message: str) -> None:
    """Raise a config error when a validation condition fails."""
    if not condition:
        raise ConfigError(message)


def load_config(cfg_path: Path) -> dict[str, Any]:
    """Load one YAML config file and require a mapping at the top level."""
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "Top-level YAML must be a mapping")
    return data


def _strip_artifact_root_prefix(path: Path, artifacts_root: Path) -> Path | None:
    """Return the artifact-relative suffix for paths starting with artifacts roots."""
    parts = path.parts
    if not parts:
        return None
    if parts[0] not in {"artifacts", artifacts_root.name}:
        return None
    if len(parts) == 1:
        return Path()
    return Path(*parts[1:])


def _resolve_path(
    path: str | Path,
    *,
    artifacts_root: Path,
    cfg_dir: Path | None = None,
    use_artifacts_root: bool = False,
    fallback_root: Path | None = None,
) -> str:
    """Resolve one path against artifacts, config, and optional fallback roots."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    artifact_relative = _strip_artifact_root_prefix(candidate, artifacts_root)
    if artifact_relative is not None:
        return str((artifacts_root / artifact_relative).resolve())
    primary_root = artifacts_root if use_artifacts_root or cfg_dir is None else cfg_dir
    resolved = (primary_root / candidate).resolve()
    if fallback_root is None or resolved.exists():
        return str(resolved)
    fallback = (fallback_root / candidate).resolve()
    return str(fallback if fallback.exists() else resolved)


def _validate_pipeline_stages(
    pipeline_raw: dict[str, Any],
    *,
    stage_modules: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Validate the pipeline mapping and return stages in canonical order."""
    unknown_stages = sorted(set(pipeline_raw).difference(stage_modules))
    require(not unknown_stages, f"Unknown stage(s): {', '.join(unknown_stages)}")

    stages: list[tuple[str, dict[str, Any]]] = []
    for stage_name in PIPELINE_STAGE_ORDER:
        if stage_name not in pipeline_raw:
            continue
        stage_cfg = pipeline_raw[stage_name]
        require(isinstance(stage_cfg, dict), f"pipeline.{stage_name} must be a mapping")
        stages.append((stage_name, stage_cfg))

    require(stages, "'pipeline' must define at least one stage")
    enabled_stage_names = {stage_name for stage_name, stage_cfg in stages if stage_cfg.get("enabled", True)}
    has_standard_stage = any(stage in enabled_stage_names for stage in STANDARD_PIPELINE_STAGES)
    has_systematization_stage = any(stage in enabled_stage_names for stage in SYSTEMATIZATION_PIPELINE_STAGES)
    require(
        not (has_standard_stage and has_systematization_stage),
        "pipeline must use either standard stages or systematization stages, not both",
    )
    return stages


def _load_risk_text(
    raw: dict[str, Any],
    *,
    has_enabled_suite_stage: bool,
    cfg_path: Path,
    artifacts_root: Path,
) -> tuple[str | None, str]:
    """Load the configured risk text when any suite stage requires it."""
    risk_name = raw.get("risk")
    if risk_name is None:
        require(not has_enabled_suite_stage, "risk is required when suite stages are enabled")
        return None, ""

    risks_dir = None
    risks_dir_raw = raw.get("risks_dir")
    if risks_dir_raw:
        risks_dir = Path(
            _resolve_path(
                risks_dir_raw,
                artifacts_root=artifacts_root,
                cfg_dir=cfg_path.parent,
                fallback_root=ROOT,
            )
        )
    search_dir = risks_dir or RISKS_DIR
    candidate = search_dir / f"{risk_name}.md"
    if not candidate.exists():
        available = ", ".join(path.stem for path in sorted(search_dir.glob("*.md")) if path.stem != "README")
        raise ConfigError(f"Unknown risk '{risk_name}'. Available: {available}")
    risk_text = candidate.read_text(encoding="utf-8").strip()
    require(bool(risk_text), f"Risk file '{risk_name}' is empty")
    return str(risk_name), risk_text


def load_runtime_context(
    raw: dict[str, Any],
    cfg_path: Path,
    *,
    stage_modules: dict[str, Any],
) -> dict[str, Any]:
    """Build the shared runtime context used by every stage."""
    pipeline = parse_pipeline_config(raw, cfg_path)
    target = pipeline.target if pipeline else None

    artifacts_root = Path(raw.get("artifacts_root") or "artifacts").expanduser()
    if not artifacts_root.is_absolute():
        artifacts_root = (ROOT / artifacts_root).resolve()
    else:
        artifacts_root = artifacts_root.resolve()

    results_dir_raw = raw.get("results_dir")
    if results_dir_raw:
        results_dir = Path(
            _resolve_path(
                results_dir_raw,
                artifacts_root=artifacts_root,
                use_artifacts_root=True,
            )
        )
    else:
        results_dir = (artifacts_root / "results").resolve()

    suite_id = str(raw.get("suite") or raw.get("suite_id") or datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%S"))
    pipeline_raw = raw.get("pipeline")
    require(isinstance(pipeline_raw, dict), "'pipeline' must be a mapping")
    stages = _validate_pipeline_stages(pipeline_raw, stage_modules=stage_modules)
    enabled_stage_names = [
        stage_name
        for stage_name, stage_cfg in stages
        if stage_cfg.get("enabled", True)
    ]

    has_enabled_run_stage = any(stage_modules[name].SCOPE == "run" for name in enabled_stage_names)
    has_enabled_suite_stage = any(stage_modules[name].SCOPE == "suite" for name in enabled_stage_names)

    run_id = raw.get("run") or raw.get("run_id")
    if has_enabled_run_stage and not run_id:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_id = str(run_id) if run_id else None

    risk_name, risk_text = _load_risk_text(
        raw,
        has_enabled_suite_stage=has_enabled_suite_stage,
        cfg_path=cfg_path,
        artifacts_root=artifacts_root,
    )

    suite_root = (results_dir / suite_id).resolve()
    run_root = (suite_root / run_id).resolve() if run_id else None

    return {
        "config_path": cfg_path,
        "suite_id": suite_id,
        "run_id": run_id,
        "risk_name": risk_name,
        "risk": risk_text,
        "artifacts_root": artifacts_root,
        "results_dir": results_dir,
        "suite_root": suite_root,
        "run_root": run_root,
        "stages": stages,
        "target": target,
        "evaluation": pipeline.evaluation if pipeline else None,
    }


def resolve_stage_paths(
    cfg: dict[str, Any],
    *,
    cfg_path: Path,
    artifacts_root: Path,
) -> dict[str, Any]:
    """Resolve all *_path and *_dir values in one stage config mapping."""
    resolved = dict(cfg)
    for key, value in list(resolved.items()):
        if not value or not key.endswith(("_path", "_dir")):
            continue
        resolved[key] = _resolve_path(
            value,
            artifacts_root=artifacts_root,
            cfg_dir=cfg_path.parent,
            use_artifacts_root=key in OUTPUT_PATH_KEYS,
        )
    return resolved


# ── Config parsing ─────────────────────────────────────────────


def _optional_str(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    stripped = value.strip()
    return stripped or None


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
    raise ValueError(f"{field_name} must be numeric")


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
    raise ValueError(f"{field_name} must be an integer")


def _reject_unknown_keys(raw: dict[str, Any], *, field_name: str, allowed: set[str]) -> None:
    unknown = sorted(set(raw).difference(allowed))
    if unknown:
        raise ValueError(f"{field_name} has unsupported field(s): {', '.join(unknown)}")


def parse_model_config(
    raw: Any,
    *,
    field_name: str,
    default_temperature: float | None = None,
    default_max_tokens: int | None = None,
) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a mapping")
    _reject_unknown_keys(
        raw,
        field_name=field_name,
        allowed={"name", "temperature", "max_tokens"},
    )
    name = _optional_str(raw.get("name"), field_name=f"{field_name}.name")
    if not name:
        raise ValueError(f"{field_name}.name is required")
    temperature = _optional_float(
        raw.get("temperature"),
        field_name=f"{field_name}.temperature",
    )
    max_tokens = _optional_int(
        raw.get("max_tokens"),
        field_name=f"{field_name}.max_tokens",
    )
    return ModelConfig(
        name=name,
        temperature=temperature if temperature is not None else default_temperature,
        max_tokens=max_tokens if max_tokens is not None else default_max_tokens,
    )


def parse_tools_config(raw: dict[str, Any], *, field_name: str) -> ToolsConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a mapping")
    _reject_unknown_keys(raw, field_name=field_name, allowed={"module", "toolset", "simulator"})
    return ToolsConfig(
        module=_optional_str(raw.get("module"), field_name=f"{field_name}.module"),
        toolset=_optional_str(raw.get("toolset"), field_name=f"{field_name}.toolset"),
        simulator=_optional_str(raw.get("simulator"), field_name=f"{field_name}.simulator"),
    )


def parse_target_config(raw: dict[str, Any], *, field_name: str) -> TargetConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"{field_name} must be a mapping")
    _reject_unknown_keys(raw, field_name=field_name, allowed={"model", "system_prompt", "tools", "connector"})
    tools_raw = raw.get("tools")
    tools = None
    if tools_raw is not None:
        tools = parse_tools_config(tools_raw, field_name=f"{field_name}.tools")
    return TargetConfig(
        model=(
            parse_model_config(
                raw.get("model"),
                field_name=f"{field_name}.model",
                default_temperature=DEFAULT_ROLLOUT_TEMPERATURE,
                default_max_tokens=DEFAULT_ROLLOUT_MAX_TOKENS,
            )
            if raw.get("model") is not None
            else None
        ),
        system_prompt=_optional_str(raw.get("system_prompt"), field_name=f"{field_name}.system_prompt"),
        tools=tools,
        connector=_optional_str(raw.get("connector"), field_name=f"{field_name}.connector"),
    )


def parse_pipeline_config(raw: dict[str, Any], cfg_path: Path) -> PipelineConfig | None:
    pipeline_raw = raw.get("pipeline")
    if pipeline_raw is None:
        return None
    if not isinstance(pipeline_raw, dict):
        raise ValueError("pipeline must be a mapping")

    rollout_stage = pipeline_raw.get("rollout")
    scorer_stage = pipeline_raw.get("judge")

    if rollout_stage is not None and not isinstance(rollout_stage, dict):
        raise ValueError("pipeline.rollout must be a mapping")
    if scorer_stage is not None and not isinstance(scorer_stage, dict):
        raise ValueError("pipeline.judge must be a mapping")

    target = None
    rollout_cfg = RolloutConfig()
    auditor = None
    judge = None

    if rollout_stage is not None:
        if "environment" in rollout_stage:
            raise ValueError("pipeline.rollout.environment is no longer supported; use pipeline.rollout.target.tools")
        if "temperature" in rollout_stage:
            raise ValueError("pipeline.rollout.temperature is no longer supported; move it under target.model or auditor.model")
        if "max_tokens" in rollout_stage:
            raise ValueError("pipeline.rollout.max_tokens is no longer supported; move it under target.model or auditor.model")
        target_raw = rollout_stage.get("target")
        require(target_raw is not None, "pipeline.rollout.target is required when rollout stage is enabled")
        if not isinstance(target_raw, dict):
            raise ValueError("pipeline.rollout.target must be a mapping")
        target = parse_target_config(target_raw, field_name="pipeline.rollout.target")

        auditor_raw = rollout_stage.get("auditor")
        if auditor_raw is not None:
            if not isinstance(auditor_raw, dict):
                raise ValueError("pipeline.rollout.auditor must be a mapping")
            if "max_turns" in auditor_raw:
                raise ValueError("pipeline.rollout.auditor.max_turns is no longer supported; use pipeline.rollout.max_turns")
            auditor = AuditorConfig(
                model=parse_model_config(
                    auditor_raw.get("model"),
                    field_name="pipeline.rollout.auditor.model",
                    default_temperature=DEFAULT_ROLLOUT_TEMPERATURE,
                    default_max_tokens=DEFAULT_ROLLOUT_MAX_TOKENS,
                ),
            )

        rollout_cfg = RolloutConfig(
            max_tool_calls=_optional_int(
                rollout_stage.get("max_tool_calls"),
                field_name="pipeline.rollout.max_tool_calls",
            )
            or DEFAULT_ROLLOUT_MAX_TOOL_CALLS,
            max_turns=_optional_int(
                rollout_stage.get("max_turns"),
                field_name="pipeline.rollout.max_turns",
            )
            or DEFAULT_AUDITOR_MAX_TURNS,
            tool_timeout_s=_optional_float(
                rollout_stage.get("tool_timeout_s"),
                field_name="pipeline.rollout.tool_timeout_s",
            ),
            startup_timeout_s=_optional_float(
                rollout_stage.get("startup_timeout_s"),
                field_name="pipeline.rollout.startup_timeout_s",
            ),
            concurrency=_optional_int(
                rollout_stage.get("concurrency"),
                field_name="pipeline.rollout.concurrency",
            )
            or DEFAULT_ROLLOUT_CONCURRENCY,
        )

    if scorer_stage is not None:
        judge_raw = scorer_stage.get("judge")
        require(judge_raw is not None, "pipeline.judge.judge is required when judge is configured")
        if not isinstance(judge_raw, dict):
            raise ValueError("pipeline.judge.judge must be a mapping")
        judge = JudgeConfig(
            model=parse_model_config(
                judge_raw.get("model"),
                field_name="pipeline.judge.judge.model",
                default_temperature=DEFAULT_JUDGE_TEMPERATURE,
                default_max_tokens=DEFAULT_JUDGE_MAX_TOKENS,
            ),
            n=_optional_int(judge_raw.get("n"), field_name="pipeline.judge.judge.n") or 1,
            dimensions=list(judge_raw.get("dimensions") or []),
        )

    evaluation = None
    if judge is not None or rollout_stage is not None:
        evaluation = EvaluationConfig(judge=judge, auditor=auditor, rollout=rollout_cfg)

    return PipelineConfig(target=target, evaluation=evaluation)
