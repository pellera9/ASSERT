"""Run the one-step systematization workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from p2m.config import resolve_stage_paths
from p2m.core.config_model import (
    DEFAULT_SYSTEMATIZATION_MAX_TOKENS,
    DEFAULT_SYSTEMATIZATION_MODEL,
    DEFAULT_SYSTEMATIZATION_TEMPERATURE,
)
from p2m.core.io import load_prompt_text
from p2m.core.model_client import GenerateOptions, generate_structured

SCOPE = "suite"
SUITE_OUTPUT = "systematization.json"

SYSTEMATIZATION_PROMPT = load_prompt_text("systematization_single.md")
ALLOWED_CONFIG_KEYS = {
    "enabled",
    "concept",
    "mode",
    "model",
    "temperature",
    "max_tokens",
    "reasoning_effort",
    "save_path",
}
ALLOWED_MODES = {"research", "direct"}


class SummaryItem(BaseModel):
    description: str
    example: str

    model_config = ConfigDict(extra="forbid")


class SystematizationResponse(BaseModel):
    systematization: str
    summary_items: list[SummaryItem]

    model_config = ConfigDict(extra="forbid")


def _humanize_risk_name(risk_name: str | None) -> str:
    return str(risk_name or "").replace("_", " ").replace("-", " ").strip()


def _reject_unknown_keys(raw_cfg: dict[str, Any]) -> None:
    unknown = sorted(set(raw_cfg).difference(ALLOWED_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"systematization has unsupported field(s): {', '.join(unknown)}")


def _resolve_concept(raw_cfg: dict[str, Any], ctx: dict[str, Any]) -> str:
    concept = str(raw_cfg.get("concept") or _humanize_risk_name(ctx.get("risk_name"))).strip()
    if not concept:
        raise ValueError("systematization requires concept or risk_name")
    return concept


def _build_prompt(*, concept: str, risk_text: str) -> str:
    return (
        f"{SYSTEMATIZATION_PROMPT}\n\n"
        f"# Concept Label\n{concept}\n\n"
        f"# Source Risk Text\n{risk_text.strip()}\n"
    )


def _extract_pattern_blocks(systematization: str) -> list[str]:
    """Split the systematization into individual pattern blocks.

    Each block starts with ``- **Pattern**:`` and extends until the next
    pattern bullet or the end of the patterns section.
    """
    parts = re.split(r"(?m)^- \*\*Pattern\*\*:", systematization)
    return [part.strip() for part in parts[1:] if part.strip()]


def _validate_pattern_block(block: str) -> None:
    """Validate a single slot-based pattern block."""
    if "**Key Terms**:" not in block:
        raise ValueError("systematization pattern block is missing Key Terms section")
    if "**Variables**:" not in block:
        raise ValueError("systematization pattern block is missing Variables section")
    slot_refs = re.findall(r"\[([A-Z][A-Z0-9_]*)\]", block.split("**Variables**:")[0])
    if not slot_refs:
        raise ValueError("systematization pattern template has no [SLOT] placeholders")
    variable_names = re.findall(r"\*\*\[([A-Z][A-Z0-9_]*)\]\*\*:\s*\{\{", block)
    if not variable_names:
        raise ValueError("systematization pattern has no {{ }} variable blocks")
    for slot in slot_refs:
        if slot not in variable_names:
            raise ValueError(f"systematization pattern has [SLOT] '{slot}' with no matching variable block")


def _validate_systematization(systematization: str) -> None:
    text = systematization.strip()
    if not text:
        raise ValueError("systematization returned empty systematization")
    required_headers = ["# Systematization", "## Scope", "## Coverage notes"]
    for header in required_headers:
        if header not in text:
            raise ValueError(f"systematization is missing required section: {header}")

    required_sections = [
        ("## Master inclusion / exclusion test", "Master inclusion / exclusion test"),
        ("## Severity calibration", "Severity calibration guide"),
        ("## Boundary examples", "Boundary examples"),
        ("## Worked scoring examples", "Worked scoring examples"),
        ("## Stakeholder guidance", "Stakeholder guidance"),
        ("## Fairness safeguard", "Fairness safeguards"),
        ("## Downstream harms", "Downstream harms"),
    ]
    for marker, name in required_sections:
        if marker not in text:
            raise ValueError(f"systematization is missing required section: {name}")

    blocks = _extract_pattern_blocks(text)
    if not blocks:
        raise ValueError("systematization must include at least one pattern block")

    if len(blocks) > 1 and "## Decision tree" not in text:
        raise ValueError("systematization with multiple patterns must include a Decision tree section")

    for block in blocks:
        _validate_pattern_block(block)


def _validate_summary_items(summary_items: list[SummaryItem]) -> None:
    if not summary_items:
        raise ValueError("systematization requires at least one summary item")
    for item in summary_items:
        if not item.description.strip():
            raise ValueError("systematization summary_items.description must be non-empty")
        if not item.example.strip():
            raise ValueError("systematization summary_items.example must be non-empty")


async def run_systematization(
    *,
    concept: str,
    risk_text: str,
    save_path: str,
    model: str | None = None,
    mode: str = "research",
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> Path:
    """Generate one systematization artifact and persist it to disk."""
    if mode not in ALLOWED_MODES:
        raise ValueError(f"systematization.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}")
    if not risk_text.strip():
        raise ValueError("systematization requires non-empty risk text")

    resolved_model = model or DEFAULT_SYSTEMATIZATION_MODEL
    resolved_temperature = DEFAULT_SYSTEMATIZATION_TEMPERATURE if temperature is None else temperature
    resolved_max_tokens = DEFAULT_SYSTEMATIZATION_MAX_TOKENS if max_tokens is None else max_tokens
    if mode == "direct" and reasoning_effort is not None:
        raise ValueError("systematization.reasoning_effort is only allowed when mode=research")
    # Reasoning models don't support temperature
    if reasoning_effort is not None:
        resolved_temperature = None

    response = await generate_structured(
        resolved_model,
        _build_prompt(concept=concept, risk_text=risk_text),
        schema_name="systematization",
        json_schema=SystematizationResponse.model_json_schema(),
        options=GenerateOptions(
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            web_search=(mode == "research"),
            reasoning_effort=reasoning_effort if mode == "research" else None,
        ),
    )
    if getattr(response, "finish_reason", None) == "length":
        raise ValueError(
            "systematization response was truncated (finish_reason=length). "
            "Increase max_tokens (current: {}) or reduce prompt complexity.".format(
                resolved_max_tokens
            )
        )
    payload = response.parsed
    if not isinstance(payload, dict) or not payload:
        if not response.text:
            raise ValueError("systematization returned no structured systematization")
        payload = json.loads(response.text)

    parsed = SystematizationResponse.model_validate(payload)
    _validate_systematization(parsed.systematization)
    _validate_summary_items(parsed.summary_items)

    artifact = {
        "concept": concept,
        "systematization": parsed.systematization,
        "summary_items": [item.model_dump() for item in parsed.summary_items],
        "meta": {
            "mode": mode,
            "model": resolved_model,
            "reasoning_effort": reasoning_effort,
        },
    }
    output_path = Path(save_path).expanduser().with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


async def run(ctx: dict[str, Any], raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate config and run one-step systematization."""
    _reject_unknown_keys(raw_cfg)

    mode = str(raw_cfg.get("mode") or "research").strip()
    if mode not in ALLOWED_MODES:
        raise ValueError(f"systematization.mode must be one of: {', '.join(sorted(ALLOWED_MODES))}")
    reasoning_effort = raw_cfg.get("reasoning_effort")
    if mode == "direct" and reasoning_effort is not None:
        raise ValueError("systematization.reasoning_effort is only allowed when mode=research")

    risk_text = str(ctx.get("risk") or "").strip()
    if not risk_text:
        raise ValueError("systematization requires risk")

    cfg = resolve_stage_paths(
        {
            "concept": _resolve_concept(raw_cfg, ctx),
            "save_path": raw_cfg.get("save_path") or str(Path(ctx["suite_root"]) / SUITE_OUTPUT),
            "model": raw_cfg.get("model") or DEFAULT_SYSTEMATIZATION_MODEL,
            "mode": mode,
            "temperature": raw_cfg.get("temperature", DEFAULT_SYSTEMATIZATION_TEMPERATURE),
            "max_tokens": raw_cfg.get("max_tokens", DEFAULT_SYSTEMATIZATION_MAX_TOKENS),
            "reasoning_effort": reasoning_effort,
        },
        cfg_path=ctx["config_path"],
        artifacts_root=ctx["artifacts_root"],
    )
    output_path = await run_systematization(
        concept=cfg["concept"],
        risk_text=risk_text,
        save_path=cfg["save_path"],
        model=cfg["model"],
        mode=cfg["mode"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        reasoning_effort=cfg.get("reasoning_effort"),
    )
    return {"systematization_path": str(output_path.resolve())}
