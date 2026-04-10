"""Convert a systematization artifact to a structured policy JSON."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from p2m.config import resolve_stage_paths
from p2m.core.config_model import (
    DEFAULT_SYSTEMATIZATION_CONVERT_MAX_TOKENS,
    DEFAULT_SYSTEMATIZATION_CONVERT_TEMPERATURE,
    DEFAULT_SYSTEMATIZATION_MODEL,
)
from p2m.core.model_client import GenerateOptions, generate_structured
from p2m.stages.policy import policy_schema

BASE_DIR = Path(__file__).resolve().parents[2]
GUIDELINE_PROMPT = (BASE_DIR / "prompts" / "systematization_convert_single.md").read_text()

POLICY_SCHEMA: dict[str, Any] = policy_schema()
SCOPE = "suite"
SUITE_OUTPUT = "policy.json"
DEFAULT_SUB_RISK_COUNT_HINT = 30


def _require_nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"systematization_convert returned invalid {field}")
    return value.strip()


def _require_examples(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"systematization_convert returned invalid {field}")
    examples = []
    for example in value:
        if not isinstance(example, str) or not example.strip():
            raise ValueError(f"systematization_convert returned invalid {field}")
        examples.append(example.strip())
    if not examples:
        raise ValueError(f"systematization_convert returned invalid {field}")
    return examples


def _normalize_definition_of_terms(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("systematization_convert returned invalid definition_of_terms")
    terms = []
    for term_payload in payload:
        if not isinstance(term_payload, dict):
            raise ValueError("systematization_convert returned invalid definition_of_terms")
        terms.append(
            {
                "term": _require_nonempty_string(term_payload.get("term"), field="definition_of_terms.term"),
                "definition": _require_nonempty_string(
                    term_payload.get("definition"),
                    field="definition_of_terms.definition",
                ),
                "examples": _require_examples(
                    term_payload.get("examples"),
                    field="definition_of_terms.examples",
                ),
            }
        )
    return terms


def _normalize_sub_risks(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError("systematization_convert returned invalid sub_risks")
    sub_risks = []
    for sub_risk_payload in payload:
        if not isinstance(sub_risk_payload, dict):
            raise ValueError("systematization_convert returned invalid sub_risks")
        permissible = sub_risk_payload.get("permissible")
        if not isinstance(permissible, bool):
            raise ValueError("systematization_convert returned invalid sub_risks.permissible")
        sub_risks.append(
            {
                "name": _require_nonempty_string(sub_risk_payload.get("name"), field="sub_risks.name"),
                "definition": _require_nonempty_string(
                    sub_risk_payload.get("definition"),
                    field="sub_risks.definition",
                ),
                "examples": _require_examples(
                    sub_risk_payload.get("examples"),
                    field="sub_risks.examples",
                ),
                "permissible": permissible,
            }
        )
    if not sub_risks:
        raise ValueError("systematization_convert returned no sub_risks")
    return sub_risks


async def run_systematization_to_policy(
    *,
    systematization_path: str,
    save_path: str = f"artifacts/taxonomies/{SUITE_OUTPUT}",
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    sub_risk_count_hint: int = DEFAULT_SUB_RISK_COUNT_HINT,
) -> Path:
    """Convert the systematization artifact into the policy JSON artifact."""
    model = model or DEFAULT_SYSTEMATIZATION_MODEL
    max_tokens = max_tokens if max_tokens is not None else DEFAULT_SYSTEMATIZATION_CONVERT_MAX_TOKENS
    temperature = temperature if temperature is not None else DEFAULT_SYSTEMATIZATION_CONVERT_TEMPERATURE
    data_path = Path(systematization_path).expanduser()
    if not data_path.is_absolute():
        data_path = BASE_DIR / data_path
    data = json.loads(data_path.read_text(encoding="utf-8"))
    concept = str(data.get("concept") or "").strip()
    if not concept:
        raise ValueError("systematization_convert requires systematization.json to include concept")
    systematization_text = str(data.get("systematization") or "").strip()
    if not systematization_text:
        raise ValueError("systematization_convert requires a non-empty systematization")
    summary_items = data.get("summary_items")
    if summary_items is not None and not isinstance(summary_items, list):
        raise ValueError("systematization_convert requires summary_items to be a list when present")

    prompt = (
        GUIDELINE_PROMPT.replace("{{sub_risk_count}}", str(sub_risk_count_hint))
        + "\n\n# SYSTEMATIZATION\n"
        + systematization_text
    )
    if summary_items:
        prompt += "\n\n# SUMMARY ITEMS\n" + json.dumps(summary_items, ensure_ascii=False, indent=2)
    response = await generate_structured(
        model,
        prompt,
        schema_name="policy",
        json_schema=POLICY_SCHEMA,
        options=GenerateOptions(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
    )
    policy_payload = response.parsed
    if not isinstance(policy_payload, dict) or not policy_payload:
        raise ValueError("systematization_convert returned no structured policy")

    risk = policy_payload.get("risk")
    if not isinstance(risk, dict):
        raise ValueError("systematization_convert returned invalid risk")
    risk_definition = _require_nonempty_string(risk.get("definition"), field="risk.definition")
    terms = _normalize_definition_of_terms(policy_payload.get("definition_of_terms"))
    sub_risks = _normalize_sub_risks(policy_payload.get("sub_risks"))
    policy = {
        "risk": {
            "name": concept,
            "definition": risk_definition,
        },
        "definition_of_terms": terms,
        "sub_risks": sub_risks,
        "meta": {
            "source": "systematization",
            "systematization_path": systematization_path,
            "slug": concept,
            "run_id": uuid.uuid4().hex[:8],
        },
    }
    output_path = Path(save_path).expanduser()
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


async def run(ctx: dict[str, Any], raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate config and run the systematization conversion workflow."""
    cfg = resolve_stage_paths(
        {
            "systematization_path": raw_cfg.get("systematization_path")
            or str(Path(ctx["suite_root"]) / "systematization.json"),
            "save_path": raw_cfg.get("save_path") or str(Path(ctx["suite_root"]) / SUITE_OUTPUT),
            "model": raw_cfg.get("model"),
            "max_tokens": raw_cfg.get("max_tokens", DEFAULT_SYSTEMATIZATION_CONVERT_MAX_TOKENS),
            "temperature": raw_cfg.get("temperature", DEFAULT_SYSTEMATIZATION_CONVERT_TEMPERATURE),
            "sub_risk_count_hint": raw_cfg.get("sub_risk_count_hint", DEFAULT_SUB_RISK_COUNT_HINT),
        },
        cfg_path=ctx["config_path"],
        artifacts_root=ctx["artifacts_root"],
    )
    generated_path = await run_systematization_to_policy(
        systematization_path=cfg["systematization_path"],
        save_path=cfg["save_path"],
        model=cfg.get("model"),
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        sub_risk_count_hint=int(cfg["sub_risk_count_hint"]),
    )
    return {"policy_generated_path": str(generated_path.resolve())}
