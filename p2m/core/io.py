"""Path, JSON, and JSONL helpers used across p2m workflows."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve relative paths against CWD, then repo root."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return BASE_DIR / p


def write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    _atomic_write_text(path, text)


def append_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            tmp_name = handle.name
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass


def load_seeds(
    path: str | Path,
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Load seed records from a JSONL file."""
    resolved = resolve_path(path)
    if not resolved.is_file():
        tried = [str(path), str(resolved)]
        raise FileNotFoundError(f"Seed file not found. Tried: {tried}")

    records: list[dict[str, Any]] = []
    bad_lines: list[int] = []
    for lineno, line in _iter_nonempty_lines(resolved):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if strict:
                raise ValueError(
                    f"Malformed JSON at line {lineno} in {resolved}: {line[:120]}"
                )
            bad_lines.append(lineno)
    if bad_lines:
        log.warning(
            "Skipped %d malformed line(s) in %s (lines: %s)",
            len(bad_lines), resolved, bad_lines[:10],
        )
    return records


def normalize_seed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign canonical opaque seed IDs and stable parent-based variation IDs."""
    normalized = [dict(row) for row in rows]
    base_id_by_original_id: dict[str, str] = {}
    base_counter = 1

    for row in normalized:
        if str(row.get("parent_seed_id") or ""):
            continue
        original_seed_id = str(row.get("seed_id") or "")
        seed_id = f"seed_{base_counter:06d}"
        if original_seed_id:
            if original_seed_id in base_id_by_original_id:
                raise ValueError(f"duplicate base seed_id: {original_seed_id}")
            base_id_by_original_id[original_seed_id] = seed_id
        row["seed_id"] = seed_id
        base_counter += 1

    variation_counts: dict[str, int] = {}
    for row in normalized:
        original_parent_seed_id = str(row.get("parent_seed_id") or "")
        if not original_parent_seed_id:
            row.pop("parent_seed_id", None)
            continue
        parent_seed_id = base_id_by_original_id.get(original_parent_seed_id)
        if parent_seed_id is None:
            raise ValueError("seed variation parent_seed_id must reference a base seed")
        variation_counts[parent_seed_id] = variation_counts.get(parent_seed_id, 0) + 1
        row["parent_seed_id"] = parent_seed_id
        row["seed_id"] = f"{parent_seed_id}-v{variation_counts[parent_seed_id]}"

    return normalized


def dump_yaml(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def slugify(text: str) -> str:
    """Collapse free text into a filesystem-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk. Returns empty list if file missing."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for i, line in _iter_nonempty_lines(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("%s:%d: %s", path, i, exc)
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file. Returns None if missing or not a dict."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return data if isinstance(data, dict) else None


# ── Prompt loading ─────────────────────────────────────────────

PROMPTS_DIR = BASE_DIR / "prompts"


def load_prompt_text(filename: str) -> str:
    """Load a prompt file from the prompts/ directory."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def normalize_seed_context(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


# ── Item helpers ───────────────────────────────────────────────

def get_permissible_flag(payload: dict[str, Any], default: bool | None = None) -> bool | None:
    """Read the canonical permissibility flag."""
    if "permissible" in payload and payload.get("permissible") is not None:
        return bool(payload["permissible"])
    return default


# ── Output filenames (written by run stages, read by viewer) ──

TRANSCRIPTS_FILE = "transcripts.jsonl"
SCORES_FILE = "scores.jsonl"
METRICS_FILE = "metrics.json"


def _iter_nonempty_lines(path: Path) -> Iterable[tuple[int, str]]:
    with open(path, encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if line:
                yield lineno, line
