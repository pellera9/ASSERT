"""OTel trace parser — converts OTLP JSON to p2m transcript events.

Supports traces exported from Phoenix, Jaeger, or any OpenTelemetry collector.
Follows OpenInference semantic conventions for LLM/agent span attributes.

Usage:
    from p2m.core.otel import parse_otel_traces

    transcripts = parse_otel_traces(
        "traces.json",
        group_by="session.id",
    )
    # Returns list[dict] in p2m transcript row format (same as transcripts.jsonl)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# OpenInference semantic conventions
# https://arize-ai.github.io/openinference/

_SPAN_KIND_KEY = "openinference.span.kind"
_INPUT_VALUE_KEY = "input.value"
_OUTPUT_VALUE_KEY = "output.value"
_LLM_MODEL_KEY = "llm.model_name"
_LLM_INPUT_TOKENS_KEY = "llm.token_count.prompt"
_LLM_OUTPUT_TOKENS_KEY = "llm.token_count.completion"
_TOOL_NAME_KEY = "tool.name"
_LANGGRAPH_NODE_KEY = "langgraph.node"


@dataclass
class OTelSpan:
    """Minimal representation of an OpenTelemetry span."""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time_ns: int
    end_time_ns: int
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "OK"

    @property
    def latency_ms(self) -> float:
        return (self.end_time_ns - self.start_time_ns) / 1_000_000


def parse_otel_traces(
    path: str | Path,
    *,
    group_by: str = "session.id",
) -> list[dict[str, Any]]:
    """Parse OTLP JSON export into p2m transcript rows.

    Args:
        path: Path to OTLP JSON file.
        group_by: Span attribute key to group spans into conversations.

    Returns:
        List of transcript row dicts, one per conversation. Each row has the
        same schema as a row in transcripts.jsonl:
        {
            "metadata": {...},
            "events": [...],
            "raw": {...},
        }
    """
    spans = _parse_otlp_json(Path(path))
    grouped = _group_spans(spans, group_by)

    rows = []
    for session_id, session_spans in grouped.items():
        session_spans.sort(key=lambda s: s.start_time_ns)
        events, aggregate = _spans_to_events(session_spans)

        rows.append({
            "metadata": {
                "kind": "otel_import",
                "session_id": session_id,
                "runtime_mode": "otel_traced",
            },
            "events": events,
            "raw": aggregate,
        })

    return rows


def _parse_otlp_json(path: Path) -> list[OTelSpan]:
    """Parse OTLP JSON export format."""
    data = json.loads(path.read_text(encoding="utf-8"))

    spans: list[OTelSpan] = []
    for resource_span in data.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for raw in scope_span.get("spans", []):
                attrs = _flatten_attributes(raw.get("attributes", []))
                spans.append(OTelSpan(
                    trace_id=raw.get("traceId", ""),
                    span_id=raw.get("spanId", ""),
                    parent_span_id=raw.get("parentSpanId"),
                    name=raw.get("name", ""),
                    kind=attrs.get(_SPAN_KIND_KEY, "UNKNOWN"),
                    start_time_ns=int(raw.get("startTimeUnixNano", 0)),
                    end_time_ns=int(raw.get("endTimeUnixNano", 0)),
                    attributes=attrs,
                    status=raw.get("status", {}).get("code", "OK"),
                ))
    return spans


def _flatten_attributes(attrs: list[dict]) -> dict[str, Any]:
    """Convert OTLP attribute array [{key, value}] to flat dict."""
    result: dict[str, Any] = {}
    for attr in attrs:
        key = attr.get("key", "")
        value = attr.get("value", {})
        if "stringValue" in value:
            result[key] = value["stringValue"]
        elif "intValue" in value:
            result[key] = int(value["intValue"])
        elif "doubleValue" in value:
            result[key] = float(value["doubleValue"])
        elif "boolValue" in value:
            result[key] = value["boolValue"]
        elif "arrayValue" in value:
            result[key] = [
                _extract_value(v) for v in value["arrayValue"].get("values", [])
            ]
    return result


def _extract_value(value_obj: dict) -> Any:
    """Extract a scalar value from an OTLP Value object."""
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value_obj:
            return value_obj[key]
    return None


def _group_spans(
    spans: list[OTelSpan],
    group_key: str,
) -> dict[str, list[OTelSpan]]:
    """Group spans by a session/conversation attribute."""
    groups: dict[str, list[OTelSpan]] = {}
    for span in spans:
        session_id = str(span.attributes.get(group_key, span.trace_id))
        groups.setdefault(session_id, []).append(span)
    return groups


def _spans_to_events(
    spans: list[OTelSpan],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert a group of spans into p2m transcript events + aggregate metadata.

    Returns:
        (events, aggregate) where events is a list of transcript event dicts
        and aggregate is summary metadata for the conversation.
    """
    events: list[dict[str, Any]] = []
    nodes_visited: list[str] = []
    tools_called: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency_ms = 0.0
    llm_call_count = 0

    for span in spans:
        if span.kind == "LLM":
            output_text = span.attributes.get(_OUTPUT_VALUE_KEY, "")
            model = span.attributes.get(_LLM_MODEL_KEY, "")
            input_tokens = span.attributes.get(_LLM_INPUT_TOKENS_KEY, 0)
            output_tokens = span.attributes.get(_LLM_OUTPUT_TOKENS_KEY, 0)
            node_name = span.attributes.get(_LANGGRAPH_NODE_KEY, span.name)

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_latency_ms += span.latency_ms
            llm_call_count += 1

            if node_name and node_name not in nodes_visited:
                nodes_visited.append(node_name)

            if output_text:
                events.append({
                    "view": ["target", "combined"],
                    "actor": "target",
                    "edit": {
                        "type": "add_message",
                        "message": {"role": "assistant", "content": output_text},
                    },
                    "raw": {
                        "_node": node_name,
                        "_model": model,
                        "_tokens": {"input": input_tokens, "output": output_tokens},
                        "_latency_ms": span.latency_ms,
                    },
                })

        elif span.kind == "TOOL":
            tool_name = span.attributes.get(_TOOL_NAME_KEY, span.name)
            tool_input = span.attributes.get(_INPUT_VALUE_KEY, "")
            tool_output = span.attributes.get(_OUTPUT_VALUE_KEY, "")

            if tool_name and tool_name not in tools_called:
                tools_called.append(tool_name)

            try:
                tool_args = json.loads(tool_input) if tool_input else {}
            except (json.JSONDecodeError, TypeError):
                tool_args = {"raw": tool_input}

            events.append({
                "view": ["target", "combined"],
                "actor": "tool",
                "edit": {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_result": tool_output or "",
                },
            })

        elif span.kind == "CHAIN":
            node_name = span.attributes.get(_LANGGRAPH_NODE_KEY, span.name)
            if node_name and node_name not in nodes_visited:
                nodes_visited.append(node_name)

    aggregate = {
        "nodes_visited": nodes_visited,
        "tools_called": tools_called,
        "total_tokens": {
            "input": total_input_tokens,
            "output": total_output_tokens,
        },
        "total_latency_ms": total_latency_ms,
        "llm_call_count": llm_call_count,
    }

    return events, aggregate


# ── Span validation ───────────────────────────────────────────────


@dataclass
class SpanValidationResult:
    """Result of validating spans for eval readiness."""

    valid: bool
    missing_attributes: list[str]
    warnings: list[str]


def validate_spans(spans: list[OTelSpan]) -> SpanValidationResult:
    """Check spans have required OpenInference attributes for quality eval.

    Required: ``openinference.span.kind``, ``output.value`` (for LLM spans).
    Recommended: ``llm.model_name``, ``llm.token_count.*``, ``tool.name``
    (for TOOL spans).
    """
    missing: list[str] = []
    warnings: list[str] = []

    for span in spans:
        if span.kind == "UNKNOWN":
            missing.append(f"span {span.span_id}: {_SPAN_KIND_KEY}")

        if span.kind == "LLM":
            if not span.attributes.get(_OUTPUT_VALUE_KEY):
                missing.append(f"span {span.span_id}: {_OUTPUT_VALUE_KEY}")
            if not span.attributes.get(_LLM_MODEL_KEY):
                warnings.append(f"span {span.span_id}: missing recommended {_LLM_MODEL_KEY}")
            if (
                not span.attributes.get(_LLM_INPUT_TOKENS_KEY)
                and not span.attributes.get(_LLM_OUTPUT_TOKENS_KEY)
            ):
                warnings.append(f"span {span.span_id}: missing recommended token counts")

        if span.kind == "TOOL":
            if not span.attributes.get(_TOOL_NAME_KEY):
                warnings.append(f"span {span.span_id}: missing recommended {_TOOL_NAME_KEY}")

    return SpanValidationResult(
        valid=len(missing) == 0,
        missing_attributes=missing,
        warnings=warnings,
    )


# ── Trace compression ────────────────────────────────────────────


def compress_trace_for_judge(
    events: list[dict[str, Any]],
    *,
    max_events: int = 50,
    include_tool_args: bool = True,
    include_token_counts: bool = True,
) -> list[dict[str, Any]]:
    """Compress a trace to fit within judge token budget.

    Strategy: Keep all tool call events (they're evidence), keep first and
    last LLM events per node, drop intermediate LLM events if over budget.
    """
    if len(events) <= max_events:
        result = events
    else:
        # Partition: tool events are always kept; LLM events may be trimmed
        tool_events = [e for e in events if e.get("actor") == "tool"]
        llm_events = [e for e in events if e.get("actor") != "tool"]

        remaining = max_events - len(tool_events)
        if remaining < 0:
            remaining = 0

        if remaining >= len(llm_events):
            trimmed_llm = llm_events
        else:
            # Keep first and last LLM event per node, drop middle ones
            by_node: dict[str, list[dict[str, Any]]] = {}
            for ev in llm_events:
                node = (ev.get("raw") or {}).get("_node", "_default")
                by_node.setdefault(node, []).append(ev)

            trimmed_llm = []
            for node_events in by_node.values():
                if len(node_events) <= 2:
                    trimmed_llm.extend(node_events)
                else:
                    trimmed_llm.append(node_events[0])
                    trimmed_llm.append(node_events[-1])

            # If still over budget, hard-truncate
            if len(trimmed_llm) > remaining:
                trimmed_llm = trimmed_llm[:remaining]

        result = tool_events + trimmed_llm

    # Optionally strip tool args or token counts to save tokens
    if not include_tool_args or not include_token_counts:
        compressed = []
        for event in result:
            event = dict(event)
            if not include_tool_args and event.get("actor") == "tool":
                edit = dict(event.get("edit", {}))
                edit.pop("tool_args", None)
                event["edit"] = edit
            if not include_token_counts and "raw" in event:
                raw = dict(event["raw"])
                raw.pop("_tokens", None)
                event["raw"] = raw
            compressed.append(event)
        return compressed

    return result


# ── Trace exporters ──────────────────────────────────────────────


@runtime_checkable
class TraceExporter(Protocol):
    """Interface for exporting OTel traces. Decouples from Phoenix."""

    def export_session(self, session_id: str) -> list[OTelSpan]: ...


class FileTraceExporter:
    """Reads OTLP JSON from file. No external dependencies."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def export_session(self, session_id: str) -> list[OTelSpan]:
        """Return spans whose ``session.id`` attribute matches *session_id*.

        Falls back to matching on ``trace_id`` when no ``session.id``
        attribute is present.
        """
        all_spans = _parse_otlp_json(self._path)
        return [
            s for s in all_spans
            if s.attributes.get("session.id", s.trace_id) == session_id
        ]


class InMemoryTraceExporter:
    """Collects spans in-memory during a rollout. For testing and CI."""

    def __init__(self) -> None:
        self._spans: list[OTelSpan] = []

    def add_span(self, span: OTelSpan) -> None:
        self._spans.append(span)

    def export_session(self, session_id: str) -> list[OTelSpan]:
        """Return spans whose ``session.id`` attribute or trace_id matches."""
        return [
            s for s in self._spans
            if s.attributes.get("session.id", s.trace_id) == session_id
        ]


# ── Extraction APIs (3 granularities) ─────────────────────────────
# These productize Arize's notebook utility functions as typed, tested APIs.


def extract_span_inputs(
    spans: list[OTelSpan],
    *,
    span_kind: str = "LLM",
) -> list[dict[str, Any]]:
    """Extract eval inputs from individual spans.

    Returns one dict per matching span with:
    - query: the input to this span
    - response: the output from this span
    - model: the LLM model used (if available)
    - tokens: input/output token counts (if available)
    """
    results = []
    for span in spans:
        if span.kind != span_kind:
            continue
        results.append({
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "query": span.attributes.get(_INPUT_VALUE_KEY, ""),
            "response": span.attributes.get(_OUTPUT_VALUE_KEY, ""),
            "model": span.attributes.get(_LLM_MODEL_KEY, ""),
            "input_tokens": span.attributes.get(_LLM_INPUT_TOKENS_KEY, 0),
            "output_tokens": span.attributes.get(_LLM_OUTPUT_TOKENS_KEY, 0),
            "latency_ms": span.latency_ms,
            "node": span.attributes.get(_LANGGRAPH_NODE_KEY, span.name),
        })
    return results


def extract_trajectory_inputs(
    spans: list[OTelSpan],
    *,
    group_by: str = "trace_id",
) -> list[dict[str, Any]]:
    """Extract trajectory eval inputs — one row per trace.

    Groups spans by trace, extracts tool calls in order, collapses
    to the format needed for trajectory evaluation prompts.

    Returns one dict per trace with:
    - trace_id: the trace identifier
    - user_input: the first user input in the trace
    - tool_calls: ordered list of {name, arguments} dicts
    - tool_defs: tool definitions (if available)
    - node_path: ordered list of node names visited
    - total_tokens: aggregate token usage
    """
    if group_by == "trace_id":
        grouped = _group_spans_by_trace(spans)
    else:
        grouped = _group_spans(spans, group_by)

    results = []
    for group_id, group_spans in grouped.items():
        group_spans.sort(key=lambda s: s.start_time_ns)

        user_input = ""
        tool_calls: list[dict[str, Any]] = []
        tool_defs: list[Any] = []
        node_path: list[str] = []
        total_input_tokens = 0
        total_output_tokens = 0

        for span in group_spans:
            if span.kind == "LLM":
                if not user_input:
                    user_input = span.attributes.get(_INPUT_VALUE_KEY, "")
                node = span.attributes.get(_LANGGRAPH_NODE_KEY, span.name)
                if node and node not in node_path:
                    node_path.append(node)
                total_input_tokens += span.attributes.get(_LLM_INPUT_TOKENS_KEY, 0)
                total_output_tokens += span.attributes.get(_LLM_OUTPUT_TOKENS_KEY, 0)
            elif span.kind == "TOOL":
                tool_name = span.attributes.get(_TOOL_NAME_KEY, span.name)
                tool_input = span.attributes.get(_INPUT_VALUE_KEY, "")
                try:
                    args = json.loads(tool_input) if tool_input else {}
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tool_input}
                tool_calls.append({"name": tool_name, "arguments": args})

        results.append({
            "trace_id": group_id,
            "user_input": user_input,
            "tool_calls": json.dumps(tool_calls),
            "tool_defs": json.dumps(tool_defs),
            "node_path": json.dumps(node_path),
            "total_tokens": {"input": total_input_tokens, "output": total_output_tokens},
        })

    return results


def extract_session_inputs(
    spans: list[OTelSpan],
    *,
    session_id_key: str = "session.id",
) -> list[dict[str, Any]]:
    """Extract session eval inputs — one row per session.

    Groups spans by session ID, orders traces chronologically,
    extracts user inputs and outputs across the session.

    Returns one dict per session with:
    - session_id: the session identifier
    - user_inputs: chronological list of user inputs
    - output_messages: chronological list of agent outputs
    - trace_count: number of traces in the session
    - tool_calls: all tool calls across the session
    """
    grouped = _group_spans(spans, session_id_key)

    results = []
    for session_id, session_spans in grouped.items():
        session_spans.sort(key=lambda s: s.start_time_ns)

        user_inputs: list[str] = []
        output_messages: list[str] = []
        tool_calls: list[str] = []
        trace_ids: set[str] = set()

        for span in session_spans:
            trace_ids.add(span.trace_id)
            if span.kind == "LLM":
                inp = span.attributes.get(_INPUT_VALUE_KEY, "")
                out = span.attributes.get(_OUTPUT_VALUE_KEY, "")
                if inp and inp not in user_inputs:
                    user_inputs.append(inp)
                if out:
                    output_messages.append(out)
            elif span.kind == "TOOL":
                tool_name = span.attributes.get(_TOOL_NAME_KEY, span.name)
                tool_input = span.attributes.get(_INPUT_VALUE_KEY, "")
                tool_calls.append(f"{tool_name}({tool_input})")

        results.append({
            "session_id": session_id,
            "user_inputs": json.dumps(user_inputs),
            "output_messages": json.dumps(output_messages),
            "trace_count": len(trace_ids),
            "tool_calls": json.dumps(tool_calls),
        })

    return results


def _group_spans_by_trace(spans: list[OTelSpan]) -> dict[str, list[OTelSpan]]:
    """Group spans by trace_id."""
    groups: dict[str, list[OTelSpan]] = {}
    for span in spans:
        groups.setdefault(span.trace_id, []).append(span)
    return groups
