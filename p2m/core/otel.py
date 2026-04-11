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
from typing import Any


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
