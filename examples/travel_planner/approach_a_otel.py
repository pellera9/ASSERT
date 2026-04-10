"""Approach A: Phoenix OTel auto-instrumentation + universal trace adapter.

USER-SIDE CODE: 2 lines to instrument, then run p2m normally.
P2M-SIDE CODE: Conversation.from_otel() parser + OTel-aware rollout.

This file contains both sides annotated clearly.
"""

from __future__ import annotations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USER-SIDE: What the developer adds to their agent (2 lines)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from phoenix.otel import register
register(auto_instrument=True)
# That's it. Phoenix auto-discovers LangChain, LangGraph, OpenAI, MCP tool calls.
# No code changes to the agent itself.

from examples.travel_planner.agent import chat_sync  # noqa: E402 — unchanged agent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P2M-SIDE: New code p2m ships — OTel trace parser + session
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from p2m.core.async_utils import invoke_callable
from p2m.core.model_client import Message
from p2m.core.session import (
    AdapterEvent,
    ConnectorResponse,
    TurnResult,
)


# ── Conversation.from_otel() — parse OTLP JSON into p2m Conversations ──

@dataclass
class OTelSpan:
    """Minimal representation of an OpenTelemetry span."""
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str  # "LLM", "CHAIN", "TOOL", "AGENT", etc.
    start_time: int  # unix nanos
    end_time: int
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    status: str = "OK"


@dataclass
class ConversationTurn:
    """One turn of a conversation extracted from traces."""
    role: str  # "user", "assistant", "tool_call", "tool_result"
    content: str
    timestamp: int = 0
    # Rich metadata only available via OTel (not in approaches B/C)
    node_name: str | None = None       # e.g. "intent_classifier", "flight_search"
    model_name: str | None = None      # e.g. "gpt-4o"
    token_usage: dict[str, int] | None = None  # {"input": 150, "output": 42}
    latency_ms: float | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


@dataclass
class Conversation:
    """A structured conversation extracted from OTel traces.

    This is what the judge receives. Compare the richness of data here
    vs. what Approach B/C can provide.
    """
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    # Aggregate metadata
    total_tokens: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    total_latency_ms: float = 0.0
    nodes_visited: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_otel(
        cls,
        path: str | Path,
        *,
        group_by: str = "session.id",
    ) -> list["Conversation"]:
        """Parse OTLP JSON export into Conversation objects.

        Args:
            path: Path to OTLP JSON file (exported from Phoenix or any collector)
            group_by: Attribute key to group spans into conversations.
                      Common values: "session.id", "gen_ai.conversation.id"

        Returns:
            List of Conversation objects, one per unique group_by value.
        """
        raw_path = Path(path)
        spans = _parse_otlp_json(raw_path)
        grouped = _group_spans_by_attribute(spans, group_by)

        conversations = []
        for session_id, session_spans in grouped.items():
            conv = cls(session_id=session_id)
            # Sort spans by start time to reconstruct conversation order
            session_spans.sort(key=lambda s: s.start_time)

            for span in session_spans:
                _process_span_into_conversation(span, conv)

            conversations.append(conv)

        return conversations

    def to_judge_input(self) -> list[dict[str, Any]]:
        """Format as input for p2m's LLM judge.

        This is the key differentiator: the judge sees EVERYTHING —
        not just the final user/assistant text, but every internal
        node, tool call, LLM reasoning step, and token cost.
        """
        messages = []
        for turn in self.turns:
            entry: dict[str, Any] = {
                "role": turn.role,
                "content": turn.content,
            }
            if turn.node_name:
                entry["_node"] = turn.node_name
            if turn.model_name:
                entry["_model"] = turn.model_name
            if turn.token_usage:
                entry["_tokens"] = turn.token_usage
            if turn.latency_ms is not None:
                entry["_latency_ms"] = turn.latency_ms
            if turn.tool_name:
                entry["_tool"] = turn.tool_name
            if turn.tool_args:
                entry["_tool_args"] = turn.tool_args
            messages.append(entry)

        # Append aggregate metadata as a system context message
        messages.append({
            "role": "system",
            "content": "[Execution metadata]",
            "_total_tokens": self.total_tokens,
            "_total_latency_ms": self.total_latency_ms,
            "_nodes_visited": self.nodes_visited,
            "_tools_called": self.tools_called,
            "_llm_call_count": len(self.llm_calls),
        })
        return messages


def _parse_otlp_json(path: Path) -> list[OTelSpan]:
    """Parse OTLP JSON export format into OTelSpan objects."""
    data = json.loads(path.read_text(encoding="utf-8"))

    spans = []
    # OTLP JSON has: {"resourceSpans": [{"scopeSpans": [{"spans": [...]}]}]}
    for resource_span in data.get("resourceSpans", []):
        for scope_span in resource_span.get("scopeSpans", []):
            for raw_span in scope_span.get("spans", []):
                attrs = _flatten_attributes(raw_span.get("attributes", []))
                spans.append(OTelSpan(
                    trace_id=raw_span.get("traceId", ""),
                    span_id=raw_span.get("spanId", ""),
                    parent_span_id=raw_span.get("parentSpanId"),
                    name=raw_span.get("name", ""),
                    kind=attrs.get("openinference.span.kind", "UNKNOWN"),
                    start_time=int(raw_span.get("startTimeUnixNano", 0)),
                    end_time=int(raw_span.get("endTimeUnixNano", 0)),
                    attributes=attrs,
                    events=raw_span.get("events", []),
                    status=raw_span.get("status", {}).get("code", "OK"),
                ))
    return spans


def _flatten_attributes(attrs: list[dict]) -> dict[str, Any]:
    """Convert OTLP attribute array to flat dict."""
    result = {}
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
    return result


def _group_spans_by_attribute(
    spans: list[OTelSpan],
    group_key: str,
) -> dict[str, list[OTelSpan]]:
    """Group spans by a session/conversation attribute."""
    groups: dict[str, list[OTelSpan]] = {}
    for span in spans:
        session_id = span.attributes.get(group_key, span.trace_id)
        groups.setdefault(str(session_id), []).append(span)
    return groups


def _process_span_into_conversation(span: OTelSpan, conv: Conversation) -> None:
    """Extract conversation turns from a single OTel span."""
    latency_ms = (span.end_time - span.start_time) / 1_000_000

    if span.kind == "LLM":
        # Extract input/output from OpenInference conventions
        input_text = span.attributes.get("input.value", "")
        output_text = span.attributes.get("output.value", "")
        model = span.attributes.get("llm.model_name", "")
        input_tokens = span.attributes.get("llm.token_count.prompt", 0)
        output_tokens = span.attributes.get("llm.token_count.completion", 0)

        node_name = span.attributes.get("langgraph.node", span.name)

        # Record the LLM call
        conv.llm_calls.append({
            "node": node_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        })
        conv.total_tokens["input"] += input_tokens
        conv.total_tokens["output"] += output_tokens
        conv.total_latency_ms += latency_ms

        if node_name and node_name not in conv.nodes_visited:
            conv.nodes_visited.append(node_name)

        # Add assistant turn with full context
        if output_text:
            conv.turns.append(ConversationTurn(
                role="assistant",
                content=output_text,
                timestamp=span.start_time,
                node_name=node_name,
                model_name=model,
                token_usage={"input": input_tokens, "output": output_tokens},
                latency_ms=latency_ms,
            ))

    elif span.kind == "TOOL":
        tool_name = span.attributes.get("tool.name", span.name)
        tool_input = span.attributes.get("input.value", "")
        tool_output = span.attributes.get("output.value", "")

        if tool_name not in conv.tools_called:
            conv.tools_called.append(tool_name)

        # Tool call
        try:
            tool_args = json.loads(tool_input) if tool_input else {}
        except json.JSONDecodeError:
            tool_args = {"raw": tool_input}

        conv.turns.append(ConversationTurn(
            role="tool_call",
            content=f"Calling {tool_name}",
            timestamp=span.start_time,
            tool_name=tool_name,
            tool_args=tool_args,
            latency_ms=latency_ms,
        ))

        # Tool result
        conv.turns.append(ConversationTurn(
            role="tool_result",
            content=tool_output or "(no output)",
            timestamp=span.end_time,
            tool_name=tool_name,
        ))

    elif span.kind == "CHAIN":
        # LangGraph node execution — record the node visit
        node_name = span.attributes.get("langgraph.node", span.name)
        if node_name and node_name not in conv.nodes_visited:
            conv.nodes_visited.append(node_name)


# ── OTel-aware session for p2m rollout ─────────────────────────

class OTelTracedSession:
    """A p2m session that invokes a callable and captures OTel traces.

    This replaces ExternalSession when the user configures target.callable + trace.
    The session:
    1. Calls the user's function (which triggers the instrumented graph)
    2. Exports traces from Phoenix
    3. Parses traces into rich conversation data
    4. Returns TurnResult with full internal visibility
    """

    def __init__(
        self,
        *,
        callable_ref: str,
        trace_backend: str = "phoenix",
        group_by: str = "session.id",
    ) -> None:
        self._callable_ref = callable_ref
        self._trace_backend = trace_backend
        self._group_by = group_by
        self._callable = None
        self._session_id = None

    @property
    def runtime_mode(self) -> str:
        return "otel_traced"

    async def open(self) -> None:
        # Import the user's callable
        module_path, func_name = self._callable_ref.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(module_path)
        self._callable = getattr(mod, func_name)

    async def close(self) -> None:
        self._callable = None

    async def run_turn(self, messages: list[Message]) -> TurnResult:
        """Invoke callable, capture OTel traces, return rich TurnResult."""
        user_text = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_text = msg.text
                break

        # 1. Generate a unique session ID for trace grouping
        import uuid
        self._session_id = uuid.uuid4().hex[:12]

        # 2. Set the session context for Phoenix to tag traces
        import contextvars
        _session_var = contextvars.ContextVar("otel_session_id")
        _session_var.set(self._session_id)

        # 3. Invoke the user's callable — Phoenix captures ALL traces automatically
        response_text = await invoke_callable(self._callable, user_text)

        # 4. Export traces from Phoenix for this session
        traces_json = await self._export_traces()

        # 5. Parse traces into rich conversation data
        conversations = Conversation.from_otel(
            traces_json,
            group_by=self._group_by,
        )
        conv = conversations[0] if conversations else None

        # 6. Build rich interaction messages from the trace data
        interaction_messages = [
            {"role": "user", "content": user_text},
        ]

        if conv:
            # Include EVERY internal step — this is what the judge sees
            for turn in conv.turns:
                if turn.role == "assistant":
                    interaction_messages.append({
                        "role": "assistant",
                        "content": turn.content,
                        "raw": {
                            "_node": turn.node_name,
                            "_model": turn.model_name,
                            "_tokens": turn.token_usage,
                            "_latency_ms": turn.latency_ms,
                        },
                    })
                elif turn.role == "tool_call":
                    interaction_messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": f"tc_{turn.tool_name}",
                            "function": turn.tool_name,
                            "arguments": turn.tool_args or {},
                        }],
                    })
                elif turn.role == "tool_result":
                    interaction_messages.append({
                        "role": "tool",
                        "content": turn.content,
                        "function": turn.tool_name,
                        "tool_call_id": f"tc_{turn.tool_name}",
                    })

            # Final assistant message (the actual response to the user)
            if not any(m.get("role") == "assistant" and m.get("content") for m in interaction_messages[1:]):
                interaction_messages.append({
                    "role": "assistant",
                    "content": response_text,
                })

        else:
            interaction_messages.append({
                "role": "assistant",
                "content": response_text,
            })

        return TurnResult(
            text=response_text,
            state_messages=list(messages) + [Message(role="assistant", content=response_text)],
            interaction_messages=interaction_messages,
            raw={
                "session_id": self._session_id,
                "trace_backend": self._trace_backend,
                "nodes_visited": conv.nodes_visited if conv else [],
                "tools_called": conv.tools_called if conv else [],
                "total_tokens": conv.total_tokens if conv else {},
                "total_latency_ms": conv.total_latency_ms if conv else 0,
                "llm_call_count": len(conv.llm_calls) if conv else 0,
            },
        )

    async def _export_traces(self) -> Path:
        """Export traces from Phoenix for the current session."""
        import tempfile
        # Phoenix client API to export traces
        import phoenix as px

        traces_df = px.Client().get_spans_dataframe(
            filter_condition=f"attributes['session.id'] == '{self._session_id}'"
        )

        # Convert to OTLP JSON format
        export_path = Path(tempfile.mktemp(suffix=".json"))
        otlp_json = _dataframe_to_otlp_json(traces_df)
        export_path.write_text(json.dumps(otlp_json), encoding="utf-8")
        return export_path


def _dataframe_to_otlp_json(df) -> dict:
    """Convert Phoenix DataFrame to OTLP JSON format."""
    spans = []
    for _, row in df.iterrows():
        spans.append({
            "traceId": str(row.get("context.trace_id", "")),
            "spanId": str(row.get("context.span_id", "")),
            "parentSpanId": str(row.get("parent_id", "")) or None,
            "name": str(row.get("name", "")),
            "startTimeUnixNano": int(row.get("start_time", 0)),
            "endTimeUnixNano": int(row.get("end_time", 0)),
            "attributes": [
                {"key": k, "value": {"stringValue": str(v)}}
                for k, v in row.items()
                if k.startswith(("input.", "output.", "llm.", "tool.", "openinference.", "langgraph."))
            ],
        })
    return {"resourceSpans": [{"scopeSpans": [{"spans": spans}]}]}
