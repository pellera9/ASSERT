"""Tests for framework-agnostic agent support POC.

Tests both Approach A (OTel trace parser) and Approach B (CallableSession).
"""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from p2m.core.otel import parse_otel_traces, OTelSpan, _flatten_attributes

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_TRACES = FIXTURES / "sample_otel_traces.json"


# ── Approach A: OTel trace parser tests ──────────────────────────


class TestOTelParser(unittest.TestCase):
    """Validates that OTLP JSON traces are correctly converted to p2m transcript rows."""

    def test_parse_groups_by_session_id(self):
        """Two sessions in the fixture should produce two transcript rows."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        self.assertEqual(len(rows), 2)
        session_ids = {r["metadata"]["session_id"] for r in rows}
        self.assertEqual(session_ids, {"sess_tokyo_trip", "sess_paris_trip"})

    def test_tokyo_session_has_all_events(self):
        """The Tokyo session should have 3 LLM + 2 tool = 5 events."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        tokyo = next(r for r in rows if r["metadata"]["session_id"] == "sess_tokyo_trip")

        events = tokyo["events"]
        llm_events = [e for e in events if e["actor"] == "target"]
        tool_events = [e for e in events if e["actor"] == "tool"]
        self.assertEqual(len(llm_events), 3)
        self.assertEqual(len(tool_events), 2)

    def test_aggregate_metadata(self):
        """Aggregate metadata should reflect all spans in the session."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        tokyo = next(r for r in rows if r["metadata"]["session_id"] == "sess_tokyo_trip")
        agg = tokyo["raw"]

        self.assertEqual(agg["llm_call_count"], 3)
        self.assertEqual(agg["total_tokens"]["input"], 85 + 210 + 420)
        self.assertEqual(agg["total_tokens"]["output"], 42 + 35 + 95)
        self.assertIn("intent_classifier", agg["nodes_visited"])
        self.assertIn("flight_search", agg["nodes_visited"])
        self.assertIn("itinerary_optimizer", agg["nodes_visited"])
        self.assertIn("search_flights", agg["tools_called"])
        self.assertIn("search_hotels", agg["tools_called"])

    def test_tool_events_have_parsed_args(self):
        """Tool call events should contain parsed JSON args."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        tokyo = next(r for r in rows if r["metadata"]["session_id"] == "sess_tokyo_trip")

        tool_events = [e for e in tokyo["events"] if e["actor"] == "tool"]
        flight_event = next(e for e in tool_events if e["edit"]["tool_name"] == "search_flights")
        self.assertEqual(flight_event["edit"]["tool_args"]["destination"], "NRT")
        self.assertEqual(flight_event["edit"]["tool_args"]["max_price"], 1500)

    def test_llm_events_have_node_metadata(self):
        """LLM events should carry node name, model, tokens, and latency."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        tokyo = next(r for r in rows if r["metadata"]["session_id"] == "sess_tokyo_trip")

        llm_events = [e for e in tokyo["events"] if e["actor"] == "target"]
        first = llm_events[0]
        self.assertEqual(first["raw"]["_node"], "intent_classifier")
        self.assertEqual(first["raw"]["_model"], "gpt-4o")
        self.assertEqual(first["raw"]["_tokens"]["input"], 85)
        self.assertGreater(first["raw"]["_latency_ms"], 0)

    def test_paris_session_is_minimal(self):
        """The Paris session has only 1 LLM span, no tools."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        paris = next(r for r in rows if r["metadata"]["session_id"] == "sess_paris_trip")

        self.assertEqual(len(paris["events"]), 1)
        self.assertEqual(paris["raw"]["llm_call_count"], 1)
        self.assertEqual(paris["raw"]["tools_called"], [])

    def test_events_are_time_ordered(self):
        """Events within a session should be in chronological order."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        tokyo = next(r for r in rows if r["metadata"]["session_id"] == "sess_tokyo_trip")

        events = tokyo["events"]
        # First event should be intent_classifier, last should be itinerary_optimizer
        first_llm = next(e for e in events if e["actor"] == "target")
        self.assertEqual(first_llm["raw"]["_node"], "intent_classifier")

    def test_group_by_trace_id(self):
        """Grouping by trace_id should produce different groupings."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="nonexistent.key")
        # Falls back to trace_id grouping
        trace_ids = {r["metadata"]["session_id"] for r in rows}
        self.assertIn("abc123", trace_ids)
        self.assertIn("def456", trace_ids)

    def test_transcript_row_schema(self):
        """Each row should have metadata, events, and raw keys."""
        rows = parse_otel_traces(SAMPLE_TRACES, group_by="session.id")
        for row in rows:
            self.assertIn("metadata", row)
            self.assertIn("events", row)
            self.assertIn("raw", row)
            self.assertEqual(row["metadata"]["runtime_mode"], "otel_traced")
            self.assertEqual(row["metadata"]["kind"], "otel_import")


class TestFlattenAttributes(unittest.TestCase):
    """Test OTLP attribute parsing edge cases."""

    def test_string_value(self):
        attrs = [{"key": "k", "value": {"stringValue": "hello"}}]
        self.assertEqual(_flatten_attributes(attrs), {"k": "hello"})

    def test_int_value(self):
        attrs = [{"key": "k", "value": {"intValue": "42"}}]
        self.assertEqual(_flatten_attributes(attrs), {"k": 42})

    def test_double_value(self):
        attrs = [{"key": "k", "value": {"doubleValue": 3.14}}]
        self.assertAlmostEqual(_flatten_attributes(attrs)["k"], 3.14)

    def test_bool_value(self):
        attrs = [{"key": "k", "value": {"boolValue": True}}]
        self.assertEqual(_flatten_attributes(attrs), {"k": True})

    def test_empty_attributes(self):
        self.assertEqual(_flatten_attributes([]), {})


# ── Approach B: CallableSession tests ────────────────────────────


class TestCallableSession(unittest.TestCase):
    """Validates CallableSession can invoke sync/async callables."""

    def test_import(self):
        """CallableSession should be importable from p2m.core.session."""
        from p2m.core.session import CallableSession
        self.assertTrue(callable(CallableSession))

    def test_sync_callable(self):
        """CallableSession should handle a sync fn(str) -> str."""
        from p2m.core.session import CallableSession
        from p2m.core.model_client import Message

        # Create a temp module with a sync callable
        import types
        mod = types.ModuleType("_test_sync_target")
        mod.target = lambda msg: f"echo: {msg}"

        import sys
        sys.modules["_test_sync_target"] = mod

        try:
            session = CallableSession(callable_ref="_test_sync_target:target")

            async def _run():
                await session.open()
                result = await session.run_turn([Message(role="user", content="hello")])
                await session.close()
                return result

            result = asyncio.run(_run())
            self.assertEqual(result.text, "echo: hello")
            self.assertEqual(len(result.interaction_messages), 2)
            self.assertEqual(result.interaction_messages[0]["role"], "user")
            self.assertEqual(result.interaction_messages[1]["role"], "assistant")
        finally:
            del sys.modules["_test_sync_target"]

    def test_async_callable(self):
        """CallableSession should handle an async fn(str) -> str."""
        from p2m.core.session import CallableSession
        from p2m.core.model_client import Message

        import types
        mod = types.ModuleType("_test_async_target")

        async def async_target(msg: str) -> str:
            return f"async: {msg}"

        mod.target = async_target

        import sys
        sys.modules["_test_async_target"] = mod

        try:
            session = CallableSession(callable_ref="_test_async_target:target")

            async def _run():
                await session.open()
                result = await session.run_turn([Message(role="user", content="world")])
                await session.close()
                return result

            result = asyncio.run(_run())
            self.assertEqual(result.text, "async: world")
        finally:
            del sys.modules["_test_async_target"]

    def test_callable_with_history(self):
        """CallableSession should detect and pass history parameter."""
        from p2m.core.session import CallableSession
        from p2m.core.model_client import Message

        import types
        mod = types.ModuleType("_test_history_target")

        def target_with_history(msg: str, history: list = None) -> str:
            return f"got {len(history or [])} history items"

        mod.target = target_with_history

        import sys
        sys.modules["_test_history_target"] = mod

        try:
            session = CallableSession(callable_ref="_test_history_target:target")

            async def _run():
                await session.open()
                messages = [
                    Message(role="user", content="first"),
                    Message(role="assistant", content="reply"),
                    Message(role="user", content="second"),
                ]
                result = await session.run_turn(messages)
                await session.close()
                return result

            result = asyncio.run(_run())
            self.assertIn("3 history items", result.text)
        finally:
            del sys.modules["_test_history_target"]

    def test_runtime_mode(self):
        """CallableSession.runtime_mode should be 'callable'."""
        from p2m.core.session import CallableSession
        session = CallableSession(callable_ref="some.module:fn")
        self.assertEqual(session.runtime_mode, "callable")


# ── TargetConfig validation tests ────────────────────────────────


class TestTargetConfigCallable(unittest.TestCase):
    """Validates TargetConfig accepts callable field."""

    def test_callable_target_is_valid(self):
        """TargetConfig with callable should not raise."""
        from p2m.core.config_model import TargetConfig
        tc = TargetConfig(callable="my_module:my_fn")
        self.assertTrue(tc.is_callable)
        self.assertFalse(tc.is_external)

    def test_callable_and_model_conflicts(self):
        """TargetConfig with both callable and model should raise."""
        from p2m.core.config_model import TargetConfig
        with self.assertRaises(ValueError):
            TargetConfig(callable="my_module:my_fn", model="openai/gpt-4o")

    def test_callable_and_connector_conflicts(self):
        """TargetConfig with both callable and connector should raise."""
        from p2m.core.config_model import TargetConfig
        with self.assertRaises(ValueError):
            TargetConfig(callable="my_module:my_fn", connector="some.connector")


if __name__ == "__main__":
    unittest.main()
