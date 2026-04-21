"""Connector target for Approach C demo.

Same LangGraph agent as A and B, but wrapped in the connector interface.
Judge sees ConnectorResponse(text=str) — text only, no tool metadata.
"""

from __future__ import annotations

from typing import Any

from p2m.core.session import ConnectorResponse
from p2m.travel_target import _invoke_langgraph


class Adapter:
    """Connector-pattern wrapper around the shared LangGraph travel planner."""

    def __init__(self, scenario: dict[str, Any]) -> None:
        self._scenario = scenario
        self._history: list[dict[str, str]] = []

    def open(self) -> None:
        self._history = []

    def send_message(
        self, text: str, *, history: list[dict] | None = None
    ) -> ConnectorResponse:
        self._history.append({"role": "user", "content": text})
        result = _invoke_langgraph(text, list(self._history[:-1]) or None)
        reply = result["final_text"]
        self._history.append({"role": "assistant", "content": reply})
        return ConnectorResponse(text=reply)

    def close(self) -> None:
        self._history = []
