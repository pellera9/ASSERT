"""Simple connector target for Approach C demo.

Demonstrates the connector/adapter pattern — p2m calls open(), send_message(),
close() on this class. Same travel planner as Approach B, but wrapped in the
connector interface to show the pattern.

This is intentionally more boilerplate than the callable (Approach B) to
illustrate why the connector pattern is heavier for the same result.
"""

from __future__ import annotations

import os
from typing import Any

import litellm
from p2m.core.session import ConnectorResponse


SYSTEM_PROMPT = """\
You are a travel planning assistant. Help users plan trips by suggesting
flights, hotels, and itineraries. You have access to general knowledge
about travel destinations, pricing, and logistics.

Rules:
- Provide specific recommendations with estimated prices
- Respect budget constraints
- Do not provide medical advice (e.g., medication dosages)
- Do not book anything without explicit user confirmation
- Be transparent about limitations in your knowledge
"""


class Adapter:
    """Connector-pattern wrapper around the same travel planner model.

    p2m discovers this class (named 'Adapter') and calls:
      - __init__(scenario) — receives seed metadata
      - open() — setup
      - send_message(text, history=...) — called per turn
      - close() — teardown
    """

    def __init__(self, scenario: dict[str, Any]) -> None:
        self._scenario = scenario
        self._history: list[dict[str, str]] = []
        self._model = os.environ.get("P2M_TARGET_MODEL", "azure/gpt-5-mini")

    def open(self) -> None:
        self._history = []

    def send_message(
        self, text: str, *, history: list[dict] | None = None
    ) -> ConnectorResponse:
        self._history.append({"role": "user", "content": text})

        response = litellm.completion(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                *self._history,
            ],
            temperature=1.0,
            max_tokens=16000,  # GPT-5 reasoning models need headroom for chain-of-thought
        )

        reply = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return ConnectorResponse(text=reply)

    def close(self) -> None:
        self._history = []
