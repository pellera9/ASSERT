"""Approach B: Callable / HTTP target wrapper.

USER-SIDE CODE: 3-line wrapper function.
P2M-SIDE CODE: CallableSession that imports and invokes the function.

This is the simplest approach — black-box input/output only.
"""

from __future__ import annotations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USER-SIDE: What the developer writes (3 lines)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from examples.travel_planner.agent import chat_sync


def target(message: str) -> str:
    """Thin wrapper — p2m calls this with a seed prompt, gets a string back."""
    return chat_sync(message)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P2M-SIDE: CallableSession — new session type for target.callable
# This is what p2m ships as part of P0-2 (backlog item).
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import asyncio
import importlib
import inspect
from typing import Any

from p2m.core.async_utils import invoke_callable
from p2m.core.model_client import Message
from p2m.core.session import TurnResult


class CallableSession:
    """Invokes a user-provided callable as the eval target.

    Supports:
    - Sync functions: fn(str) -> str
    - Async functions: async fn(str) -> str
    - Sync with history: fn(str, history=list) -> str
    - Async with history: async fn(str, history=list) -> str

    The callable receives the latest user message as a string.
    It returns the agent's response as a string.

    This is a black box: p2m sees input/output only. No internal visibility.
    """

    def __init__(
        self,
        *,
        callable_ref: str,
        system_prompt: str | None = None,
        message_timeout_s: float | None = None,
    ) -> None:
        self._callable_ref = callable_ref
        self._system_prompt = system_prompt
        self._message_timeout_s = message_timeout_s
        self._callable = None
        self._supports_history = False

    @property
    def runtime_mode(self) -> str:
        return "callable"

    async def open(self) -> None:
        """Import and resolve the callable from module:function format."""
        module_path, func_name = self._callable_ref.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        self._callable = getattr(mod, func_name)

        # Detect if callable accepts a history parameter
        sig = inspect.signature(self._callable)
        self._supports_history = "history" in sig.parameters

    async def close(self) -> None:
        self._callable = None

    async def run_turn(self, messages: list[Message]) -> TurnResult:
        """Invoke the callable and wrap the response."""
        # Extract latest user message
        user_text = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_text = msg.text
                break

        # Build history if callable supports it
        if self._supports_history:
            history = [
                {"role": msg.role, "content": msg.text}
                for msg in messages
                if msg.role in ("user", "assistant")
            ]
            response_text = await invoke_callable(
                self._callable,
                user_text,
                history=history,
                timeout_s=self._message_timeout_s,
            )
        else:
            response_text = await invoke_callable(
                self._callable,
                user_text,
                timeout_s=self._message_timeout_s,
            )

        # Normalize response
        if not isinstance(response_text, str):
            response_text = str(response_text)

        # Build interaction messages — INPUT/OUTPUT ONLY, NO INTERNALS
        interaction_messages = [
            {
                "role": "user",
                "content": user_text,
                "raw": {"message": {"role": "user", "content": user_text}},
            },
            {
                "role": "assistant",
                "content": response_text,
                "raw": {"message": {"role": "assistant", "content": response_text}},
            },
        ]

        return TurnResult(
            text=response_text,
            state_messages=list(messages) + [Message(role="assistant", content=response_text)],
            interaction_messages=interaction_messages,
            raw={"callable": self._callable_ref},
        )


# ── HTTP variant (same black-box visibility) ──────────────────

import aiohttp


class HTTPEndpointSession:
    """Invokes an HTTP endpoint as the eval target.

    POST {"message": text, "history": [...]} to the URL.
    Expects {"response": "..."} back.

    Same black-box visibility as CallableSession.
    """

    def __init__(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
        message_timeout_s: float | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._headers = headers or {}
        self._timeout_s = message_timeout_s
        self._session = None

    @property
    def runtime_mode(self) -> str:
        return "http_endpoint"

    async def open(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self._timeout_s or 60)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def run_turn(self, messages: list[Message]) -> TurnResult:
        user_text = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_text = msg.text
                break

        history = [
            {"role": msg.role, "content": msg.text}
            for msg in messages
            if msg.role in ("user", "assistant")
        ]

        payload = {"message": user_text, "history": history}

        async with self._session.post(
            self._endpoint,
            json=payload,
            headers=self._headers,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            response_text = data.get("response", "")

        interaction_messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": response_text},
        ]

        return TurnResult(
            text=response_text,
            state_messages=list(messages) + [Message(role="assistant", content=response_text)],
            interaction_messages=interaction_messages,
            raw={"endpoint": self._endpoint},
        )
