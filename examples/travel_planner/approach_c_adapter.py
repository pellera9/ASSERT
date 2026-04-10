"""Approach C: Framework-specific LangChain/LangGraph adapter.

USER-SIDE CODE: Zero (just config — IF their graph matches the adapter's assumptions).
P2M-SIDE CODE: Full adapter that knows about LangGraph internals.

This file shows both the happy path AND the failure modes.
"""

from __future__ import annotations

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USER-SIDE: What the developer writes (nothing — config only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# The user just points the config at their agent module:
#   target:
#     connector: p2m.adapters.langchain
#     agent_module: examples.travel_planner.agent
#     agent_attr: get_graph


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# P2M-SIDE: The framework adapter p2m would ship
# p2m/adapters/langchain.py — this is what p2m must build and maintain
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import importlib
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from p2m.core.session import ConnectorResponse, AdapterEvent


class Adapter:
    """p2m connector for LangChain/LangGraph agents.

    Attempts to handle multiple LangChain patterns:
    - AgentExecutor (invoke with {"input": text})
    - StateGraph (invoke with {"messages": [...]})
    - LCEL Runnable (invoke with text)

    PROBLEM: Every graph has a different state schema. This adapter must
    guess the right invocation pattern and output extraction method.
    """

    def __init__(self, scenario: dict[str, Any]) -> None:
        self._scenario = scenario
        self._agent = None
        self._agent_type = None  # "agent_executor", "state_graph", "runnable"
        self._history: list[BaseMessage] = []

    def open(self) -> None:
        """Import the user's agent and detect its type."""
        # These come from the YAML config via scenario passthrough
        # (p2m would need to plumb these through — another config schema change)
        agent_module = self._scenario.get("_agent_module", "")
        agent_attr = self._scenario.get("_agent_attr", "agent")

        if not agent_module:
            raise ValueError(
                "LangChain adapter requires agent_module in config. "
                "Set target.agent_module in your YAML."
            )

        mod = importlib.import_module(agent_module)
        agent_or_factory = getattr(mod, agent_attr)

        # If it's a callable factory, call it to get the agent
        if callable(agent_or_factory) and not hasattr(agent_or_factory, "invoke"):
            self._agent = agent_or_factory()
        else:
            self._agent = agent_or_factory

        # Detect agent type — THIS IS THE FRAGILE PART
        self._agent_type = self._detect_agent_type()

    def _detect_agent_type(self) -> str:
        """Attempt to detect what kind of LangChain agent this is.

        This is inherently fragile because:
        1. LangGraph CompiledGraph and AgentExecutor both have .invoke()
        2. StateGraph state schemas are user-defined (no standard keys)
        3. LCEL chains have .invoke() but different signatures
        4. LangGraph is pre-1.0 and the API surface changes frequently
        """
        agent = self._agent
        cls_name = type(agent).__name__

        # Check for LangGraph CompiledStateGraph
        if hasattr(agent, "get_graph") and hasattr(agent, "nodes"):
            return "state_graph"

        # Check for AgentExecutor
        if cls_name == "AgentExecutor" or hasattr(agent, "agent"):
            return "agent_executor"

        # Check for LCEL Runnable
        if hasattr(agent, "invoke") and hasattr(agent, "batch"):
            return "runnable"

        raise ValueError(
            f"Cannot detect agent type for {cls_name}. "
            f"The LangChain adapter supports AgentExecutor, StateGraph, "
            f"and LCEL Runnables. For custom agents, use target.callable instead."
        )

    def send_message(self, text: str, *, history: list[dict] | None = None) -> ConnectorResponse:
        """Send a message to the agent and return the response.

        THIS IS WHERE APPROACH C BREAKS for non-standard graphs.
        """
        if self._agent_type == "agent_executor":
            return self._invoke_agent_executor(text)
        elif self._agent_type == "state_graph":
            return self._invoke_state_graph(text)
        elif self._agent_type == "runnable":
            return self._invoke_runnable(text)
        else:
            raise ValueError(f"Unknown agent type: {self._agent_type}")

    def _invoke_agent_executor(self, text: str) -> ConnectorResponse:
        """AgentExecutor: the happy path. Standard input/output keys."""
        result = self._agent.invoke({"input": text})
        output = result.get("output", "")
        return ConnectorResponse(text=output)

    def _invoke_state_graph(self, text: str) -> ConnectorResponse:
        """StateGraph: THE PROBLEM CASE.

        LangGraph StateGraph has NO standard state schema. The user defines
        the state class with arbitrary keys. Our travel planner uses:

            class TravelState(dict):
                messages: Annotated[Sequence[BaseMessage], add_messages]
                intent: str
                destination: str
                dates: dict
                budget: float
                flights: list
                hotels: list
                itinerary: str

        But another graph might use:
            - {"input": str, "output": str}
            - {"query": str, "results": list}
            - {"chat_history": list, "question": str, "answer": str}

        The adapter MUST know the state schema to:
        1. Construct the input dict correctly
        2. Extract the output from the result dict

        This is fundamentally impossible to generalize.
        """
        # Try common patterns — each is a guess that may fail
        input_variants = [
            # Pattern 1: LangGraph message-based (our travel planner uses this)
            {"messages": [HumanMessage(content=text)]},
            # Pattern 2: Simple input/output
            {"input": text},
            # Pattern 3: Chat-style
            {"chat_history": self._history, "question": text},
            # Pattern 4: Query-style
            {"query": text},
        ]

        last_error = None
        for input_dict in input_variants:
            try:
                result = self._agent.invoke(input_dict)
                output = self._extract_output_from_state_graph(result, text)
                if output:
                    return ConnectorResponse(text=output)
            except Exception as e:
                last_error = e
                continue

        # ALL patterns failed — this is the common failure mode
        raise ValueError(
            f"LangChain adapter could not invoke StateGraph. "
            f"Tried {len(input_variants)} input patterns, all failed. "
            f"Last error: {last_error}\n\n"
            f"Your graph has a custom state schema that this adapter doesn't support. "
            f"Use target.callable instead:\n"
            f"  target:\n"
            f"    callable: your_module:chat_function\n"
        )

    def _extract_output_from_state_graph(self, result: dict, original_input: str) -> str:
        """Try to extract the agent's response from a StateGraph result.

        Another guessing game. The result dict has the full state, and we
        need to find which field contains the response text.
        """
        # Pattern 1: Check messages list for last AI message
        messages = result.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    self._history.append(HumanMessage(content=original_input))
                    self._history.append(msg)
                    return msg.content

        # Pattern 2: Check "output" key
        output = result.get("output")
        if isinstance(output, str) and output:
            return output

        # Pattern 3: Check "answer" key
        answer = result.get("answer")
        if isinstance(answer, str) and answer:
            return answer

        # Pattern 4: Check "itinerary" key (specific to our travel planner)
        # NOTE: This is exactly the problem — we'd need to add special cases
        # for every possible state schema
        itinerary = result.get("itinerary")
        if isinstance(itinerary, str) and itinerary:
            return itinerary

        # Pattern 5: Check "response" key
        response = result.get("response")
        if isinstance(response, str) and response:
            return response

        return ""

    def _invoke_runnable(self, text: str) -> ConnectorResponse:
        """LCEL Runnable: try string input, then dict input."""
        try:
            result = self._agent.invoke(text)
        except Exception:
            result = self._agent.invoke({"input": text})

        if isinstance(result, str):
            return ConnectorResponse(text=result)
        elif isinstance(result, AIMessage):
            return ConnectorResponse(text=result.content)
        elif isinstance(result, dict):
            return ConnectorResponse(text=str(result.get("output", result)))
        else:
            return ConnectorResponse(text=str(result))

    def close(self) -> None:
        """Clean up."""
        self._agent = None
        self._history.clear()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THE MAINTENANCE BURDEN — what p2m must track and update
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KNOWN_ISSUES = """
Known issues with the framework adapter approach:

1. STATE SCHEMA GUESSING
   Every LangGraph StateGraph has a unique state schema. The adapter must
   guess input/output keys. Our travel planner has 8 state fields — the
   adapter has no way to know which ones matter.

2. LANGGRAPH API CHURN
   LangGraph is pre-1.0. Between v0.1 and v0.2:
   - StateGraph.compile() return type changed
   - Message annotation format changed
   - ToolNode API changed
   - Checkpoint interface changed completely
   Each change breaks the adapter.

3. MCP TOOL OPACITY
   The adapter invokes the graph but cannot see MCP tool calls happening
   inside nodes. The judge sees the same black-box output as Approach B,
   but with more complexity and fragility.

4. MULTIPLICATIVE MAINTENANCE
   To support N frameworks at the same quality level:
   - LangChain AgentExecutor (stable, straightforward)
   - LangGraph StateGraph (unstable, highly variable)
   - LangGraph LCEL (stable, but 3+ invocation patterns)
   - Semantic Kernel (different plugin model entirely)
   - OpenAI Agents SDK (Runner.run() + different message format)
   - CrewAI (task-based, not message-based)
   - AutoGen (multi-agent conversation protocol)
   That's 7 adapters × ongoing API changes = permanent maintenance.

5. FALSE SENSE OF SUPPORT
   If the adapter works for AgentExecutor but fails for StateGraph,
   users see "LangChain supported" and expect it to work. When it doesn't,
   trust erodes faster than if we never claimed support.
"""
