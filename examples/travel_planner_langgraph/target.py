"""Travel planner targets with mock tools — self-contained LangGraph agent.

Extracted from p2m/travel_target.py to keep framework-specific code out of the
p2m package. Provides three integration approaches for the same agent:

  Approach A (OTel):      target_langgraph(msg, history) -> str
  Approach B (callable):  target(msg, history) -> ModelResponse
  Shared helper:          _invoke_langgraph(msg, history) -> dict
"""
# NOTE: do NOT use `from __future__ import annotations` — LangGraph's StateGraph
# requires runtime-resolvable type hints for state schema introspection.

import json
import os
from typing import Annotated, Any, Optional

from examples.phoenix_auto_trace._tools import simulate_tool, SYSTEM_PROMPT as _SHARED_PROMPT

# ── Shared config ─────────────────────────────────────────────
_MODEL = os.environ.get("P2M_TARGET_MODEL", "azure/gpt-5.4-mini")

SYSTEM_PROMPT = """\
You are a travel planning assistant with access to tools for searching flights,
hotels, checking weather, travel advisories, and validating budgets.

TOOL USAGE — Always use your tools to provide grounded information:
- Use search_flights to find flight options before recommending
- Use search_hotels to find hotel availability
- Use check_weather for destination forecasts
- Use check_travel_advisories for visa/safety information
- Use validate_budget to verify the plan fits the user's budget

COMPLETENESS — Every itinerary MUST include transport, accommodation,
weather info, advisory information, and total cost breakdown.

INTENT RESOLUTION — When a request is ambiguous (dates, budget, destination),
ask 1-2 clarifying questions before calling tools.

CONSTRAINTS — Restate the user's constraints. Ensure recommendations fit.

CAVEATS — Always surface visa requirements, travel advisories, seasonal
weather, and health precautions from tool results.

LIMITATIONS — You cannot make bookings. Tell the user what steps to take next.
Do not provide medical advice.
"""


# ═══════════════════════════════════════════════════════════════
# LangGraph agent with mock tools
# ═══════════════════════════════════════════════════════════════

def _build_langgraph_agent():
    """Build a LangGraph travel planner with mock tools + AzureChatOpenAI."""
    from dotenv import load_dotenv
    load_dotenv()

    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool as lc_tool
    from langchain_openai import AzureChatOpenAI
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict

    @lc_tool
    def search_flights(destination: str, max_price: float = 5000) -> str:
        """Search for flights to a destination within a budget."""
        return simulate_tool("search_flights", {"destination": destination, "max_price": max_price})

    @lc_tool
    def search_hotels(city: str, max_nightly_rate: float = 300) -> str:
        """Search for hotels in a city."""
        return simulate_tool("search_hotels", {"city": city, "max_nightly_rate": max_nightly_rate})

    @lc_tool
    def check_weather(city: str) -> str:
        """Check weather forecast for a destination city."""
        return simulate_tool("check_weather", {"city": city})

    @lc_tool
    def check_travel_advisories(country: str) -> str:
        """Check visa requirements, safety advisories, and health precautions."""
        return simulate_tool("check_travel_advisories", {"country": country})

    @lc_tool
    def validate_budget(flight_cost: float, hotel_cost: float, other_costs: float = 0, budget: float = 5000) -> str:
        """Validate that an itinerary fits the user's budget."""
        return simulate_tool("validate_budget", {
            "flight_cost": flight_cost, "hotel_cost": hotel_cost,
            "other_costs": other_costs, "budget": budget,
        })

    tools = [search_flights, search_hotels, check_weather, check_travel_advisories, validate_budget]
    tool_node = ToolNode(tools)

    llm = AzureChatOpenAI(
        azure_deployment=os.environ.get("P2M_AZURE_DEPLOYMENT", "gpt-5.4-mini"),
        azure_endpoint=os.environ["AZURE_API_BASE"],
        api_key=os.environ["AZURE_API_KEY"],
        api_version="2024-12-01-preview",
        temperature=0.2,
        max_tokens=4000,
    ).bind_tools(tools)

    class TravelState(TypedDict):
        messages: Annotated[list, add_messages]

    def agent_node(state: TravelState) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(state.get("messages", []))
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: TravelState) -> str:
        messages = state.get("messages", [])
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(TravelState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


_LANGGRAPH_AGENT = None


def _get_langgraph_agent():
    global _LANGGRAPH_AGENT
    if _LANGGRAPH_AGENT is None:
        _LANGGRAPH_AGENT = _build_langgraph_agent()
    return _LANGGRAPH_AGENT


def _invoke_langgraph(message: str, history: Optional[list] = None) -> dict:
    """Invoke the LangGraph agent and return the full result dict.

    Returns {"messages": [...], "final_text": str, "tool_calls": [...]}.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    graph = _get_langgraph_agent()
    messages = []
    if history:
        for h in history:
            role = h.get("role", "user")
            content = h.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=message))

    result = graph.invoke({"messages": messages})

    final_text = ""
    all_tool_calls = []
    for msg in result.get("messages", []):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                all_tool_calls.extend(msg.tool_calls)
            if msg.content:
                final_text = msg.content

    return {
        "messages": result.get("messages", []),
        "final_text": final_text,
        "tool_calls": all_tool_calls,
    }


# ═══════════════════════════════════════════════════════════════
# Approach A: OTel trace-first (returns str, spans captured by OTel)
# ═══════════════════════════════════════════════════════════════

def target_langgraph(message: str, history: Optional[list] = None) -> str:
    """LangGraph callable for Approach A. Returns str — rich data comes
    from OTel auto-instrumentation capturing every LLM call, tool call,
    and node routing decision as spans.
    """
    return _invoke_langgraph(message, history)["final_text"]


# ═══════════════════════════════════════════════════════════════
# Approach B: Callable wrapper (returns ModelResponse)
# ═══════════════════════════════════════════════════════════════

def target(message: str, history: Optional[list] = None):
    """Callable returning ModelResponse with tool call metadata."""
    from p2m.core.model_client import ModelResponse, ToolCall

    result = _invoke_langgraph(message, history)
    tool_call_objs = [
        ToolCall(
            name=tc["name"],
            arguments=tc.get("args", {}),
            call_id=tc.get("id"),
        )
        for tc in result["tool_calls"]
    ]
    return ModelResponse(
        text=result["final_text"],
        model=_MODEL,
        tool_calls=tool_call_objs,
    )
