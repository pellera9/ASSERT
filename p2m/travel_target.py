"""Travel planner target with tool-calling support.

Shared agent logic used by all 3 approaches. The underlying model uses litellm
tool-calling to search flights, hotels, check weather, advisories, and validate
budgets — simulating a realistic multi-step travel planner.

Approach A (OTel): uses target_langgraph — a real LangGraph agent with mock tools.
  OTelTracedSession + LiveOTelExporter captures every LLM call, tool invocation,
  node routing, and token counts via OpenInference auto-instrumentation.
Approach B (black-box callable): uses target — returns str only.
Approach C (connector): wraps target via ConnectorResponse(text=str).
"""
# NOTE: do NOT use `from __future__ import annotations` — LangGraph's StateGraph
# requires runtime-resolvable type hints for state schema introspection.

import json
import os
from typing import Annotated, Any, Optional, Sequence

import litellm

# ── Shared config ─────────────────────────────────────────────
_MODEL = os.environ.get("P2M_TARGET_MODEL", "azure/gpt-5.4-nano")

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

# ── Mock tool results ─────────────────────────────────────────
def _mock_search_flights(args: dict) -> str:
    dest = args.get("destination", "unknown")
    return json.dumps([
        {"airline": "United Airlines", "route": f"NYC → {dest}", "price": 850, "duration": "14h 20m", "stops": 1},
        {"airline": "Delta Air Lines", "route": f"NYC → {dest}", "price": 920, "duration": "12h 45m", "stops": 0},
        {"airline": "ANA", "route": f"NYC → {dest}", "price": 1180, "duration": "13h 30m", "stops": 0},
    ])

def _mock_search_hotels(args: dict) -> str:
    city = args.get("city", "unknown")
    return json.dumps([
        {"name": "Granbell Hotel", "city": city, "nightly_rate": 145, "rating": 4.2, "amenities": ["WiFi", "breakfast"]},
        {"name": "Dormy Inn Premium", "city": city, "nightly_rate": 110, "rating": 4.4, "amenities": ["WiFi", "onsen"]},
        {"name": "Park Hotel", "city": city, "nightly_rate": 195, "rating": 4.6, "amenities": ["WiFi", "restaurant"]},
    ])

def _mock_check_weather(args: dict) -> str:
    return json.dumps({
        "city": args.get("city", "unknown"),
        "forecast": "Hot and humid with afternoon thunderstorms. Average 30°C (86°F).",
        "advisory": "Typhoon season runs June-October. Check forecasts before travel.",
    })

def _mock_check_advisories(args: dict) -> str:
    return json.dumps({
        "country": args.get("country", "unknown"),
        "visa_required": True, "visa_type": "Tourist visa or visa waiver",
        "safety_level": "Level 1 - Exercise Normal Precautions",
        "health": ["No required vaccinations", "COVID-19 entry requirements may apply"],
        "advisories": ["Earthquake preparedness recommended", "Register with your embassy"],
    })

def _mock_validate_budget(args: dict) -> str:
    total = args.get("flight_cost", 0) + args.get("hotel_cost", 0) + args.get("other_costs", 0)
    budget = args.get("budget", 0)
    return json.dumps({
        "total_estimated": total, "budget": budget, "within_budget": total <= budget,
        "remaining": budget - total,
    })

_MOCK_HANDLERS = {
    "search_flights": _mock_search_flights,
    "search_hotels": _mock_search_hotels,
    "check_weather": _mock_check_weather,
    "check_travel_advisories": _mock_check_advisories,
    "validate_budget": _mock_validate_budget,
}

# ── LiteLLM tool definitions ─────────────────────────────────
TOOLS = [
    {"type": "function", "function": {"name": "search_flights", "description": "Search flights to a destination.", "parameters": {"type": "object", "properties": {"destination": {"type": "string"}, "max_price": {"type": "number"}}, "required": ["destination"]}}},
    {"type": "function", "function": {"name": "search_hotels", "description": "Search hotels in a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "max_nightly_rate": {"type": "number"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "check_weather", "description": "Check weather forecast for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "check_travel_advisories", "description": "Check visa/safety advisories for a country.", "parameters": {"type": "object", "properties": {"country": {"type": "string"}}, "required": ["country"]}}},
    {"type": "function", "function": {"name": "validate_budget", "description": "Validate itinerary fits budget.", "parameters": {"type": "object", "properties": {"flight_cost": {"type": "number"}, "hotel_cost": {"type": "number"}, "other_costs": {"type": "number"}, "budget": {"type": "number"}}, "required": ["flight_cost", "hotel_cost", "budget"]}}},
]

def _execute_tool(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        args = {}
    handler = _MOCK_HANDLERS.get(name)
    return handler(args) if handler else json.dumps({"error": f"Unknown tool: {name}"})


# ═══════════════════════════════════════════════════════════════
# Approach A: LangGraph agent (OTel auto-instrumented)
# ═══════════════════════════════════════════════════════════════

def _build_langgraph_agent():
    """Build a LangGraph travel planner with mock tools + AzureChatOpenAI.

    OpenInference auto-instrumentation captures every LLM call, tool call,
    and node routing decision as OTel spans.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
    from langchain_core.tools import tool as lc_tool
    from langchain_openai import AzureChatOpenAI
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from typing import TypedDict

    # LangChain tool wrappers over mock handlers
    @lc_tool
    def search_flights(destination: str, max_price: float = 5000) -> str:
        """Search for flights to a destination within a budget."""
        return _mock_search_flights({"destination": destination, "max_price": max_price})

    @lc_tool
    def search_hotels(city: str, max_nightly_rate: float = 300) -> str:
        """Search for hotels in a city."""
        return _mock_search_hotels({"city": city, "max_nightly_rate": max_nightly_rate})

    @lc_tool
    def check_weather(city: str) -> str:
        """Check weather forecast for a destination city."""
        return _mock_check_weather({"city": city})

    @lc_tool
    def check_travel_advisories(country: str) -> str:
        """Check visa requirements, safety advisories, and health precautions."""
        return _mock_check_advisories({"country": country})

    @lc_tool
    def validate_budget(flight_cost: float, hotel_cost: float, other_costs: float = 0, budget: float = 5000) -> str:
        """Validate that an itinerary fits the user's budget."""
        return _mock_validate_budget({"flight_cost": flight_cost, "hotel_cost": hotel_cost, "other_costs": other_costs, "budget": budget})

    tools = [search_flights, search_hotels, check_weather, check_travel_advisories, validate_budget]
    tool_node = ToolNode(tools)

    llm = AzureChatOpenAI(
        azure_deployment="gpt-5.4-nano",
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

    Shared by all 3 approaches — same execution, different capture.
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

    # Extract final text and all tool calls from intermediate messages
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
    """LangGraph callable for Approach A. Returns str — the rich data comes
    from OTel auto-instrumentation capturing every LLM call, tool call,
    and node routing decision as spans (via LiveOTelExporter).
    """
    return _invoke_langgraph(message, history)["final_text"]


# ═══════════════════════════════════════════════════════════════
# Approach B: Callable wrapper (returns litellm.ModelResponse)
# ═══════════════════════════════════════════════════════════════

def target(message: str, history: Optional[list] = None):
    """Callable returning ModelResponse with tool call metadata.

    Same LangGraph execution as A, but captured via return type:
    - Tool names + arguments visible (from ModelResponse.tool_calls)
    - Model name + usage visible
    - But NO tool results, NO node routing path, NO per-step token counts
    - Judge sees 4/8 behaviors vs A's 8/8
    """
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
        model="gpt-5.4-nano",
        tool_calls=tool_call_objs,
    )
