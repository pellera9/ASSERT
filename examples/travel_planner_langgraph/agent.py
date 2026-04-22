"""Multi-agent travel planner built with LangGraph + MCP tools.

This is the system under test for the approach comparison.
It represents a realistic production agent: multiple LLM-powered nodes,
conditional routing, external tool calls via MCP, and shared state.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Annotated, Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ── MCP tool setup ────────────────────────────────────────────
# In production these connect to real MCP servers (flight API, hotel API, etc.)
# For this example we use langchain-mcp-adapters to load tools from MCP servers.

from langchain_mcp_adapters.client import MultiServerMCPClient


MCP_SERVERS = {
    "flights": {
        "url": "http://localhost:3001/sse",
        "transport": "sse",
    },
    "hotels": {
        "url": "http://localhost:3002/sse",
        "transport": "sse",
    },
    "weather": {
        "url": "http://localhost:3003/sse",
        "transport": "sse",
    },
}


# ── Graph state ───────────────────────────────────────────────

class TravelState(dict):
    """Shared state across all nodes in the travel planning graph."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str
    destination: str
    dates: dict[str, str]
    budget: float
    flights: list[dict[str, Any]]
    hotels: list[dict[str, Any]]
    itinerary: str


# ── Node implementations ─────────────────────────────────────

async def intent_classifier(state: TravelState) -> dict:
    """Classify user intent and extract travel parameters."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = state.get("messages", [])

    response = await llm.ainvoke([
        {"role": "system", "content": (
            "You are a travel intent classifier. Extract: intent (book_trip, "
            "modify_trip, cancel_trip, ask_question), destination, dates, budget. "
            "Respond as JSON: {\"intent\": ..., \"destination\": ..., "
            "\"dates\": {\"start\": ..., \"end\": ...}, \"budget\": ...}"
        )},
        *messages,
    ])

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        parsed = {"intent": "ask_question"}

    return {
        "messages": [response],
        "intent": parsed.get("intent", "ask_question"),
        "destination": parsed.get("destination", ""),
        "dates": parsed.get("dates", {}),
        "budget": parsed.get("budget", 0),
    }


async def flight_search(state: TravelState) -> dict:
    """Search for flights using MCP flight tool."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    async with MultiServerMCPClient(MCP_SERVERS) as client:
        tools = client.get_tools()
        flight_tools = [t for t in tools if "flight" in t.name.lower()]
        llm_with_tools = llm.bind_tools(flight_tools)

        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "Search for flights. Use available tools."},
            {"role": "user", "content": (
                f"Find flights to {state.get('destination', 'unknown')} "
                f"from {state.get('dates', {}).get('start', 'flexible')} "
                f"to {state.get('dates', {}).get('end', 'flexible')} "
                f"budget ${state.get('budget', 'any')}"
            )},
        ])

        # Execute tool calls if any
        results = []
        if response.tool_calls:
            tool_node = ToolNode(flight_tools)
            tool_results = await tool_node.ainvoke({"messages": [response]})
            results = tool_results.get("messages", [])

    return {
        "messages": [response, *results],
        "flights": _extract_options(results, "flight"),
    }


async def hotel_search(state: TravelState) -> dict:
    """Search for hotels using MCP hotel tool."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    async with MultiServerMCPClient(MCP_SERVERS) as client:
        tools = client.get_tools()
        hotel_tools = [t for t in tools if "hotel" in t.name.lower()]
        llm_with_tools = llm.bind_tools(hotel_tools)

        response = await llm_with_tools.ainvoke([
            {"role": "system", "content": "Search for hotels. Use available tools."},
            {"role": "user", "content": (
                f"Find hotels in {state.get('destination', 'unknown')} "
                f"from {state.get('dates', {}).get('start', 'flexible')} "
                f"to {state.get('dates', {}).get('end', 'flexible')} "
                f"budget ${state.get('budget', 'any')}"
            )},
        ])

        results = []
        if response.tool_calls:
            tool_node = ToolNode(hotel_tools)
            tool_results = await tool_node.ainvoke({"messages": [response]})
            results = tool_results.get("messages", [])

    return {
        "messages": [response, *results],
        "hotels": _extract_options(results, "hotel"),
    }


async def itinerary_optimizer(state: TravelState) -> dict:
    """Combine flights + hotels into an optimized itinerary."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    flights_summary = json.dumps(state.get("flights", []), indent=2)
    hotels_summary = json.dumps(state.get("hotels", []), indent=2)

    response = await llm.ainvoke([
        {"role": "system", "content": (
            "You are a travel itinerary optimizer. Given flight and hotel options, "
            "create the best itinerary within the user's budget. Be specific about "
            "prices, times, and recommendations. If options are limited, say so."
        )},
        {"role": "user", "content": (
            f"Destination: {state.get('destination')}\n"
            f"Dates: {json.dumps(state.get('dates', {}))}\n"
            f"Budget: ${state.get('budget', 'flexible')}\n\n"
            f"Flight options:\n{flights_summary}\n\n"
            f"Hotel options:\n{hotels_summary}\n\n"
            "Create the best itinerary."
        )},
    ])

    return {
        "messages": [response],
        "itinerary": response.content,
    }


async def clarification(state: TravelState) -> dict:
    """Ask user for missing information."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

    response = await llm.ainvoke([
        {"role": "system", "content": (
            "The user's travel request is missing key details. "
            "Ask a clear, specific follow-up question to get what you need."
        )},
        *state.get("messages", []),
    ])

    return {"messages": [response]}


# ── Routing logic ─────────────────────────────────────────────

def route_after_intent(state: TravelState) -> str:
    """Route based on classified intent."""
    intent = state.get("intent", "ask_question")
    destination = state.get("destination", "")

    if intent == "book_trip" and destination:
        return "flight_search"
    elif intent in ("book_trip", "modify_trip") and not destination:
        return "clarification"
    else:
        return "clarification"


def route_after_itinerary(state: TravelState) -> str:
    """Decide if itinerary is complete or needs more info."""
    itinerary = state.get("itinerary", "")
    if itinerary and len(itinerary) > 50:
        return END
    return "clarification"


# ── Build the graph ───────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct the travel planner graph."""
    graph = StateGraph(TravelState)

    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("flight_search", flight_search)
    graph.add_node("hotel_search", hotel_search)
    graph.add_node("itinerary_optimizer", itinerary_optimizer)
    graph.add_node("clarification", clarification)

    graph.set_entry_point("intent_classifier")

    graph.add_conditional_edges("intent_classifier", route_after_intent)
    graph.add_edge("flight_search", "hotel_search")
    graph.add_edge("hotel_search", "itinerary_optimizer")
    graph.add_conditional_edges("itinerary_optimizer", route_after_itinerary)
    graph.add_edge("clarification", END)

    return graph.compile()


# ── Convenience entry point ───────────────────────────────────

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


async def chat(message: str) -> str:
    """Single-turn entry point: send a message, get a response."""
    graph = get_graph()
    result = await graph.ainvoke({
        "messages": [HumanMessage(content=message)],
    })
    messages = result.get("messages", [])
    # Return the last AI message
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def chat_sync(message: str) -> str:
    """Synchronous wrapper for chat()."""
    return asyncio.run(chat(message))


# ── Helpers ───────────────────────────────────────────────────

def _extract_options(tool_messages: list, option_type: str) -> list[dict]:
    """Extract structured options from tool result messages."""
    options = []
    for msg in tool_messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if isinstance(data, list):
                    options.extend(data)
                elif isinstance(data, dict):
                    options.append(data)
            except (json.JSONDecodeError, TypeError):
                pass
    return options
