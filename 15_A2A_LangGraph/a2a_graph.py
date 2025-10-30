import asyncio
import logging
from typing import Dict, Any, Annotated, List, TypedDict, Literal
from uuid import uuid4

# --- 1. Imports for LangGraph and LangChain ---
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field

# *** NEW IMPORT: Replaced Placeholder with Real OpenAI Model ***
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv


import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import AgentCard, MessageSendParams, SendMessageRequest
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, EXTENDED_AGENT_CARD_PATH


load_dotenv()
# --- 3. Agent State Definition ---
class AgentState(TypedDict):
    """State schema for the LangGraph. Messages are appended automatically."""
    messages: Annotated[List[BaseMessage], add_messages]


# --- 4. The Real A2A Client Tool ---

async def _async_a2a_send_message(query: str, base_url: str = 'http://localhost:10000') -> str:
    """The asynchronous core logic to send a message via A2A protocol."""
    
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("A2ATool")
    logger.setLevel(logging.INFO)

    final_response_text = "ERROR: A2A Agent communication failed."
    
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
            final_agent_card_to_use: AgentCard | None = None
            
            # --- Fetch Agent Card ---
            _public_card = await resolver.get_agent_card()
            final_agent_card_to_use = _public_card
            
            # --- Initialize Client and Send Message ---
            client = A2AClient(httpx_client=httpx_client, agent_card=final_agent_card_to_use)

            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': query}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

           

            response = await client.send_message(request)
            print("PRINTING RAW RESPONSE")

            print(response)
            # Extract the final response text
            result_message = response.root.result.artifacts[0]
            
            if result_message and result_message.parts:
                for part in result_message.parts:
                    if part.root and part.root.kind == 'text' and part.root.text:
                        final_response_text = part.root.text
                        break
            else:
                 final_response_text = f"A2A Agent responded but message structure was unexpected: {response.model_dump_json(indent=2)}"


    except Exception as e:
        logger.error(f'Error during A2A message send to {base_url}: {type(e).__name__}: {str(e)}', exc_info=True)
        final_response_text = f"ERROR communicating with A2A Agent at {base_url}. Is the service running? Details: {type(e).__name__}: {str(e)}"

    return final_response_text


@tool
def a2a_agent_tool(query: str) -> str:
    """
    Sends a query to the external A2A Agent running at http://localhost:10000 
    and returns the agent's final text response. Use this for complex queries requiring
    the specialized A2A Agent's capabilities.
    """
    print(f"\n--- A2A TOOL INVOKED ---")
    print(f"Query sent to A2A Agent: {query}")
    
    # Executes the async A2A client logic synchronously within the tool's execution.
    result = asyncio.run(_async_a2a_send_message(query))
    
    print(f"--- A2A TOOL FINISHED ---")
    
    return result


# --- 5. Initialize the Real LLM for Tool Calling (OpenAI) ---


llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Get the available tools and bind them to the LLM
tools = [a2a_agent_tool]
llm_with_tools = llm.bind_tools(tools)


# --- 6. LangGraph Nodes and Router ---

def agent_node(state: AgentState) -> Dict[str, Any]:
    """The main agent's decision-making node (LLM invocation)."""
    # The LLM determines if it should call the tool or give a final answer
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def route_to_action_or_end(state: AgentState) -> str:
    """Decide whether to execute tools or finish the conversation."""
    last_message: AIMessage = state["messages"][-1]
    # Check if the LLM's response included tool_calls
    if getattr(last_message, "tool_calls", None):
        return "action"
    # If not, the LLM generated the final answer
    return END

# --- 7. Build and Compile the LangGraph ---

def build_a2a_user_agent_graph(checkpointer=None):
    """
    Builds a LangGraph agent that uses an A2A Agent as a tool.
    """
    graph = StateGraph(AgentState)
    
    tool_node = ToolNode(tools)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("action", tool_node)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Conditional edge from agent: tool call or end
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_end,
        {"action": "action", END: END},
    )
    
    # Edge from action back to agent for processing tool output
    graph.add_edge("action", "agent")
    
    return graph.compile(checkpointer=checkpointer)

# --- 8. Example Usage (Run this locally) ---
if __name__ == '__main__':
    # Initialize the graph
    app_graph = build_a2a_user_agent_graph()

    # Query that the LLM should recognize as needing the A2A tool
    query_a2a_needed = "Can you use your specialized agent to find the latest developments in large language models for me?"
    
    config = {"configurable": {"thread_id": "openai_a2a_call_thread"}}
    
    print(f"--- Running Query: {query_a2a_needed} (Will attempt real A2A call) ---")
    inputs = {"messages": [HumanMessage(content=query_a2a_needed)]}
    
    # Run the graph in stream mode
    try:
        for s in app_graph.stream(inputs, config):
            print(s)
    except ImportError:
        print("\n\nERROR: Necessary libraries ('a2a' or 'httpx') are not installed. Cannot run the A2A client tool.")
    except Exception as e:
        print(f"\nAN ERROR OCCURRED DURING GRAPH EXECUTION: {e}")
        print("\nEnsure your A2A service is running on http://localhost:10000 and the OPENAI_API_KEY is set.")