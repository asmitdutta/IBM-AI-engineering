import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_mcp_adapters.tools import load_mcp_tools

# MCP server launch config
server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"]
)

# LangGraph state definition
class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]

# --- Subclass ClientSession to Handle Notifications ---
class LoggingClientSession(ClientSession):
    """
    A custom ClientSession that overrides the notification handler
    to print log messages from the server.
    """
    async def _received_notification(self, notification: types.JSONRPCNotification) -> None:
        """
        This method is automatically called by the base class's internal
        event loop whenever a notification is received.
        """
        # The library pre-parses the notification. The actual notification object
        # is in the .root attribute of the object passed to this handler
        if hasattr(notification, 'root') and notification.root.method == "notifications/message":
            try:
                # The `notification.root.params` object is already a parsed Pydantic model
                # We can access its attributes like `level` and `data` directly
                log_params = notification.root.params
                level = log_params.level.upper()
                message = log_params.data
                print(f"[{level} from Server]: {message}")
            except AttributeError as e:
                print(f"[CLIENT PARSE ERROR]: Log message params have an unexpected structure: {notification.root.params}. Error: {e}")
            except Exception as e:
                print(f"[CLIENT PARSE ERROR]: An unexpected error occurred while processing log message: {e}")


async def create_graph(session):
    # Load tools from MCP server
    tools = await load_mcp_tools(session)

    # LLM configuration (system prompt can be added later)
    llm = ChatOllama(model="llama3.2", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Prompt template with user/assistant chat only
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful RAG assistant. Your role is to answer questions using the content of documents provided by the user. When a user gives you a URL path, use your tool to ingest it into your memory. When they ask a question, use your search tool to find the relevant context within the ingested documents and use that context to form a clear answer."),
        MessagesPlaceholder("messages")
    ])

    chat_llm = prompt_template | llm_with_tools

    # Define chat node
    def chat_node(state: State) -> State:
        state["messages"] = chat_llm.invoke({"messages": state["messages"]})
        return state

    # Build LangGraph with tool routing
    graph = StateGraph(State)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tool_node", ToolNode(tools=tools))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition, {
        "tools": "tool_node",
        "__end__": END
    })
    graph.add_edge("tool_node", "chat_node")

    return graph.compile(checkpointer=MemorySaver())


# Entry point
async def main():
    async with stdio_client(server_params) as (read, write):
        async with LoggingClientSession(read, write) as session:
            await session.initialize()

            agent = await create_graph(session)
            # --- Print statement for clarity ---
            print("""
            Hello! I'm your document assistant.

            To get started, tell me which document to read by providing its URL, like:
                Example: ingest_document https://example.com/employee_handbook.txt

            Once it's loaded, I can answer any questions you have about it!
            """)


            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                try:
                    response = await agent.ainvoke(
                        {"messages": user_input},
                        config={"configurable": {"thread_id": "weather-session"}}
                    )
                    print("AI:", response["messages"][-1].content)
                except Exception as e:
                    print("Error:", e)


if __name__ == "__main__":
    asyncio.run(main())
