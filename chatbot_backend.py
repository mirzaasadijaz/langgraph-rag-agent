from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from prompt import system_instruction
from tools import tools_list

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()

# Main LLM
groq_llm = init_chat_model(
    model="openai/gpt-oss-20b", 
    model_provider="groq",
    temperature=0 
)

gemini_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

llm_with_tools = groq_llm.bind_tools(tools_list)

def chat_node(state: ChatState):
    messages = state['messages']

    # Prepend the system prompt to enforce rules (including the language rule)
    full_messages = [system_instruction] + messages

    response = llm_with_tools.invoke(full_messages)

    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

memory = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
tool_node = ToolNode(tools=tools_list)
graph.add_node("tools", tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)