from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.tools.customer_profiler import customer_profiler
from src.tools.transaction_inspector import transaction_inspector

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

tools = [customer_profiler, transaction_inspector]

llm = ChatGroq(
    model = "llama-3.3-70b-versatile", temperature = 0
).bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    response = llm.invoke(state['messages'])
    return {'messages': [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> str:
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node('agent', agent_node)
graph.add_node('tools', tool_node)
graph.set_entry_point('agent')
graph.add_conditional_edges('agent', should_continue, {'tools': 'tools', END:END})
graph.add_edge('tools', 'agent')

app = graph.compile()

SYSTEM_PROMPT = """You are a fraud investigation agent at a bank. Give a transaction ID, investigate if the transation is suspicious.
Your Process:
1. First, inspect the transaction details using transaction_inspector.
2. Use the cc_num from the result to look up the customer's baseline using the customer_profiler.
3. Compare the transaction against the customer's normal behaviour.
4. Write a brief investigation summary: what you found, what's normal vs abnormal and your assessment.

Be specific - cite actual numbers from the tools. Do not guess. """

def investigate(trans_num: str) -> str:
    result = app.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Investigate transaction {trans_num}")
        ]}
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    print(investigate("e8a81877ae9a0a7f883e15cb39dc4022"))
