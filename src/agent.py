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
from src.tools.risk_scorer import risk_scorer
from src.tools.report_generator import report_generator

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

tools = [customer_profiler, transaction_inspector, risk_scorer, report_generator]

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


SYSTEM_PROMPT = """You are a fraud investigation agent. You investigate one transaction at a time.

Follow this sequence:
1. Call transaction_inspector with the trans_num to get transaction facts.
2. Extract cc_num from the inspector output.
3. Call customer_profiler with the cc_num to get the customer baseline.
4. Call risk_scorer with the trans_num and cc_num to get the risk assessment.
5. Call report_generator with the trans_num and cc_num to get the structured report.
6. Write a ONE-SENTENCE summary of your findings, referencing the risk_level and the strongest signal from the report.

Do NOT skip steps. Do NOT invent numbers — use only what the tools return."""




def investigate(trans_num: str) -> str:
    result = app.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Investigate transaction {trans_num}")
        ]}
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    print(investigate("d96c1d6c9551870ef62686b070a8e7db"))
