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

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

tools = [customer_profiler, transaction_inspector, risk_scorer]

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


SYSTEM_PROMPT = """You are a fraud investigation agent. Given a transaction identifier, investigate it methodically:

1. Call transaction_inspector with the trans_num to get transaction facts.
2. Extract the cc_num from the inspector result.
3. Call customer_profiler with the cc_num to get the customer's baseline behavior.
4. Call risk_scorer with the trans_num and cc_num to get a quantified risk assessment.
5. Write your final summary using the risk_scorer output as your anchor.

When writing your summary:
- State the risk level and score first.
- For LOW risk (0-25): note the transaction appears normal, briefly mention any minor anomalies.
- For MEDIUM risk (26-50): flag as worth monitoring, explain which signals contributed.
- For HIGH risk (51-75): recommend investigation, cite specific evidence with numbers.
- For CRITICAL risk (76-100): urgent flag, detail every signal with exact figures.
- Always cite specific numbers from the tools. Never invent figures.
- Do NOT call a transaction suspicious unless the risk score supports it."""




def investigate(trans_num: str) -> str:
    result = app.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Investigate transaction {trans_num}")
        ]}
    )
    return result["messages"][-1].content

if __name__ == "__main__":
    print(investigate("397894a5c4c02e3c61c784001f0f14e4"))
