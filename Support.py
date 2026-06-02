import os
import operator
from typing import Annotated, TypedDict
from dotenv import load_dotenv
import streamlit as st

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

# -----------------------------
# API KEY
# -----------------------------

st.sidebar.header("ðŸ”‘ API Configuration")

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Create the UI Input field
st.sidebar.header("🔑 API Configuration")
api_key_input = st.sidebar.text_input("Enter Gemini API Key", type="password")

# 2. Prevent the application from sending a request if the field is empty
if not api_key_input:
    st.info("Please enter your Gemini API Key to proceed.")
else:
    # 3. Explicitly pass the user-provided text directly to the model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key_input  # Forces the use of the typed key string
        )
        
        # Run your chain execution safely below...
        
    except Exception as e:
        st.error(f"Initialization error: {e}")

# =====================================================================
# 1. CORE GRAPH STATE SCHEMATICS
# =====================================================================
class SupportAgentState(TypedDict):
    """Encapsulates telemetry state parameters across node transition pathways."""
    messages: Annotated[list[BaseMessage], operator.add]
    category: str          # "technical", "billing", or "general"
    has_error: bool        # Operational circuit breaker status flag
    error_log: str         # Captured exception raw trace strings


# =====================================================================
# 2. STATE LOGIC NODES (WORKERS)
# =====================================================================

def router_node(state: SupportAgentState):
    """Leverages LLM structural intent analysis to dictate execution steps."""
    latest_user_message = state["messages"][-1].content
    
    prompt = f"""You are a structural intent classification router. 
    Analyze this user request: '{latest_user_message}'
    Categorize it into exactly ONE of these options: 'technical', 'billing', or 'general'.
    Response formatting requirement: Output ONLY the lowercase word. Zero punctuation. Do not explain.
    """
    
    response = llm.invoke([HumanMessage(content=prompt)])
    category = response.content.strip().lower()
    
    # Structural safety layer for unexpected model variations
    if category not in ["technical", "billing", "general"]:
        category = "general"
        
    return {"category": category, "has_error": False, "error_log": ""}


def execution_node(state: SupportAgentState):
    """Executes business transactions or catches structural environment errors."""
    category = state["category"]
    latest_user_message = state["messages"][-1].content
    
    try:
        # SIMULATION: Mimics a critical production database timeout on billing routes
        if category == "billing":
            raise ConnectionError("Billing API Failure: Connection pool leak detected on microservice tier.")
            
        # Successful operational path running through standard generation
        prompt = f"Provide a brief, supportive customer care resolution response for a {category} issue regarding: {latest_user_message}"
        response = llm.invoke([HumanMessage(content=prompt)])
        
        return {
            "messages": [AIMessage(content=response.content)],
            "has_error": False
        }
        
    except Exception as system_fault:
        # Intercept live failures, preserving the state variables without crashing the application
        return {
            "has_error": True,
            "error_log": str(system_fault)
        }


def error_mitigation_node(state: SupportAgentState):
    """Executes recovery procedures when the execution circuit breaker trips."""
    failed_category = state["category"]
    internal_error = state["error_log"]
    
    # Craft a resilient response that logs details while protecting user experience
    safe_remediation_text = (
        f"🚨 [System Notice: Circuit Breaker Recovered] Our internal {failed_category} processing "
        "subsystems are undergoing unexpected database maintenance. Your event log has been systematically "
        f"preserved and prioritized for engineering triage. Debug trace: '{internal_error}'"
    )
    
    return {
        "messages": [AIMessage(content=safe_remediation_text)],
        "has_error": False,  # Resiliency mechanism completely resolves the state error flag
        "error_log": ""
    }


# =====================================================================
# 3. CONDITIONAL ROUTING MAPS
# =====================================================================
def evaluate_runtime_health(state: SupportAgentState):
    """Evaluates error conditions to branch execution to recovery paths."""
    if state.get("has_error"):
        return "remediate"
    return "terminate"


# =====================================================================
# 4. GRAPH ASSEMBLY & ACTIVE COMPILATION
# =====================================================================
workflow_builder = StateGraph(SupportAgentState)

# Register active state node structures
workflow_builder.add_node("intent_router", router_node)
workflow_builder.add_node("execution_handler", execution_node)
workflow_builder.add_node("error_mitigator", error_mitigation_node)

# Map hardwired execution transitions
workflow_builder.set_entry_point("intent_router")
workflow_builder.add_edge("intent_router", "execution_handler")

# Inject adaptive routing path conditional checks
workflow_builder.add_conditional_edges(
    "execution_handler",
    evaluate_runtime_health,
    {
        "remediate": "error_mitigator",
        "terminate": END
    }
)

# Connect remediation logic securely to termination node
workflow_builder.add_edge("error_mitigator", END)

# Compile the workflow configuration into a production runtime object
compiled_engine = workflow_builder.compile()


# =====================================================================
# 5. ENTERPRISE STREAMLIT USER INTERFACE
# =====================================================================
st.set_page_config(page_title="SentineIFlow AI", page_icon="♊", layout="centered")

st.title("♊ Intelligent Customer Support Workflow Automation")
st.caption("Orchestrated via LangGraph Framework utilizing gemini-2.5-flash")
st.write("Demonstrating real-time error interception, structured tracking, and automatic recovery workflows.")

# Initialize persistent session tracking structures
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Redraw historic messages on frame refreshes
for msg in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

# Intercept and process raw conversational input strings
if user_input := st.chat_input("Enter your support query..."):
    
    # Display the user input instantly
    with st.chat_message("user"):
        st.markdown(user_input)
    
    user_msg_wrapper = HumanMessage(content=user_input)
    st.session_state.chat_history.append(user_msg_wrapper)
    
    # Pack parameters into initial state structure payload
    graph_payload = {"messages": [user_msg_wrapper]}
    
    with st.spinner("Executing backend LangGraph nodes..."):
        # Execute the graph processing loop synchronously 
        terminal_execution_state = compiled_engine.invoke(graph_payload)
    
    # Parse final payload assets out of state variables
    system_response = terminal_execution_state["messages"][-1]
    runtime_intent = terminal_execution_state.get("category", "undefined")
    
    # Render system status metrics cleanly inside the app sidebar
    with st.sidebar:
        st.subheader("📊 Graph Operations Metrics")
        st.info(f"**Intent Categorization:** {runtime_intent.upper()}")
        if "Circuit Breaker" in system_response.content:
            st.warning("⚠️ **Circuit Gate:** Tripped & Healed")
        else:
            st.success("✅ **Circuit Gate:** Clean Pass")
            
    # Print the system's generated output string
    with st.chat_message("assistant"):
        st.markdown(system_response.content)
        
    st.session_state.chat_history.append(system_response)
