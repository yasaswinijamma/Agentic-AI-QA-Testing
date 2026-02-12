import os
from typing import TypedDict
from dotenv import load_dotenv

# LangChain & LangGraph
from langgraph.graph import StateGraph, START, END
# Updated Import Location for newer LangGraph versions
from langgraph.constants import CONF
from langchain_openai import ChatOpenAI
from langchain_core.language_models.fake import FakeListLLM

# DeepEval (The Evaluation Framework)
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

# 1. Configuration
load_dotenv()

# --- TOGGLE THIS ---
# Change to True to test for free / Change to False to use real OpenAI
USE_MOCK = True 
# -------------------

class AgentState(TypedDict):
    query: str
    research_notes: str
    final_draft: str

# 2. Initialize the "Brain"
if USE_MOCK:
    print("⚠️  MOCK MODE: Simulating LLM responses (Free)")
    llm = FakeListLLM(responses=[
        "Autonomous agents use LLMs to plan and execute tasks independently.",
        "Summary: AI agents are independent systems driven by LLM reasoning."
    ])
else:
    print("🚀 PRODUCTION MODE: Connecting to OpenAI")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 3. Define Agent Nodes
def researcher_node(state: AgentState):
    print("--- 🔍 Step 1: Researcher agent gathering data ---")
    prompt = f"Facts about: {state['query']}"
    response = llm.invoke(prompt)
    content = response if isinstance(response, str) else response.content
    return {"research_notes": content}

def writer_node(state: AgentState):
    print("--- ✍️  Step 2: Writer agent synthesizing summary ---")
    prompt = f"Summarize: {state['research_notes']}"
    response = llm.invoke(prompt)
    content = response if isinstance(response, str) else response.content
    return {"final_draft": content}

# 4. Orchestration (The Agent Graph)
workflow = StateGraph(AgentState)

# Define retry logic as a simple dictionary (Universal compatibility)
# This handles the 429 errors by retrying 3 times
retry_config = {
    "retry_on": Exception,
    "max_attempts": 3,
    "backoff_factor": 2.0
}

workflow.add_node("researcher", researcher_node, retry=retry_config)
workflow.add_node("writer", writer_node, retry=retry_config)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# 5. QA Test Suite
def run_agentic_test():
    test_query = "How do autonomous agents work?"
    print(f"\n[TEST START] Query: {test_query}\n" + "="*40)
    
    # Run the Agentic Workflow
    result = app.invoke({"query": test_query})
    actual_output = result.get("final_draft", "No output generated")
    
    print(f"FINAL AGENT OUTPUT: {actual_output}")

    # 6. DeepEval Validation (Only if real LLM is used)
    if not USE_MOCK and os.getenv("OPENAI_API_KEY"):
        print("\n--- ✅ Validating Accuracy (DeepEval) ---")
        metric = AnswerRelevancyMetric(threshold=0.5)
        test_case = LLMTestCase(input=test_query, actual_output=actual_output)
        assert_test(test_case, [metric])
        print("QA Check: PASSED")
    else:
        print("\nQA Check: Skipped (Mock Mode or Missing API Key)")

if __name__ == "__main__":
    run_agentic_test()