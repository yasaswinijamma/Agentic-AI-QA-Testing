# Agentic AI QA Framework - Multi-Agent Testing

This repository demonstrates a production-grade testing harness for agentic AI systems using **LangGraph** for orchestration and **DeepEval** for LLM-as-a-judge validation.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph (StateGraph)
* **LLM Integration:** LangChain / OpenAI
* **Validation:** DeepEval (Semantic Relevancy)
* **Resiliency:** Exponential Backoff & Retry Policies

## 🚀 Key Features Demonstrated
### 1. Multi-Agent Orchestration
The system utilizes a `StateGraph` to manage transitions between a **Researcher Agent** and a **Writer Agent**, ensuring state consistency across the workflow.

### 2. Resiliency & Reliability
Addressing the challenges of non-deterministic systems, the framework implements a custom `retry_config` to handle transient failures (e.g., OpenAI 429 Rate Limits) using exponential backoff.

### 3. Cost-Effective Simulation
Includes a **Mock Mode** using `FakeListLLM`. This allows for high-velocity CI/CD testing without incurring API costs or dependency on external LLM availability.

### 4. Semantic Validation
Integrated **DeepEval's AnswerRelevancyMetric** to move beyond exact-match assertions, grading the agent's output based on semantic alignment with the original user intent.

## 💻 How to Run
1. Activate environment: `source venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run test: `python code.py`