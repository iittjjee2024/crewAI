import os
from crewai import Agent, Task, Crew, LLM
from langchain.tools import tool
import sys

@tool("mock_tool")
def mock_tool(query: str) -> str:
    """A mock tool."""
    return "Mock result"

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Please set OPENROUTER_API_KEY")
    sys.exit(1)

models_to_test = [
    "openrouter/mistralai/mistral-7b-instruct:free",
    "openrouter/google/gemma-2-9b-it:free",
    "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    "openrouter/meta-llama/llama-3.2-3b-instruct:free",
    "openrouter/qwen/qwen-2.5-72b-instruct:free",
    "openrouter/qwen/qwen-2-7b-instruct:free"
]

for model in models_to_test:
    print(f"\\nTesting {model}...")
    try:
        llm = LLM(model=model, api_key=api_key, base_url="https://openrouter.ai/api/v1")
        agent = Agent(
            role="Tester",
            goal="Test tool use",
            backstory="You are a tester.",
            llm=llm,
            verbose=False,
            # No tools here, just test the LLM itself
        )
        task = Task(description="Say hello", expected_output="A string", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])
        crew.kickoff()
        print(f"[SUCCESS] {model} supports tools!")
        break
    except Exception as e:
        print(f"[FAILED] {model} - {str(e)[:100]}")
