from crewai import Agent
from tools import google_search_tool

def create_researcher(config, llm):
    return Agent(
        config=config,
        verbose=True,
        allow_delegation=False,
        tools=[google_search_tool],
        llm=llm
    )
