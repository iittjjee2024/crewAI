from crewai import Agent

def create_architect(config, llm):
    return Agent(
        config=config,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
