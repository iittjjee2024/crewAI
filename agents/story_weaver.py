from crewai import Agent

def create_story_weaver(config, llm):
    return Agent(
        config=config,
        verbose=True,
        allow_delegation=False,
        llm=llm
    )
