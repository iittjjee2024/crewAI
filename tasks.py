from crewai import Task

def create_tasks(configs, researcher, architect, moral_guide, story_weaver, validator, story_prompt: str):
    """
    Creates and returns the 5 sequential tasks by mapping YAML configurations to agents.
    """
    
    # Task 1: Research
    research_task = Task(
        config=configs['research_task'],
        agent=researcher,
        # Format the description dynamically since it has a placeholder {story_prompt}
        description=configs['research_task']['description'].format(story_prompt=story_prompt)
    )
    
    # Task 2: Blueprint
    blueprint_task = Task(
        config=configs['blueprint_task'],
        agent=architect,
        context=[research_task]
    )
    
    # Task 3: Moral Lesson
    moral_task = Task(
        config=configs['moral_task'],
        agent=moral_guide,
        context=[blueprint_task]
    )
    
    # Task 4: Story Writing
    story_task = Task(
        config=configs['story_task'],
        agent=story_weaver,
        context=[blueprint_task, moral_task]
    )
    
    # Task 5: Validation
    validation_task = Task(
        config=configs['validation_task'],
        agent=validator,
        context=[story_task]
    )
    
    return [research_task, blueprint_task, moral_task, story_task, validation_task]
