from crewai import Task


def create_tasks(configs, car_buyer, story_prompt: str):
    """
    Creates and returns tasks for the car buying pipeline.
    """

    # Car buying task
    car_buying_task = Task(
        config=configs['car_buying_task'],
        agent=car_buyer,
        # Format the description dynamically since it has a placeholder {story_prompt}
        description=configs['car_buying_task']['description'].format(story_prompt=story_prompt)
    )

    return [car_buying_task]