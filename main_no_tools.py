import os
import yaml
import argparse
from dotenv import load_dotenv
from crewai import Crew, Process, LLM

# Import modular agents and tasks
from agents.researcher import create_researcher
from agents.architect import create_architect
from agents.moral_guide import create_moral_guide
from agents.story_weaver import create_story_weaver
from agents.validator import create_validator
from tasks import create_tasks


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Check for required API keys
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key or openrouter_api_key == "your_openrouter_api_key_here":
        print("Error: OPENROUTER_API_KEY is missing or not configured in .env")
        return

    # Configure the LLM to use OpenRouter's free models via CrewAI's native LLM class
    llm = LLM(
        model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Parse command line arguments for the story prompt
    parser = argparse.ArgumentParser(
        description="Car Buying Assistant using CrewAI"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Find me a reliable used sedan within my budget.",
        help="The car buying prompt to generate",
    )
    args = parser.parse_args()

    story_prompt = args.prompt
    print(f"\\n=== Starting Car Buying Pipeline ===")
    print(f"Prompt: '{story_prompt}'\\n")

    # Load YAML configs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agents_config = load_yaml(os.path.join(base_dir, "config", "agents.yaml"))
    tasks_config = load_yaml(os.path.join(base_dir, "config", "tasks.yaml"))

    # 1. Initialize Agents
    car_buyer = create_car_buyer(agents_config["car_buyer"], llm)

    # ------------------------------------------------------------------
    # HOTFIX: OpenRouter's FREE tier models do not support Tool Calling.
    # We must explicitly clear the tools list so the Researcher uses its
    # internal knowledge instead of crashing OpenRouter's API.
    # ------------------------------------------------------------------
    car_buyer.tools = []

    architect = create_architect(agents_config["blueprint_architect"], llm)
    moral_guide = create_moral_guide(agents_config["moral_guide"], llm)
    story_weaver = create_story_weaver(agents_config["story_weaver"], llm)
    validator = create_validator(agents_config["quality_validator"], llm)

    # 2. Initialize Tasks
    tasks = create_tasks(
        tasks_config,
        car_buyer,
        architect,
        moral_guide,
        story_weaver,
        validator,
        story_prompt,
    )

    # 3. Create the Crew
    story_crew = Crew(
        agents=[car_buyer, architect, moral_guide, story_weaver, validator],
        tasks=tasks,
        process=Process.sequential,  # Tasks will be executed one after the other
        verbose=True,
    )

    # 4. Kickoff the process
    print("Initiating CrewAI sequence...\\n")
    result = story_crew.kickoff()

    # 5. Output the result
    print("\\n==================================================")
    print("============= FINAL CAR RECOMMENDATIONS ==============")
    print("==================================================\\n")
    print(result)
    print("\\n==================================================")


if __name__ == "__main__":
    main()
