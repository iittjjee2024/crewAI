import os
import yaml
import argparse
from dotenv import load_dotenv
from crewai import Crew, Process, LLM

# ── LiteLLM compatibility patch ───────────────────────────────────────────────
# LiteLLM 1.67+ injects a 'cache_breakpoint' field inside message objects for
# Anthropic's prompt-caching feature. Groq's API does NOT support this field
# and returns a 400 error. We patch litellm.completion() to strip it out
# before the request is sent, so the rest of the code stays clean.
import litellm
_original_completion = litellm.completion
def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                msg.pop("cache_breakpoint", None)
    return _original_completion(*args, **kwargs)
litellm.completion = _patched_completion
# ─────────────────────────────────────────────────────────────────────────────

# Import modular agents and tasks
from agents.researcher import create_researcher
from agents.architect import create_architect
from agents.moral_guide import create_moral_guide
from agents.story_weaver import create_story_weaver
from agents.validator import create_validator
from tasks import create_tasks


def load_yaml(file_path):
    """Helper to load a YAML configuration file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def configure_llm():
    """
    Configure the LLM. Tries Groq first (free, fast, supports tool calling),
    then falls back to OpenRouter if Groq key is missing.

    WHY GROQ?
    - Groq offers a genuinely free API with very generous rate limits.
    - It supports Tool Calling (function calling), which is needed for web search.
    - It is much faster and more reliable than OpenRouter's free tier.

    Get your free Groq key at: https://console.groq.com
    """

    groq_api_key = os.getenv("GROQ_API_KEY")

    if groq_api_key and groq_api_key != "your_groq_api_key_here":
        print("✅ Using Groq (free tier, supports tool calling)...")
        # Groq uses its own base URL. 'groq/' prefix tells CrewAI to use it.
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=groq_api_key,
            drop_params=True,   # drop unsupported params (e.g. cache_breakpoint)
        )
    
    # Fallback: OpenRouter (reliable but free tier has stricter limits)
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_api_key and openrouter_api_key != "your_openrouter_api_key_here":
        print("⚠️  Groq key not set. Falling back to OpenRouter (no tool calling on free tier)...")
        return LLM(
            model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    
    raise ValueError(
        "No API key found! Please add GROQ_API_KEY to your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )


def main():
    # ── 0. Load .env ──────────────────────────────────────────────────────────
    load_dotenv()

    # ── 1. Set up the LLM ─────────────────────────────────────────────────────
    llm = configure_llm()

    # ── 2. Parse the story prompt ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Mythological Story Generator using CrewAI"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The Tale of young Hercules and his first trial",
        help="The mythological story prompt to generate",
    )
    args = parser.parse_args()

    story_prompt = args.prompt
    print(f"\n=== Starting Mythology Story Generation Pipeline ===")
    print(f"Prompt: '{story_prompt}'\n")

    # ── 3. Load YAML configs ──────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agents_config = load_yaml(os.path.join(base_dir, "config", "agents.yaml"))
    tasks_config = load_yaml(os.path.join(base_dir, "config", "tasks.yaml"))

    # ── 4. Initialize Agents ──────────────────────────────────────────────────
    researcher  = create_researcher(agents_config["mythology_researcher"], llm)
    architect   = create_architect(agents_config["blueprint_architect"], llm)
    moral_guide = create_moral_guide(agents_config["moral_guide"], llm)
    story_weaver = create_story_weaver(agents_config["story_weaver"], llm)
    validator   = create_validator(agents_config["quality_validator"], llm)

    # NOTE: If you are NOT using Groq (i.e. using OpenRouter's free tier),
    # the Researcher cannot use the Google Search tool because OpenRouter's
    # free tier does not support tool calling. In that case, uncomment this:
    # researcher.tools = []

    # ── 5. Initialize Tasks ───────────────────────────────────────────────────
    tasks = create_tasks(
        tasks_config,
        researcher,
        architect,
        moral_guide,
        story_weaver,
        validator,
        story_prompt,
    )

    # ── 6. Create the Crew ────────────────────────────────────────────────────
    story_crew = Crew(
        agents=[researcher, architect, moral_guide, story_weaver, validator],
        tasks=tasks,
        process=Process.sequential,  # Tasks run one after the other
        verbose=True,
    )

    # ── 7. Kick off and display results ───────────────────────────────────────
    print("Initiating CrewAI sequence...\n")
    result = story_crew.kickoff()

    print("\n==================================================")
    print("============= FINAL GENERATED STORY ==============")
    print("==================================================\n")
    print(result)
    print("\n==================================================")


if __name__ == "__main__":
    main()
