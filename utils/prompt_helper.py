from pathlib import Path


def load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory."""
    prompts_dir = Path(__file__).parent
    prompt_path = prompts_dir / f"{prompt_name}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_analysis_prompt(text: str) -> str:
    """Get the formatted analysis prompt."""
    prompt = load_prompt("analysis")
    return prompt.format(text=text)


def get_metadata_prompt(text: str) -> str:
    """Get the formatted metadata extraction prompt."""
    prompt = load_prompt("metadata")
    return prompt.format(text=text)


def get_date_extraction_prompt(text: str) -> str:
    """Get the formatted date extraction prompt."""
    prompt = load_prompt("date_extraction")
    return prompt.format(text=text)
