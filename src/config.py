"""Configuration and prompt-file loading."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """Read config.yaml into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Resolve a path from config.yaml against the project root."""
    return PROJECT_ROOT / relative


def load_prompts(cfg: dict) -> dict:
    """Read the prompt file and check it has the keys the code needs."""
    with open(project_path(cfg["paths"]["prompts"]), "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    missing = {"system", "user_template"} - set(prompts)
    if missing:
        raise KeyError(f"Prompt file is missing keys: {sorted(missing)}")
    return prompts
