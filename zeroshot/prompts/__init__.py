"""Versioned zero-shot prompt assets."""

from .loader import (
    DEFAULT_PROMPT_PATH,
    load_prompt_dictionary,
    prompts_for_labels,
)

__all__ = ["DEFAULT_PROMPT_PATH", "load_prompt_dictionary", "prompts_for_labels"]

