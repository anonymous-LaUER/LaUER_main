"""Load the frozen MELP CKEPE prompt snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

DEFAULT_PROMPT_PATH = Path(__file__).with_name("melp_ckepe.json")
OFFICIAL_PROMPT_PATHS = {name: Path(__file__).with_name(f"official_{name}.json") for name in ("CSN", "CPSC", "ptbxl")}


def official_prompt_path(task: str) -> Path:
    """Dataset context is mandatory: the same acronym may name a different target scope."""
    name = str(task).lower()
    dataset = "ptbxl" if name.startswith("ptbxl") else "CSN" if name in ("csn", "chapman") else "CPSC" if name in ("cpsc", "cpsc2018", "icbeb") else None
    if dataset is None:
        raise ValueError(f"No official annotation query contract for {task!r}")
    return OFFICIAL_PROMPT_PATHS[dataset]

def load_prompt_dictionary(
    path: str | Path = DEFAULT_PROMPT_PATH,
) -> Dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prompt snapshot is missing: {path}. Run "
            "`python -m zeroshot.prompts.prepare_prompt_snapshot`."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Invalid prompt dictionary: {path}")
    invalid = [
        key
        for key, value in payload.items()
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip()
    ]
    if invalid:
        raise ValueError(f"Invalid prompt entries: {invalid[:10]}")
    return payload

def prompts_for_labels(
    labels: Sequence[str],
    path: str | Path = DEFAULT_PROMPT_PATH,
    *, task: str | None = None,
) -> Tuple[str, ...]:
    if task is not None:
        expected = official_prompt_path(task)
        historical_defaults = {DEFAULT_PROMPT_PATH.resolve(), Path(__file__).with_name("eval_prompt.json").resolve()}
        supplied = Path(path).resolve()
        if supplied in historical_defaults:
            path = expected
        elif supplied != expected.resolve():
            raise ValueError("Official annotation evaluation requires its dataset-specific query registry; custom/hierarchical prompts need a separately named protocol")
    dictionary = load_prompt_dictionary(path)
    missing = [label for label in labels if label not in dictionary]
    if missing:
        raise KeyError(f"Prompt snapshot has no entries for {missing}")
    return tuple(dictionary[label] for label in labels)

