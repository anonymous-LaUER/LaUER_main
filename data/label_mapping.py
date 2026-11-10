"""Canonical LaUER concept vocabulary, independent of private annotation files."""

import json
from pathlib import Path


def load_canonical_labels(annotation_csv=None):
    """Return the 169 training concepts; annotation CSV order remains authoritative."""
    with Path(__file__).with_name("concepts.json").open(encoding="utf-8") as stream:
        return json.load(stream)
