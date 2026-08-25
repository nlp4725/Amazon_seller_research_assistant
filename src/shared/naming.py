"""
Filename slugs for run records.

`safe_name` was copy-pasted verbatim into main_1.py, main_2.py and
classify_agent.py, and cat_selector.py imported the private copy out of
classify_agent -- a dependency that had nothing to do with classification.
One definition here instead.
"""

import re


def safe_name(text: str) -> str:
    """Lowercase slug safe to use as a filename: "Dog Drinking Bowl" -> "dog_drinking_bowl"."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
