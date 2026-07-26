from __future__ import annotations

import re


def step_zero_extract_entities(user_raw_input: str) -> list[str]:
    """
    Extract explicit structural document identifiers from a raw user query.

    Matches examples such as:
    - "Figure 4.1"
    - "Table 2.2"
    - "fig 3.7"
    - standalone identifiers like "4.1" or "3.7"

    Returns:
        A deduplicated list of clean string matches in order of appearance.
        Returns [] if no identifiers are found.
    """
    if not user_raw_input:
        return []

    pattern = re.compile(
        r"\b(?:"
        r"(?:fig(?:ure)?|tab(?:le|el)?|table|tabel)\s*[Oo0]?\s*\.?\s*\d+(?:\.\d+)*"
        r"|"
        r"[Oo0]?\s*\.?\s*\d+\.\d+(?:\.\d+)*"
        r")\b",
        flags=re.IGNORECASE,
    )

    matches: list[str] = []
    seen: set[str] = set()

    for match in pattern.finditer(user_raw_input):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        key = value.lower()

        if key not in seen:
            seen.add(key)
            matches.append(value)

    return matches
