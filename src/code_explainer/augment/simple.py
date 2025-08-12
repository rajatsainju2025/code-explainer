"""Simple data augmentation strategies for code/explanation pairs."""

from __future__ import annotations

from typing import Dict, List


def swap_lines(code: str) -> str:
    lines = code.splitlines()
    if len(lines) >= 2:
        lines[0], lines[1] = lines[1], lines[0]
    return "\n".join(lines)


def augment_dataset(items: List[Dict[str, str]], ratio: float = 0.2) -> List[Dict[str, str]]:
    if ratio <= 0:
        return items
    n = max(1, int(len(items) * ratio))
    augmented = []
    for i in range(n):
        it = items[i % len(items)]
        augmented.append({"code": swap_lines(it["code"]), "explanation": it["explanation"]})
    return items + augmented
