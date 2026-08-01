from __future__ import annotations

import re


SUPERSCRIPTS = str.maketrans({
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "n": "ⁿ",
})


def format_math_notation(text: str) -> str:
    """Convert common raw LaTeX fragments into readable Unicode notation."""
    formatted = text.replace("\\(", "").replace("\\)", "").replace("$", "")
    formatted = formatted.replace("\\left", "").replace("\\right", "")
    formatted = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", " × ", formatted)
    formatted = re.sub(
        r"\\frac\{x\^\{([^{}]+)\}\}\{([^{}]+)\}",
        lambda match: f"x^{{{match.group(1)}}}/({match.group(2)})",
        formatted,
    )
    formatted = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        lambda match: f"{match.group(1)}/{match.group(2)}",
        formatted,
    )
    formatted = re.sub(r"\\sqrt\{([^{}]+)\}", lambda match: f"√({match.group(1)})", formatted)
    replacements = {
        r"\int": "∫",
        r"\times": "×",
        r"\cdot": "·",
        r"\div": "÷",
        r"\pi": "π",
        r"\neq": "≠",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\sin": "sin",
        r"\cos": "cos",
        r"\tan": "tan",
        r"\ln": "ln",
        r"\mathrm": "",
    }
    for latex, unicode_symbol in replacements.items():
        formatted = formatted.replace(latex, unicode_symbol)
    formatted = re.sub(
        r"\^\{([0-9n+\-]+)\}",
        lambda match: match.group(1).translate(SUPERSCRIPTS),
        formatted,
    )
    formatted = re.sub(
        r"\^([0-9n])\b",
        lambda match: match.group(1).translate(SUPERSCRIPTS),
        formatted,
    )
    return re.sub(r"[ \t]+\n", "\n", formatted).strip()
