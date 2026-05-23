"""AST validation and wrapping for LLM-generated predict() method bodies.

validate_predict_body() checks for dangerous constructs without executing code.
wrap_predict_body() produces a complete Strategy subclass source string.
"""
from __future__ import annotations

import ast
import textwrap

# ---------------------------------------------------------------------------
# Disallowed AST node types
# ---------------------------------------------------------------------------

_BANNED_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Delete,
)

# Names that must not appear in any Name or Attribute node
_BANNED_NAMES: frozenset[str] = frozenset({
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "importlib",
    "subprocess",
    "os",
    "sys",
    "socket",
    "requests",
    "urllib",
    "http",
    "builtins",
})

_MAX_BODY_CHARS: int = 4_000


class CodeValidationError(ValueError):
    """Raised when generated predict() body fails AST validation."""


def validate_predict_body(body: str) -> None:
    """Raise CodeValidationError if body contains disallowed constructs.

    Checks performed:
    1. Source length cap (prevents runaway LLM output).
    2. Parseable as valid Python.
    3. No import statements.
    4. No use of dangerous built-in names.
    5. No sys.exit / os.system / subprocess calls (via Attribute inspection).
    """
    if len(body) > _MAX_BODY_CHARS:
        raise CodeValidationError(
            f"predict() body too long: {len(body)} chars (max {_MAX_BODY_CHARS})"
        )

    # Wrap in a function so the parser has a valid module to work with
    wrapped = _wrap_for_parse(body)
    try:
        tree = ast.parse(wrapped)
    except SyntaxError as exc:
        raise CodeValidationError(f"Syntax error in predict() body: {exc}") from exc

    for node in ast.walk(tree):
        # Banned node types
        if isinstance(node, _BANNED_NODES):
            raise CodeValidationError(
                f"Disallowed construct in predict() body: {type(node).__name__}"
            )
        # Banned names in Name nodes
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise CodeValidationError(
                f"Disallowed name in predict() body: '{node.id}'"
            )
        # Banned names in Attribute nodes (e.g. os.system, sys.exit)
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_NAMES:
            raise CodeValidationError(
                f"Disallowed attribute in predict() body: '{node.attr}'"
            )


def wrap_predict_body(body: str, class_name: str = "GeneratedStrategy") -> str:
    """Return a complete Python module string containing a Strategy subclass.

    The generated class overrides predict() with the supplied body.
    The module exposes a top-level `strategy` instance ready for use.
    """
    # Normalise indentation: ensure body is indented exactly 8 spaces
    dedented = textwrap.dedent(body)
    indented = textwrap.indent(dedented, "        ")  # 8 spaces

    return f"""\
from __future__ import annotations

import numpy as np
import pandas as pd

from autoalpha.core.strategy import Strategy


class {class_name}(Strategy):
    def fit(self, history: pd.DataFrame) -> None:
        pass

    def predict(self, bar_data: pd.DataFrame, bar_date: pd.Timestamp | None = None) -> dict[str, float]:
{indented}


strategy = {class_name}()
"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _wrap_for_parse(body: str) -> str:
    """Wrap body in a dummy function definition for AST parsing."""
    dedented = textwrap.dedent(body)
    indented = textwrap.indent(dedented, "    ")
    return f"def _predict(self, bar_data, bar_date=None):\n{indented}\n"
