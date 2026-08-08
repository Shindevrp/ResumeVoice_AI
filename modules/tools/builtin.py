from __future__ import annotations

import datetime
import math
import random

from modules.tools.registry import ToolRegistry, ToolSpec


def _get_time() -> str:
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p").lstrip("0")


def _get_date() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y")


def _calculate(expression: str) -> str:
    allowed = {
        "abs",
        "acos",
        "asin",
        "atan",
        "ceil",
        "cos",
        "degrees",
        "exp",
        "factorial",
        "floor",
        "fmod",
        "log",
        "log10",
        "pow",
        "radians",
        "sin",
        "sqrt",
        "tan",
        "pi",
        "e",
    }
    safe_dict = {k: getattr(math, k, None) for k in allowed}
    safe_dict.update(
        {
            "abs": abs,
            "int": int,
            "float": float,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "str": str,
        }
    )
    safe_dict["__builtins__"] = None
    result = eval(expression, safe_dict)
    return str(result)


def _roll_dice(sides: str = "6") -> str:
    n = max(1, int(sides))
    return str(random.randint(1, n))


def _echo(text: str) -> str:
    return text


def get_builtin_tools() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolSpec(
            name="get_time",
            description="Get the current time",
            parameters={"properties": {}},
            handler=_get_time,
        )
    )

    registry.register(
        ToolSpec(
            name="get_date",
            description="Get today's date",
            parameters={"properties": {}},
            handler=_get_date,
        )
    )

    registry.register(
        ToolSpec(
            name="calculate",
            description="Evaluate a mathematical expression",
            parameters={
                "properties": {
                    "expression": {"description": "math expression to evaluate"},
                },
            },
            handler=_calculate,
        )
    )

    registry.register(
        ToolSpec(
            name="roll_dice",
            description="Roll a dice with the given number of sides",
            parameters={
                "properties": {
                    "sides": {"description": "number of sides (default 6)"},
                },
            },
            handler=_roll_dice,
        )
    )

    return registry
