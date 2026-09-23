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
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    allowed_names = {n: getattr(math, n, None) for n in allowed}
    allowed_names["abs"] = abs  # builtin; math has no abs

    def _eval_node(node: ast.AST) -> int | float:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("unsupported literal")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(f"unsafe name: {node.id}")
            return allowed_names[node.id]
        if isinstance(node, ast.BinOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"unsafe operator: {type(node.op).__name__}")
            return op(_eval_node(node.left), _eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op = allowed_ops.get(type(node.op))
            if op is None:
                raise ValueError(f"unsafe unary: {type(node.op).__name__}")
            return op(_eval_node(node.operand))
        raise ValueError(f"unsafe syntax: {type(node).__name__}")

    root = ast.parse(expression, mode="eval")
    result = _eval_node(root.body)
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
