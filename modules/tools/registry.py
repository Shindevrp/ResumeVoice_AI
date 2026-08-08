from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str | Coroutine[Any, Any, str]]


TOOL_CALL_RE = r"\{tool:(\w+)\(([^}]*)\)\}"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def system_prompt_block(self) -> str:
        if not self._tools:
            return ""
        names = ", ".join(sorted(self._tools.keys()))
        return (
            f"\nYou have access to tools: {{{names}}}. "
            f"Use format: {{tool:name(args)}}. "
            f"Only use them if explicitly asked."
        )

    def find_calls(self, text: str) -> list[dict[str, str]]:
        calls = []
        for match in re.finditer(TOOL_CALL_RE, text):
            calls.append(
                {
                    "name": match.group(1),
                    "args": [
                        a.strip().strip('"').strip("'")
                        for a in match.group(2).split(",")
                        if a.strip()
                    ],
                }
            )
        return calls

    def strip_calls(self, text: str) -> str:
        return re.sub(TOOL_CALL_RE, "", text).strip()

    async def execute_call(self, call: dict[str, Any]) -> dict[str, str]:
        spec = self._tools.get(call["name"])
        if not spec:
            return {"tool": call["name"], "result": f"Unknown tool: {call['name']}"}
        try:
            if asyncio.iscoroutinefunction(spec.handler):
                result = await spec.handler(*call["args"])
            else:
                result = spec.handler(*call["args"])
        except Exception as e:
            result = f"Error: {e}"
        return {"tool": call["name"], "result": result}

    async def execute_all(self, text: str) -> tuple[str, list[dict[str, str]]]:
        calls = self.find_calls(text)
        if not calls:
            return text, []
        results = await asyncio.gather(*[self.execute_call(c) for c in calls])
        cleaned = self.strip_calls(text)
        return cleaned, results
