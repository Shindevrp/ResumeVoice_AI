from __future__ import annotations

import asyncio

from modules.tools.builtin import get_builtin_tools
from modules.tools.registry import ToolRegistry, ToolSpec


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        reg = ToolRegistry()
        spec = ToolSpec(
            name="ping",
            description="pings",
            parameters={"properties": {}},
            handler=lambda: "pong",
        )
        reg.register(spec)
        assert reg.get("ping") is spec
        assert reg.list_tools() == [spec]

    def test_system_prompt_block_empty(self) -> None:
        assert ToolRegistry().system_prompt_block() == ""

    def test_system_prompt_block_lists_tools(self) -> None:
        block = get_builtin_tools().system_prompt_block()
        assert "get_time" in block
        assert "calculate" in block
        assert "{tool:" in block

    def test_find_calls(self) -> None:
        reg = get_builtin_tools()
        calls = reg.find_calls(
            "the time is {tool:get_time()}, then {tool:calculate(2+2)}"
        )
        assert calls[0] == {"name": "get_time", "args": []}
        assert calls[1] == {"name": "calculate", "args": ["2+2"]}

    def test_find_calls_quoted_args(self) -> None:
        reg = get_builtin_tools()
        calls = reg.find_calls("{tool:echo('hi there')}")
        assert calls == [{"name": "echo", "args": ["hi there"]}]

    def test_strip_calls(self) -> None:
        reg = get_builtin_tools()
        assert reg.strip_calls("say {tool:echo(hi)} now") == "say  now"
        assert reg.strip_calls("no calls here") == "no calls here"


class TestToolExecution:
    def test_sync_handler(self) -> None:
        async def run() -> dict[str, str]:
            return await get_builtin_tools().execute_call(
                {"name": "calculate", "args": ["2+2"]}
            )

        result = asyncio.run(run())
        assert result == {"tool": "calculate", "result": "4"}

    def test_async_handler(self) -> None:
        async def handler(*args: str) -> str:
            return f"async:{','.join(args)}"

        reg = ToolRegistry()
        reg.register(ToolSpec("atool", "", {"properties": {}}, handler))

        async def run() -> dict[str, str]:
            return await reg.execute_call({"name": "atool", "args": ["x"]})

        assert asyncio.run(run()) == {"tool": "atool", "result": "async:x"}

    def test_unknown_tool(self) -> None:
        async def run() -> dict[str, str]:
            return await get_builtin_tools().execute_call({"name": "nope", "args": []})

        result = asyncio.run(run())
        assert "Unknown tool" in result["result"]

    def test_handler_error_is_caught(self) -> None:
        def boom() -> str:
            raise ValueError("bad")

        reg = ToolRegistry()
        reg.register(ToolSpec("boom", "", {"properties": {}}, boom))

        async def run() -> dict[str, str]:
            return await reg.execute_call({"name": "boom", "args": []})

        result = asyncio.run(run())
        assert result["result"] == "Error: bad"

    def test_calculate_blocks_builtins(self) -> None:
        async def run() -> dict[str, str]:
            return await get_builtin_tools().execute_call(
                {"name": "calculate", "args": ["__import__('os')"]}
            )

        result = asyncio.run(run())
        assert result["result"].startswith("Error:")

    def test_roll_dice_range(self) -> None:
        async def run() -> int:
            result = await get_builtin_tools().execute_call(
                {"name": "roll_dice", "args": ["6"]}
            )
            return int(result["result"])

        for _ in range(20):
            assert 1 <= asyncio.run(run()) <= 6

    def test_get_date(self) -> None:
        async def run() -> str:
            result = await get_builtin_tools().execute_call(
                {"name": "get_date", "args": []}
            )
            return result["result"]

        date = asyncio.run(run())
        assert "," in date and date.split(",")[0].strip()


class TestExecuteAll:
    def test_cleans_text_and_returns_results(self) -> None:
        async def run() -> tuple[str, list[dict[str, str]]]:
            return await get_builtin_tools().execute_all(
                "result is {tool:calculate(1+1)}!"
            )

        cleaned, results = asyncio.run(run())
        assert cleaned == "result is !"
        assert results == [{"tool": "calculate", "result": "2"}]
