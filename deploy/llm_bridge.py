#!/usr/bin/env python3
"""LLM bridge: expose host-only Ollama (127.0.0.1:11434) to Docker containers.

The compose `resumevoice` service calls LLM at http://host.docker.internal:11434
(= the host's docker bridge gateway). If Ollama only listens on loopback
(default OLLAMA_HOST), that connection is refused. This proxy listens on the
bridge gateway IP and forwards to loopback so containers can reach Ollama
WITHOUT a systemd/sudo change.

Preferred permanent fix (needs sudo, run once):
    sudo systemctl edit ollama
    # [Service]
    # Environment="OLLAMA_HOST=0.0.0.0"
    sudo systemctl restart ollama

Run:  python3 deploy/llm_bridge.py [--listen 172.17.0.1 --listen-port 11434]
"""
from __future__ import annotations

import argparse
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("llm_bridge")


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65_536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def _handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter,
                  target_host: str, target_port: int) -> None:
    try:
        server_r, server_w = await asyncio.open_connection(target_host, target_port)
    except (OSError, ConnectionError) as e:
        logger.warning(f"upstream {target_host}:{target_port} unreachable: {e}")
        client_w.close()
        return
    tasks = [
        asyncio.create_task(_pump(client_r, server_w)),
        asyncio.create_task(_pump(server_r, client_w)),
    ]
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="172.17.0.1")
    parser.add_argument("--listen-port", type=int, default=11434)
    parser.add_argument("--target", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=11434)
    args = parser.parse_args()

    server = await asyncio.start_server(
        lambda r, w: _handle(r, w, args.target, args.target_port),
        host=args.listen,
        port=args.listen_port,
    )
    logger.info(f"llm bridge {args.listen}:{args.listen_port} -> {args.target}:{args.target_port}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())