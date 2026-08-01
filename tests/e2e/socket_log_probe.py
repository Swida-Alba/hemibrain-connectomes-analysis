#!/usr/bin/env python3
"""Probe the running DROCAT UI like a browser (socket.io) to verify live log streaming.

Usage (while the app runs on port 8765):
    python tests/e2e/socket_log_probe.py
"""

import asyncio
import json
import re
import sys
import time

import httpx
import socketio

BASE = "http://127.0.0.1:8765"


def walk(payloads: dict, predicate):
    """Yield payloads matching predicate, traversing the element tree."""
    for element_id, payload in payloads.items():
        if not isinstance(payload, dict):
            continue
        if predicate(payload):
            yield element_id, payload
        for child_id in payload.get("children", []) or []:
            child = payloads.get(child_id)
            if isinstance(child, dict):
                yield from walk({child_id: child}, predicate)


async def main() -> int:
    async with httpx.AsyncClient(timeout=15) as http:
        resp = await http.get(BASE + "/")
        html = resp.text
    match = re.search(r"'client_id': '([0-9a-f-]+)'", html)
    if not match:
        print("FAIL: client_id not found in page HTML")
        return 1
    client_id = match.group(1)
    print(f"client_id: {client_id}")

    sio = socketio.AsyncClient()
    updates: list[tuple[float, dict]] = []

    @sio.on("update")
    async def on_update(data):
        if isinstance(data, dict):
            updates.append((time.time(), data))

    await sio.connect(
        BASE + "?client_id=" + client_id,
        socketio_path="_nicegui_ws/socket.io",
        wait_timeout=10,
    )
    ack_holder = {}
    await sio.emit(
        "handshake",
        {"client_id": client_id, "tab_id": "probe-tab", "document_id": "probe-doc"},
        callback=lambda data: ack_holder.setdefault("ack", data),
    )
    await asyncio.sleep(0.5)
    print("handshake ack:", ack_holder.get("ack"))

    # The server-rendered HTML embeds the element tree as JSON with numeric ids.
    decoder = json.JSONDecoder()
    tree_start = html.find('{"0":{"tag"')
    if tree_start < 0:
        print("FAIL: element tree JSON not found in SSR HTML")
        return 1
    tree, _ = decoder.raw_decode(html, tree_start)

    def find_element(label: str) -> tuple[str, dict] | None:
        for element_id, payload in tree.items():
            props = payload.get("props", {}) or {}
            if props.get("label") == label:
                return element_id, payload
        return None

    def listener_id(payload: dict, event_type: str) -> int | None:
        for listener in payload.get("events", []) or []:
            if listener.get("type") == event_type:
                return listener.get("listener_id")
        return None

    source = find_element("Source Neurons")
    target = find_element("Target Neurons")
    run = find_element("Find All Paths")
    print(
        "source_id=", source[0] if source else None,
        "target_id=", target[0] if target else None,
        "run_id=", run[0] if run else None,
    )
    if not (source and target and run):
        print("FAIL: could not locate inputs/run button in SSR HTML")
        return 1
    source_listener = listener_id(source[1], "update:value")
    target_listener = listener_id(target[1], "update:value")
    run_listener = listener_id(run[1], "click")
    print(f"listeners: source={source_listener} target={target_listener} run={run_listener}")
    if not (source_listener and target_listener and run_listener):
        print("FAIL: listener ids not found")
        return 1

    def emit_event(element_id: str, listener: int, event_type: str, args: list):
        return sio.emit(
            "event",
            {
                "client_id": client_id,
                "id": int(element_id),
                "listener_id": listener,
                "type": event_type,
                "args": args,
            },
        )

    await emit_event(source[0], source_listener, "update:value", [json.dumps("aMe12")])
    await emit_event(target[0], target_listener, "update:value", [json.dumps("aMe10")])
    await asyncio.sleep(0.3)
    await emit_event(run[0], run_listener, "click", [])
    print("Run clicked, watching log updates...")

    started = time.time()
    log_lines: list[tuple[float, str, str]] = []
    seen_function_banner = False
    seen_finished = False
    while time.time() - started < 150:
        await asyncio.sleep(0.2)
        for ts, tree in list(updates):
            for element_id, payload in walk(
                tree, lambda p: isinstance(p, dict) and "text" in p
            ):
                text = payload.get("text", "")
                if not text or not isinstance(text, str):
                    continue
                if "UI FUNCTION" in text:
                    seen_function_banner = True
                if "FINISHED" in text:
                    seen_finished = True
                if (text, element_id) not in [(t, i) for _, t, i in log_lines]:
                    log_lines.append((time.time() - started, text, element_id))
                    print(f"[{log_lines[-1][0]:6.1f}s] {text}")
            updates.remove((ts, tree))
        if seen_finished:
            break

    print()
    print(f"log lines captured: {len(log_lines)}")
    print(f"UI FUNCTION banner seen: {seen_function_banner}")
    print(f"FINISHED line seen: {seen_finished}")
    if log_lines:
        print(f"first log line arrived {log_lines[0][0]:.1f}s after Run click")
    await sio.disconnect()
    return 0 if (seen_function_banner and seen_finished) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
