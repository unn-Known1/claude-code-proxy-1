#!/usr/bin/env python3
import os, json, time, httpx, argparse, asyncio, sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = "http://localhost:8082/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MODEL = "claude-3-sonnet-20240229"

headers = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": ANTHROPIC_VERSION,
    "content-type": "application/json",
}

calculator_tool = {
    "name": "calculator",
    "description": "Evaluate math",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "Expression"}},
        "required": ["expression"],
    },
}
weather_tool = {
    "name": "weather",
    "description": "Get weather",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string", "description": "City"}},
        "required": ["location"],
    },
}
search_tool = {
    "name": "search",
    "description": "Search web",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Query"}},
        "required": ["query"],
    },
}

TEST_SCENARIOS = {
    "simple": {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": "Hello!"}],
    },
    "calculator": {
        "model": MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": "What is 135 + 7.5 divided by 2.5?"}],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"},
    },
    "multi_tool": {
        "model": MODEL,
        "max_tokens": 500,
        "temperature": 0.7,
        "system": "Use tools when appropriate.",
        "messages": [{"role": "user", "content": "What's the weather in NYC?"}],
        "tools": [weather_tool, search_tool],
        "tool_choice": {"type": "auto"},
    },
    "content_blocks": {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Calculate 75.5 / 5"}],
            }
        ],
        "tools": [calculator_tool],
    },
    "simple_stream": {
        "model": MODEL,
        "max_tokens": 100,
        "stream": True,
        "messages": [{"role": "user", "content": "Count 1 to 5"}],
    },
    "calculator_stream": {
        "model": MODEL,
        "max_tokens": 300,
        "stream": True,
        "messages": [{"role": "user", "content": "What is 135 + 17.5 / 2.5?"}],
        "tools": [calculator_tool],
        "tool_choice": {"type": "auto"},
    },
}

REQUIRED_EVENT_TYPES = {
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
}


def get_response(url, data):
    start = time.time()
    r = httpx.post(url, headers=headers, json=data, timeout=30)
    print(f"Response time: {time.time() - start:.2f}s")
    return r


def compare_responses(ar, pr, check_tools=False):
    aj, pj = ar.json(), pr.json()
    print(
        f"Anthropic: {json.dumps({k: v for k, v in aj.items() if k != 'content'}, indent=2)}"
    )
    print(
        f"Proxy: {json.dumps({k: v for k, v in pj.items() if k != 'content'}, indent=2)}"
    )
    assert pj.get("role") == "assistant" and pj.get("type") == "message"
    assert pj.get("stop_reason") in [
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        None,
    ]
    ac, pc = aj["content"], pj["content"]
    assert isinstance(ac, list) and isinstance(pc, list) and len(pc) > 0

    if check_tools:
        at = next((i for i in ac if i.get("type") == "tool_use"), None)
        pt = next((i for i in pc if i.get("type") == "tool_use"), None)
        if at:
            print(f"Anthropic tool: {json.dumps(at, indent=2)}")
        if pt:
            print(f"Proxy tool: {json.dumps(pt, indent=2)}")

    at = next((i.get("text") for i in ac if i.get("type") == "text"), None)
    pt = next((i.get("text") for i in pc if i.get("type") == "text"), None)
    if check_tools and (at is None or pt is None):
        return True
    assert at is not None and pt is not None
    print(f"Anthropic: {at[:200]}...")
    print(f"Proxy: {pt[:200]}...")
    return True


def test_request(name, data, check_tools=False):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    ad, pd = data.copy(), data.copy()
    try:
        ar = get_response(ANTHROPIC_API_URL, ad)
        pr = get_response(PROXY_API_URL, pd)
        print(f"Anthropic: {ar.status_code}, Proxy: {pr.status_code}")
        if ar.status_code != 200 or pr.status_code != 200:
            print(f"Error - Anthropic: {ar.text}, Proxy: {pr.text}")
            return False
        return compare_responses(ar, pr, check_tools)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


class StreamStats:
    def __init__(self):
        self.event_types, self.event_counts = set(), {}
        self.total_chunks = 0
        self.text_content = ""
        self.has_tool_use = False
        self.has_error = False
        self.error_message = ""

    def add(self, event_data):
        self.total_chunks += 1
        if "type" in event_data:
            t = event_data["type"]
            self.event_types.add(t)
            self.event_counts[t] = self.event_counts.get(t, 0) + 1
            if (
                t == "content_block_start"
                and event_data.get("content_block", {}).get("type") == "tool_use"
            ):
                self.has_tool_use = True
            if t == "content_block_delta":
                d = event_data.get("delta", {})
                if d.get("type") == "text_delta":
                    self.text_content += d.get("text", "")

    def summarize(self):
        print(
            f"Chunks: {self.total_chunks}, Events: {sorted(self.event_types)}, Tool use: {self.has_tool_use}, Text: {self.text_content[:100]}..."
        )
        if self.has_error:
            print(f"Error: {self.error_message}")


async def stream_response(url, data, name):
    stats = StreamStats()
    try:
        async with httpx.AsyncClient() as client:
            data["stream"] = True
            start = time.time()
            async with client.stream(
                "POST", url, json=data, headers=headers, timeout=30
            ) as r:
                if r.status_code != 200:
                    stats.has_error = True
                    stats.error_message = f"HTTP {r.status_code}"
                    return stats, stats.error_message
                buffer = ""
                async for chunk in r.aiter_text():
                    if not chunk.strip():
                        continue
                    buffer += chunk
                    events = buffer.split("\n\n")
                    for e in events[:-1]:
                        if "data: " in e:
                            for line in e.split("\n"):
                                if line.startswith("data: "):
                                    d = line[6:]
                                    if d == "[DONE]":
                                        break
                                    try:
                                        stats.add(json.loads(d))
                                    except:
                                        pass
                    buffer = events[-1] if events else ""
            print(f"{name} stream: {time.time() - start:.2f}s")
    except Exception as e:
        stats.has_error = True
        stats.error_message = str(e)
    return stats, stats.error_message


def compare_stream_stats(as_, ps):
    pm = REQUIRED_EVENT_TYPES - ps.event_types
    print(f"Proxy missing: {pm if pm else 'None'}")
    return not ps.has_error and (len(ps.text_content) > 0 or ps.has_tool_use)


async def test_streaming(name, data):
    print(f"\n{'=' * 20} {name} STREAM {'=' * 20}")
    ad, pd = data.copy(), data.copy()
    try:
        as_, ae = await stream_response(ANTHROPIC_API_URL, ad, "Anthropic")
        ps, pe = await stream_response(PROXY_API_URL, pd, "Proxy")
        print(f"\n--- Anthropic ---")
        as_.summarize()
        print(f"\n--- Proxy ---")
        ps.summarize()
        if ae:
            print(f"Anthropic error: {ae}")
        if pe:
            print(f"Proxy error: {pe}")
            return False
        return compare_stream_stats(as_, ps)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_tests(args):
    results = {}
    if not args.streaming_only:
        print("\n=== NON-STREAMING TESTS ===")
        for name, data in TEST_SCENARIOS.items():
            if data.get("stream"):
                continue
            if args.simple and "tools" in data:
                continue
            if args.tools_only and "tools" not in data:
                continue
            results[name] = test_request(name, data, "tools" in data)

    if not args.no_streaming:
        print("\n=== STREAMING TESTS ===")
        for name, data in TEST_SCENARIOS.items():
            if not data.get("stream") and not name.endswith("_stream"):
                continue
            if args.simple and "tools" in data:
                continue
            if args.tools_only and "tools" not in data:
                continue
            results[f"{name}_stream"] = await test_streaming(name, data)

    print(f"\n=== SUMMARY ===")
    for t, r in results.items():
        print(f"{t}: {'PASS' if r else 'FAIL'}")
    passed = sum(1 for v in results.values() if v)
    print(f"Total: {passed}/{len(results)}")
    return passed == len(results)


async def main():
    if not ANTHROPIC_API_KEY:
        print("Error: ANTHROPIC_API_KEY not set")
        return
    p = argparse.ArgumentParser()
    p.add_argument("--no_streaming", action="store_true")
    p.add_argument("--streaming_only", action="store_true")
    p.add_argument("--simple", action="store_true")
    p.add_argument("--tools_only", action="store_true")
    success = await run_tests(p.parse_args())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
