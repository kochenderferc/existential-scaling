#!/usr/bin/env python3
"""
Quick local test: a server failure in the response stream must raise, not be
recorded as an empty wrong answer.

vLLM reports a mid-stream failure as HTTP 200 with an error object and no
"choices", so resp.raise_for_status() does not see it. Before the check this
test covers, such a run logged every question as FAIL with 0 tokens in 0s and
still wrote a summary.json, which looks like a model scoring 0% rather than a
server that never answered.

Needs no GPU and no model: it serves canned responses from localhost.

Usage (from repo root):
    python scripts/test_stream_errors.py
"""
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import eval_base

# What the mock server replies with, all of it inside an HTTP 200.
BODIES = {
    # vLLM rejecting a request field, reported after the stream has opened
    "error_chunk": 'data: {"error":{"message":"1 validation error: thinking_token_budget '
                   'extra fields not permitted","type":"BadRequestError"}}\n\ndata: [DONE]\n\n',
    # a body that is not SSE at all
    "plain_json":  '{"object":"error","message":"model not found","code":404}\n',
    # a normal short completion
    "ok":          'data: {"choices":[{"delta":{"content":"ASC 606-10-25-15"},"finish_reason":null}]}\n\n'
                   'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                   'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":7}}\n\n'
                   'data: [DONE]\n\n',
}

CASES = [
    ("error_chunk", "server error in stream"),
    ("plain_json",  "server returned no completion"),
    ("ok",          None),                            # must not raise
]

mode = {"name": "ok"}


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = BODIES[mode["name"]].encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class DummyBenchmark:
    """Duck-typed: importing src/benchmarks needs the dataset libraries."""

    def get_question_text(self, row):
        return row["question"]

    def build_result(self, row, thinking, answer, metrics, elapsed):
        return {"raw_answer": answer, "correct": bool(answer)}


def main() -> int:
    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    cfg = {
        "model": "mock", "port": port, "benchmark": "test", "context_window": 4096,
        "max_output_tokens": 64, "thinking_budget": 0, "enable_thinking": False,
        "pre_prompt": "", "post_prompt": "", "system_prompt": "", "request_timeout": 10,
    }
    eval_base._engine_cache[port] = "vllm"       # skip the engine probe

    failures = 0
    for name, expected in CASES:
        mode["name"] = name
        try:
            result = eval_base.run_one(DummyBenchmark(), {"question": "x"}, cfg, 0.1, quiet=True)
        except Exception as exc:
            if expected and expected in str(exc):
                print(f"  ok    {name}: raised -> {str(exc)[:90]}")
            else:
                print(f"  FAIL  {name}: {type(exc).__name__}: {str(exc)[:140]}")
                failures += 1
            continue
        if expected:
            print(f"  FAIL  {name}: expected a raise, recorded {result.get('raw_answer')!r}")
            failures += 1
        else:
            print(f"  ok    {name}: answer={result['raw_answer']!r} "
                  f"tokens={result.get('completion_tokens')}")

    print("all stream-error cases pass" if not failures else f"{failures} case(s) failed")
    return failures


if __name__ == "__main__":
    sys.exit(main())
