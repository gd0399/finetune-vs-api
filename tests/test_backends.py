"""Both real backends against a fake local server that speaks the Anthropic and OpenAI wire formats."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ftva import backends  # noqa: E402
from ftva.prompt import SYSTEM, parse  # noqa: E402

SEEN = {}
ANSWER = {"age": 50, "sex": "male"}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN[self.path] = body
        if self.path.endswith("/messages"):
            out = {"id": "m1", "type": "message", "role": "assistant", "model": body["model"],
                   "content": [{"type": "text", "text": json.dumps(ANSWER)}], "stop_reason": "end_turn",
                   "stop_sequence": None, "usage": {"input_tokens": 812, "output_tokens": 140}}
        else:
            out = {"choices": [{"message": {"role": "assistant", "content": "```json\n" + json.dumps(ANSWER) + "\n```"}}],
                   "usage": {"prompt_tokens": 790, "completion_tokens": 131}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server():
    s = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"
    s.shutdown()


def test_anthropic_backend(server, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server)
    text, tin, tout = backends.make("anthropic:claude-haiku-5-5")("54yo F c/o cough.")
    assert parse(text) == ANSWER and (tin, tout) == (812, 140)
    sent = SEEN["/v1/messages"]
    assert sent["model"] == "claude-haiku-5-5" and sent["system"] == SYSTEM
    assert "54yo F c/o cough." in sent["messages"][0]["content"]


def test_openai_backend(server):
    text, tin, tout = backends.make(f"openai:qwen3-8b-ft@{server}/v1")("54yo F c/o cough.")
    assert parse(text) == ANSWER and (tin, tout) == (790, 131)
    sent = SEEN["/v1/chat/completions"]
    assert sent["model"] == "qwen3-8b-ft"
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert sent["messages"][0] == {"role": "system", "content": SYSTEM}
