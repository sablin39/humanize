"""Kimi Code, driven through `KimiCodeCLIAgent` against a stand-in `kimi web` on PATH.

The stand-in prints its listening address and serves the current session REST routes. It
serves no `/ws`, so ordinary turns use the driver's polling path. Started with `KIMI_MODEL_*`,
it asks a local LiteLLM gateway, in front of an `Endpoint`, for its answer.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from hmz.coganchor import providers
from hmz.coganchor.agents import Failed, KimiCodeCLIAgent, KimiCodeCLIAgentConfig
from hmz.coganchor.gateway import gatewayable
from hmz.coganchor.providers import Provider
from tests.integration.doubles_agents import (
    CHATTING,
    Endpoint,
    Standins,
    kinds,
    serving,
    standins,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

CONFIG = KimiCodeCLIAgentConfig(model="kimi-code/stand-in", effort="high")

KIMI = r"""
import itertools, socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

note()
SESSIONS = itertools.count(1)
PROMPTS = {}


class Handler(BaseHTTPRequestHandler):
    def reply(self, data, status=200):
        body = json.dumps({"code": 0, "msg": "ok", "data": data}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        sent = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0))
                          or b"null")
        path = self.path.removeprefix("/api/v1")
        if path == "/sessions":
            self.reply({"id": "session-%d" % next(SESSIONS)})
            return
        session = path.split("/")[2]
        if path.endswith("/prompts"):
            said = sent["content"][0]["text"]
            note(json.dumps({"session": session, "prompt": said,
                             "thinking": sent.get("thinking")}))
            if said == "refused":
                self.reply(None, status=400)
                return
            if os.environ.get("KIMI_MODEL_BASE_URL"):
                said = chatted(os.environ["KIMI_MODEL_BASE_URL"],
                               os.environ["KIMI_MODEL_API_KEY"],
                               os.environ["KIMI_MODEL_NAME"], said)
            PROMPTS[session] = said
            self.reply({"prompt_id": "p-1", "user_message_id": "u-1", "status": "running"})
        else:
            self.reply({})

    def do_GET(self):
        path, _, query = self.path.removeprefix("/api/v1").partition("?")
        parts = path.split("/")
        if path == "/ws" or len(parts) < 3:
            self.send_error(404, "Not Found")
            return
        session = parts[2]
        said = PROMPTS.get(session, "")
        if path.endswith("/messages"):
            self.reply({"items": [{"id": "m-1", "role": "assistant", "content": [
                {"type": "thinking", "thinking": "thinking about " + said},
                {"type": "tool_use", "tool_name": "Shell"},
                {"type": "text", "text": said}]}]})
        elif path.endswith("/agents/main/turns"):
            turns = [{"turn_id": "t-1", "status": "ended", "outcome": "completed",
                      "usage": {"input_other": 30, "output": 10,
                                "input_cache_read": 0, "input_cache_creation": 0}}]
            self.reply({"turns": turns, "has_more": False})
        elif path == f"/sessions/{session}":
            self.reply({"id": session, "busy": False, "main_turn_active": False,
                        "usage": {"input_tokens": 0, "output_tokens": 0,
                                  "cache_read_tokens": 0,
                                  "cache_creation_tokens": 0}})

    def log_message(self, *ignored):
        pass


class Server(HTTPServer):
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


server = Server(("127.0.0.1", 0), Handler)
print(f"Kimi server: http://127.0.0.1:{server.server_port}/#token=secret", flush=True)
server.serve_forever()
"""


@pytest.fixture
def kimi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Standins:
    held = standins(tmp_path, monkeypatch)
    held.install("kimi", CHATTING + KIMI)
    return held


def _prompts(kimi: Standins) -> list[dict[str, Any]]:
    return [
        json.loads(said)
        for said in kimi.said()
        if said.startswith("{") and "prompt" in json.loads(said)
    ]


def test_a_turn_says_what_it_did_and_what_it_spent(kimi: Standins) -> None:
    said = list(KimiCodeCLIAgent(CONFIG).new().stream("hello"))

    assert kinds(said) == ["reasoning", "tool", "text", "result"]
    assert said[-1].text == "hello"
    assert said[-1].spent.output == 10
    assert dict(said[-1].tokens) == {"kimi-code/stand-in": 40}


def test_one_daemon_holds_the_session_across_turns(kimi: Standins) -> None:
    agent = KimiCodeCLIAgent(CONFIG)
    session = agent.new()

    assert session("one") == "one"
    assert session("two") == "two"

    launch = kimi.calls()[0]
    assert launch.argv[0] == "web"
    assert len({one.pid for one in kimi.calls()}) == 1
    assert [(one["session"], one["prompt"]) for one in _prompts(kimi)] == [
        ("session-1", "one"),
        ("session-1", "two"),
    ]
    assert agent.opened == ["session-1"]


def test_a_prompt_the_daemon_refuses_is_a_failed_turn_leaving_nothing_open(
    kimi: Standins,
) -> None:
    agent = KimiCodeCLIAgent(CONFIG)
    session = agent.new()

    with pytest.raises(Failed):
        session("refused")
    with pytest.raises(RuntimeError):
        _ = session.id
    assert agent.opened == []


@pytest.fixture
def endpoint() -> Iterator[Endpoint]:
    yield from serving()


def _unswapped(self: Provider) -> tuple[tuple[str, str], ...]:
    """No credential path answered by another, which the supervisor's own tests cover."""
    return ()


@pytest.mark.skipif(not gatewayable(), reason="the [litellm] extra is not installed")
@pytest.mark.timeout(240)
def test_the_daemon_under_an_account_a_gateway_holds_is_pointed_at_it(
    endpoint: Endpoint, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    held = standins(
        tmp_path,
        monkeypatch,
        "KIMI_MODEL_BASE_URL",
        "KIMI_MODEL_API_KEY",
        "KIMI_MODEL_NAME",
        "KIMI_MODEL_PROVIDER_TYPE",
        "LITELLM_GATEWAY_KEY",
    )
    held.install("kimi", CHATTING + KIMI)
    monkeypatch.setenv("HUMANIZE_HOME", str(tmp_path / "hmz"))
    monkeypatch.setenv("LITELLM_GATEWAY_KEY", "lying-about")
    monkeypatch.setenv("KIMI_MODEL_NAME", "lying-about")
    monkeypatch.setattr(Provider, "swaps", _unswapped)
    providers.add(
        "kimi",
        "gw",
        way="litellm-openai-gateway",
        env={
            "LITELLM_GATEWAY_URL": endpoint.url,
            "LITELLM_GATEWAY_KEY": "upstream-secret",
            "LITELLM_GATEWAY_API": "openai",
        },
    )
    agent = KimiCodeCLIAgent(
        KimiCodeCLIAgentConfig(model="stand-in", effort="", provider="gw")
    )
    try:
        said = agent.new()("hello")
        through = agent.gateway
        assert through is not None
    finally:
        agent.stop()

    assert said == "hello"
    assert [one["model"] for one in endpoint.asked] == ["stand-in"]
    assert held.calls("kimi")[0].env == {
        "HMZ_TEST_INHERITED": "from-the-flow",
        "KIMI_MODEL_BASE_URL": through.url,
        "KIMI_MODEL_API_KEY": through.key,
        "KIMI_MODEL_NAME": "stand-in",
        "KIMI_MODEL_PROVIDER_TYPE": "openai",
        "LITELLM_GATEWAY_KEY": None,
    }
    assert not through.running
