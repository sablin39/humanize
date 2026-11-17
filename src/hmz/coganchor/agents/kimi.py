"""Kimi Code: the app server it serves itself from, where a session is more than its prompts.

``kimi --prompt`` takes a prompt and nothing else. A ``/goal`` written into one is text the model
reads rather than a goal its runtime keeps, there is no flag for swarm mode, the effort an agent
is configured with has nowhere to go, and a turn already running has nowhere to be talked to.
``kimi web`` is the same binary serving the sessions its own browser client drives, and there
all four are things done to the session a turn is submitted to.

Which is the whole answer to why this backend is not driven through its headless mode. 0.42.0
has a real one -- ``kimi -p <prompt> --output-format stream-json`` prints a turn as it happens,
and for a driver that only wanted the words that would be the shorter road. It is not taken
because three of the things this driver is for are not in it. There is no route to a turn that
is already running, so :meth:`KimiCodeCLISession.interject` -- the reason ``steers`` is True
here at all -- would have nothing to write to. There is no per-turn body, so the rung an agent
runs at, its thinking level and its swarm width would have to be the process's own flags,
fixed for the life of a prompt that is one turn long anyway, and ``--plan``/``--yolo``/``--auto``
are the only three of them there are. And a turn that stops -- to ask, or for a tool it wants
approving -- stops against a terminal there, not against an interaction with an id that
something else can resolve through ``/interactions``. That is what lets a flow answer or
refuse one with nobody at a keyboard.
The server is the only surface in this CLI where a session is a thing
rather than an invocation, so the daemon it is, and the cost of that choice -- one ``kimi web``
per agent, and its argv -- is paid below.

What a person types at Kimi's own prompt reaches a chat here as far as that server reaches,
and no further (:mod:`hmz.coganchor.agents.native`). Read off the routes 0.42.0 and 2.6.2
both serve, which are what Kimi's own web client sends them as:

- `/<skill>` for every skill `GET /sessions/<id>/skills` lists that a person may activate is
  called, as the prompt's `skills` -- the activation the server renders the skill into, with
  what follows the name as its arguments.
- `/compact [instructions]` is called, as `POST /sessions/<id>:compact`, and the turn is over
  once the session stops being busy with it.

Both only once there is a conversation: until then a `/` line is text, as the catalog says.

And nothing else, because nothing else has a route: there is no shell route for a `!`, no
mention for an `@` beyond the text the model reads, and Kimi's other slash commands are its
terminal's. Until the server holds a conversation there are no skills to list, so the catalog
is empty and every line is text.
"""

# A session and the agent holding it are two halves of one object declared in one
# file, which is what the underscore keeps out of the package rather than out of them.
# pyright: reportPrivateUsage=false

from __future__ import annotations

import collections
import contextlib
import importlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import weakref
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, cast

from hmz.coganchor.backends import SWARM

from .base import AgentBase, SessionBase
from .config import UNSAID, AgentConfig
from .event import Event, Failed, Question, Usage, say
from .hooks import EVERYWHERE, Moment
from .native import NativeCommand
from .preload import preloaded
from .watchdog import Watchdog

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from types import ModuleType

    from pydantic import BaseModel
    from websockets.sync.client import ClientConnection

    from hmz.coganchor.fence import Fence
    from hmz.coganchor.gateway import Gateway
    from hmz.coganchor.homes import Home

#: What to say when the websocket client a session is told about its own turn over is not
#: in this Python environment. Kimi Code is an extra rather than part of every install, so
#: this is a choice somebody made rather than an install gone wrong, and the way back is
#: named where it is missed.
_EXTRA = (
    "Kimi Code needs the websocket client its app server's notifications ride on, which is "
    "the [kimi] extra and is not installed in this Python environment: uv sync --extra kimi "
    "from a checkout, or pip install 'websockets>=15,<18'"
)

#: The one line `kimi web` prints once it is listening, and the only place the port it took --
#: asked for as 0, so that two flows on one machine cannot collide -- and its token are said.
#:
#: Printed only by a server whose log level is *not* `silent`, which is the whole reason
#: :attr:`KimiCodeCLIAgentConfig.log_level` exists and does not default to the CLI's own
#: default. `kimi web` writes one of two ready outputs: the compact line this reads at any
#: level from `fatal` to `trace`, and a drawn banner -- logo, ANSI colour, the token on a
#: line of its own -- at `silent`. Omitting `--log-level` is `silent`, so the bare command a
#: person runs is the one shape a machine cannot read, and a daemon started without the flag
#: would leave this loop reading banner lines that never match until the agent was collected.
#: Verified against 0.42.0 by starting both.
_LISTENING = re.compile(r"^Kimi server: (\S+)/#token=(\S+)$")

#: The model the provider a daemon builds out of `KIMI_MODEL_*` is made with.
_MODEL_NAME = "KIMI_MODEL_NAME"

#: What `kimi web --log-level` takes, and the one of them this driver cannot be run at. The
#: level is the daemon's own logging, which nothing here reads -- what it decides for us is
#: the shape of the ready line above, so `silent` is refused where the config arrives rather
#: than hung on where the daemon starts.
LOG_LEVELS = ("fatal", "error", "warn", "info", "debug", "trace", "silent")
_BANNERED = "silent"

#: The levels left once the unreadable one is out, which is what a flow that named something
#: Kimi has never heard of is offered. Listing `silent` there would be offering the refusal
#: below as the way out of the refusal above.
READABLE = tuple(level for level in LOG_LEVELS if level != _BANNERED)

#: The highest port there is, which is as much as this driver can say about one: whether the
#: number is free is the system's answer and it gives it by refusing to bind, not by refusing
#: the config. 0 is the useful end of the range and the default -- it means ask for any.
_PORT_MAX = 65535


@dataclass
class _Running:
    """The turn under way, if one is: which session it is in, and what it is running at.

    Written as the turn opens and cleared when it is over, read by whoever wants to put a word
    in. A prompt sent to a session that is working is queued rather than run, and steering it
    is what moves it into the turn already running instead of leaving it for the next one.
    """

    session: str | None = None
    config: dict[str, Any] = field(default_factory=dict[str, Any])
    #: The daemon the turn is running on, and the prompt it is: what a cut aborts, and what
    #: the turn asks what it is waiting on -- never a daemon started after it was put down.
    server: _AppServer | None = None
    prompt: str | None = None


#: How often a running turn is asked whether it is still running, how long one call may take,
#: and how long a daemon being taken down is given to go before it is left to the system.
_POLL_SECONDS = 1.0
_CALL_SECONDS = 60.0
_STOP_SECONDS = 5.0

#: The one line `kimi fork` prints, which is where the session it cut is named.
_FORKED = re.compile(r"Forked to (\S+)")

#: Event-capable daemons need polling only as recovery while the model is thinking.
_RECOVERY_SECONDS = 10.0

#: Pending approvals and questions arrive as flat `interaction` frames. The daemon offers no
#: REST list route, so these frames are both the wake-up and the authoritative payload.


def _websockets(name: str) -> ModuleType:
    """Loads part of the websocket client only when a kimi session needs one."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as why:
        if why.name != "websockets":
            raise
        raise ModuleNotFoundError(_EXTRA) from why


def _failure() -> type[Exception]:
    """What the websocket client raises, which every read of one here is caught by."""
    module = _websockets("websockets.exceptions")
    return cast("type[Exception]", vars(module)["WebSocketException"])


def _update(socket: ClientConnection, deadline: float) -> dict[str, Any]:
    """Read one bounded event frame, refusing malformed or stalled event streams."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    message = json.loads(socket.recv(timeout=remaining))
    if not isinstance(message, dict):
        raise TypeError("Kimi event frames must be objects")
    return cast("dict[str, Any]", message)


class _Updates:
    """Wake a session reader when its official daemon has something new to read.

    REST remains authoritative for history and goals. Pending approvals and questions arrive
    as flat interaction frames, and are answered through their interaction ids. A lost
    subscription falls back to polling history and usage without resubmitting the prompt.

    Spending is carried here rather than merely announced: each turn's completed total replaces
    its running per-step figures. The REST turn list supplies the same totals on recovery or
    when a subscription is unavailable; :meth:`KimiCodeCLISession._counting` reads both on its
    usual second, without counting a completed turn's steps twice.
    """

    def __init__(
        self,
        base: str,
        token: str,
        session: str,
        turns: dict[tuple[str, str], Counter[str]],
        steps: dict[tuple[str, str], dict[int, Counter[str]]],
    ) -> None:
        """Subscribes to one session of the daemon, falling back to polling if it cannot.

        Args:
          base: Where the daemon is listening.
          token: What it takes to be let in.
          session: The one session of it this listener carries, and the only one it counts.
          turns: Each agent turn's latest cost, shared with the session's REST reads.
          steps: Each running turn's per-step costs, replaced as notifications arrive.
        """
        # Loaded here rather than where this module is read, so that an install without
        # the `kimi` extra still holds every other backend. Before the try below, because
        # what the client raises is half of what that try catches.
        connect = cast(
            "Callable[..., ClientConnection]",
            vars(_websockets("websockets.sync.client"))["connect"],
        )
        self._failed = _failure()
        self._session = session
        self._turns = turns
        self._steps = steps
        self._agents = {"main"}
        self._socket: ClientConnection | None = None
        self._contexts = contextlib.ExitStack()
        self.ended = False
        #: What the daemon said the last turn failed of, or "" for one that did not: a turn
        #: that failed stops the session as a turn that answered does, with nothing to say.
        self.failed = ""
        #: A flat approval or question waiting for its response.
        self.pending: list[dict[str, Any]] = []
        try:
            socket = self._contexts.enter_context(
                connect(
                    base.replace("http:", "ws:", 1) + "/ws",
                    additional_headers={"Authorization": f"Bearer {token}"},
                    proxy=None,
                    compression=None,
                    ping_interval=None,
                    open_timeout=1,
                    close_timeout=0.1,
                )
            )
            self._socket = socket
            socket.send(
                json.dumps(
                    {
                        "type": "subscribe",
                        "request_id": "hmz",
                        "session_id": session,
                    }
                )
            )
            deadline = time.monotonic() + 1
            while True:
                message = _update(socket, deadline)
                if (
                    message.get("type") == "response"
                    and message.get("request_id") == "hmz"
                ):
                    if message.get("code") != 0:
                        self.close()
                    break
        except (OSError, self._failed, ValueError, TypeError):
            self.close()

    def wait(self, *, settled: bool) -> None:
        """Wait for progress, falling back to ordinary polling after a disconnect."""
        socket = self._socket
        if socket is None:
            time.sleep(_POLL_SECONDS)
            return
        # Keep the extra REST read after completion, without its polling delay once the
        # daemon has confirmed that this turn's final events have been published.
        if settled and self.ended:
            return
        started = time.monotonic()
        deadline = started + (_POLL_SECONDS if settled else _RECOVERY_SECONDS)
        try:
            while True:
                message = _update(socket, deadline)
                if message.get("type") == "ping":
                    socket.send(json.dumps({"type": "pong"}))
                    continue
                if message.get("session_id") != self._session:
                    continue
                kind = message.get("type")
                payload = message
                agent = str(message.get("agent_id") or "main")
                main = agent == "main"
                if kind == "task":
                    child = str(message.get("child_agent_id") or "")
                    if child and child != "main" and child not in self._agents:
                        self._agents.add(child)
                        socket.send(
                            json.dumps(
                                {
                                    "type": "subscribe",
                                    "request_id": f"hmz-{len(self._agents)}",
                                    "session_id": self._session,
                                    "agent_ids": [child],
                                }
                            )
                        )
                if kind in ("turn.started", "turn.ended"):
                    turn = payload.get("turn")
                    if isinstance(turn, dict):
                        turn = cast("dict[str, Any]", turn)
                        _remember_turn(
                            self._turns,
                            self._steps,
                            str(message.get("agent_id") or "main"),
                            turn,
                        )
                elif kind == "turn.updated":
                    step = payload.get("step")
                    if isinstance(step, dict):
                        step = cast("dict[str, Any]", step)
                        _remember_step(
                            self._turns,
                            self._steps,
                            str(message.get("agent_id") or "main"),
                            str(payload.get("turn_id") or ""),
                            step,
                        )
                if kind == "turn.started":
                    if main:
                        self.ended, self.failed = False, ""
                elif kind == "turn.ended":
                    if main:
                        self.ended = True
                        ending = cast("dict[str, Any]", payload.get("turn") or {})
                        if ending.get("outcome") == "failed":
                            self.failed = _erred(ending.get("error"))
                    return
                elif kind == "interaction":
                    named = str(message.get("interaction_id") or "")
                    self.pending = [
                        one
                        for one in self.pending
                        if one.get("interaction_id") != named
                    ]
                    if message.get("status") == "pending" and named:
                        self.pending.append(message)
                    if not settled:
                        return
                elif kind == "turn.updated":
                    deadline = min(deadline, started + _POLL_SECONDS)
                elif not settled:
                    return
        except TimeoutError:
            return  # REST polling is always available to finish the turn.
        except (OSError, self._failed, ValueError, TypeError):
            self.close()

    def close(self) -> None:
        """Release the subscription, including a partially completed handshake."""
        self._socket = None
        with contextlib.suppress(OSError, self._failed):
            self._contexts.close()


#: Every kind of token counts. `input_other` excludes cached input, so adding these four
#: fields accounts for all traffic exactly once.
_KINDS = {
    "input": "input_other",
    "output": "output",
    "cache_read": "input_cache_read",
    "cache_write": "input_cache_creation",
}


def _cost(turns: Mapping[tuple[str, str], Counter[str]]) -> Counter[str]:
    """The latest cost of every turn this session has learned about."""
    return sum(turns.values(), Counter[str]())


def _remember_step(
    turns: dict[tuple[str, str], Counter[str]],
    steps: dict[tuple[str, str], dict[int, Counter[str]]],
    agent: str,
    turn: str,
    step: Mapping[str, Any],
) -> None:
    """Replace a step's cost, keeping work still being retried out of the final total."""
    if not turn:
        return
    current = steps.setdefault((agent, turn), {})
    index = step.get("index")
    at = int(index) if isinstance(index, (int, float)) else len(current)
    current[at] = _spent(step)
    turns[(agent, turn)] = sum(current.values(), Counter[str]())


def _remember_turn(
    turns: dict[tuple[str, str], Counter[str]],
    steps: dict[tuple[str, str], dict[int, Counter[str]]],
    agent: str,
    turn: Mapping[str, Any],
) -> None:
    """Replace a turn's cost, rather than add another copy of it."""
    named = agent, str(turn.get("turn_id") or "")
    if not named[1]:
        return
    if turn.get("status") == "ended":
        turns[named] = _spent(turn)
        steps.pop(named, None)
        return
    current = steps.setdefault(named, {})
    for index, step in enumerate(turn.get("steps") or []):
        if not isinstance(step, dict):
            continue
        step = cast("dict[str, Any]", step)
        at = step.get("index")
        current[int(at) if isinstance(at, (int, float)) else index] = _spent(step)
    turns[named] = sum(current.values(), Counter[str]())


def _spent(payload: Mapping[str, Any]) -> Counter[str]:
    """What one completed step of a turn says it cost, by the kind this package counts in.

    Args:
      payload: A turn or one of its steps, with an optional `usage` object.

    Returns:
      Its four counts, and nothing at all for a step that carried none -- `usage` is optional
      in the daemon's own schema, and a step that does not say is a step that adds nothing
      rather than one that cost nothing.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return Counter()
    said = cast("dict[str, Any]", usage)
    return Counter(
        {
            kind: tokens
            for kind, named in _KINDS.items()
            # Nothing below zero. A running total these are added into is a total a turn's
            # spending is the rise across, so one negative count would be this session buying
            # back tokens it had already been charged for -- and the rise it hid would never
            # be reported at all.
            if isinstance(count := said.get(named), (int, float))
            and (tokens := int(count)) > 0
        }
    )


#: What each kind of block a message is written in reads as. A block of a kind that is not here
#: is not shown: an image is not a line of a transcript.
_BLOCKS = {"text": "text", "thinking": "reasoning", "tool_use": "tool"}

#: The blocks whose words go on arriving after the block does, which is what makes the last of
#: them half a sentence rather than a thing to show. A tool block is named the moment it is
#: there and grows no further, so a turn that reaches for one says so as it reaches.
_GROWS = ("text", "thinking")

#: Approvals and questions are flat `interaction` frames. Pending ones are re-sent after
#: every accepted subscription, and `/interactions/<id>` resolves them with their kind and
#: the answer. An approval carries its tool, action and input in `request`; a question carries
#: the questions and their options there instead. Resolution is one-shot rather than a stored
#: session-wide grant: `scope` is deliberately omitted.
#:
#: `scope` is the one optional field with teeth, and it is left off deliberately. `session`
#: there writes the approval down as a rule the session keeps, so every later call matching it
#: is approved by the daemon without ever reaching this route -- which would be a hook hung on
#: `PERMISSION_REQUEST` seeing the first `Bash` of a turn and none of the twenty after it. An
#: approval answered here is answered once, for the call that raised it. `selected_label` is
#: the one field this driver has nothing to put in: it names which of the options a plan
#: review offered was taken, and a plan review is a thing shown to a person.
#:
#: A rejected answer carries the reason to the model as a tool refusal, so the turn can
#: continue without executing the command or waiting for somebody at a keyboard.
_ANSWERS = ("approved", "rejected")

#: How each rung of the ladder is set on a Kimi session. The daemon takes one of `manual`,
#: `yolo` and `auto`, and plan mode beside it, and the two are said together because neither
#: means much alone.
#:
#: The names mislead, so here is 0.42.0's own ladder. Kimi calls them Always Ask (`manual`),
#: Ask When Needed (`yolo`) and Never Ask (`auto`), loosest last -- `yolo` is the middle rung
#: and not the top one the word suggests. What decides it is the order its permission policies
#: are consulted in, the first to answer winning: the one that approves everything on `auto`
#: is consulted fourth, ahead of every policy that would ask, while the one that approves
#: everything on `yolo` is consulted tenth, behind them. So a `yolo` turn still stops to ask
#: before a Bash command the parser rates dangerous *or cannot parse at all*, before a
#: sensitive file, before a path under `.git`, and -- in plan mode -- before the `ExitPlanMode`
#: the model is told to end a plan with. `manual` adds no policy of its own at all: it is the
#: absence of the other two, so what survives it is the list of tools approved by default --
#: `Read`, `Grep`, `Glob`, `WebSearch`, `FetchURL`, `Agent`, `Skill`, `AskUserQuestion` and
#: the plan and todo tools -- plus a `Write` or `Edit` inside a git worktree's own workspace.
#: Bash asks, every time.
#:
#: Each of those is an approval and not a question, and until this driver answered its
#: interaction that made all three of the tighter modes unreachable: a turn stopped on one
#: stayed busy with nothing being answered, and ended as the watchdog's stall. That is
#: what :meth:`KimiCodeCLISession._approved` buys back, so the table below is the ladder
#: rather than one rung written four times.
#:
#: What each row is, then, and why:
#:
#: `read-only` is `manual` with plan mode on. Plan mode is the hard half -- it vetoes `Write`,
#: `Edit`, `TaskStop` and the two cron tools outright, with no approval offered and nothing to
#: answer -- and `manual` is the half that catches what plan mode does not. Bash is the one
#: that matters: Kimi's own plan-mode reminder tells the model "Use Bash only when needed;
#: Bash follows the normal permission mode and rules", so at `auto` a rung named read-only
#: would run a command that writes a file. At `manual` that command is an approval instead --
#: and an approval this driver answers `rejected`, because this rung is not in
#: :data:`_GRANTED`. That is what makes the name true with no hook hung at all: the write is
#: refused by the rung itself, and a flow that wants the write allowed asks for a looser rung
#: rather than for a hook -- a hook can refuse what would have been granted, and there is
#: nothing here for it to widen. The same goes for the `ExitPlanMode` the model is told to end
#: a plan with: gated at every mode but `auto`, so at this rung the agent cannot quietly leave
#: the rung either.
#:
#: `auto` is `yolo`, for the reason Humanize's `auto` exists: the agent may ask for more than
#: it has, and the asking is granted. That is `yolo` exactly -- the dangerous command, the
#: sensitive file, the path under `.git`, each one asked for and each one answered -- and it
#: is the same row Codex's table writes as `on-request`.
#:
#: `workspace-write` and `bypass` are `auto`, which is the mode where nothing is asked. There
#: is no sandbox in this CLI, so `workspace-write` cannot be the fenced thing it is on Codex;
#: what it can be is the rung that changes the workspace without stopping, and that is `auto`.
#: `bypass` means nothing asked and nothing checked, which is what `auto` is.
#:
#: The one thing `auto` costs is the question. It denies `AskUserQuestion` outright -- "Make a
#: reasonable decision and continue without asking the user" -- so a turn at either of those
#: two rungs does not stop to ask a person, and `NOTIFICATION` does not fire there. At
#: `read-only` and `auto` it does: `AskUserQuestion` is approved by default under both `manual`
#: and `yolo`, and question interactions are answered.
#:
#: What would make `read-only` bite harder still is a rung that cannot reach for `Bash` at
#: all: no call to refuse is a fence, where an approval answered is a gate somebody has to
#: hold. It is not the `tools` key of this body, which is the thing to say plainly because
#: this file said the opposite. `sessionAgentConfigSchema` does declare `tools`, so a body
#: carrying one is taken with a 200 -- and `applySessionAgentConfig` then reads `model`,
#: `thinking`, `permission_mode`, `plan_mode`, `swarm_mode`, `tower_mode` and the two goal
#: keys, and nothing else. The allow-list is accepted and dropped, and an agent sent one
#: keeps all of its tools. What the daemon really takes is `disabled_tools`, on the *prompt*
#: body rather than this one, which `setSessionDisabledTools` applies to the session. Both
#: checked against 0.42.0, the first of them live. Which is somebody else's change to make
#: and is being made: it is a lever on the prompt, and this table sets the session.
_PERMITTED = {
    "read-only": {"permission_mode": "manual", "plan_mode": True},
    "workspace-write": {"permission_mode": "auto", "plan_mode": False},
    "auto": {"permission_mode": "yolo", "plan_mode": False},
    "bypass": {"permission_mode": "auto", "plan_mode": False},
    # An agent nobody wrote a rung for: neither key is sent, and the session runs at whatever
    # `kimi web` would have run it at on its own. Which is this file's own rule about defaults
    # applied to the one setting that had been breaking it -- an install that configures
    # nothing behaves as the bare CLI does. 0.42.0 reads `default_permission_mode` out of the
    # install's config as it bootstraps a session and leaves the mode at `manual` where there
    # is none, so what an install that says nothing gets is Always Ask, the same as the person
    # who starts `kimi web` at a terminal and answers it from the browser.
    #
    # Nobody is at the browser here, which is why this row used to be the one that wedged: the
    # first tool call of such a turn stopped the session somewhere the poll could not move it,
    # and the watchdog ended it as a stall a quarter of an hour later. It no longer does.
    # `manual` is a rung this driver can answer now, and saying nothing lands on it the same
    # way `read-only` does -- the approval is read, a hook gets it, and the turn goes on. On
    # `rejected`, where no hook said otherwise: what Humanize was told about this agent is
    # nothing, and nothing is not a yes. See :data:`_GRANTED`.
    UNSAID: {},
}

#: The two tools a Kimi agent reaches the web with, by the names 0.42.0 registers them under.
#: `WebSearch` is the search -- registered only where the install has a search provider, which
#: is why a daemon signed into nothing does not list it -- and `FetchURL` is the one that reads
#: a page it is given. Both are approved by default at every rung this driver sets, so neither
#: is a tool the permission ladder above ever takes away.
#:
#: They are withheld through `disabled_tools`, which the prompt body takes and the profile body
#: does not: `promptSubmissionSchema` declares it, and the route hands it to
#: `AgentToolPolicyService.setSessionDisabledTools` before the prompt is enqueued. What that
#: sets is session state rather than a word to the model. Two things read it, and both are the
#: whole of what the model is given: `activeEntries` filters the tool list a request carries by
#: `isToolActive`, so a disabled tool is not in the request at all; and the executor's own
#: call guard answers `Tool "<name>" is disabled by the active tool policy` for one reached for
#: anyway. The policy is session-scoped rather than agent-scoped, so a swarm's subagents are
#: inside it too.
#:
#: Said in both directions, because Kimi's is a list rather than a switch. `setSessionDisabledTools`
#: replaces the whole list and the state it replaces is written to the session's own
#: `state.json`, so a session resumed or forked from one that had the web withheld comes back
#: with it still withheld -- and an agent that may search would then be an agent that does not.
#: An empty list is what a session that was never told carries, so saying it costs a turn
#: nothing. An agent nobody was asked about says neither, the way it sends no rung.
#:
#: One thing it cannot do, which is true of every backend here: `Bash` can still `curl`. What
#: this takes away is the web the CLI hands its agent, which is what `web_search` is about.
_WEB_TOOLS = ("WebSearch", "FetchURL")

#: The kinds of skill a person may activate, as Kimi's `isUserActivatableSkillType` reads them:
#: an untyped skill, a prompt or inline one, and a flow. A `reference` is the model's to read,
#: and the server refuses it activated.
_ACTIVATED = (None, "prompt", "inline", "flow")

#: Kimi's own compaction, which has a route of its own.
_COMPACT = NativeCommand(
    "/", "compact", "compact the conversation, with optional instructions", "call"
)

#: The permission modes in which what the agent reaches for is granted rather than asked
#: about, which is the whole of what decides whether this driver may answer an approval yes.
#: `auto` approves before any policy can ask, so an approval under it is one Kimi did not
#: raise; `yolo` is the mode whose bargain is that the agent asks and is granted. `manual` is
#: neither: it withholds approval for everything but the reads, and an approval arriving
#: under it is the session doing exactly what the rung told it to.
_GRANTS = ("auto", "yolo")

#: The rungs at which an approval this driver reads is answered yes. Derived from the table
#: above rather than written beside it, so that a row whose mode moves takes its answer with
#: it and the two cannot drift apart.
#:
#: Two rows are out, and for one reason said two ways. `read-only` is `manual`: its whole
#: meaning is that the agent may look at anything and change nothing, and at `manual` the
#: reads are the part the daemon approves by itself -- so every approval that reaches this
#: driver at that rung is, by construction, a tool that is not a read. Answering those yes
#: would make the rung name a thing it does not do: plan mode already refuses the edits, and
#: the Bash that plan mode lets through is precisely what the approval is for. The silence is
#: out because it names no mode at all: an agent nobody wrote a rung for bootstraps at
#: `manual`, and granting there would be Humanize handing out, one call at a time, the `yolo`
#: that an agent declared at `auto` gets by asking for it.
#:
#: Refusing is not the same as leaving it: the daemon holds the tool until somebody resolves
#: it either way, so saying nothing would be the fifteen-minute stall this reading exists to
#: remove. `rejected` hands the model a line saying the tool was not run, with the reason --
#: so a turn at a withholding rung is one the agent is told no by and works around, which is
#: an outcome a flow can read.
_GRANTED = frozenset(
    rung
    for rung, settings in _PERMITTED.items()
    if settings.get("permission_mode") in _GRANTS
)


class _AppServer:
    """A `kimi web` daemon of our own, and the calls one turn of a session is made of."""

    def __init__(
        self,
        argv: list[str],
        env: Mapping[str, str] | None = None,
        home: Home | None = None,
    ) -> None:
        """Starts the daemon and waits for it to say where it is listening.

        Args:
          argv: The command that starts it, already wrapped for wherever its work is to land.
          env: The whole environment to start it in, which is this process's own less what the
            agent's provider hushes and plus what it sets, or None to inherit this one. The
            daemon is the agent's, so its account is the agent's too.
          home: The virtual home it is started in, which is dropped once it is down; None for
            one started in the user's own.

        Raises:
          subprocess.CalledProcessError: If it stops without ever saying, which would leave a
            flow waiting on a server that is not there. Reported as a failed turn, because it
            is the turn that starts it that has nowhere to run.
        """
        self._argv = argv
        self._stopping = threading.Lock()
        self._stopped = False
        #: The virtual home it was started in, dropped once it is down.
        self.home = home
        self._proc = subprocess.Popen(
            argv,
            # Its log is nobody's to read; what is wanted from it is the one line below.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            env=dict(env) if env else None,
            start_new_session=os.name != "nt",
        )
        assert self._proc.stdout is not None  # noqa: S101
        for line in self._proc.stdout:
            if (listening := _LISTENING.match(line.strip())) is not None:
                self._base = f"{listening[1]}/api/v1"
                self._token = listening[2]
                break
        else:
            raise Failed(
                self._proc.wait(), argv, "", f"{argv[0]} stopped without listening"
            )
        # A pipe nobody drains stops the daemon writing to it, so the rest of the log is read
        # and dropped rather than left to fill.
        threading.Thread(
            target=collections.deque, args=(self._proc.stdout, 0), daemon=True
        ).start()

    def call(
        self, method: str, path: str, body: Any = None, seconds: float = _CALL_SECONDS
    ) -> Any:
        """Makes one call to the daemon.

        Args:
          method: The HTTP method to make it with.
          path: The path under the daemon's API root.
          body: What to send as JSON, or None to send nothing.
          seconds: How long it is given.

        Returns:
          What the daemon answered with, unwrapped from its envelope.

        Raises:
          subprocess.CalledProcessError: If it refuses the call or cannot be reached, which is
            a failed turn however it failed -- reported the way every other backend reports one,
            so that a flow catches turns rather than transports.
        """
        # The address is the one this process just watched its own server announce, so the
        # scheme is http and the host is loopback whatever the audit rule fears.
        request = urllib.request.Request(  # noqa: S310
            self._base + path,
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=seconds) as response:  # noqa: S310
                said: dict[str, Any] = json.load(response)
        except urllib.error.HTTPError as refused:
            raise Failed(
                refused.code, self._argv, "", refused.read().decode(errors="replace")
            ) from refused
        except OSError as unreachable:  # a daemon that died, or a call that timed out
            raise Failed(1, self._argv, "", str(unreachable)) from unreachable
        # A refusal arrives inside a 200: a word steered into a turn that has already ended
        # comes back as `{"code": 40402, "msg": ...}` with the status still OK. Read as an
        # answer, that is a word which never landed reading as one that did.
        if said.get("code"):
            raise Failed(
                1, self._argv, "", f"{path}: {said.get('msg') or said['code']}"
            )
        return said.get("data")

    def busy(self, session: str) -> bool:
        """Whether the session has a main turn, subagent, or background task running."""
        route = f"/sessions/{session}"
        held = self.call("GET", route)
        if "busy" not in held:
            raise Failed(1, self._argv, "", f"{route}: no session status in its answer")
        return bool(held["busy"] or held.get("main_turn_active", False))

    def stop(self) -> None:
        """Takes the daemon and its children down, leaving its sessions on disk."""
        with self._stopping:
            if self._stopped:
                return
            self._stopped = True
            if os.name == "nt":
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=_STOP_SECONDS)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                self._proc.wait()
                if self.home is not None:
                    self.home.drop()
                return

            # Provider wrappers and Kimi share this dedicated group. Taking down the group
            # prevents a stopped flow from leaving either wrapper or daemon behind. And the
            # rest of the tree after it: Kimi starts each command a tool runs in a session of
            # its own, which the group does not reach, so a command a turn was in when its
            # daemon went would otherwise run on to its end with nothing waiting for it.
            import psutil

            kin: list[psutil.Process] = []
            if self._proc.poll() is None:
                # Only while it is ours: the pid of one already gone and reaped is anybody's.
                with contextlib.suppress(psutil.Error):
                    kin = psutil.Process(self._proc.pid).children(recursive=True)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._proc.wait(timeout=_STOP_SECONDS)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self._proc.pid, signal.SIGKILL)
            self._proc.wait()
            for one in kin:
                with contextlib.suppress(psutil.Error):
                    one.kill()
            if self.home is not None:
                self.home.drop()


@dataclass(frozen=True, kw_only=True)
class KimiCodeCLIAgentConfig(AgentConfig):
    """What Kimi Code is configured with: a model, an effort saying width too, and a daemon.

    The daemon fields are here because of a rule this repository holds every backend to: the
    default of an option is the harness's own default, so that an install which sets nothing
    behaves as the bare CLI does, and anything Humanize puts on top is a field somebody can
    take back off. Three of the four below break that rule on purpose, and say which way.
    ``kimi web`` is written for a person who runs one of them at a terminal and wants a
    browser; this driver runs one per agent, unattended, and reads its first line. Left at the
    CLI's own defaults, two agents would collide on port 58627, eight would open eight browser
    windows, and none of them would print an address a machine can read. So the defaults here
    are the unattended ones, the departure is named in each attribute, and a flow that wants
    the CLI's behaviour back can have it -- except at ``silent``, which is the one setting that
    would leave the daemon unreachable rather than merely different.

    Attributes:
      effort: How hard to think, in Kimi's own wording, optionally prefixed `swarm` to run
        every turn as a fleet of subagents rather than as one agent -- `max` and `swarmmax` are
        the same thinking at either width.
      port: What to bind the daemon to. 0, not the CLI's 58627, and this is the one departure
        that is not a preference: one daemon per agent means a flow with two Kimi agents
        starts two, and a fixed port makes the second one fail to bind. 0 asks the system for
        a free one and the ready line says which it got. Setting it -- for a flow that means
        to reach the same daemon from outside, say -- is taking on both of the ways that can
        go wrong: that nothing else on the machine wants that port, and that this agent never
        starts a second daemon. It starts one whenever its account changes, and lets go of the
        old one rather than stopping it, so an agent that falls back on a fixed port tries to
        bind a port its own predecessor is still holding and fails the turn that asked.
      open_browser: Whether to open the web UI as the daemon comes up. False, where the CLI
        opens one, because a flow running unattended has no browser to open and a run on a
        headless machine would be asking `xdg-open` to do something about it. True is for
        watching a flow work: the UI is a real client of the same session the turns go to.
      log_level: How loudly the daemon logs, out of :data:`READABLE`. `error`, where the CLI
        logs nothing, and this is the departure that is load-bearing rather than merely
        sensible: at `silent` -- which is what omitting the flag means -- `kimi web` draws its
        ready output as a banner instead of printing the one line :data:`_LISTENING` reads, and
        a daemon whose address cannot be read is a daemon no turn can be submitted to. `error`
        is the quietest level that still prints the line, so it buys the address at the least
        noise. Raise it to watch the daemon; `silent` is refused.
      web_title: What the web UI calls itself in a browser tab, or None for the CLI's own
        `<workspace dir> | Kimi Code`. Worth having only because there is one daemon per agent:
        opened side by side, eight agents on one project are eight identical tabs, and the
        agent's name in the title is what tells them apart.
    """

    port: int = 0
    open_browser: bool = False
    log_level: str = "error"
    web_title: str | None = None

    def __post_init__(self) -> None:
        """Checks the daemon settings, where a flow can still be told which one it got wrong.

        Raises:
          ValueError: If the port is not one, if the log level is not one Kimi takes, or if it
            is `silent` -- which the daemon accepts and this driver cannot read back.
        """
        super().__post_init__()
        if not 0 <= self.port <= _PORT_MAX:
            raise ValueError(f"port must be between 0 and {_PORT_MAX}, not {self.port}")
        if self.log_level == _BANNERED:
            raise ValueError(
                f"log_level {_BANNERED!r} makes `kimi web` draw its ready banner instead of "
                "printing the line this driver reads its address from"
            )
        if self.log_level not in READABLE:
            raise ValueError(
                f"log_level must be one of {', '.join(READABLE)}, "
                f"not {self.log_level!r}"
            )
        if self.web_title is not None and not self.web_title.strip():
            raise ValueError("web_title must say something, or be None for Kimi's own")


class KimiCodeCLISession(SessionBase):
    """A Kimi Code conversation, held by the app server and named by it as it opens.

    The id is the server's, handed out before the turn rather than read back out of a resume
    hint, so a second agent working alongside cannot be resumed by mistake.
    """

    _agent: KimiCodeCLIAgent  # every turn is submitted to the server this agent holds

    #: A prompt sent to a session the daemon is already working is queued behind as a turn of
    #: its own; steering moves it into the turn that is running, which is what makes it a word
    #: put in rather than one waiting.
    steers: ClassVar[bool] = True

    #: `kimi fork --cwd` cuts the child where it is told to work, out of the account's own
    #: store rather than one kept beside the directory the parent is in.
    forks_elsewhere: ClassVar[bool] = True

    #: The daemon holds every conversation of the agent, and nothing reaches into one of its
    #: turns to stop it.
    cuts_transport: ClassVar[bool] = True

    def __init__(
        self, agent: AgentBase, cwd: str | os.PathLike[str] | None = None
    ) -> None:
        """Initializes a session running nothing yet.

        Args:
          agent: The agent whose config every turn of this session runs at.
          cwd: The directory this conversation works in, as for `SessionBase`.
        """
        super().__init__(agent, cwd)
        #: The turn under way, which is what a word put in is steered into.
        self._running = _Running()
        #: What this session has cost so far, by kind, as the daemon counts it: a running
        #: total for the whole conversation, so what one turn cost is the rise across it.
        self._counted: Counter[str] = Counter()
        #: Each agent turn's latest cost, for the whole conversation rather than one turn.
        #: Running steps are replaced by their turn's final total rather than counted twice.
        self._turn_usage: dict[tuple[str, str], Counter[str]] = {}
        self._step_usage: dict[tuple[str, str], dict[int, Counter[str]]] = {}
        #: What the daemon was last told this session runs at, so a turn at the same
        #: settings does not tell it again -- and which daemon was told it. Both, because a
        #: profile is state that daemon holds rather than a fact about the session: an agent
        #: that fell back, or whose daemon the watchdog put down, is served by one that was
        #: never told, and a cache that outlived the thing it describes would leave that
        #: turn running at the CLI's own settings instead of at this agent's.
        self._profile: dict[str, Any] = {}
        self._profiled: _AppServer | None = None
        #: Interactions already answered: a later subscription replays pending interactions,
        #: but the driver must not post a second answer or fire the hook twice.
        self._answered: set[str] = set()
        #: The skills the server last listed for this conversation, by name, as native
        #: entries: kept for when the daemon is down, and what a native turn is read against.
        self._activatable: dict[str, NativeCommand] = {}

    def natives(self) -> list[NativeCommand]:
        """`/compact`, and every skill the server lists that a person may activate.

        Asked of the server holding the conversation each time, since a skill installed
        mid-chat is one Kimi offers at once; what it listed last stands while it will not say,
        or while no daemon is up to ask. Nothing before there is a conversation: there is
        nothing to list skills for, and a `/` line then is text.

        Returns:
          The catalog, as far as the server has said it.
        """
        session, server = self._id, self._agent._server
        if session is None:
            return []
        if server is not None:
            # A daemon that will not say, or says something this cannot read, leaves the
            # catalog as it last stood rather than failing whoever asked for it.
            with contextlib.suppress(subprocess.CalledProcessError, ValueError):
                said = server.call("GET", f"/sessions/{session}/skills")
                listed = (
                    cast("dict[str, Any]", said).get("skills")
                    if isinstance(said, dict)
                    else None
                )
                if not isinstance(listed, list):
                    return [_COMPACT, *self._activatable.values()]
                skills: dict[str, NativeCommand] = {}
                for one in cast("list[Any]", listed):
                    if not isinstance(one, dict):
                        continue
                    told = cast("dict[str, Any]", one)
                    name = str(told.get("name") or "")
                    # `/compact` is Kimi's compaction, whatever a skill is called.
                    if (
                        name not in ("", _COMPACT.name)
                        and told.get("type") in _ACTIVATED
                    ):
                        about = str(told.get("description") or "")
                        skills[name] = NativeCommand("/", name, about, "call")
                self._activatable = skills
        return [_COMPACT, *self._activatable.values()]

    def interject(self, text: str) -> None:
        """Puts a word into the turn already running, rather than into the next one.

        A prompt sent to a session that is working is queued, and would be answered as a turn
        of its own once this one ended. Steering moves it into the turn that is running, which
        is what makes it a word put in rather than a turn queued behind.

        Args:
          text: What to say to the agent.

        Raises:
          RuntimeError: If no turn is running, so there is none to steer it into.
          subprocess.CalledProcessError: If the daemon refuses either call.
        """
        running = self._running
        if running.session is None:
            raise RuntimeError("no turn is running to be talked to")
        # Named by its own words: Kimi mints a fresh id for a steered prompt, so the id it
        # answers with is not the one it took, and the words are what both ends have.
        self.steering(text, ticket=text)
        # The turn's own daemon, never one started for this: a word for a turn whose daemon
        # has been put down is a word with no turn to go into.
        server = running.server or self._agent.server
        queued = server.call(
            "POST",
            f"/sessions/{running.session}/prompts",
            {
                "content": [{"type": "text", "text": text}],
                **{k: v for k, v in running.config.items() if k != "plan_mode"},
            },
        )
        try:
            server.call(
                "POST",
                f"/sessions/{running.session}/prompts:steer",
                {"prompt_ids": [queued["prompt_id"]]},
            )
        except BaseException:
            self.unsteered(text)  # nothing is coming back for a word that never went in
            raise

    def _stream(
        self,
        prompt: str,
        *,
        schema: type[BaseModel] | None = None,  # noqa: ARG002
    ) -> Iterator[Event]:
        """Sends one turn, opening the session on the first call and resuming it after.

        The daemon answers a turn whole, but it writes the turn down as it goes: what the
        agent said is read back as it is written, which is what makes the turn watchable.

        A shape is not a setting of a turn here -- the daemon takes a prompt and nothing
        else about the answer -- so a turn asked for one has already been asked in the
        prompt, which is what :attr:`SessionBase.shapes` says of this backend.

        A native turn's `/compact` is the server's compaction, and a `/<skill>` the prompt
        with that skill activated -- the line as typed beside the activation, as Kimi's own
        prompt sends a skill typed into it; any other line, and any line before there is a
        conversation, is a prompt like any other.
        """
        name, _, rest = prompt.strip().partition(" ")
        rest = rest.strip()
        if self.native_turn and name.startswith("/") and self._id is not None:
            if name == f"/{_COMPACT.name}":
                yield from self._compact(self._id, rest)
                return
            named = name.removeprefix("/")
            if named not in self._activatable:
                # A skill installed since the catalog was last read, or a daemon not up
                # yet in this process to have read it from.
                self._agent.server  # noqa: B018 -- started, for the catalog to be asked of
                self.natives()
            if named in self._activatable:
                activated = {"name": named} | ({"args": rest} if rest else {})
                yield from self._submit(prompt, goal=False, skills=[activated])
                return
        yield from self._submit(prompt, goal=False)

    def _compact(self, session: str, instruction: str) -> Iterator[Event]:
        """Compacts the conversation as Kimi's own `/compact` does, and waits for it to finish.

        The route starts the compaction and answers at once; the session says it is busy for
        as long as the compaction runs, which is what the turn waits on. Over once it has
        been seen busy and then not, or not busy twice a poll apart: a compaction too quick
        to be seen running is one that has finished, and a session read once as idle before
        the server had marked it is not.

        Args:
          session: The conversation, by the id the daemon gave it.
          instruction: What to keep, or "" for Kimi's own say.

        Yields:
          The `result` saying it is done, with what it cost.

        Raises:
          subprocess.CalledProcessError: If the daemon refuses -- one compaction already
            running is refused -- or the turn is cut off while it waits.
        """
        with self._lock:
            server = self._agent.server
            server.call(
                "POST",
                f"/sessions/{session}:compact",
                {"instruction": instruction} if instruction else {},
            )
            seen, idle = False, 0
            while idle < (1 if seen else 2):
                if self._cut:
                    raise Failed(1, server._argv, "", f"cut off: {self._cut}")
                if server.busy(session):
                    seen, idle = True, 0
                else:
                    idle += 1
                if idle < (1 if seen else 2):
                    time.sleep(_POLL_SECONDS)
            costing = Usage()
            with contextlib.suppress(subprocess.CalledProcessError):
                costing = self._counting(server, session)
            spent = (
                {self._agent.config.model: int(costing.total)}
                if costing.total > 0
                else {}
            )
            yield Event(kind="result", text="compacted", tokens=spent, spent=costing)

    def _interact(self, session: str, updates: _Updates) -> None:
        """Answer the pending questions and approvals carried by interaction frames.

        The frames carry everything a response needs; there is no REST list to poll. Each is
        answered once, and remains queued for the existing bounded retry if its answer fails.
        """
        pending, updates.pending = updates.pending, []
        for interaction in pending:
            named = str(interaction.get("interaction_id") or "")
            if not named or named in self._answered:
                continue
            if interaction.get("kind") == "approval":
                self._approved(session, interaction)
            elif interaction.get("kind") == "question":
                self._asked(session, interaction)
            else:
                continue
            self._answered.add(named)

    def _approved(self, session: str, pending: dict[str, Any]) -> None:
        """Answers an approval interaction, and lets a hook refuse it.

        The daemon holds the tool until this interaction is resolved. Modes other than `auto`
        can therefore pause here indefinitely unless the driver answers.

        Answered yes at the rungs that mean granting -- :data:`_GRANTED` -- because nobody is
        at a prompt there and the rung has already said what the agent may do. Answered no at
        the two that do not: `read-only`, where every approval that gets this far is by
        construction a tool that is not a read, and the silence, where nothing said what this
        agent may do and so nothing said yes.

        Put to `PERMISSION_REQUEST` first either way, because this is the moment the backend
        actually waits on and so the one place a hook here can stop an agent doing something --
        and because a flow that hung one at a withholding rung asked to be the one deciding.
        A refusal, the hook's or the rung's, is `rejected` with the reason, which the daemon
        hands the model as a line saying the tool was not run: the turn goes on, having been
        refused, rather than ending or hanging.

        Once apiece, by the id the daemon gave it: an approval answered a second time is
        refused rather than resolved, and a refusal a second is a working turn dying inside
        the recovery window for having done its job twice.

        Args:
          session: The session the turn is running in.
          pending: The flat approval frame, with its request under `request`.

        Raises:
          subprocess.CalledProcessError: If the daemon refuses the answer.
        """
        if not (named := pending.get("interaction_id")):
            return
        request = cast("dict[str, Any]", pending.get("request") or {})
        asking = self._fire(
            Moment.PERMISSION_REQUEST,
            tool=str(request.get("tool_name") or ""),
            about=str(request.get("action") or ""),
            called=pending,
        )
        allowed, refused = _ANSWERS
        if asking.refused:
            answer = {
                "decision": refused,
                "feedback": asking.because or "refused by a hook",
            }
        elif (rung := self._agent.config.permission) in _GRANTED:
            answer = {"decision": allowed}
        else:
            answer = {
                "decision": refused,
                "feedback": f"this agent runs at {rung}, which withholds approval"
                if rung
                else "nothing said what this agent may do",
            }
        (self._running.server or self._agent.server).call(
            "POST",
            f"/sessions/{session}/interactions/{named}",
            {"kind": "approval", **answer},
        )

    def _asked(self, session: str, pending: dict[str, Any]) -> None:
        """Answers a question interaction, if the turn has stopped to ask anything.

        An answer is matched to an option label where possible. With nobody to answer, it is
        skipped, which lets the tool carry on.

        Args:
          session: The session the turn is running in.
          pending: The flat question frame, with its request under `request`.

        Raises:
          subprocess.CalledProcessError: If the daemon refuses the answer.
        """
        if not pending.get("interaction_id"):
            return
        answers: dict[str, dict[str, Any]] = {}
        request = cast("dict[str, Any]", pending.get("request") or {})
        for asked in cast("list[Any]", request.get("questions") or []):
            question = cast("dict[str, Any]", asked)
            offers: list[Any] = question.get("options") or []
            options = {
                str(cast("dict[str, Any]", option).get("label", "")).lower(): cast(
                    "dict[str, Any]", option
                ).get("id")
                for option in offers
                if isinstance(option, dict)
            }
            said = self._agent.asked(
                Question(
                    text=str(question.get("question") or question.get("header") or ""),
                    options=tuple(
                        str(cast("dict[str, Any]", option)["label"])
                        for option in offers
                        if isinstance(option, dict)
                        and cast("dict[str, Any]", option).get("label")
                    ),
                )
            )
            if said is None:
                answers[str(question["id"])] = {"kind": "skipped"}
            elif (chosen := options.get(said.strip().lower())) is not None:
                answers[str(question["id"])] = {
                    "kind": "single",
                    "option_id": chosen,
                }
            else:
                answers[str(question["id"])] = {"kind": "other", "text": said}
        (self._running.server or self._agent.server).call(
            "POST",
            f"/sessions/{session}/interactions/{pending['interaction_id']}",
            {"kind": "question", "answers": answers},
        )

    def _cuts(self) -> None:
        """Aborts the prompt the turn is running, which is what stops a command it is in.

        Before the daemon is put down, which is what cutting a turn here goes on to do:
        Kimi starts a tool's command in a session of its own, and aborting the prompt is how
        it is told to take that down -- the daemon going takes down only its own group.
        """
        running = self._running
        if running.server is None or running.session is None or not running.prompt:
            return
        with contextlib.suppress(subprocess.CalledProcessError, ValueError):
            running.server.call(
                "POST",
                f"/sessions/{running.session}/prompts/{running.prompt}:abort",
                seconds=_STOP_SECONDS,
            )

    def _lets_go(self) -> None:
        """Takes down the daemon, which is what a wedged turn here is waiting on.

        The agent's rather than this session's: one daemon serves every conversation with it,
        so there is nothing of this one's own to put down, and every other turn on this agent
        goes with it. The sessions survive: the next turn starts another daemon and resumes
        this one by the id it already has.
        """
        self._agent._down()

    def _counting(self, server: _AppServer, session: str) -> Usage:
        """What this session has spent since the last time it was asked.

        The daemon counts the whole conversation, so what has been spent since is the rise
        across it. Told to the meters as it is read, which is what a rate taken while the turn
        is still running is made of.

        The session aggregate is zero on current Kimi Code. Main-agent totals come from its
        turn list, which also recovers usage after a lost subscription; per-step frames and
        subscribed child agents fill in work as it runs. Completed turn totals replace their
        steps rather than adding another copy of them. Only the increase is reported.

        Args:
          server: The daemon holding the session.
          session: The session to ask about.

        Returns:
          The rise since the last reading, by kind, which is nothing at all where it has not
          moved.

        Raises:
          subprocess.CalledProcessError: If the daemon will not say, which is not a failed
            turn: what a run costs is worth nothing at the price of the run.
        """
        page = server.call("GET", f"/sessions/{session}/agents/main/turns")
        for one in cast("list[Any]", page.get("turns") or []):
            if isinstance(one, dict):
                _remember_turn(
                    self._turn_usage,
                    self._step_usage,
                    "main",
                    cast("dict[str, Any]", one),
                )
        counted = _cost(self._turn_usage)
        risen = Usage(
            {
                kind: tokens
                for kind in set(counted) | set(self._counted)
                if (tokens := counted[kind] - self._counted[kind]) > 0
            }
        )
        self._counted = counted
        self._spends(risen)
        return risen

    def _pursue(self, objective: str) -> str:
        """Runs the turn under a goal of Kimi's own, which its runtime steers until it is met.

        The objective is the prompt as well as the goal, which is what ``/goal`` does: the
        agent is told what to do, and the runtime is told what it is for.

        Returns:
          The agent's response once the goal is done with, stripped.
        """
        said = ""
        for event in self._submit(objective, goal=True):
            # Told to whoever is watching, since a goal does not run through `stream`: it is
            # one prompt and as many turns of the model as the objective takes.
            self._agent._heard(event)
            if event.kind == "result":
                said = event.text
        return said.strip()

    def _fork(self, whence: str) -> str:
        """Cuts a second conversation from one the daemon is holding, as `kimi fork` does.

        The CLI's own command rather than the daemon's route for it: `POST /sessions/<id>
        ::fork` is dispatched to a manager that knows none of the workspaces the sessions are
        actually in, and refuses every one of them as a session that does not exist. The
        command is the same engine doing the same thing from the outside, and it says which
        session it made on the one line it prints -- which is the whole of what is read back.

        Run as this agent's account and in this conversation's directory, so that a fork is
        cut where the conversation is and out of the store that account writes.

        Args:
          whence: The conversation to cut from, by the id the daemon gave it.

        Returns:
          The new session's id, which is this session's from here on.

        Raises:
          subprocess.CalledProcessError: If the fork will not run, will not answer, or runs
            and names no session -- which leaves this session unopened, so the next turn
            tries again.
        """
        argv = self._agent.spawned(
            ["kimi", "fork", whence, "--yes", "--cwd", self._workspace()], self.cwd
        )
        try:
            ran = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                check=False,
                env=self._environ(),
                # The same ceiling every call to this backend is under: a fork is one read
                # and one write of a conversation, and a flow waiting forever on one is a
                # flow that has stopped.
                timeout=_CALL_SECONDS,
            )
        except subprocess.TimeoutExpired as slow:
            out = slow.stdout or ""
            if found := _FORKED.search(
                out.decode(errors="replace") if isinstance(out, bytes) else out
            ):
                # Named before it ran out of time: a session the daemon holds and logs,
                # which is the run's whether or not the fork then finished.
                self._tells(found.group(1))
            raise Failed(
                1, argv, "", f"kimi fork took longer than {_CALL_SECONDS:.0f}s"
            ) from slow
        found = _FORKED.search(ran.stdout)
        if ran.returncode != 0 or found is None:
            if found is not None:
                # Named before it failed: the run's all the same, as one that works is.
                self._tells(found.group(1))
            raise Failed(
                ran.returncode or 1,
                argv,
                ran.stdout,
                ran.stderr.strip() or "kimi fork named no session",
            )
        return found.group(1)

    def _told(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """What one turn of this session runs at, in the profile and prompt halves.

        Two halves rather than one body because the two routes a turn is made of do not take
        the same keys. The session profile reads the model, the thinking level, the rung and
        the width; the prompt reads those except `plan_mode`, plus `disabled_tools`. In
        particular, `plan_mode: false` in a prompt does not leave read-only mode: the profile
        alone changes that. Both are `object()` schemas, which drop undeclared keys rather
        than refusing them, so filtering here matters more than what a request accepts.

        Returns:
          The settings both routes take, and beside them the ones only the prompt body has
          anywhere to put.
        """
        effort = self.effort
        turn: dict[str, Any] = {
            "model": self._agent.config.model,
            # The rung where there is one, and the width beside it. An agent at no rung sends
            # neither: Kimi then runs the model at its own thinking level, where a `thinking`
            # of "" is a level it has no word for. A fleet is a width rather than a rung, so
            # it is asked for only where a rung said so.
            **({"thinking": effort.removeprefix(SWARM)} if effort else {}),
            "swarm_mode": effort.startswith(SWARM),
            # What it may do without being asked, where a rung said. An agent at no rung
            # sends neither key and is left wherever this install's own `kimi web` leaves it,
            # which is what a config written before there was a rung to write comes back off
            # the disk as too.
            **_PERMITTED.get(self._agent.config.permission, _PERMITTED[UNSAID]),
        }
        # And whether the web is the agent's, where anybody said. An agent nobody was asked
        # about sends neither the key nor an empty list: the session is then left carrying
        # whatever it already carried, which for one this driver opened is nothing.
        #
        # A fence that cuts the network takes the web away whatever was said about searching.
        # `FetchURL` is stopped by the proxy, but `WebSearch` is a search a Kimi install asks
        # its own service for, and that service is on the hosts the fence has to keep open for
        # the model. So a search from inside the fence would get out through the one door that
        # was left. Saying the tools are disabled closes that door.
        searching = self._agent.config.web_search
        fence = self._agent.config.fence
        if fence is not None and not fence.online:
            searching = False
        web: dict[str, Any] = (
            {}
            if searching is None
            else {"disabled_tools": list(_WEB_TOOLS) if searching is False else []}
        )
        return turn, web

    def _submit(
        self,
        prompt: str,
        *,
        goal: bool,
        skills: list[dict[str, str]] | None = None,
    ) -> Iterator[Event]:
        """Runs one turn, saying what the agent says as it says it.

        A goal-driven turn is many turns of the model, and it is over when the session falls
        idle rather than when the first of them ends. The session is asked whether it is still
        running before its messages are read, so that nothing said between the two is missed.

        Args:
          prompt: The input prompt for this turn, which is the objective as well when it is
            a goal the session is being set.
          goal: Whether to set the prompt as the session's goal before sending it.
          skills: The skills to activate with the prompt, each by `name` and with the `args`
            it was typed with, as Kimi's own `/<skill>` activates one; None for none.

        Yields:
          What the agent said, in the order it said it, and the answer it ended on.

        Raises:
          subprocess.CalledProcessError: If the daemon refuses any of the calls a turn is made
            of, leaving the session unopened so that the next call retries the turn.
        """
        turn, web = self._told()
        updates: _Updates | None = None
        with self._lock:  # a conversation is a sequence: one turn at a time
            # A turn that failed is as over as one that landed: neither leaves
            # anything for a word to be steered into.
            try:
                server = self._agent.server
                # A session of its own per attempt while this one is unopened: an opening turn that
                # failed leaves the daemon holding a conversation nothing landed in, and resuming
                # that one would be resuming a turn that never happened.
                if (session := self._id) is None:
                    # A fork cuts its conversation from the one it came from rather than
                    # starting one; anything else starts one here.
                    session = (
                        self._fork(self._forked_from)
                        if self._forked_from is not None
                        else server.call(
                            "POST",
                            "/sessions",
                            {"metadata": {"cwd": self._workspace()}},
                        )["id"]
                    )
                    # The run's from here, as the daemon names it: setting the session up,
                    # prompting it and pursuing a goal in it may each fail yet, and what the
                    # daemon holds and logs under this id is the run's whether or not they do.
                    self._tells(session)
                    # A session of its own is a session set nothing yet.
                    self._profile, self._profiled = {}, None
                # The settings are the session's rather than the turn's, which is the whole
                # reason there is a server here -- so a second turn at the same settings is
                # a session that already has them. Said again only when it would say
                # something different, because one daemon serves every session of its agent
                # and a call that changes nothing is a call the others wait behind. A goal
                # is the exception: it is something the runtime is set going on rather than
                # a setting it holds, and the same objective again is asking for it again.
                profile = turn | ({"goal_objective": prompt} if goal else {})
                if goal or profile != self._profile or server is not self._profiled:
                    server.call(
                        "POST",
                        f"/sessions/{session}/profile",
                        {"agent_config": profile},
                    )
                    self._profile, self._profiled = profile, server
                updates = _Updates(
                    server._base,
                    server._token,
                    session,
                    self._turn_usage,
                    self._step_usage,
                )
                # Said before the prompt goes in, so that a word put in has a session to be
                # steered into from the moment there is a turn to interrupt.
                # The web is part of what the turn is running at rather than part of what the
                # session was set to, so it is carried here: a word steered in is a prompt
                # submitted to the same route, and one that said nothing about the web would
                # be the one body of this session that did not.
                self._running = _Running(
                    session=session, config=turn | web, server=server
                )
                posted = server.call(
                    "POST",
                    f"/sessions/{session}/prompts",
                    {
                        "content": [{"type": "text", "text": prompt}],
                        **{k: v for k, v in (turn | web).items() if k != "plan_mode"},
                        **({"skills": skills} if skills else {}),
                    },
                )
                since = posted["user_message_id"]
                self._running.prompt = str(posted.get("prompt_id") or "") or None
                answer = ""
                shown: dict[
                    str, int
                ] = {}  # how much of each message has been passed on
                settled = False
                refused = 0.0  # since when the daemon has not answered about questions
                # Whether anything at all has been seen of this turn: the session working,
                # the daemon saying the turn ended, or a word of it written down. Until one
                # of them a session that is not busy is a session that has not started --
                # the daemon takes a prompt before it runs it -- and a turn read back there
                # would answer with what had not been said. A daemon that says none of the
                # three for a whole recovery window has said all it is going to.
                stirred = False
                opened = time.monotonic()
                costing = Usage()  # what this turn has come to, added up as it goes
                # When each of the two readings that are not made every time round was last
                # made, so that a turn told nothing still makes them at the cadence it made
                # them at before there were notifications at all. What the daemon says brings
                # one forward; nothing the daemon says can put one off. One `kimi web` serves
                # every session of its agent from one thread, so a reading skipped here is
                # not this turn going faster -- it is the other seven not queueing behind it.
                read_at = {"asked": 0.0, "counted": 0.0}

                def due(what: str, *, told: bool = False) -> bool:
                    if not told and time.monotonic() - read_at[what] < _POLL_SECONDS:
                        return False
                    read_at[what] = time.monotonic()
                    return True

                # Under a clock. Nothing here blocks forever on its own -- every call the
                # daemon takes is bounded -- but a session that answers `busy` and never says
                # another word is a turn that polls until somebody stops it, which is the same
                # hang read off a different pipe. So the clock is patted by what the agent
                # says rather than by the daemon answering, and looks at the daemon's own
                # process before believing anything is wrong.
                with Watchdog(self, riding=lambda: server._proc) as watch:
                    while True:
                        if self._cut:
                            # Ended here rather than on the next call to a daemon that has
                            # been put down, which is a call that fails -- or, for one that
                            # asks the agent for its daemon, one that starts another.
                            raise Failed(1, server._argv, "", f"cut off: {self._cut}")
                        # Interactions are answered before anything else: the turn is waiting
                        # on one, not merely streaming past it. An answer refused for a short
                        # while is retried; one refused for a whole recovery window is a failed
                        # turn rather than an unattended session waiting forever.
                        if updates.pending or due("asked"):
                            try:
                                self._interact(session, updates)
                            except subprocess.CalledProcessError:
                                refused = refused or time.monotonic()
                                if time.monotonic() - refused >= _RECOVERY_SECONDS:
                                    raise
                            else:
                                refused = 0.0
                        # Whether it is still working, which is the reading a turn ends on --
                        # and the one read here that is made every time round, because it is
                        # also what paces the round. A session read as stopped is a wait of a
                        # second below rather than another notification, so a reader that put
                        # this on a cadence of its own would spend that second reading the
                        # history over and over instead of waiting the frames out. Measured, and
                        # written down in `bench/cli-concurrency/REJECTED.md`.
                        busy = server.busy(session)
                        # What it has come to so far, asked for each time round rather than once
                        # at the end: a turn is minutes long, and a rate that only moved when one
                        # ended would stand still for all of them. A daemon that will not say is
                        # not a failed turn. A second apart and no faster, whatever the daemon
                        # says has landed: a meter is not something a turn waits on, and what a
                        # skipped reading would have added is still in the total, which is read
                        # once more as the turn settles.
                        if due("counted"):
                            with contextlib.suppress(subprocess.CalledProcessError):
                                costing = costing + self._counting(server, session)
                        if goal and not busy:
                            # A goal runs through the quiet between its turns: Kimi starts
                            # the next one itself once the session falls still, so a session
                            # that has stopped is a goal that has stopped only when the goal
                            # is no longer being pursued.
                            pursued = server.call("GET", f"/sessions/{session}/goal")
                            busy = pursued is not None and pursued["status"] == "active"
                        said = server.call(
                            "GET", f"/sessions/{session}/messages?after_id={since}"
                        )["items"]
                        stirred = (
                            stirred
                            or busy
                            or updates.ended
                            or bool(said)
                            or time.monotonic() - opened >= _RECOVERY_SECONDS
                        )
                        # A message is readable while it is still being written, so the turn
                        # is read again from its own first message every poll rather than
                        # once: a message put aside as seen would be the one the agent had
                        # only started saying. Newest first, and a turn reads forwards; what
                        # has been passed on is not passed on twice.
                        #
                        # And a message readable while it is being written grows its last
                        # block in place, so a block of words at the end of the newest one is
                        # half a sentence until something follows it: shown as it stands it is
                        # a paragraph broken across as many parts as the turn was polled, with
                        # the rest of it never shown at all, the array having grown no longer.
                        # So that one waits for the block after it, or for the turn to be over.
                        # Only that one: every earlier block is finished, and a tool block is
                        # whole the moment it is there -- a turn that reached for something
                        # says so as it reaches.
                        writing = (
                            ""
                            if settled
                            else next(
                                (
                                    str(one["id"])
                                    for one in said
                                    if one["role"] == "assistant"
                                ),
                                "",
                            )
                        )
                        for message in reversed(said):
                            if message["role"] != "assistant":
                                # A word put into the turn, spliced into the conversation at the
                                # step that takes it in: that splice is the agent saying it has
                                # it, and is the only thing here that is not the agent talking.
                                # Read as one, it would show your own words back as the agent's.
                                if message["id"] not in shown:
                                    shown[message["id"]] = len(message["content"])
                                    words = "".join(
                                        block.get("text") or ""
                                        for block in message["content"]
                                        if block.get("type") == "text"
                                    )
                                    if self.took(words) is not None:
                                        watch.saw()
                                        with watch.held():
                                            yield Event(kind="took", text=words)
                                continue
                            content = message["content"]
                            ready = len(content) - (
                                1
                                if content
                                and message["id"] == writing
                                and str(content[-1].get("type")) in _GROWS
                                else 0
                            )
                            for block in content[shown.get(message["id"], 0) : ready]:
                                kind = _BLOCKS.get(str(block.get("type")))
                                # A tool is named by what it is; everything else is what it says,
                                # which a block keeps under its own name -- text under `text`.
                                words = str(
                                    (
                                        block.get("tool_name")
                                        if kind == "tool"
                                        else block.get(str(block.get("type")))
                                    )
                                    or ""
                                )
                                if kind is None or not words.strip():
                                    continue
                                if not self._agent._watchers:
                                    # On stderr, where every other backend puts its progress:
                                    # a turn nobody can watch is a flow that reads as hung for
                                    # as long as the turn takes. Something watching the agent
                                    # shows the turn itself, and would be showing it twice.
                                    say(words, sys.stderr)
                                watch.saw()
                                with watch.held():  # a pause of ours is not its silence
                                    yield Event(kind=kind, text=words)
                            shown[message["id"]] = ready
                        # And the answer is taken fresh each time, so that it is what the
                        # agent ended up saying rather than what it had said when it was
                        # first readable.
                        for message in (
                            said
                        ):  # newest first: the last thing said that has any words
                            text = "".join(
                                block["text"]
                                for block in message["content"]
                                if block["type"] == "text"
                            )
                            if message["role"] == "assistant" and text:
                                answer = text
                                break
                        if stirred and settled and not busy and updates.failed:
                            # Stopped as a turn that answered stops, and with nothing to say:
                            # read as one, it was a loop going round on an empty turn as the
                            # work of the turn before it.
                            raise Failed(1, server._argv, answer, updates.failed)
                        if stirred and settled and not busy:
                            # Taken note of before it is passed on: a turn that landed is a session
                            # this agent opened, whether or not there is anywhere left to say so.
                            self._adopt(session)
                            if not self._agent._watchers:
                                # Where the CLI would have put the response. Something watching
                                # the agent has had it already, as the turn said it.
                                say(answer, sys.stdout)
                            # What the turn cost: the daemon counts the whole session, so what
                            # this turn spent is the rise across it, added up as the turn went.
                            # Asked once more here and never at the cost of the answer -- a turn
                            # that landed has landed, whatever the daemon then says it came to.
                            with contextlib.suppress(subprocess.CalledProcessError):
                                costing = costing + self._counting(server, session)
                            spent = (
                                {self._agent.config.model: int(costing.total)}
                                if costing.total > 0
                                else {}
                            )
                            yield Event(
                                kind="result",
                                text=answer.strip(),
                                tokens=spent,
                                spent=costing,
                            )
                            return
                        # A session says it has stopped before the last thing it said can be read
                        # back, so what it said is read once more after it stops rather than at the
                        # moment it does -- otherwise a turn returns everything but its answer.
                        # And a session that has not started yet is a session that is not busy
                        # too: the daemon takes a prompt before it runs it, so one still reading
                        # is not a turn that is over. It takes two of them, a wait apart, and a
                        # turn that has stirred, for what is read back to be this turn rather
                        # than the moment before it.
                        settled = not busy
                        updates.wait(settled=settled)

            finally:
                if updates is not None:
                    updates.close()
                self._running = _Running()


def _erred(error: object) -> str:
    """What the daemon said a failed turn failed of, in its own words where it gave any.

    Args:
      error: The `error` of a `turn.ended` frame, which the protocol leaves unshaped.

    Returns:
      Its message, or the whole of it written out, and something to say where it said nothing.
    """
    if isinstance(error, dict):
        said = cast("dict[str, Any]", error)
        if message := said.get("message") or said.get("errorMessage"):
            return str(message)
    if error:
        return error if isinstance(error, str) else json.dumps(error)
    return "the turn failed, and the daemon did not say why"


def _free() -> int:
    """A TCP port on loopback that nothing is listening on as this is asked.

    Asked of the system rather than guessed, and let go of at once for the daemon to take. A
    port some other process takes in between is a daemon that fails to bind, which fails the
    turn that started it rather than a turn run anywhere but where it was told.
    """
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class KimiCodeCLIAgent(AgentBase):
    """Kimi Code, driven through an app server of its own so a whole session is settable.

    Every moment a turn passes through, and one more: the daemon holds the tool it is about
    to run on an approval interaction until somebody resolves it, and that is the one place
    a hook here can say no to something and have the agent hear it.
    """

    #: Said of the backend and not of the rung, by the rule this repository settles the
    #: question with: a backend that holds a tool until its client answers is a backend where
    #: this moment is real. Kimi holds it -- the interaction keeps the call, the turn waits, and
    #: a `rejected` is a line the model reads -- so the moment is declared, once, for the
    #: class. What a given rung does with it is a fact about the run, the way a turn that
    #: calls no tool is a run in which `PRE_TOOL_USE` never arrives. :class:`Unhooked` is for
    #: a hook hung where the backend could never have answered it, which is not this.
    #:
    #: Being exact about what a hook here buys, rung by rung, because the grant gate narrowed
    #: it. At `workspace-write` and `bypass` -- Kimi's `auto`, which approves before any
    #: policy can ask -- no approval is raised and the hook never fires. At `auto`, Kimi's
    #: `yolo`, it is the whole decision: the dangerous command, the sensitive file and the
    #: `.git` path are asked about, and :data:`_GRANTED` says yes where the hook says
    #: nothing. At `read-only` and at no rung at all the answer was already no, so what a
    #: hook there gets is the sight of the tool and the chance to say why -- not the casting
    #: vote. It cannot turn that no into a yes, and should not be able to: a
    #: :class:`~hmz.coganchor.agents.hooks.Verdict` can refuse or say nothing, and a hook that
    #: only watches must never be the thing that widens a rung.
    #:
    #: So one rung of the five is where this changes an outcome. Declared for all of them
    #: anyway, because the rung is the thing that moves: a place may narrow one after the
    #: agent is built, and the rung an agent comes at is a default rather than a fact about
    #: this backend. A capability that went false and true as a setting moved under it would
    #: be a worse promise than one that is occasionally generous -- what this is read for is
    #: refusing a flow before its first turn, and a promise that changes between the reading
    #: and the turn refuses nothing.
    moments: ClassVar[frozenset[Moment]] = EVERYWHERE | {Moment.PERMISSION_REQUEST}

    #: Kimi keeps itself going toward an objective, which is what `pursue` reaches for.
    pursues: ClassVar[bool] = True

    #: What it counts, read off the same table its driver reads a usage with, so that
    #: what a run is told this backend reports is what its driver actually parses.
    counts: ClassVar[frozenset[str]] = frozenset(_KINDS)

    def __init__(self, config: AgentConfig, *, name: str | None = None) -> None:
        """Initializes an agent whose server is not running yet.

        Args:
          config: The model and effort every session of this agent runs at.
          name: What to call this agent, defaulting to one nothing else answers to.
        """
        #: The one port the daemon starting now was told to listen on inside a fence that
        #: cuts the network, which :meth:`natively` hands the wrapper to let it bind. Set
        #: before the base class is, which asks :meth:`natively` about the fence it is given.
        self._listening: int | None = None
        super().__init__(config, name=name)
        self._server: _AppServer | None = None
        #: Which account the daemon up now was started as, and inside which fence: an agent
        #: that has fallen back starts another rather than going on submitting turns as
        #: somebody else, and one set up with another fence starts another rather than going
        #: on running turns inside the walls the old one was put up with -- the fence is put
        #: around the daemon once, where it starts, and every session of the agent is a
        #: session of that one process. With the flow's command lines on its PATH from
        #: where the shelf was. And, for an account a local LiteLLM gateway holds, pointed at
        #: which gateway and naming which model (see :meth:`environment`): one started again
        #: is at another port under another key. And the flow's skills its virtual home was
        #: built carrying: one carrying others is started again in a home carrying those.
        self._server_as: (
            tuple[str, Fence | None, str, Gateway | None, str, tuple[str, ...]] | None
        ) = None
        self._serving = threading.Lock()

    def environment(self) -> Mapping[str, str]:
        """What this agent's daemon is started with, plus the preload where one is wanted.

        On the agent rather than on a session because the daemon is the agent's: one process holds
        every conversation with it, so there is one runtime to load anything into and it is started
        once, with this. :mod:`hmz.coganchor.agents.preload` is what decides whether one is wanted
        -- Kimi Code is a Node program, so what a turn of it runs, reads, writes and opens can be
        read from inside the process taking it.

        Started once is also the whole of the catch: the question is asked where the daemon is
        started, so a hook hung after that gets nothing from this backend for as long as the
        agent lives. A backend that is a process a turn asks it again every turn.

        And, for an account a local LiteLLM gateway holds, the model the daemon's one provider
        is made with: the account's way of saying it is asked of the person rather than
        answered by the gateway, and what it is to be is the model this agent runs, which the
        gateway sends on to whatever is behind it.

        Returns:
          The provider's own variables, and the preload beside them for an agent something is
          listening to.
        """
        held = dict(super().environment())
        if self.gateway is not None:
            held[_MODEL_NAME] = self.config.model
        return preloaded(self, held)

    def natively(self, fence: Fence) -> Fence:
        """None of the fence, which Kimi has no way of holding itself to; plus its port.

        Kimi Code 0.42 has no sandbox. Its permission modes decide what is asked about rather
        than what can be reached -- a session's `permission_rules` can only approve -- and the
        one guard that reads paths, the workspace access policy its file tools consult, is a
        constant the CLI does not let be configured and that neither `Bash` nor anything
        `Bash` starts ever passes through. So the whole fence is put around it from outside.

        What the driver adds is the port. The daemon is reached over loopback, and a fence
        that cuts the network governs what may be bound as well as what may be connected to,
        so the one port :attr:`server` told it to listen on is granted to bind -- that port
        and no other, and nothing where the network is left alone.

        Args:
          fence: The fence this agent is held to.

        Returns:
          `fence`, with the daemon's port to bind where the network is cut and one is
          starting.
        """
        if self._listening is None or fence.online:
            return fence
        return fence.granting(listen=[self._listening])

    @property
    def server(self) -> _AppServer:
        """The daemon this agent's turns are submitted to, started the first time it is asked for.

        One per agent rather than one per session, so that a flow dropping a session a turn
        does not start a server a turn; it is taken down when the agent is collected, or at
        exit for one held to the end. An anchored agent starts it through coganchor, which
        leaves the server here, holding the conversation, and its work on the target -- the
        same split the CLI ran under.
        """
        with (
            self._serving
        ):  # two sessions of one agent share the server rather than start two
            if self._server is not None and self._server_as != (
                self.node().name,
                self.config.fence,
                self.shelf.where(),
                *self._through(),
                tuple(one.name for one in self.loaded),
            ):
                # Started as an account this agent has since left, or inside a fence it has
                # since been set up with another of. Let go of rather than taken down: a turn
                # on another thread may still be talking to it, and its own finalizer stops it
                # when the agent is collected either way.
                self._server, self._server_as = None, None
            if self._server is None:
                # Read off the config with defaults beside them, because an agent may be
                # given the common `AgentConfig` rather than Kimi's own: a flow that never
                # asked for any of this gets the daemon this driver has always started.
                # `--no-open` and the level are written out rather than left off, since
                # what the CLI does when they are missing is a browser and a banner.
                argv = ["kimi", "web"]
                if not getattr(self.config, "open_browser", False):
                    argv.append("--no-open")
                port: int = getattr(self.config, "port", 0)
                fence = self.config.fence
                if fence is not None and not fence.online:
                    # A fence that cuts the network lets bind only the ports it names, and
                    # "any" is not one of them: the port is chosen here, before the daemon is
                    # spawned, so that the one it binds is the one the wrapper was told.
                    port = port or _free()
                argv += [
                    "--port",
                    str(port),
                    "--log-level",
                    getattr(self.config, "log_level", "error"),
                ]
                if title := getattr(self.config, "web_title", None):
                    argv += ["--web-title", title]
                # Read before the environment is built out of it: a fallback landing
                # between the two reads would name the account this server is *not* signed
                # into, and a server that believes it is already elsewhere is one nothing ever
                # starts again.
                account, pathed = self.node().name, self.shelf.where()
                through = self._through()
                lodged = tuple(one.name for one in self.loaded)
                # In a virtual home of its own, carrying every skill this agent was given --
                # the flow's own beside links to the user's, and none in the workspace -- which
                # goes once the daemon is down.
                home = self.homed()
                self._listening = port or None
                try:
                    spawned = self.spawned(argv, home=home)
                    environ = self._environ()
                    self._server = _AppServer(
                        spawned,
                        environ if home is None else home.environ(environ),
                        home,
                    )
                except BaseException:
                    if home is not None:
                        home.drop()
                    raise
                finally:
                    self._listening = None
                self._server_as = (account, fence, pathed, *through, lodged)
                # Held by the finalizer alone, which is what takes the daemon down: when the
                # agent is collected, and at exit for one held to the end.
                weakref.finalize(self, self._server.stop)
            return self._server

    def _through(self) -> tuple[Gateway | None, str]:
        """The gateway a daemon is pointed at and the model it names, or neither."""
        through = self.gateway
        return through, self.config.model if through is not None else ""

    def stop(self) -> None:
        """Takes no further turn, and takes down the server the turn under way is waiting on."""
        super().stop()
        self._down()

    def _down(self) -> None:
        """Takes down the daemon this agent holds, if it is holding one."""
        with self._serving:
            server, self._server, self._server_as = self._server, None, None
        if server is not None:
            server.stop()

    def new(self, cwd: str | os.PathLike[str] | None = None) -> KimiCodeCLISession:
        """Opens a new Kimi session, in the directory it is given or in this one."""
        return KimiCodeCLISession(self, cwd)
