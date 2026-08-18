"""Structured logging for the voice pipeline.

Every stage already printed something, but in its own shape and with no way to tie
lines together, so reconstructing one bad turn meant eyeballing interleaved output from
several concurrent calls. This gives every line the same prefix, a millisecond
timestamp, a level, and - crucially - a turn id, so one conversation can be grepped out
of a log holding twenty simultaneous ones.

Levels come from LOG_LEVEL (default INFO). Set LOG_LEVEL=DEBUG to see per-chunk detail.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager

_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Each turn gets an id that follows it across stt -> llm -> tts, including across the
# threadpool hops, so concurrent calls stay separable in the log.
_local = threading.local()

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)-5s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("voice")
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, _LEVEL, logging.INFO))
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Logger for one pipeline stage, e.g. get_logger("stt")."""
    _configure()
    return logging.getLogger(f"voice.{name}")


def turn_id() -> str:
    """Id of the turn being handled on this thread, or '-' outside a turn."""
    return getattr(_local, "turn_id", "-")


def set_turn_id(value: str) -> None:
    """Adopt a turn id on this thread - needed after a run_in_threadpool hop."""
    _local.turn_id = value


def new_turn_id() -> str:
    return uuid.uuid4().hex[:8]


def kv(**fields: object) -> str:
    """Render fields as key=value, so lines stay greppable and parseable.

    Floats are rounded to 3 decimals; strings holding spaces are quoted.
    """
    parts = []
    for key, value in fields.items():
        if isinstance(value, float):
            value = f"{value:.3f}"
        elif isinstance(value, str) and (" " in value or not value):
            value = repr(value)
        parts.append(f"{key}={value}")
    return " ".join(parts)


@contextmanager
def turn(log: logging.Logger, kind: str, **fields: object):
    """Scope one turn: assign an id, log start and end, and never swallow the error.

    Emits a `done` line with elapsed time on success and a `failed` line with the
    exception type on error, so a turn that dies is as visible as one that completes.
    """
    tid = new_turn_id()
    set_turn_id(tid)
    started = time.perf_counter()
    log.info("%s start %s", kind, kv(turn=tid, **fields))
    try:
        yield tid
    except Exception as exc:
        log.error(
            "%s failed %s",
            kind,
            kv(turn=tid, elapsed=time.perf_counter() - started,
               error=f"{type(exc).__name__}: {exc}"),
        )
        raise
    else:
        log.info("%s done %s", kind, kv(turn=tid, elapsed=time.perf_counter() - started))
    finally:
        set_turn_id("-")
