"""A minimal RFC 6455 WebSocket, over ``http.server``.

``fastapi``/``uvicorn`` are not installed on this box and adding a dependency is the
most common way this hardware eats an afternoon (CLAUDE.md), so the §4.3 push channel is
written out: the handshake is a SHA-1 and a base64, and a frame is a short header plus
an optionally masked payload. That is the whole protocol we need — the client sends
almost nothing and we send small JSON messages.

What this deliberately does **not** implement: extensions (``permessage-deflate`` is
declined by not negotiating it), continuation frames on *send* (our messages are one
frame), and subprotocols. Fragmented and binary frames from the client are read and
discarded — ``ui/static/data.js`` only ever listens.

Message shapes, fixed by the UI's ``connect()`` handler:

    {"type": "refinement",    "turn_id": ..., "job":   DeepJob.to_dict()}
    {"type": "monitor_state", "state":   ...}
    {"type": "action",        "entry":   ActionLogEntry.to_dict()}
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import threading
from typing import Any, BinaryIO

from .telemetry import log_event

__all__ = ["WS_GUID", "accept_key", "WebSocketConnection", "WebSocketHub", "WebSocketClosed"]

#: RFC 6455 §1.3. The magic string every implementation concatenates before hashing.
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_OP_CONT = 0x0
_OP_TEXT = 0x1
_OP_BINARY = 0x2
_OP_CLOSE = 0x8
_OP_PING = 0x9
_OP_PONG = 0xA

_FIN = 0x80
_MASKED = 0x80
_LEN_16 = 126
_LEN_64 = 127

#: A frame larger than this from a client is refused rather than buffered. The UI sends
#: control frames and nothing else; anything bigger is a bug or an attack, and neither
#: deserves a megabyte of this process's memory.
_MAX_CLIENT_FRAME_BYTES = 1 << 16


class WebSocketClosed(Exception):
    """The peer went away. Normal — a browser tab closing is not an error."""


def accept_key(client_key: str) -> str:
    """``Sec-WebSocket-Accept`` for a client's ``Sec-WebSocket-Key`` (RFC 6455 §4.2.2)."""
    digest = hashlib.sha1((client_key.strip() + WS_GUID).encode("ascii")).digest()  # noqa: S324
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, opcode: int = _OP_TEXT) -> bytes:
    """One unmasked server frame. Server→client frames must not be masked (§5.1)."""
    header = bytearray([_FIN | opcode])
    length = len(payload)
    if length < _LEN_16:
        header.append(length)
    elif length < (1 << 16):
        header.append(_LEN_16)
        header += struct.pack(">H", length)
    else:
        header.append(_LEN_64)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


class WebSocketConnection:
    """One upgraded connection. Sends are serialised; reads happen on its own thread.

    The handler thread owns the read loop and the hub broadcasts from the executor
    thread, so ``_send_lock`` is not optional: two interleaved frames on one socket is a
    protocol violation the browser reports as a corrupt frame and nothing else.
    """

    def __init__(self, rfile: BinaryIO, wfile: BinaryIO, peer: str = "") -> None:
        self._rfile = rfile
        self._wfile = wfile
        self._send_lock = threading.Lock()
        self.peer = peer
        self.open = True

    # -- sending ---------------------------------------------------------------------

    def send_json(self, payload: dict[str, Any]) -> None:
        self.send_text(json.dumps(payload, default=str))

    def send_text(self, text: str) -> None:
        self._send(encode_frame(text.encode("utf-8"), _OP_TEXT))

    def ping(self) -> None:
        self._send(encode_frame(b"", _OP_PING))

    def close(self, code: int = 1000) -> None:
        if not self.open:
            return
        try:
            self._send(encode_frame(struct.pack(">H", code), _OP_CLOSE))
        except WebSocketClosed:
            pass
        self.open = False

    def _send(self, frame: bytes) -> None:
        if not self.open:
            raise WebSocketClosed(self.peer)
        with self._send_lock:
            try:
                self._wfile.write(frame)
                self._wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError, ValueError) as exc:
                self.open = False
                raise WebSocketClosed(f"{self.peer}: {exc}") from exc

    # -- receiving --------------------------------------------------------------------

    def read_frame(self) -> tuple[int, bytes]:
        """Read one client frame. Raises :class:`WebSocketClosed` at end of stream."""
        header = self._read_exactly(2)
        opcode = header[0] & 0x0F
        masked = bool(header[1] & _MASKED)
        length = header[1] & 0x7F
        if length == _LEN_16:
            length = struct.unpack(">H", self._read_exactly(2))[0]
        elif length == _LEN_64:
            length = struct.unpack(">Q", self._read_exactly(8))[0]
        if length > _MAX_CLIENT_FRAME_BYTES:
            self.close(1009)
            raise WebSocketClosed(f"{self.peer}: client frame of {length} bytes refused")
        mask = self._read_exactly(4) if masked else b""
        payload = self._read_exactly(length) if length else b""
        if masked:
            payload = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        return opcode, payload

    def serve(self) -> None:
        """Read until the peer closes. Answers pings; ignores everything else.

        Nothing the client sends means anything to us — ``data.js`` never sends — but the
        read loop has to exist: it is how a closed tab is noticed, and an unnoticed close
        leaks a connection the hub keeps broadcasting to.
        """
        try:
            while self.open:
                opcode, payload = self.read_frame()
                if opcode == _OP_CLOSE:
                    self.close()
                    return
                if opcode == _OP_PING:
                    self._send(encode_frame(payload, _OP_PONG))
                elif opcode in (_OP_PONG, _OP_TEXT, _OP_BINARY, _OP_CONT):
                    continue
        except (WebSocketClosed, OSError, struct.error):
            self.open = False

    def _read_exactly(self, count: int) -> bytes:
        chunks: list[bytes] = []
        remaining = count
        while remaining > 0:
            try:
                chunk = self._rfile.read(remaining)
            except (OSError, ValueError) as exc:
                self.open = False
                raise WebSocketClosed(f"{self.peer}: {exc}") from exc
            if not chunk:
                self.open = False
                raise WebSocketClosed(f"{self.peer}: peer closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


class WebSocketHub:
    """Every open socket, and the one place a message is fanned out to them.

    A send failure drops that connection and never propagates: a browser closing during
    a 34-second deep job must not take down the job's notification to everyone else.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connections: list[WebSocketConnection] = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._connections)

    def add(self, connection: WebSocketConnection) -> None:
        with self._lock:
            self._connections.append(connection)
        log_event("agent.ws.open", peer=connection.peer, connections=len(self))

    def remove(self, connection: WebSocketConnection) -> None:
        with self._lock:
            if connection in self._connections:
                self._connections.remove(connection)
        log_event("agent.ws.close", peer=connection.peer, connections=len(self))

    def broadcast(self, message: dict[str, Any]) -> int:
        """Send one message to every open socket. Returns how many received it."""
        with self._lock:
            targets = list(self._connections)
        delivered = 0
        for connection in targets:
            try:
                connection.send_json(message)
                delivered += 1
            except WebSocketClosed:
                self.remove(connection)
        log_event(
            "agent.ws.broadcast",
            type=message.get("type"),
            delivered=delivered,
            targets=len(targets),
        )
        return delivered

    def ping_all(self) -> None:
        with self._lock:
            targets = list(self._connections)
        for connection in targets:
            try:
                connection.ping()
            except WebSocketClosed:
                self.remove(connection)

    def close_all(self) -> None:
        with self._lock:
            targets = list(self._connections)
            self._connections.clear()
        for connection in targets:
            try:
                connection.close(1001)
            except WebSocketClosed:
                pass


def client_key() -> str:
    """A random ``Sec-WebSocket-Key``. Used by the tests, which speak the protocol."""
    return base64.b64encode(os.urandom(16)).decode("ascii")


def mask_frame(payload: bytes, opcode: int = _OP_TEXT) -> bytes:
    """A masked client frame, for tests and for anything that speaks as a client."""
    mask = os.urandom(4)
    masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
    header = bytearray([_FIN | opcode])
    length = len(payload)
    if length < _LEN_16:
        header.append(_MASKED | length)
    elif length < (1 << 16):
        header.append(_MASKED | _LEN_16)
        header += struct.pack(">H", length)
    else:
        header.append(_MASKED | _LEN_64)
        header += struct.pack(">Q", length)
    return bytes(header) + mask + masked


def read_message(sock: socket.socket) -> tuple[int, bytes]:
    """Read one server frame off a raw socket. Test-side counterpart of the encoder."""
    connection = WebSocketConnection(sock.makefile("rb"), sock.makefile("wb"), peer="client")
    return connection.read_frame()
