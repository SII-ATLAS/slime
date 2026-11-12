"""Authenticated CONNECT relay for sandbox HTTPS traffic on a reachable host.

Run with ``python -m examples.ava_rl.relay --help``. The relay does not
terminate TLS or receive YiCloud signing keys. It permits one gateway only.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import logging
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = logging.getLogger(__name__)


def read_private_token(path):
    path = Path(path)
    if path.stat().st_mode & 0o077:
        raise ValueError("relay token file must have mode 0600")
    token = path.read_text().strip()
    if len(token) < 32 or any(c.isspace() for c in token):
        raise ValueError("relay token must contain at least 32 non-whitespace characters")
    return token


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024

    def __init__(
        self,
        address,
        token,
        *,
        target="sandbox.yicloud.com.cn:443",
        max_connections=128,
        idle_timeout=2100,
        lifetime=12000,
    ):
        self.authorization = "Basic " + base64.b64encode(("slime:" + token).encode()).decode()
        self.target = target
        self.slots = threading.BoundedSemaphore(max_connections)
        self.idle_timeout = idle_timeout
        self.lifetime = lifetime
        super().__init__(address, RelayHandler)


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 10

    def log_message(self, *_):
        # Base request logs can expose untrusted request paths or userinfo.
        pass

    def do_CONNECT(self):
        self.close_connection = True
        supplied = self.headers.get("Proxy-Authorization", "")
        if not hmac.compare_digest(supplied.encode(), self.server.authorization.encode()):
            self.send_response(407)
            self.send_header("Proxy-Authenticate", 'Basic realm="slime-sandbox"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path != self.server.target:
            self.send_error(403, "Gateway not allowed")
            return
        if not self.server.slots.acquire(blocking=False):
            self.send_error(503, "Relay capacity reached")
            return
        try:
            host, port = self.server.target.rsplit(":", 1)
            try:
                upstream = socket.create_connection((host, int(port)), timeout=10)
            except OSError:
                self.send_error(502, "Gateway unavailable")
                return
            with upstream:
                self.send_response(200, "Connection Established")
                self.end_headers()
                self.wfile.flush()
                self.connection.settimeout(30)
                upstream.settimeout(30)
                deadline = time.monotonic() + self.server.lifetime
                while time.monotonic() < deadline:
                    ready, _, _ = select.select(
                        [self.connection, upstream],
                        [],
                        [],
                        min(self.server.idle_timeout, max(0, deadline - time.monotonic())),
                    )
                    if not ready:
                        break
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = upstream if source is self.connection else self.connection
                        destination.sendall(data)
        except OSError:
            # A disconnected client is normal; never log tunnel contents.
            pass
        finally:
            self.server.slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True, help="private interface address")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--max-connections", type=int, default=128)
    args = parser.parse_args()
    if args.max_connections < 1:
        parser.error("--max-connections must be positive")
    logging.basicConfig(level=logging.INFO)
    with RelayServer(
        (args.bind, args.port), read_private_token(args.token_file), max_connections=args.max_connections
    ) as server:
        LOG.info("Sandbox relay listening on %s:%s; target=%s", *server.server_address, server.target)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
