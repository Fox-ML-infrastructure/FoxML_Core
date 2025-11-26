import contextlib
import json
import os
import time
import urllib.request


def kill_switch(path: str = "kill.flag", interval_s: float = 5.0):
    """Yield control repeatedly; return True if kill flag appears.

    Usage:
        for stopped in kill_switch():
            if stopped: break
            # unit of work
    """
    last_poll = 0.0
    while True:
        now = time.monotonic()
        if now - last_poll >= interval_s:
            last_poll = now
            if os.path.exists(path):
                yield True
                return
        yield False


def notify_ntfy(title: str, msg):
    url = os.getenv("NTFY_URL")
    if not url:
        return
    payload = msg if isinstance(msg, str) else json.dumps(msg)
    req = urllib.request.Request(url, data=payload.encode(), method="POST")
    req.add_header("Title", title)
    req.add_header("Content-Type", "application/json" if isinstance(msg, dict) else "text/plain")
    with contextlib.suppress(Exception):
        urllib.request.urlopen(req, timeout=5)
