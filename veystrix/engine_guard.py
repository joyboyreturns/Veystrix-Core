"""Terminate an owned engine process group if its router disappears."""
from __future__ import annotations
import os
import signal
import subprocess
import sys
import threading


def main():
    fd = int(sys.argv[1])
    command = sys.argv[2:]
    if not command:
        return 2
    # Child inherits this guardian's private process group, but not the pipe.
    child = subprocess.Popen(command, close_fds=True)
    def watch():
        try:
            while os.read(fd, 1):
                pass
        except OSError:
            pass
        try:
            os.killpg(os.getpgrp(), signal.SIGTERM)
        except ProcessLookupError:
            pass
    threading.Thread(target=watch, daemon=True).start()
    try:
        return child.wait()
    finally:
        os.close(fd)

if __name__ == "__main__":
    raise SystemExit(main())
