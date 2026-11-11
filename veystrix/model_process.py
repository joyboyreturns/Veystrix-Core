"""One supervised engine process, owned exclusively by the local router."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from argparse import Namespace
from pathlib import Path

from models.flashnext.overnight import server_command


class ProcessDriver:
    """Start one guarded CLI engine and prove its process group has exited on unload."""
    def __init__(self, *, pool_gb="0", prefix_dir=None, startup_timeout=1800, stop_timeout=60,
                 command_factory=None, flashnext_kv_cache="bf16", flashnext_memory_policy=None, flashnext_speed=False):
        self.pool_gb = pool_gb
        self.prefix_dir = prefix_dir
        self.startup_timeout = startup_timeout
        self.stop_timeout = stop_timeout
        self.flashnext_kv_cache = flashnext_kv_cache
        self.flashnext_memory_policy = flashnext_memory_policy
        self.flashnext_speed = flashnext_speed
        self.command_factory = command_factory or self._command
        self.process = None
        self.port = None
        self.instance = None
        self.api_key = uuid.uuid4().hex + uuid.uuid4().hex
        self._pgid = None
        self._guard_pipe = None

    def _command(self, model, port):
        is_flashnext = model.family == "flashnext"
        tiered = is_flashnext and self.flashnext_kv_cache == "tiered4"
        features = ["kv-tiered4"] if tiered else []
        if is_flashnext and self.flashnext_speed:
            features.append("short-throughput")
        args = Namespace(model=model.family, checkpoint=model.checkpoint, host="127.0.0.1", port=port,
                         pool_gb=("auto" if model.family == "nemotron35-lightning" and self.pool_gb == "0" else self.pool_gb),
                         enable=features, prefix_dir=self.prefix_dir,
                         preferences_file=None, no_auto_purge=is_flashnext and self.flashnext_memory_policy == "absolute-swap-only",
                         threshold=1.0 if tiered else None)
        command = server_command(args)
        # Run the same CLI directly; no launcher shell or additional watchdog process is left behind.
        if command[0].endswith("chat.sh"):
            command = [sys.executable, "-m", "veystrix.cli", *command[1:]]
        return command

    def health(self, timeout=2):
        if self.port is None:
            return None
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/health",
                                         headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except (OSError, ValueError, urllib.error.URLError):
            return None

    def ready(self, model):
        if self.process is None or self.process.poll() is not None:
            return False
        h = self.health()
        return bool(h and h.get("ready") and h.get("instance_id") == self.instance
                    and h.get("installed_model_id") == model.id)

    def load(self, model, cancelled):
        if self.process is not None:
            raise RuntimeError("previous engine process has not been released")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.port = port
        self.instance = uuid.uuid4().hex
        env = dict(os.environ)
        env.update(VEYSTRIX_SERVER_API_KEY=self.api_key, VEYSTRIX_ENGINE_INSTANCE_ID=self.instance,
                   VEYSTRIX_PYTHON=sys.executable)
        if model.family == "flashnext" and self.flashnext_memory_policy:
            env["FLASHNEXT_MEMORY_POLICY"] = self.flashnext_memory_policy
        env.setdefault("VEYSTRIX_NEMOTRON_AUTHORIZED", "1")
        read_fd, write_fd = os.pipe()
        command = [sys.executable, "-m", "veystrix.engine_guard", str(read_fd), *self.command_factory(model, port)]
        try:
            self.process = subprocess.Popen(command, env=env, start_new_session=True, pass_fds=(read_fd,))
        except BaseException:
            os.close(write_fd)
            raise
        finally:
            os.close(read_fd)
        self._guard_pipe = write_fd
        self._pgid = self.process.pid
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if cancelled():
                raise RuntimeError("model loading cancelled")
            if self.process.poll() is not None:
                raise RuntimeError(f"engine exited during load ({self.process.returncode})")
            if self.ready(model):
                return
            time.sleep(0.2)
        raise TimeoutError("timed out waiting for selected model to load")

    @staticmethod
    def _group_alive(pgid):
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def unload(self):
        if self.process is None:
            return
        process, pgid = self.process, self._pgid
        # Signal only while the process we launched is still alive. If it
        # already exited, its numeric process-group ID could have been reused.
        if process.poll() is None and self._group_alive(pgid):
            try:
                os.killpg(pgid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + self.stop_timeout
        while time.monotonic() < deadline:
            process.poll()  # reap the group leader
            if not self._group_alive(pgid):
                if self._guard_pipe is not None:
                    os.close(self._guard_pipe)
                    self._guard_pipe = None
                self.process = None
                self.port = None
                self.instance = None
                self._pgid = None
                return
            time.sleep(0.1)
        # Do not start another model if anything in the previous group survives.
        raise RuntimeError("the previous engine process group did not exit")
