"""Single-resident model lifecycle coordination for the local model router.

The driver owns an engine process and must not report unload complete until every
model worker has exited. The coordinator grants compatible requests shared
leases and waits for all leases to end before changing the resident model.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Callable, Protocol

from veystrix.model_registry import InstalledModel


PRELOAD_OWNER = "internal:preloaded"


class LifecycleError(RuntimeError):
    code = "model_lifecycle_error"


class ModelNotFound(LifecycleError):
    code = "model_not_found"


class ModelBusy(LifecycleError):
    code = "model_busy"


class HandoffRequired(LifecycleError):
    code = "model_handoff_required"


class LifecycleFault(LifecycleError):
    code = "model_lifecycle_fault"


class Driver(Protocol):
    def load(self, model: InstalledModel, cancelled: Callable[[], bool]) -> None:
        """Return only after this exact model is ready; clean up before raising."""
    def unload(self) -> None:
        """Return only once the previous model and all its workers have exited."""
    def ready(self, model: InstalledModel) -> bool:
        """Check the process and exact checkpoint identity, not just its family."""


@dataclass(frozen=True)
class Snapshot:
    state: str
    model_id: str | None
    owner: str | None
    error: str | None
    active: bool


class ModelCoordinator:
    """Own request leases and the resident model identity.

    An owner is a caller-provided stable session identity. An unowned model is
    not silently evicted; a different session must explicitly authorize handoff.
    This intentionally favors preserving long-running coding work over convenience.
    The HTTP router is responsible for authenticating the caller and only granting
    explicit handoff based on an authorized user action.
    """
    def __init__(self, resolve: Callable[[str], InstalledModel | None], driver: Driver):
        self.resolve = resolve
        self.driver = driver
        self._gate = threading.Lock()
        self._leases_changed = threading.Condition(self._gate)
        self._leases = 0
        self._exclusive_lease = False
        self._state_lock = threading.Lock()
        self._state = "unloaded"
        self._model: InstalledModel | None = None
        self._owner: str | None = None
        self._error: str | None = None
        self._active = False

    def _set(self, *, state=None, model=..., owner=..., error=..., active=None):
        with self._state_lock:
            if state is not None:
                self._state = state
            if model is not ...:
                self._model = model
            if owner is not ...:
                self._owner = owner
            if error is not ...:
                self._error = error
            if active is not None:
                self._active = active

    def snapshot(self) -> Snapshot:
        with self._state_lock:
            return Snapshot(self._state, self._model.id if self._model else None,
                            self._owner, self._error, self._active)

    def _acquire(self, deadline: float, cancelled: Callable[[], bool]) -> None:
        while True:
            if cancelled():
                raise ModelBusy("request was cancelled while waiting for the model")
            remaining = deadline - time.monotonic()
            if self._gate.acquire(timeout=max(0.0, min(0.1, remaining))):
                if cancelled():
                    self._gate.release()
                    raise ModelBusy("request was cancelled while waiting for the model")
                return
            if remaining <= 0:
                raise ModelBusy("another request or model handoff is in progress")

    @contextmanager
    def request(self, model_id: str, *, owner: str | None = None, handoff: bool = False,
                exclusive_owner: bool = False, timeout: float = 0.0,
                cancelled: Callable[[], bool] = lambda: False):
        """Hold the model for one entire request; caller must not retry generation."""
        if not isinstance(model_id, str) or not model_id:
            raise ModelNotFound("a model installation ID is required")
        deadline = time.monotonic() + max(0.0, timeout)
        self._acquire(deadline, cancelled)
        try:
            while self._leases and (
                self._exclusive_lease or exclusive_owner or self._model.id != model_id
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ModelBusy("another request or model handoff is in progress")
                self._leases_changed.wait(min(0.1, remaining))
                if cancelled():
                    raise ModelBusy("request was cancelled while waiting for the model")
            if self._state == "fault":
                raise LifecycleFault(self._error or "model lifecycle is faulted")
            if self._state == "cooldown":
                raise ModelBusy("the engine is cooling down after an unexpected exit")
            # Resolve while holding the gate and before unloading the current model.
            model = self.resolve(model_id)
            if model is None or model.id != model_id:
                raise ModelNotFound(f"model {model_id!r} is not installed")
            current = self._model
            if current and exclusive_owner and self._owner not in {PRELOAD_OWNER, owner}:
                raise HandoffRequired("another session owns the engine; start a fresh idle engine for this workflow")
            if current and current.id != model.id:
                if not handoff and self._owner != PRELOAD_OWNER and (not owner or not self._owner or owner != self._owner):
                    raise HandoffRequired("another session owns the current model; explicit handoff is required")
                self._set(state="unloading", error=None)
                try:
                    self.driver.unload()
                except Exception as exc:
                    self._set(state="fault", error=f"unload could not be verified: {exc}")
                    raise LifecycleFault("previous model may still be resident; refusing to load another") from exc
                self._set(state="unloaded", model=None, owner=None)
                current = None
            if current is not None:
                if self._owner == PRELOAD_OWNER:
                    self._set(owner=owner)
                try:
                    ready = self.driver.ready(current)
                except Exception as exc:
                    if self._leases:
                        raise ModelBusy("resident worker readiness could not be verified during an active request") from exc
                    self._set(state="fault", error=f"readiness verification failed: {exc}")
                    raise LifecycleFault("loaded model readiness could not be verified") from exc
                if not ready:
                    if self._leases:
                        raise ModelBusy("resident worker is busy or temporarily unavailable")
                    self._set(state="fault", error="loaded model readiness or identity was lost")
                    raise LifecycleFault("loaded model is no longer verified; refusing to route")
            else:
                if cancelled():
                    raise ModelBusy("request was cancelled before model load")
                self._set(state="loading", error=None)
                try:
                    self.driver.load(model, cancelled)
                    if not self.driver.ready(model):
                        raise RuntimeError("driver did not verify the selected model")
                except Exception as exc:
                    # A partial load may have created a live worker. Never assume cleanup succeeded.
                    try:
                        self.driver.unload()
                    except Exception as unload_exc:
                        self._set(state="fault", error=f"failed load cleanup could not be verified: {unload_exc}")
                        raise LifecycleFault("failed model load may still be resident") from exc
                    self._set(state="unloaded", model=None, owner=None, error=str(exc))
                    raise LifecycleFault(f"model load failed: {exc}") from exc
                self._set(state="ready", model=model, owner=owner, error=None)
            if cancelled():
                raise ModelBusy("request was cancelled before generation")
            self._leases += 1
            self._exclusive_lease = exclusive_owner
            self._set(active=True)
        finally:
            self._gate.release()
        try:
            yield model
        finally:
            with self._leases_changed:
                self._leases -= 1
                if not self._leases:
                    self._exclusive_lease = False
                    self._set(active=False)
                    self._leases_changed.notify_all()
