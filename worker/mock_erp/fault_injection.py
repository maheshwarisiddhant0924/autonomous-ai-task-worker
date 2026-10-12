"""Fault injection — test hooks for evaluation scenarios.

Each fault is a callable that wraps the normal database operation.
Faults are registered by name and activated per-scenario.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

# ── Fault registry ───────────────────────────────────────────────────────────

_active_faults: dict[str, "Fault"] = {}


@dataclass
class Fault:
    name: str
    fault_type: str        # "transient_drop", "persistent_drop", "delay", "409", "timeout"
    trigger_count: int = 1 # how many times to trigger before allowing normal behavior
    _triggered: int = field(default=0, init=False)

    def should_trigger(self) -> bool:
        if self._triggered < self.trigger_count:
            self._triggered += 1
            return True
        return False

    def reset(self):
        self._triggered = 0


def activate_fault(name: str, fault_type: str, trigger_count: int = 1):
    _active_faults[name] = Fault(name=name, fault_type=fault_type, trigger_count=trigger_count)


def deactivate_fault(name: str):
    _active_faults.pop(name, None)


def clear_all_faults():
    _active_faults.clear()


def get_active_fault(name: str) -> Fault | None:
    return _active_faults.get(name)


# ── Specific fault behaviors ─────────────────────────────────────────────────

class TransientDropFault(Exception):
    """Server returns 200 but silently doesn't persist the write."""
    pass


class PersistentDropFault(Exception):
    """Server returns 200 but never persists the write."""
    pass


class DelayedPersistenceFault(Exception):
    """Write succeeds but row appears after a delay."""
    pass


class TimeoutAfterSuccessFault(Exception):
    """Write succeeds in DB but the HTTP response times out."""
    pass


class SimulatedTimeoutError(Exception):
    """Simulated network/server timeout."""
    pass


def check_write_fault() -> str | None:
    """Check if any write fault should trigger. Returns fault type or None."""
    fault = get_active_fault("write")
    if fault and fault.should_trigger():
        return fault.fault_type
    return None


def check_read_fault() -> str | None:
    """Check if any read fault should trigger."""
    fault = get_active_fault("read")
    if fault and fault.should_trigger():
        return fault.fault_type
    return None


# ── Hook for concurrent row insertion (scenario #16) ────────────────────────

_pause_hooks: list[Callable] = []


def register_pause_hook(hook: Callable):
    """Register a function to run during a human-approval pause."""
    _pause_hooks.append(hook)


def run_pause_hooks():
    """Execute all registered pause hooks (simulates concurrent activity)."""
    for hook in _pause_hooks:
        hook()


def clear_pause_hooks():
    _pause_hooks.clear()
