"""Best-effort guards for optional telemetry components."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def warn_telemetry_failure(component: str, error: Exception) -> None:
    warnings.warn(
        f"Optional {component} telemetry disabled after "
        f"{type(error).__name__}: {error}",
        RuntimeWarning,
        stacklevel=2,
    )


def create_optional(component: str, factory: Callable[[], T]) -> T | None:
    try:
        return factory()
    except Exception as error:
        warn_telemetry_failure(component, error)
        return None


def best_effort(component: str, action: Callable[[], None]) -> bool:
    try:
        action()
        return True
    except Exception as error:
        warn_telemetry_failure(component, error)
        return False


def close_optional(component: str, logger: object | None) -> None:
    if logger is None:
        return
    close = getattr(logger, "close", None)
    if callable(close):
        best_effort(f"{component} close", close)
