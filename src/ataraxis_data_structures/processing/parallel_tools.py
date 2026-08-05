"""Provides assets for controlling the threading behavior of the worker processes used by parallel processing jobs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from contextlib import contextmanager

from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from collections.abc import Generator

_THREAD_LIMIT_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""The threading-layer environment variables that determine how wide a thread pool each numeric backend opens.

Notes:
    Each backend reads its variable once, while it is being imported, and treats the value it read as the width of the
    pool it opens. A worker process imports those backends after it is spawned, so the value it inherits from its
    parent's environment is the only value that reaches it.
"""


@contextmanager
def limit_worker_threads(thread_count: int = 1) -> Generator[None, None, None]:
    """Constrains the numeric backends imported by worker processes to the requested thread count.

    Notes:
        The numeric backends bundled with NumPy open a thread pool sized to the host's core count when they are
        imported, whatever work the importing process intends to do. A process pool that hands each worker its own
        backend therefore opens that pool once per worker, so a job running one worker per core holds the square of
        the core count in threads while using one of them.

        The limit travels to the workers through the environment a spawned child inherits, so this context has to
        enclose the pool's whole lifetime rather than its construction alone. A pool creates each worker when work is
        first submitted to it, not when the pool itself is created.

        Restoring the previous values on exit keeps the limit from leaking into whatever the calling process does next.

    Args:
        thread_count: The number of threads each worker's numeric backends may open.

    Raises:
        ValueError: If the requested thread count is less than one.
    """
    if thread_count < 1:
        message = (
            f"Unable to limit the thread count used by the worker processes. The 'thread_count' argument must be "
            f"greater than or equal to 1, but got {thread_count}."
        )
        console.error(message=message, error=ValueError)

    previous_values = {name: os.environ.get(name) for name in _THREAD_LIMIT_VARIABLES}
    os.environ.update({name: str(thread_count) for name in _THREAD_LIMIT_VARIABLES})
    try:
        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value
