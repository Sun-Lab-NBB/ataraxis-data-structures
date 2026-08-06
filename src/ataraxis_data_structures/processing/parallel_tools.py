"""Provides assets for controlling the threading behavior of the worker processes used by parallel processing jobs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from contextlib import contextmanager

import numba
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from collections.abc import Generator

_THREAD_LIMIT_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
    "OPENCV_FFMPEG_THREADS",
    "TIFFFILE_NUM_THREADS",
)
"""The threading-layer environment variables that determine how wide a thread pool each numeric backend opens.

Notes:
    The backends divide into two groups by the moment they read their variable. The numeric backends bundled with
    NumPy and the polars query engine read theirs once, while they are being imported, and treat the value they read
    as the width of the pool they open. A worker process imports them after it is spawned, so the value it inherits
    from its parent's environment is the only value that reaches them. The OpenCV FFmpeg decoder and the tifffile
    image decoder instead read theirs the first time a capture opens or a decode asks for a default width, so a
    worker that writes the value as it starts still reaches them.

    ``NUMBA_NUM_THREADS`` is deliberately absent. numba reads that variable while it is imported, treats the value it
    read as the ceiling for the rest of the process, and re-reads it on every compilation, raising once its pool has
    started and the two disagree. ``initialize_worker_threads()`` pins numba through the library's own runtime setter
    instead, which is the supported way to change the count.
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

        A worker of a pool created outside this context pins itself through ``initialize_worker_threads()``, which
        still reaches the backends that read their variable after the worker has started.

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


def initialize_worker_threads(thread_count: int = 1) -> None:
    """Constrains the numeric backends of the calling process to the requested thread count.

    Notes:
        Runs inside a worker process, as the initializer a process pool calls in each child it spawns. This covers
        the backends that read their variable the first time they are asked to do work, which a pool created outside
        ``limit_worker_threads()`` reaches no other way. A worker of a pool created inside that context already
        inherits the pinned environment, so calling this as well changes nothing while both name the same width.

        numba latches its ceiling from the environment while it is imported, which a worker does before its pool's
        initializer runs, so the environment no longer reaches it. It is pinned through its own runtime setter here,
        narrowed to the latched ceiling, since the setter rejects a count above it.

        The OpenCV core thread count is a runtime setter rather than a variable, and pinning it falls to the caller,
        since this library takes no OpenCV dependency. The FFmpeg decoder that OpenCV bundles reads its own variable
        and is covered here.

    Args:
        thread_count: The number of threads each numeric backend of the calling process may open.

    Raises:
        ValueError: If the requested thread count is less than one.
    """
    if thread_count < 1:
        message = (
            f"Unable to initialize the thread count used by the worker process. The 'thread_count' argument must be "
            f"greater than or equal to 1, but got {thread_count}."
        )
        console.error(message=message, error=ValueError)

    os.environ.update({name: str(thread_count) for name in _THREAD_LIMIT_VARIABLES})

    numba.set_num_threads(n=min(thread_count, numba.config.NUMBA_NUM_THREADS))
