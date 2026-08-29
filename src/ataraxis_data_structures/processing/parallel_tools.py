"""Provides assets for controlling the threading behavior of the worker processes used by parallel processing jobs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from contextlib import contextmanager

import numba
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from collections.abc import Mapping, Generator

_RESIZABLE_THREAD_VARIABLES: tuple[str, ...] = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "FLEXIBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENCV_FOR_THREADS_NUM",
    "OPENCV_FFMPEG_THREADS",
    "TIFFFILE_NUM_THREADS",
)
"""The threading-layer environment variables whose backends accept a different width after the process has loaded them.

Notes:
    The BLAS backends that NumPy links size their pool from their variable while they are being imported. OpenBLAS,
    MKL, BLIS, and FlexiBLAS each expose a runtime setter that ``threadpool_limits`` drives, so a worker that starts
    at one thread still runs a job at the width the job was allocated. The numexpr evaluator and the OpenCV core pool
    each carry a runtime setter of their own.

    The OpenCV FFmpeg decoder and the tifffile image decoder read their variable the first time a capture opens or a
    decode asks for a default width. A worker that writes the value as it starts therefore reaches every backend in
    this group.
"""

_IMPORT_LATCHED_THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "DUCC0_NUM_THREADS",
    "POLARS_MAX_THREADS",
)
"""The threading-layer environment variables that at least one backend reads once as it loads, fixing a pool width that
no later call can change.

Notes:
    The transform backend that SciPy vendors reads ``DUCC0_NUM_THREADS``, falling back to ``OMP_NUM_THREADS``, and
    treats the value it read as the widest pool it will ever open. A transform handed a larger worker count runs at
    the latched width, and neither that worker count nor ``threadpool_limits`` raises it. ``OMP_NUM_THREADS``
    stays here for that reason, even though ``threadpool_limits`` also resizes the OpenMP runtimes that read it, since
    the narrower behavior is the one that governs. polars builds its pool as it is imported and is not one of the
    pools ``threadpool_limits`` manages.

    A worker imports these backends after its pool's initializer has run, so the width they read is the only width
    they ever hold. They therefore carry the widest count any job in the pool raises itself to, which leaves such a
    job free to spend the cores it was allocated.

    ``NUMBA_NUM_THREADS`` belongs to this group by behavior and is deliberately left out of it. numba reads that
    variable while it is imported, treats the value it read as the ceiling for the rest of the process, and re-reads
    it on every compilation, raising once its pool has started and the two disagree. ``initialize_worker_threads()``
    pins numba through the library's own runtime setter instead, which is the supported way to change the count.
"""


@contextmanager
def limit_worker_threads(
    thread_count: int = 1,
    maximum_thread_count: int | None = None,
    additional_thread_variables: Mapping[str, int] | None = None,
) -> Generator[None, None, None]:
    """Constrains the numeric backends imported by worker processes to the requested thread counts.

    Notes:
        The numeric backends bundled with NumPy open a thread pool sized to the host's core count when they are
        imported, whatever work the importing process intends to do. A process pool that hands each worker its own
        backend therefore opens that pool once per worker, so a job running one worker per core holds the square of
        the core count in threads while using one of them.

        The limit travels to the workers through the environment a spawned child inherits, so this context has to
        enclose the pool's whole lifetime rather than its construction alone. A pool creates each worker when work is
        first submitted to it, not when the pool itself is created.

        The backends that latch their width as they load take ``maximum_thread_count``, because a job that raises its
        own width reaches them no other way. A pool whose jobs each run at a single thread leaves that argument unset,
        which pins every backend at ``thread_count``.

        Restoring the previous values on exit keeps the limit from leaking into whatever the calling process does next.

        A worker of a pool created outside this context pins itself through ``initialize_worker_threads()``, which
        still reaches the backends that read their variable after the worker has started.

    Args:
        thread_count: The number of threads each worker's resizable numeric backends may open.
        maximum_thread_count: The number of threads each worker's import-latched numeric backends may open, which is
            the widest count any job the pool runs raises itself to. Defaults to the value of ``thread_count``.
        additional_thread_variables: The threading-layer environment variables to write alongside the ones this module
            already knows, each mapped to the width it takes. Naming a variable this module already writes replaces
            the width that variable would otherwise take. The caller pairs each entry with ``thread_count`` or with
            ``maximum_thread_count``, according to whether that backend reads its variable while loading or afterward.

    Raises:
        ValueError: If the requested thread count is less than one. If the requested maximum thread count is less than
            the requested thread count. If an additional variable carries a blank name or a width less than one.
    """
    variables = _resolve_thread_variables(
        thread_count=thread_count,
        maximum_thread_count=maximum_thread_count,
        additional_thread_variables=additional_thread_variables,
        action="limit the thread count used by the worker processes",
    )

    previous_values = {name: os.environ.get(name) for name in variables}
    os.environ.update(variables)
    try:
        yield
    finally:
        for name, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def initialize_worker_threads(
    thread_count: int = 1,
    maximum_thread_count: int | None = None,
    additional_thread_variables: Mapping[str, int] | None = None,
) -> None:
    """Constrains the numeric backends of the calling process to the requested thread counts.

    Notes:
        Runs inside a worker process, as the initializer a process pool calls in each child it spawns. This covers
        the backends that read their variable the first time they are asked to do work, which a pool created outside
        ``limit_worker_threads()`` reaches no other way. A worker of a pool created inside that context already
        inherits the pinned environment, so calling this as well changes nothing while both name the same counts.

        A pool runs this initializer before its worker unpickles the first work item, so the backends that latch their
        width as they load are still unloaded at this point and the width written here is the width they take. They
        therefore take ``maximum_thread_count``, and a caller pairing this with ``limit_worker_threads()`` hands both
        of them the same arguments.

        numba latches its ceiling from the environment while it is imported, which a worker does before its pool's
        initializer runs, so the environment no longer reaches it. It is pinned through its own runtime setter here,
        narrowed to the latched ceiling, since the setter rejects a count above it.

    Args:
        thread_count: The number of threads each resizable numeric backend of the calling process may open.
        maximum_thread_count: The number of threads each import-latched numeric backend of the calling process may
            open, which is the widest count any job the worker runs raises itself to. Defaults to the value of
            ``thread_count``.
        additional_thread_variables: The threading-layer environment variables to write alongside the ones this module
            already knows, each mapped to the width it takes. Naming a variable this module already writes replaces
            the width that variable would otherwise take. The caller pairs each entry with ``thread_count`` or with
            ``maximum_thread_count``, according to whether that backend reads its variable while loading or afterward.

    Raises:
        ValueError: If the requested thread count is less than one. If the requested maximum thread count is less than
            the requested thread count. If an additional variable carries a blank name or a width less than one.
    """
    variables = _resolve_thread_variables(
        thread_count=thread_count,
        maximum_thread_count=maximum_thread_count,
        additional_thread_variables=additional_thread_variables,
        action="initialize the thread count used by the worker process",
    )

    os.environ.update(variables)

    numba.set_num_threads(n=min(thread_count, numba.config.NUMBA_NUM_THREADS))


def _resolve_thread_variables(
    thread_count: int,
    maximum_thread_count: int | None,
    additional_thread_variables: Mapping[str, int] | None,
    action: str,
) -> dict[str, str]:
    """Validates the requested thread counts and resolves the environment variables that carry them.

    Args:
        thread_count: The width requested for the resizable backends.
        maximum_thread_count: The width requested for the import-latched backends, or None to match the width the
            resizable backends take.
        additional_thread_variables: The caller-supplied variables to write alongside the known ones, or None.
        action: The action an error message names when a requested value is invalid.

    Returns:
        The environment variable names mapped to the widths they take.

    Raises:
        ValueError: If the requested thread count is less than one. If the requested maximum thread count is less than
            the requested thread count. If an additional variable carries a blank name or a width less than one.
    """
    if thread_count < 1:
        message = (
            f"Unable to {action}. The 'thread_count' argument must be greater than or equal to 1, but got "
            f"{thread_count}."
        )
        console.error(message=message, error=ValueError)

    resolved_maximum = thread_count if maximum_thread_count is None else maximum_thread_count
    if resolved_maximum < thread_count:
        message = (
            f"Unable to {action}. The 'maximum_thread_count' argument must be greater than or equal to the "
            f"'thread_count' argument ({thread_count}), but got {resolved_maximum}."
        )
        console.error(message=message, error=ValueError)

    variables = {name: str(thread_count) for name in _RESIZABLE_THREAD_VARIABLES}
    variables.update({name: str(resolved_maximum) for name in _IMPORT_LATCHED_THREAD_VARIABLES})

    for name, count in (additional_thread_variables or {}).items():
        if not name.strip():
            message = (
                f"Unable to {action}. Every name in the 'additional_thread_variables' argument must be a non-blank "
                f"environment variable name, but got {name!r}."
            )
            console.error(message=message, error=ValueError)
        if count < 1:
            message = (
                f"Unable to {action}. Every width in the 'additional_thread_variables' argument must be greater than "
                f"or equal to 1, but got {count} for {name!r}."
            )
            console.error(message=message, error=ValueError)
        variables[name] = str(count)

    return variables
