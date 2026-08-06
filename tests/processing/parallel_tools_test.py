"""Contains tests for the parallel_tools module provided by the processing package."""

import os
from collections.abc import Iterator

import numba
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import limit_worker_threads, initialize_worker_threads
from ataraxis_data_structures.processing.parallel_tools import _THREAD_LIMIT_VARIABLES


def test_limit_worker_threads_sets_and_clears_absent_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the context sets every threading variable and removes the ones that were absent on exit."""
    for variable in _THREAD_LIMIT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with limit_worker_threads():
        for variable in _THREAD_LIMIT_VARIABLES:
            assert os.environ[variable] == "1"

    for variable in _THREAD_LIMIT_VARIABLES:
        assert variable not in os.environ


def test_limit_worker_threads_restores_preexisting_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the context restores the values the caller's environment already carried."""
    for variable in _THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(variable, "7")

    with limit_worker_threads(thread_count=2):
        for variable in _THREAD_LIMIT_VARIABLES:
            assert os.environ[variable] == "2"

    for variable in _THREAD_LIMIT_VARIABLES:
        assert os.environ[variable] == "7"


def test_limit_worker_threads_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the restore runs when the wrapped block raises."""
    for variable in _THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(variable, "5")

    with pytest.raises(RuntimeError), limit_worker_threads():
        _raise_probe_error()

    for variable in _THREAD_LIMIT_VARIABLES:
        assert os.environ[variable] == "5"


def test_limit_worker_threads_rejects_invalid_thread_count() -> None:
    """Verifies that a thread count below one is rejected."""
    message = (
        "Unable to limit the thread count used by the worker processes. The 'thread_count' argument must be "
        "greater than or equal to 1, but got 0."
    )
    with pytest.raises(ValueError, match=error_format(message)), limit_worker_threads(thread_count=0):
        pass


@pytest.fixture
def restore_numba_threads() -> Iterator[None]:
    """Restores the numba thread count the session was using, so a pinning test does not throttle its siblings."""
    previous = numba.get_num_threads()
    yield
    numba.set_num_threads(previous)


def _raise_probe_error() -> None:
    """Raises an error so a test can observe how the surrounding context manager handles an exceptional exit."""
    message = "simulated failure inside the wrapped block"
    raise RuntimeError(message)


def test_limit_worker_threads_covers_the_lazily_read_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the context pins the backends reading their variable after import alongside the import-latched
    ones.
    """
    for variable in _THREAD_LIMIT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with limit_worker_threads(thread_count=4):
        assert os.environ["POLARS_MAX_THREADS"] == "4"
        assert os.environ["OPENCV_FFMPEG_THREADS"] == "4"
        assert os.environ["TIFFFILE_NUM_THREADS"] == "4"


def test_thread_limit_variables_omit_the_numba_ceiling() -> None:
    """Verifies that numba's import-latched ceiling is absent, since writing it breaks a process that imported numba."""
    assert "NUMBA_NUM_THREADS" not in _THREAD_LIMIT_VARIABLES


def test_limit_worker_threads_leaves_the_numba_count_untouched() -> None:
    """Verifies that the parent-side context does not throttle its own numba pool, which no spawned child inherits."""
    previous = numba.get_num_threads()

    with limit_worker_threads(thread_count=1):
        assert numba.get_num_threads() == previous

    assert numba.get_num_threads() == previous


def test_initialize_worker_threads_pins_every_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the worker initializer writes every threading variable in the calling process."""
    for variable in _THREAD_LIMIT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    initialize_worker_threads(thread_count=3)

    for variable in _THREAD_LIMIT_VARIABLES:
        assert os.environ[variable] == "3"


def test_initialize_worker_threads_pins_the_numba_pool(restore_numba_threads: None) -> None:
    """Verifies that the worker initializer narrows numba through its runtime setter rather than the environment."""
    initialize_worker_threads(thread_count=2)

    assert numba.get_num_threads() == 2


def test_initialize_worker_threads_narrows_a_request_above_the_numba_ceiling(restore_numba_threads: None) -> None:
    """Verifies that a request above numba's latched ceiling is narrowed to it, since the setter rejects a wider one."""
    ceiling = numba.config.NUMBA_NUM_THREADS

    initialize_worker_threads(thread_count=ceiling + 100)

    assert numba.get_num_threads() == ceiling


def test_initialize_worker_threads_overwrites_inherited_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the worker initializer replaces a width the calling process inherited."""
    for variable in _THREAD_LIMIT_VARIABLES:
        monkeypatch.setenv(variable, "12")

    initialize_worker_threads()

    for variable in _THREAD_LIMIT_VARIABLES:
        assert os.environ[variable] == "1"


def test_initialize_worker_threads_rejects_invalid_thread_count() -> None:
    """Verifies that a thread count below one is rejected."""
    message = (
        "Unable to initialize the thread count used by the worker process. The 'thread_count' argument must be "
        "greater than or equal to 1, but got 0."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        initialize_worker_threads(thread_count=0)
