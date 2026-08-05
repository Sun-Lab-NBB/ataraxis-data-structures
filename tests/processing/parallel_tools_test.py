"""Contains tests for the parallel_tools module provided by the processing package."""

import os

import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import limit_worker_threads
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


def _raise_probe_error() -> None:
    """Raises an error so a test can observe how the surrounding context manager handles an exceptional exit."""
    message = "simulated failure inside the wrapped block"
    raise RuntimeError(message)
