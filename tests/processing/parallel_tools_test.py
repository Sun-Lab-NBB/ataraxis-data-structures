"""Contains tests for the parallel_tools module provided by the processing package."""

import os
from collections.abc import Iterator
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor

import numba
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import limit_worker_threads, initialize_worker_threads
from ataraxis_data_structures.processing.parallel_tools import (
    _RESIZABLE_THREAD_VARIABLES,
    _IMPORT_LATCHED_THREAD_VARIABLES,
)

_ALL_THREAD_VARIABLES: tuple[str, ...] = _RESIZABLE_THREAD_VARIABLES + _IMPORT_LATCHED_THREAD_VARIABLES


@pytest.fixture
def restore_numba_threads() -> Iterator[None]:
    """Restores the numba thread count the session was using, so a pinning test does not throttle its siblings."""
    previous = numba.get_num_threads()
    yield
    numba.set_num_threads(previous)


def test_limit_worker_threads_sets_and_clears_absent_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the context sets every threading variable and removes the ones that were absent on exit."""
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with limit_worker_threads():
        for variable in _ALL_THREAD_VARIABLES:
            assert os.environ[variable] == "1"

    for variable in _ALL_THREAD_VARIABLES:
        assert variable not in os.environ


def test_limit_worker_threads_restores_preexisting_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the context restores the values the caller's environment already carried."""
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.setenv(variable, "7")

    with limit_worker_threads(thread_count=2):
        for variable in _ALL_THREAD_VARIABLES:
            assert os.environ[variable] == "2"

    for variable in _ALL_THREAD_VARIABLES:
        assert os.environ[variable] == "7"


def test_limit_worker_threads_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the restore runs when the wrapped block raises."""
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.setenv(variable, "5")

    with pytest.raises(RuntimeError), limit_worker_threads():
        _raise_probe_error()

    for variable in _ALL_THREAD_VARIABLES:
        assert os.environ[variable] == "5"


def test_limit_worker_threads_rejects_invalid_thread_count() -> None:
    """Verifies that a thread count below one is rejected."""
    message = (
        "Unable to limit the thread count used by the worker processes. The 'thread_count' argument must be "
        "greater than or equal to 1, but got 0."
    )
    with pytest.raises(ValueError, match=error_format(message)), limit_worker_threads(thread_count=0):
        pass


def test_limit_worker_threads_rejects_a_maximum_below_the_thread_count() -> None:
    """Verifies that a maximum thread count narrower than the per-worker count is rejected."""
    message = (
        "Unable to limit the thread count used by the worker processes. The 'maximum_thread_count' argument must be "
        "greater than or equal to the 'thread_count' argument (4), but got 2."
    )
    with (
        pytest.raises(ValueError, match=error_format(message)),
        limit_worker_threads(thread_count=4, maximum_thread_count=2),
    ):
        pass


def test_limit_worker_threads_covers_the_lazily_read_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the context pins the backends reading their variable after import alongside the ones that read
    it while importing.
    """
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with limit_worker_threads(thread_count=4):
        assert os.environ["OPENCV_FOR_THREADS_NUM"] == "4"
        assert os.environ["OPENCV_FFMPEG_THREADS"] == "4"
        assert os.environ["TIFFFILE_NUM_THREADS"] == "4"


def test_limit_worker_threads_widens_only_the_import_latched_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a maximum above the per-worker count reaches the latched backends while the resizable ones stay
    narrow.
    """
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with limit_worker_threads(thread_count=1, maximum_thread_count=8):
        for variable in _RESIZABLE_THREAD_VARIABLES:
            assert os.environ[variable] == "1"
        for variable in _IMPORT_LATCHED_THREAD_VARIABLES:
            assert os.environ[variable] == "8"

    for variable in _ALL_THREAD_VARIABLES:
        assert variable not in os.environ


def test_thread_variables_omit_the_numba_ceiling() -> None:
    """Verifies that numba's import-latched ceiling is absent, since writing it breaks a process that imported numba."""
    assert "NUMBA_NUM_THREADS" not in _ALL_THREAD_VARIABLES


def test_import_latched_group_holds_the_variables_no_runtime_setter_reaches() -> None:
    """Verifies that the transform and dataframe backends fixing their width as they load are grouped as latched."""
    assert "OMP_NUM_THREADS" in _IMPORT_LATCHED_THREAD_VARIABLES
    assert "DUCC0_NUM_THREADS" in _IMPORT_LATCHED_THREAD_VARIABLES
    assert "POLARS_MAX_THREADS" in _IMPORT_LATCHED_THREAD_VARIABLES
    assert not set(_RESIZABLE_THREAD_VARIABLES).intersection(_IMPORT_LATCHED_THREAD_VARIABLES)


def test_resizable_group_holds_every_backend_a_runtime_setter_reaches() -> None:
    """Verifies that the BLAS backends NumPy links against and the pools carrying their own setter stay resizable."""
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "FLEXIBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        assert variable in _RESIZABLE_THREAD_VARIABLES


def test_limit_worker_threads_leaves_the_numba_count_untouched() -> None:
    """Verifies that the parent-side context does not throttle its own numba pool, which no spawned child inherits."""
    previous = numba.get_num_threads()

    with limit_worker_threads(thread_count=1):
        assert numba.get_num_threads() == previous

    assert numba.get_num_threads() == previous


def test_limit_worker_threads_writes_and_restores_additional_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a caller-supplied variable is pinned alongside the known ones and restored on exit."""
    monkeypatch.delenv("PROBE_BACKEND_NUM_THREADS", raising=False)
    monkeypatch.setenv("PROBE_KEPT_NUM_THREADS", "9")

    with limit_worker_threads(
        thread_count=1,
        additional_thread_variables={"PROBE_BACKEND_NUM_THREADS": 5, "PROBE_KEPT_NUM_THREADS": 5},
    ):
        assert os.environ["PROBE_BACKEND_NUM_THREADS"] == "5"
        assert os.environ["PROBE_KEPT_NUM_THREADS"] == "5"

    assert "PROBE_BACKEND_NUM_THREADS" not in os.environ
    assert os.environ["PROBE_KEPT_NUM_THREADS"] == "9"


def test_additional_variables_override_a_known_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an entry naming a variable the module already writes replaces the width it would take."""
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with limit_worker_threads(thread_count=1, additional_thread_variables={"OMP_NUM_THREADS": 16}):
        assert os.environ["OMP_NUM_THREADS"] == "16"
        assert os.environ["DUCC0_NUM_THREADS"] == "1"


def test_additional_variables_hold_a_known_variable_at_an_exact_width(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an entry holds one variable at a single width while the counts around it widen."""
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with limit_worker_threads(
        thread_count=1, maximum_thread_count=8, additional_thread_variables={"POLARS_MAX_THREADS": 1}
    ):
        assert os.environ["POLARS_MAX_THREADS"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "8"


def test_limit_worker_threads_rejects_a_blank_additional_variable_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a blank variable name is rejected before any variable is written."""
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    message = (
        "Unable to limit the thread count used by the worker processes. Every name in the "
        "'additional_thread_variables' argument must be a non-blank environment variable name, but got '  '."
    )
    with (
        pytest.raises(ValueError, match=error_format(message)),
        limit_worker_threads(additional_thread_variables={"  ": 2}),
    ):
        pass

    for variable in _ALL_THREAD_VARIABLES:
        assert variable not in os.environ


def test_limit_worker_threads_rejects_an_additional_width_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that an invalid width is rejected and that the entries ahead of it are left unwritten."""
    monkeypatch.delenv("PROBE_VALID_NUM_THREADS", raising=False)

    message = (
        "Unable to limit the thread count used by the worker processes. Every width in the "
        "'additional_thread_variables' argument must be greater than or equal to 1, but got 0 for "
        "'PROBE_INVALID_NUM_THREADS'."
    )
    with (
        pytest.raises(ValueError, match=error_format(message)),
        limit_worker_threads(
            additional_thread_variables={"PROBE_VALID_NUM_THREADS": 2, "PROBE_INVALID_NUM_THREADS": 0}
        ),
    ):
        pass

    assert "PROBE_VALID_NUM_THREADS" not in os.environ


def test_initialize_worker_threads_pins_every_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the worker initializer writes every threading variable in the calling process."""
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    initialize_worker_threads(thread_count=3)

    for variable in _ALL_THREAD_VARIABLES:
        assert os.environ[variable] == "3"


def test_initialize_worker_threads_holds_the_import_latched_backends_at_the_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifies that the worker initializer holds the latched backends at the maximum when it inherits a width above
    the per-worker count.
    """
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.setenv(variable, "8")

    initialize_worker_threads(thread_count=1, maximum_thread_count=8)

    for variable in _RESIZABLE_THREAD_VARIABLES:
        assert os.environ[variable] == "1"
    for variable in _IMPORT_LATCHED_THREAD_VARIABLES:
        assert os.environ[variable] == "8"


def test_initialize_worker_threads_writes_additional_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the worker initializer pins a caller-supplied variable alongside the known ones."""
    monkeypatch.delenv("PROBE_BACKEND_NUM_THREADS", raising=False)

    initialize_worker_threads(
        thread_count=2, maximum_thread_count=8, additional_thread_variables={"PROBE_BACKEND_NUM_THREADS": 8}
    )

    assert os.environ["PROBE_BACKEND_NUM_THREADS"] == "8"


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
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.setenv(variable, "12")

    initialize_worker_threads()

    for variable in _ALL_THREAD_VARIABLES:
        assert os.environ[variable] == "1"


def test_initialize_worker_threads_rejects_invalid_thread_count() -> None:
    """Verifies that a thread count below one is rejected."""
    message = (
        "Unable to initialize the thread count used by the worker process. The 'thread_count' argument must be "
        "greater than or equal to 1, but got 0."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        initialize_worker_threads(thread_count=0)


def test_initialize_worker_threads_rejects_a_maximum_below_the_thread_count() -> None:
    """Verifies that a maximum thread count narrower than the per-worker count is rejected."""
    message = (
        "Unable to initialize the thread count used by the worker process. The 'maximum_thread_count' argument must "
        "be greater than or equal to the 'thread_count' argument (4), but got 2."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        initialize_worker_threads(thread_count=4, maximum_thread_count=2)


def test_spawned_worker_loads_its_backends_at_the_maximum(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the widened count survives into the moment a spawned worker's first work item runs, which is the
    point a backend imported by that item sizes its pool.
    """
    for variable in _ALL_THREAD_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    with (
        limit_worker_threads(thread_count=1, maximum_thread_count=6),
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context(method="spawn"),
            initializer=initialize_worker_threads,
            initargs=(1, 6),
        ) as executor,
    ):
        observed = executor.submit(_read_thread_variables).result()

    for variable in _RESIZABLE_THREAD_VARIABLES:
        assert observed[variable] == "1"
    for variable in _IMPORT_LATCHED_THREAD_VARIABLES:
        assert observed[variable] == "6"


def _raise_probe_error() -> None:
    """Raises an error so a test can observe how the surrounding context manager handles an exceptional exit."""
    message = "simulated failure inside the wrapped block"
    raise RuntimeError(message)


def _read_thread_variables() -> dict[str, str]:
    """Reports the threading variables the calling process carries, standing in for a backend that reads them as it
    loads.
    """
    return {variable: os.environ.get(variable, "") for variable in _ALL_THREAD_VARIABLES}
