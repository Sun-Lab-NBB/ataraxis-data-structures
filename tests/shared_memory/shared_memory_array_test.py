"""Contains tests for the shared_memory_array module provided by the shared_memory package."""

from typing import Any
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest
from numpy.typing import NDArray
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import SharedMemoryArray


@pytest.fixture
def int_array() -> NDArray[np.int32]:
    """Returns an integer NumPy array prototype used by the tests below."""
    return np.array([1, 2, 3, 4, 5], dtype=np.int32)


@pytest.fixture
def float_array() -> NDArray[np.float64]:
    """Returns a floating-point NumPy array prototype used by the tests below."""
    return np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)


@pytest.fixture
def bool_array() -> NDArray[np.bool_]:
    """Returns a boolean NumPy array prototype used by the tests below."""
    return np.array([True, False, True, False, True], dtype=bool)


@pytest.fixture
def string_array() -> NDArray[np.str_]:
    """Returns a string NumPy array prototype used by the tests below."""
    return np.array(["a", "b", "c", "d", "e"], dtype="<U1")


@pytest.fixture
def multidimensional_array() -> NDArray[np.int32]:
    """Returns a multidimensional NumPy array prototype used by the tests below."""
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)


def test_create_array(int_array: NDArray[np.int32]) -> None:
    """Verifies the functionality of the SharedMemoryArray class create_array() method.

    Verifies creating an array from a valid NumPy prototype, checks its name, shape, datatype, and connection status,
    and confirms data integrity after creation.
    """
    # Creates a SharedMemoryArray instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_create_array", prototype=int_array)
    shared_memory_array.connect()
    assert shared_memory_array.name == "test_create_array"
    assert shared_memory_array.shape == int_array.shape
    assert shared_memory_array.datatype == int_array.dtype
    assert shared_memory_array.is_connected

    # Verifies data integrity using array context manager.
    with shared_memory_array.array(with_lock=False) as shared_array:
        np.testing.assert_array_equal(actual=shared_array, desired=int_array)

    # Destroys the array, freeing up the buffer name to be used by other SMA instances.
    shared_memory_array.disconnect()
    shared_memory_array.destroy()

    # Verifies that the buffer has been freed up.
    shared_memory_array = SharedMemoryArray.create_array(name="test_create_array", prototype=int_array)
    shared_memory_array.connect()
    shared_memory_array.disconnect()

    # Verifies that exist_ok flag works as expected by recreating an already existing buffer.
    shared_memory_array = SharedMemoryArray.create_array(name="test_create_array", prototype=int_array, exists_ok=True)
    shared_memory_array.connect()

    # Cleans up after the runtime.
    shared_memory_array.disconnect()
    shared_memory_array.destroy()


def test_create_array_recreates_leftover_buffer(int_array: NDArray[np.int32], monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the SharedMemoryArray class create_array() method reclaims a leftover shared memory buffer when
    it is called with the 'exists_ok' flag enabled.

    A buffer survives its creating runtime only on Unix, so the leftover state is staged here through a patched
    unlink() to exercise the recreation path on every supported platform.
    """
    buffer_name = "test_create_array_leftover"

    # Occupies the buffer name for the duration of the test, reproducing a buffer left behind by an earlier runtime.
    leftover_buffer = SharedMemory(name=buffer_name, create=True, size=int_array.nbytes)

    class LeftoverReleasingSharedMemory(SharedMemory):
        """Closes the leftover buffer handle held by this test whenever a shared memory buffer is unlinked."""

        def unlink(self) -> None:
            """Unlinks the shared memory buffer and releases the leftover handle opened by this test."""
            super().unlink()
            # Unix frees the buffer name through unlink() alone, while Windows ignores unlink() and frees the name
            # once the last handle to the buffer is closed. Releasing the handle held by this test brings both
            # platforms to the same post-unlink state, so the recreation path below runs identically on each.
            leftover_buffer.close()

    monkeypatch.setattr(
        target="ataraxis_data_structures.shared_memory.shared_memory_array.SharedMemory",
        name=LeftoverReleasingSharedMemory,
    )

    # Recreates the buffer over the leftover one and confirms that the prototype data reached the new buffer.
    shared_memory_array = SharedMemoryArray.create_array(name=buffer_name, prototype=int_array, exists_ok=True)
    shared_memory_array.connect()
    with shared_memory_array.array(with_lock=False) as shared_array:
        np.testing.assert_array_equal(actual=shared_array, desired=int_array)

    # Cleans up after the runtime.
    shared_memory_array.disconnect()
    shared_memory_array.destroy()


def test_create_array_multidimensional(multidimensional_array: NDArray[np.int32]) -> None:
    """Verifies the SharedMemoryArray class supports multidimensional arrays.

    Verifies creating an array from a 2D NumPy prototype and confirms its shape and data integrity.
    """
    # Creates a SharedMemoryArray instance with a 2D array.
    shared_memory_array = SharedMemoryArray.create_array(name="test_multidim", prototype=multidimensional_array)
    shared_memory_array.connect()
    assert shared_memory_array.shape == multidimensional_array.shape
    assert shared_memory_array.datatype == multidimensional_array.dtype

    # Verifies data integrity.
    with shared_memory_array.array(with_lock=False) as shared_array:
        np.testing.assert_array_equal(actual=shared_array, desired=multidimensional_array)

    # Cleans up.
    shared_memory_array.destroy()


def test_repr(int_array: NDArray[np.int32]) -> None:
    """Verifies the functionality of the SharedMemoryArray class __repr__() method.

    Verifies that the string representation reflects the array name, shape, datatype, and connection status.
    """
    # Creates a SharedMemoryArray instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_repr", prototype=int_array)
    shared_memory_array.connect()
    expected_repr = (
        f"SharedMemoryArray(name='test_repr', shape={int_array.shape}, datatype={int_array.dtype}, connected=True)"
    )
    assert repr(shared_memory_array) == expected_repr

    # Cleans up.
    shared_memory_array.destroy()


@pytest.mark.parametrize(
    "array_fixture, buffer_name, index, expected, expected_type",
    [
        # Covers integer arrays.
        ("int_array", "test_getitem_int_1", 0, 1, np.int32),
        ("int_array", "test_getitem_int_2", -1, 5, np.int32),
        ("int_array", "test_getitem_int_3", slice(0, 3), np.array([1, 2, 3]), np.ndarray),
        ("int_array", "test_getitem_int_4", slice(1, None), np.array([2, 3, 4, 5]), np.ndarray),
        ("int_array", "test_getitem_int_5", slice(-3, -1), np.array([3, 4]), np.ndarray),
        # Covers float arrays.
        ("float_array", "test_getitem_float_1", 0, 1.1, np.float64),
        ("float_array", "test_getitem_float_2", -1, 5.5, np.float64),
        ("float_array", "test_getitem_float_3", slice(0, 3), np.array([1.1, 2.2, 3.3]), np.ndarray),
        # Covers boolean arrays.
        ("bool_array", "test_getitem_bool_1", 0, True, np.bool_),
        ("bool_array", "test_getitem_bool_2", 1, False, np.bool_),
        ("bool_array", "test_getitem_bool_3", slice(0, 3), np.array([True, False, True]), np.ndarray),
        # Covers string arrays.
        ("string_array", "test_getitem_string_1", 0, "a", np.str_),
        ("string_array", "test_getitem_string_2", -1, "e", np.str_),
        ("string_array", "test_getitem_string_3", slice(0, 3), np.array(["a", "b", "c"]), np.ndarray),
    ],
)
def test_getitem(
    request: pytest.FixtureRequest,
    array_fixture: str,
    buffer_name: str,
    index: int | slice,
    expected: Any,
    expected_type: type,
) -> None:
    """Verifies the functionality of the SharedMemoryArray class __getitem__() method.

    Verifies reading data at positive, negative, single, and slice indices across int32, float64, bool, and string
    arrays, and confirms the correct return type for each scenario.

    Notes:
        Uses separate buffer names to prevent name collisions when tests are spread over multiple cores during
        pytest-xdist runtime.
    """
    # Uses the test-specific fixture to get the prototype array and instantiate the SharedMemoryArray instance.
    sample_array = request.getfixturevalue(array_fixture)
    shared_memory_array = SharedMemoryArray.create_array(name=buffer_name, prototype=sample_array)
    shared_memory_array.connect()

    # Reads data using a test-specific index.
    result = shared_memory_array[index]

    # Verifies that the value returned by the test matches expectation.
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual=result, desired=expected)
    else:
        assert result == expected

    # Verifies that the type returned by the test matches expectation.
    assert isinstance(result, expected_type)

    # Cleans up.
    shared_memory_array.destroy()


@pytest.mark.parametrize(
    "array_fixture, buffer_name, index, data, expected",
    [
        # Covers integer arrays.
        ("int_array", "test_setitem_int_1", 0, 10, 10),
        ("int_array", "test_setitem_int_2", -1, 50, 50),
        ("int_array", "test_setitem_int_3", slice(0, 3), [10, 20, 30], [10, 20, 30]),
        ("int_array", "test_setitem_int_4", slice(1, None), [20, 30, 40, 50], [20, 30, 40, 50]),
        ("int_array", "test_setitem_int_5", slice(-3, -1), [30, 40], [30, 40]),
        ("int_array", "test_setitem_int_6", 0, np.int32(15), 15),
        # Covers float arrays.
        ("float_array", "test_setitem_float_1", 0, 10.5, 10.5),
        ("float_array", "test_setitem_float_2", -1, 50.5, 50.5),
        ("float_array", "test_setitem_float_3", slice(0, 3), [10.1, 20.2, 30.3], [10.1, 20.2, 30.3]),
        # Covers boolean arrays.
        ("bool_array", "test_setitem_bool_1", 0, False, False),
        ("bool_array", "test_setitem_bool_2", -1, False, False),
        ("bool_array", "test_setitem_bool_3", slice(0, 3), [False, False, False], [False, False, False]),
        # Covers string arrays.
        ("string_array", "test_setitem_string_1", 0, "x", "x"),
        ("string_array", "test_setitem_string_2", -1, "z", "z"),
        ("string_array", "test_setitem_string_3", slice(0, 3), ["x", "y", "z"], ["x", "y", "z"]),
    ],
)
def test_setitem(
    request: pytest.FixtureRequest,
    array_fixture: str,
    buffer_name: str,
    index: int | slice,
    data: Any,
    expected: Any,
) -> None:
    """Verifies the functionality of the SharedMemoryArray class __setitem__() method.

    Verifies writing single values and lists or arrays at positive, negative, single, and slice indices across int32,
    float64, bool, and string arrays, and confirms each write is applied correctly.

    Notes:
        Uses separate buffer names to prevent name collisions when tests are spread over multiple cores during
        pytest-xdist runtime.
    """
    # Uses the test-specific fixture to get the prototype array and instantiate the SharedMemoryArray instance.
    sample_array = request.getfixturevalue(array_fixture)
    shared_memory_array = SharedMemoryArray.create_array(name=buffer_name, prototype=sample_array)
    shared_memory_array.connect()

    # Writes test data using the tested combination of index and input data.
    shared_memory_array[index] = data
    result = shared_memory_array[index]  # Reads the (supposedly) modified data back.

    # Verifies that the value(s) were written correctly.
    if isinstance(expected, list):
        np.testing.assert_array_equal(actual=result, desired=expected)
    else:
        assert result == expected

    # Checks that the data type of the written data matches the original array's data type.
    if isinstance(result, np.ndarray):
        assert result.dtype == sample_array.dtype
    else:
        assert isinstance(result, type(sample_array[0]))

    # Cleans up.
    shared_memory_array.destroy()


def test_array_context_manager(int_array: NDArray[np.int32]) -> None:
    """Verifies the functionality of the SharedMemoryArray class array() context manager.

    Verifies accessing the array with and without locking, and modifying the array through the context manager.
    """
    # Creates a SharedMemoryArray instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_array_cm", prototype=int_array)
    shared_memory_array.connect()

    # Tests reading with lock.
    with shared_memory_array.array(with_lock=True) as shared_array:
        np.testing.assert_array_equal(actual=shared_array, desired=int_array)
        assert isinstance(shared_array, np.ndarray)

    # Tests reading without the lock.
    with shared_memory_array.array(with_lock=False) as shared_array:
        np.testing.assert_array_equal(actual=shared_array, desired=int_array)

    # Tests modification through context manager.
    with shared_memory_array.array(with_lock=True) as shared_array:
        shared_array[0] = 100

    # Verifies the modification persisted.
    assert shared_memory_array[0] == 100

    # Cleans up.
    shared_memory_array.destroy()


def test_disconnect_connect(int_array: NDArray[np.int32]) -> None:
    """Verifies the functionality of the SharedMemoryArray class disconnect() and connect() methods.

    Verifies disconnecting from a connected array, reconnecting to a disconnected array, and confirming data integrity
    after reconnection.
    """
    # Creates two arrays to handle Windows garbage collection behavior.
    connection_array = SharedMemoryArray.create_array(name="test_connect", prototype=int_array)
    shared_memory_array = SharedMemoryArray.create_array(name="test_disconnect", prototype=int_array)

    # Connects to tested arrays.
    shared_memory_array.connect()
    connection_array.connect()

    # Tests disconnection.
    shared_memory_array.disconnect()
    assert not shared_memory_array.is_connected

    # Tests reconnection.
    connection_array.connect()
    assert connection_array.is_connected

    # Verifies data integrity after reconnection.
    with connection_array.array(with_lock=False) as shared_array:
        np.testing.assert_array_equal(actual=shared_array, desired=int_array)

    # Cleans up.
    connection_array.destroy()


def test_enable_buffer_destruction(int_array: NDArray[np.int32]) -> None:
    """Verifies the functionality of the enable_buffer_destruction() method.

    Verifies that enabling buffer destruction sets the corresponding flag correctly.
    """
    # Creates a SharedMemoryArray instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_destruction", prototype=int_array)
    shared_memory_array.connect()

    # Enables buffer destruction.
    shared_memory_array.enable_buffer_destruction()
    assert shared_memory_array._destroy_buffer

    # Manually cleans up (to prevent automatic destruction during test).
    shared_memory_array._destroy_buffer = False
    shared_memory_array.destroy()


def test_pickle_state_round_trip(int_array: NDArray[np.int32]) -> None:
    """Verifies the functionality of the SharedMemoryArray class __getstate__() and __setstate__() pickle hooks.

    Verifies that __getstate__() reports the instance as disconnected and drops the live buffer handle, and that
    __setstate__() restores the metadata so the receiving instance can reconnect to the same shared buffer. The hooks
    are exercised directly because the underlying Lock can only be transferred to a child process through inheritance,
    rather than through an in-process pickle round-trip.
    """
    # Creates and connects a source instance, then writes a sentinel value through it.
    shared_memory_array = SharedMemoryArray.create_array(name="test_pickle", prototype=int_array)
    shared_memory_array.connect()
    shared_memory_array[0] = 99

    # Captures the picklable state and verifies the live handle and connection flags are reset.
    state = shared_memory_array.__getstate__()
    assert state["_buffer"] is None
    assert state["_array"] is None
    assert state["_connected"] is False
    assert state["_destroy_buffer"] is False

    # Restores the state into a fresh instance the same way the pickle protocol does after __new__.
    restored = SharedMemoryArray.__new__(SharedMemoryArray)
    restored.__setstate__(state)

    # Verifies that the metadata survived the transfer and the restored instance reports as disconnected.
    assert restored.name == shared_memory_array.name
    assert restored.shape == shared_memory_array.shape
    assert restored.datatype == shared_memory_array.datatype
    assert not restored.is_connected

    # Verifies that the restored instance can reconnect to the same buffer and read the sentinel value.
    restored.connect()
    assert restored.is_connected
    assert restored[0] == 99

    # Cleans up. Releases the source handle, then destroys the buffer through the restored instance.
    shared_memory_array.disconnect()
    restored.enable_buffer_destruction()
    restored.destroy()


def test_create_array_errors() -> None:
    """Verifies error handling in the SharedMemoryArray class create_array() method.

    Verifies that creating an array with an invalid prototype type and with a name that already exists raises the
    expected errors.
    """
    # Tests with an invalid prototype type.
    message = (
        f"Invalid 'prototype' argument type encountered when creating SharedMemoryArray object 'test_error'. "
        f"Expected a NumPy array but instead encountered {type([1, 2, 3]).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        SharedMemoryArray.create_array(name="test_error", prototype=[1, 2, 3])

    # Tests with existing name.
    # Maintains reference to prevent Windows garbage collection.
    _existing = SharedMemoryArray.create_array(name="existing_array", prototype=np.array([1, 2, 3]))
    _existing.connect()
    message = (
        "Unable to create the 'existing_array' SharedMemoryArray object, as an object with this name already "
        "exists. If this method is called from a child process, use the connect() method instead "
        "to connect to the existing buffer. To clean-up the buffer left over from a previous "
        "runtime, run this method with the 'exists_ok' flag set to True."
    )
    with pytest.raises(FileExistsError, match=error_format(message)):
        SharedMemoryArray.create_array(name="existing_array", prototype=np.array([4, 5, 6]))


def test_getitem_errors(int_array: NDArray[np.int32]) -> None:
    """Verifies error handling in the SharedMemoryArray class __getitem__() method.

    Verifies that reading from a disconnected array raises a ConnectionError.
    """
    # Creates the array without connecting.
    shared_memory_array = SharedMemoryArray.create_array(name="test_getitem_error", prototype=int_array)

    # Tests reading from the disconnected array.
    message = (
        "Unable to access the data stored in the test_getitem_error SharedMemoryArray instance, as the instance is "
        "not connected to the shared memory buffer. Call the connect() method prior to accessing the array's "
        "data."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = shared_memory_array[0]


def test_setitem_errors(int_array: NDArray[np.int32]) -> None:
    """Verifies error handling in the SharedMemoryArray class __setitem__() method.

    Verifies that writing to a disconnected array raises a ConnectionError.
    """
    # Creates the array without connecting.
    shared_memory_array = SharedMemoryArray.create_array(name="test_setitem_error", prototype=int_array)

    # Tests writing to the disconnected array.
    message = (
        "Unable to modify the data stored in the test_setitem_error SharedMemoryArray instance, as the instance is "
        "not connected to the shared memory buffer. Call the connect() method prior to modifying the array's "
        "data."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        shared_memory_array[0] = 10


def test_array_context_manager_errors(int_array: NDArray[np.int32]) -> None:
    """Verifies error handling in the SharedMemoryArray class array() context manager.

    Verifies that using the array() context manager on a disconnected instance raises a ConnectionError.
    """
    # Creates the array without connecting.
    shared_memory_array = SharedMemoryArray.create_array(name="test_array_error", prototype=int_array)

    # Tests using array() on disconnected instance.
    message = (
        "Unable to access the data stored in the test_array_error SharedMemoryArray instance, as it is not "
        "connected to the shared memory buffer. Call the connect() method prior to calling the array() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)), shared_memory_array.array() as _array:
        pass


def read_write_worker(shared_memory_array: SharedMemoryArray) -> None:
    """Connects to a shared array, writes a test value, verifies the write, and disconnects.

    Args:
        shared_memory_array: The SharedMemoryArray instance to test.
    """
    # Connects to the input array.
    shared_memory_array.connect()

    # Writes and verifies that the test payload has been written.
    shared_memory_array[2] = 42
    assert shared_memory_array[2] == 42

    # Disconnects from the array and terminates the process.
    shared_memory_array.disconnect()


def concurrent_worker(shared_memory_array: SharedMemoryArray, index: int) -> None:
    """Repeatedly reads, increments, and writes back the value at a specific array index.

    Args:
        shared_memory_array: The SharedMemoryArray instance to test.
        index: The array index to repeatedly increment.
    """
    # Connects to the array.
    shared_memory_array.connect()

    # Performs repeated increment operations.
    for _ in range(100):
        # Reads data from the input index.
        value = shared_memory_array[index]
        # Increments the value by one and writes it back to the array.
        shared_memory_array[index] = value + 1

    # Disconnects and terminates the process.
    shared_memory_array.disconnect()


@pytest.mark.xdist_group("cross_process")
def test_cross_process_read_write() -> None:
    """Verifies the ability of the SharedMemoryArray class to share data across processes.

    Verifies writing data from a child process and reading it back from the parent process.
    """
    # Instantiates the SharedMemoryArray instance.
    prototype = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    shared_memory_array = SharedMemoryArray.create_array(name="test_cross_process", prototype=prototype)

    # Writes (and reads) to the SMA from a different process.
    process = Process(target=read_write_worker, args=(shared_memory_array,))
    process.start()
    process.join()

    # Finishes setting up the array in the local process.
    shared_memory_array.connect()
    shared_memory_array.enable_buffer_destruction()

    # Verifies that the data written by the other process is accessible from the main process.
    assert shared_memory_array[2] == 42

    # Cleans up.
    shared_memory_array.destroy()


@pytest.mark.xdist_group("cross_process")
def test_cross_process_concurrent_access() -> None:
    """Verifies the ability of the SharedMemoryArray class to handle concurrent access from multiple processes.

    Verifies that five processes incrementing different array elements concurrently produce the expected final values.
    """
    # Instantiates the SharedMemoryArray instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_concurrent", prototype=np.zeros(5, dtype=np.int32))

    # Generates multiple processes and uses each to repeatedly increment different indices.
    processes = [Process(target=concurrent_worker, args=(shared_memory_array, index)) for index in range(5)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()

    # Finishes setting up the array in the local process.
    shared_memory_array.connect()
    shared_memory_array.enable_buffer_destruction()

    # Verifies all indices were incremented to the expected value.
    with shared_memory_array.array(with_lock=False) as shared_array:
        assert np.all(shared_array == 100)

    # Cleans up.
    shared_memory_array.destroy()


def test_create_array_releases_buffer_when_initialization_fails() -> None:
    """Verifies that a prototype the buffer cannot be initialized from leaves no shared memory segment behind."""
    buffer_name = "test_create_array_failed_init"

    # A zero-dimensional array passes the ndarray type guard, then fails the slice assignment that fills the buffer.
    with pytest.raises(IndexError):
        SharedMemoryArray.create_array(name=buffer_name, prototype=np.array(5, dtype=np.uint8))

    # No instance exists to call destroy(), so the segment must have been released by create_array itself. Recreating
    # under the same name without the 'exists_ok' escape hatch proves the name was freed.
    recreated = SharedMemoryArray.create_array(name=buffer_name, prototype=np.array([1, 2, 3], dtype=np.uint8))
    recreated.connect()
    recreated.disconnect()
    recreated.destroy()
