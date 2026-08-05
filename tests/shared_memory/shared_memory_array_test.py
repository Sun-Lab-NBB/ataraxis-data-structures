"""Contains tests for the shared_memory_array module provided by the shared_memory package."""

import gc
import pickle
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


@pytest.fixture
def spawning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reports the runtime as starting a Python process for the duration of the test.

    The pickle hooks refuse to run at any other point, and multiprocessing only reports a spawning process while it
    transfers arguments into one it is starting. Tests exercise those hooks in-process, since the instance's lock
    reaches another process through inheritance alone and therefore cannot survive an in-process pickle round-trip.
    """
    monkeypatch.setattr(
        target="ataraxis_data_structures.shared_memory.shared_memory_array.get_spawning_popen",
        name=lambda: "spawning",
    )


def test_create_array(int_array: NDArray[np.int32]) -> None:
    """Verifies the functionality of the SharedMemoryArray class create_array() method.

    Verifies creating an array from a valid NumPy prototype, checks its name, shape, datatype, and connection status,
    and confirms data integrity after creation.
    """
    # Creates a SharedMemoryArray instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_create_array", prototype=int_array)
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

    # Verifies that the buffer has been freed up. The instance is destroyed rather than merely disconnected, since
    # a live creating instance retains the power to unlink the name the recreation below claims.
    shared_memory_array = SharedMemoryArray.create_array(name="test_create_array", prototype=int_array)
    shared_memory_array.destroy()

    # Verifies that exist_ok flag works as expected by recreating an already existing buffer.
    shared_memory_array = SharedMemoryArray.create_array(name="test_create_array", prototype=int_array, exists_ok=True)

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


def test_disconnect_connect(int_array: NDArray[np.int32], spawning: None) -> None:
    """Verifies the functionality of the SharedMemoryArray class disconnect() and connect() methods.

    Verifies disconnecting from a connected array, reconnecting to a disconnected array, and confirming data integrity
    after reconnection.
    """
    shared_memory_array = SharedMemoryArray.create_array(name="test_disconnect", prototype=int_array)
    assert shared_memory_array.is_connected

    # Opens a second handle to the same buffer through the pickle hooks, which is the transfer path a child process
    # takes. Windows destroys a buffer once its last handle closes, so this handle is what keeps the buffer alive
    # across the disconnect below and lets the reconnection run identically on every supported platform.
    holder = SharedMemoryArray.__new__(SharedMemoryArray)
    holder.__setstate__(shared_memory_array.__getstate__())
    holder.connect()

    # Tests disconnection.
    shared_memory_array.disconnect()
    assert not shared_memory_array.is_connected

    # Tests reconnection.
    shared_memory_array.connect()
    assert shared_memory_array.is_connected

    # Verifies data integrity after reconnection.
    with shared_memory_array.array(with_lock=False) as shared_array:
        np.testing.assert_array_equal(actual=shared_array, desired=int_array)

    # Cleans up.
    holder.disconnect()
    shared_memory_array.destroy()


def test_create_array_returns_a_connected_instance(int_array: NDArray[np.int32]) -> None:
    """Verifies that create_array() connects the calling process to the shared memory buffer."""
    shared_memory_array = SharedMemoryArray.create_array(name="test_create_connects", prototype=int_array)

    # The instance reads and writes its data without any connect() call of its own.
    assert shared_memory_array.is_connected
    assert shared_memory_array[0] == 1
    shared_memory_array[0] = 42
    assert shared_memory_array[0] == 42

    shared_memory_array.destroy()


def test_connect_is_idempotent(int_array: NDArray[np.int32]) -> None:
    """Verifies that calling connect() on an already connected instance leaves the connection intact.

    Code written against the earlier API calls connect() in the process that created the array, so the call has to
    stay a harmless no-op rather than rebinding the array or opening a second handle to the buffer.
    """
    shared_memory_array = SharedMemoryArray.create_array(name="test_idempotent_connect", prototype=int_array)
    shared_memory_array[0] = 77

    # Captures the buffer handle and the array view established at creation.
    original_buffer = shared_memory_array._buffer
    original_array = shared_memory_array._array

    shared_memory_array.connect()
    shared_memory_array.connect()

    # The repeated calls neither reopened the buffer nor rebound the view, so the written value survives.
    assert shared_memory_array.is_connected
    assert shared_memory_array._buffer is original_buffer
    assert shared_memory_array._array is original_array
    assert shared_memory_array[0] == 77

    shared_memory_array.destroy()


def test_creating_process_arms_buffer_destruction(int_array: NDArray[np.int32]) -> None:
    """Verifies that the creating process destroys the shared memory buffer when its instance is garbage-collected."""
    buffer_name = "test_destruction"
    shared_memory_array = SharedMemoryArray.create_array(name=buffer_name, prototype=int_array)

    # The creating process arms the destruction without any call of its own.
    assert shared_memory_array._destroy_buffer

    # Dropping the only reference runs __del__, which destroys the buffer rather than merely disconnecting from it.
    del shared_memory_array
    gc.collect()

    # Reclaiming the name without the 'exists_ok' escape hatch proves the buffer was unlinked.
    recreated = SharedMemoryArray.create_array(name=buffer_name, prototype=int_array)
    recreated.destroy()


def test_recreating_a_name_revokes_the_previous_owner(int_array: NDArray[np.int32], spawning: None) -> None:
    """Verifies that recreating a buffer name strips the destruction right from the instance that held it.

    Unlinking addresses a buffer by name, so a superseded instance that kept its destruction right would unlink the
    replacement buffer rather than its own once garbage collection reached it.
    """
    buffer_name = "test_ownership_transfer"
    superseded = SharedMemoryArray.create_array(name=buffer_name, prototype=int_array)

    # Rebinds the name onto a new buffer while the first instance is still alive.
    replacement = SharedMemoryArray.create_array(name=buffer_name, prototype=int_array, exists_ok=True)
    replacement[0] = 123

    # Collecting the superseded instance runs the destruction path that would unlink the replacement's buffer.
    del superseded
    gc.collect()

    # Opening an independent handle is what detects the damage. An unlinked name leaves every existing mapping
    # readable, so the replacement's own view keeps returning the data whether or not the segment was destroyed,
    # and only a process connecting afterwards discovers that the name no longer resolves.
    observer = SharedMemoryArray.__new__(SharedMemoryArray)
    observer.__setstate__(replacement.__getstate__())
    assert observer[0] == 123
    observer.disconnect()

    assert replacement.is_connected
    assert replacement[0] == 123

    replacement.destroy()


def test_receiving_process_does_not_destroy_the_buffer(int_array: NDArray[np.int32], spawning: None) -> None:
    """Verifies that an instance transferred to another process disconnects from the buffer instead of destroying it.

    The transferred copies share the creating instance's buffer, so a copy that destroyed the buffer on garbage
    collection would end the array's lifetime for every process still using it.
    """
    shared_memory_array = SharedMemoryArray.create_array(name="test_receiver_destruction", prototype=int_array)
    shared_memory_array[0] = 55

    # Restores a copy the way the pickle protocol does when the instance reaches another process.
    received = SharedMemoryArray.__new__(SharedMemoryArray)
    received.__setstate__(shared_memory_array.__getstate__())
    assert not received._destroy_buffer

    # Dropping the copy releases its handle without unlinking the buffer, so the creating instance still reads data.
    del received
    gc.collect()

    assert shared_memory_array[0] == 55
    shared_memory_array.destroy()


def test_pickle_state_round_trip(int_array: NDArray[np.int32], spawning: None) -> None:
    """Verifies the functionality of the SharedMemoryArray class __getstate__() and __setstate__() pickle hooks.

    Verifies that __getstate__() reports the instance as disconnected and drops the live buffer handle, and that
    __setstate__() restores the metadata and connects the receiving instance to the same shared buffer. The hooks
    are exercised directly because the underlying Lock can only be transferred to a child process through inheritance,
    rather than through an in-process pickle round-trip.
    """
    # Creates and connects a source instance, then writes a sentinel value through it.
    shared_memory_array = SharedMemoryArray.create_array(name="test_pickle", prototype=int_array)
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

    # Verifies that the metadata survived the transfer and that restoring the state connected the instance, which is
    # what removes the connect() call from every process the array reaches.
    assert restored.name == shared_memory_array.name
    assert restored.shape == shared_memory_array.shape
    assert restored.datatype == shared_memory_array.datatype
    assert restored.is_connected
    assert restored[0] == 99

    # Verifies that the redundant connect() call code written against the earlier API makes remains a no-op.
    restored.connect()
    assert restored.is_connected

    # Cleans up. The restored instance only disconnects, leaving the buffer for the creating instance to destroy.
    restored.disconnect()
    shared_memory_array.destroy()


def test_serialization_outside_process_start_errors(int_array: NDArray[np.int32]) -> None:
    """Verifies that serializing an instance outside of starting a Python process raises a RuntimeError.

    A Queue serializes on its feeder thread, where the failure reaches neither the caller nor the consumer waiting on
    the transfer, so the refusal names the transfers that work instead of leaving the instance's lock to report it.
    """
    shared_memory_array = SharedMemoryArray.create_array(name="test_serialization_guard", prototype=int_array)

    message = (
        "Unable to transfer the 'test_serialization_guard' SharedMemoryArray instance to another process, as the "
        "transfer is not part of starting that process. Pass the instance through the 'args' argument of a Process "
        "or the 'initargs' argument of a process pool, both of which start a process and therefore inherit the "
        "instance's lock. A Queue or a Pipe cannot carry the instance, and it does not need to, since every process "
        "connected to the array already shares its data through the memory buffer."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        pickle.dumps(shared_memory_array)

    # The refusal leaves the instance usable, since it rejects the transfer rather than altering any state.
    assert shared_memory_array.is_connected
    shared_memory_array.destroy()


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
    # Creates the array and disconnects it, since create_array() returns a connected instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_getitem_error", prototype=int_array)
    shared_memory_array.disconnect()

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
    # Creates the array and disconnects it, since create_array() returns a connected instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_setitem_error", prototype=int_array)
    shared_memory_array.disconnect()

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
    # Creates the array and disconnects it, since create_array() returns a connected instance.
    shared_memory_array = SharedMemoryArray.create_array(name="test_array_error", prototype=int_array)
    shared_memory_array.disconnect()

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


def auto_connect_worker(shared_memory_array: SharedMemoryArray) -> None:
    """Writes a test value through an instance that connected itself while it was transferred to this process.

    Args:
        shared_memory_array: The SharedMemoryArray instance to test.
    """
    # Records whether the instance arrived connected, so the parent is able to tell an auto-connected instance from
    # one this worker connected itself.
    shared_memory_array[1] = 1 if shared_memory_array.is_connected else 0
    shared_memory_array[2] = 42

    shared_memory_array.disconnect()


def manual_connect_probe_worker(shared_memory_array: SharedMemoryArray) -> None:
    """Reports whether an instance created without the 'auto_connect' flag arrives disconnected.

    Args:
        shared_memory_array: The SharedMemoryArray instance to test.
    """
    arrived_connected = shared_memory_array.is_connected

    # Connects manually, which is the path every process takes when 'auto_connect' is disabled.
    shared_memory_array.connect()
    shared_memory_array[1] = 1 if arrived_connected else 0

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

    # Finishes setting up the array in the local process, which is connected to the buffer since its creation.

    # Verifies that the data written by the other process is accessible from the main process.
    assert shared_memory_array[2] == 42

    # Cleans up.
    shared_memory_array.destroy()


@pytest.mark.xdist_group("cross_process")
def test_cross_process_auto_connect() -> None:
    """Verifies that the 'auto_connect' flag connects a child process without a connect() call of its own."""
    prototype = np.zeros(shape=3, dtype=np.int32)
    shared_memory_array = SharedMemoryArray.create_array(
        name="test_auto_connect", prototype=prototype, auto_connect=True
    )

    # The worker writes through the instance without calling connect() itself.
    process = Process(target=auto_connect_worker, args=(shared_memory_array,))
    process.start()
    process.join()

    assert process.exitcode == 0

    # Confirms that the instance reached the worker already connected and that its write reached the shared buffer.
    assert shared_memory_array[1] == 1
    assert shared_memory_array[2] == 42

    shared_memory_array.destroy()


@pytest.mark.xdist_group("cross_process")
def test_cross_process_manual_connect_when_disabled() -> None:
    """Verifies that a child process receives a disconnected instance when 'auto_connect' is disabled."""
    prototype = np.zeros(shape=3, dtype=np.int32)
    shared_memory_array = SharedMemoryArray.create_array(
        name="test_manual_connect", prototype=prototype, auto_connect=False
    )

    process = Process(target=manual_connect_probe_worker, args=(shared_memory_array,))
    process.start()
    process.join()

    assert process.exitcode == 0

    # The worker recorded a disconnected arrival, so the connection remains the receiving process's responsibility.
    assert shared_memory_array[1] == 0

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

    # Finishes setting up the array in the local process, which is connected to the buffer since its creation.

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
    recreated.disconnect()
    recreated.destroy()


def test_create_array_reports_a_buffer_that_survives_unlinking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a buffer name still claimed after unlinking produces an explanatory error.

    Windows destroys a shared memory buffer only once its last handle closes, so unlink() leaves the name claimed
    while any handle remains open. The behavior is staged here through a stub, since POSIX frees the name outright.
    """

    class PersistentSharedMemory:
        """Reports the buffer name as permanently taken, which is the post-unlink state Windows leaves behind."""

        def __init__(self, name: str, **arguments: Any) -> None:
            if arguments.get("create", False):
                raise FileExistsError(name)
            self.name = name

        def unlink(self) -> None:
            """Accepts the unlink without freeing the name, matching the Windows no-op."""

        def close(self) -> None:
            """Accepts the close without freeing the name."""

    monkeypatch.setattr(
        target="ataraxis_data_structures.shared_memory.shared_memory_array.SharedMemory", name=PersistentSharedMemory
    )

    message = (
        "Unable to recreate the 'test_persistent_buffer' SharedMemoryArray object, as the shared memory buffer with "
        "this name is still held by an open handle. Windows destroys a buffer only once every handle to it is "
        "closed, so unlinking one that this runtime or another process still holds leaves the name claimed. "
        "Disconnect every SharedMemoryArray instance connected to this buffer, then call this method again."
    )
    with pytest.raises(FileExistsError, match=error_format(message)):
        SharedMemoryArray.create_array(
            name="test_persistent_buffer", prototype=np.array([1, 2, 3], dtype=np.uint8), exists_ok=True
        )
