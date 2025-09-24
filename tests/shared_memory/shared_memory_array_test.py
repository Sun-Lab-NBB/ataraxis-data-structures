"""Contains tests for SharedMemoryArray class and related methods, stored in the shared_memory package."""

import multiprocessing
from multiprocessing import Process

# When spawn creates child processes, they re-import this module with __name__ == '__mp_main__'
# This configures the main proces to use the 'spawn' multiprocessing method, which is the default for Windows systems.
if __name__ != "__mp_main__":
    try:
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from ataraxis_data_structures import SharedMemoryArray


@pytest.fixture
def int_array():
    """Returns an integer numpy array prototype used by the tests below."""
    return np.array([1, 2, 3, 4, 5], dtype=np.int32)


@pytest.fixture
def float_array():
    """Returns a floating numpy array prototype used by the tests below."""
    return np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)


@pytest.fixture
def bool_array():
    """Returns a boolean numpy array prototype used by the tests below."""
    return np.array([True, False, True, False, True], dtype=bool)


@pytest.fixture
def string_array():
    """Returns a string numpy array prototype used by the tests below."""
    return np.array(["a", "b", "c", "d", "e"], dtype="<U1")


@pytest.fixture
def multi_dim_array():
    """Returns a multidimensional numpy array prototype used by the tests below."""
    return np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)


def test_create_array(int_array):
    """Verifies the functionality of the SharedMemoryArray class create_array() method.

    Tested configurations:
        - 0: Creating a SharedMemoryArray with a valid numpy array
        - 1: Verifying the name, shape, datatype, and connection status of the created array
        - 2: Verifying the data integrity of the created array
    """
    # Creates a SharedMemoryArray instance
    sma = SharedMemoryArray.create_array("test_create_array", int_array)
    sma.connect()
    assert sma.name == "test_create_array"
    assert sma.shape == int_array.shape
    assert sma.datatype == int_array.dtype
    assert sma.is_connected

    # Verifies data integrity using array context manager
    with sma.array(with_lock=False) as arr:
        np.testing.assert_array_equal(arr, int_array)

    # Destroys the array, freeing up the buffer name to be used by other SMA instances
    sma.disconnect()
    sma.destroy()

    # Verifies that the buffer has been freed up
    sma = SharedMemoryArray.create_array("test_create_array", int_array)
    sma.connect()
    sma.disconnect()

    # Verifies that exist_ok flag works as expected by recreating an already existing buffer
    sma = SharedMemoryArray.create_array("test_create_array", int_array, exists_ok=True)
    sma.connect()

    # Cleans up after the runtime
    sma.disconnect()
    sma.destroy()


def test_create_array_multidimensional(multi_dim_array):
    """Verifies the SharedMemoryArray class supports multidimensional arrays.

    Tested configurations:
        - 0: Creating a SharedMemoryArray with a 2D numpy array
        - 1: Verifying the shape and data integrity of the multidimensional array
    """
    # Creates a SharedMemoryArray instance with a 2D array
    sma = SharedMemoryArray.create_array("test_multidim", multi_dim_array)
    sma.connect()
    assert sma.shape == multi_dim_array.shape
    assert sma.datatype == multi_dim_array.dtype

    # Verifies data integrity
    with sma.array(with_lock=False) as arr:
        np.testing.assert_array_equal(arr, multi_dim_array)

    # Cleans up
    sma.destroy()


def test_repr(int_array):
    """Verifies the functionality of the SharedMemoryArray class __repr__() method.

    Tested configurations:
        - 0: Creating a SharedMemoryArray and verifying its string representation
    """
    # Creates a SharedMemoryArray instance
    sma = SharedMemoryArray.create_array("test_repr", int_array)
    sma.connect()
    expected_repr = (
        f"SharedMemoryArray(name='test_repr', shape={int_array.shape}, datatype={int_array.dtype}, connected=True)"
    )
    assert repr(sma) == expected_repr

    # Cleans up
    sma.destroy()


@pytest.mark.parametrize(
    "array_fixture, buffer_name, index, expected, expected_type",
    [
        # Integer array tests
        ("int_array", "test_getitem_int_1", 0, 1, np.int32),
        ("int_array", "test_getitem_int_2", -1, 5, np.int32),
        ("int_array", "test_getitem_int_3", slice(0, 3), np.array([1, 2, 3]), np.ndarray),
        ("int_array", "test_getitem_int_4", slice(1, None), np.array([2, 3, 4, 5]), np.ndarray),
        ("int_array", "test_getitem_int_5", slice(-3, -1), np.array([3, 4]), np.ndarray),
        # Float array tests
        ("float_array", "test_getitem_float_1", 0, 1.1, np.float64),
        ("float_array", "test_getitem_float_2", -1, 5.5, np.float64),
        ("float_array", "test_getitem_float_3", slice(0, 3), np.array([1.1, 2.2, 3.3]), np.ndarray),
        # Boolean array tests
        ("bool_array", "test_getitem_bool_1", 0, True, np.bool_),
        ("bool_array", "test_getitem_bool_2", 1, False, np.bool_),
        ("bool_array", "test_getitem_bool_3", slice(0, 3), np.array([True, False, True]), np.ndarray),
        # String array tests
        ("string_array", "test_getitem_string_1", 0, "a", np.str_),
        ("string_array", "test_getitem_string_2", -1, "e", np.str_),
        ("string_array", "test_getitem_string_3", slice(0, 3), np.array(["a", "b", "c"]), np.ndarray),
    ],
)
def test_getitem(request, array_fixture, buffer_name, index, expected, expected_type):
    """Verifies the functionality of the SharedMemoryArray class __getitem__() method.

    Notes:
        Uses separate buffer names to prevent name collisions when tests are spread over multiple cores during
        pytest-xdist runtime.

    Tested configurations:
        - Reading data at various indices (positive, negative, single, slices)
        - Reading from different data types (int32, float64, bool, string)
        - Verifying correct return types for all scenarios
    """
    # Uses the test-specific fixture to get the prototype array and instantiate the SMA instance
    sample_array = request.getfixturevalue(array_fixture)
    sma = SharedMemoryArray.create_array(buffer_name, sample_array)
    sma.connect()

    # Reads data using a test-specific index
    result = sma[index]

    # Verifies that the value returned by the test matches expectation
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected

    # Verifies that the type returned by the test matches expectation
    assert isinstance(result, expected_type)

    # Cleans up
    sma.destroy()


@pytest.mark.parametrize(
    "array_fixture, buffer_name, index, data, expected",
    [
        # Integer array tests
        ("int_array", "test_setitem_int_1", 0, 10, 10),
        ("int_array", "test_setitem_int_2", -1, 50, 50),
        ("int_array", "test_setitem_int_3", slice(0, 3), [10, 20, 30], [10, 20, 30]),
        ("int_array", "test_setitem_int_4", slice(1, None), [20, 30, 40, 50], [20, 30, 40, 50]),
        ("int_array", "test_setitem_int_5", slice(-3, -1), [30, 40], [30, 40]),
        ("int_array", "test_setitem_int_6", 0, np.int32(15), 15),
        # Float array tests
        ("float_array", "test_setitem_float_1", 0, 10.5, 10.5),
        ("float_array", "test_setitem_float_2", -1, 50.5, 50.5),
        ("float_array", "test_setitem_float_3", slice(0, 3), [10.1, 20.2, 30.3], [10.1, 20.2, 30.3]),
        # Boolean array tests
        ("bool_array", "test_setitem_bool_1", 0, False, False),
        ("bool_array", "test_setitem_bool_2", -1, False, False),
        ("bool_array", "test_setitem_bool_3", slice(0, 3), [False, False, False], [False, False, False]),
        # String array tests
        ("string_array", "test_setitem_string_1", 0, "x", "x"),
        ("string_array", "test_setitem_string_2", -1, "z", "z"),
        ("string_array", "test_setitem_string_3", slice(0, 3), ["x", "y", "z"], ["x", "y", "z"]),
    ],
)
def test_setitem(request, array_fixture, buffer_name, index, data, expected):
    """Verifies the functionality of the SharedMemoryArray class __setitem__() method.

    Notes:
        Uses separate buffer names to prevent name collisions when tests are spread over multiple cores during
        pytest-xdist runtime.

    Tested configurations:
        - Writing data at various indices (positive, negative, single, slices)
        - Writing to different data types (int32, float64, bool, string)
        - Writing single values and lists/arrays of values
        - Verifying correct data writing for all scenarios
    """
    # Uses the test-specific fixture to get the prototype array and instantiate the SMA object
    sample_array = request.getfixturevalue(array_fixture)
    sma = SharedMemoryArray.create_array(buffer_name, sample_array)
    sma.connect()

    # Writes test data using the tested combination of index and input data
    sma[index] = data
    result = sma[index]  # Reads the (supposedly) modified data back

    # Verifies that the value(s) were written correctly
    if isinstance(expected, list):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected

    # Checks that the data type of the written data matches the original array's data type
    if isinstance(result, np.ndarray):
        assert result.dtype == sample_array.dtype
    else:
        assert isinstance(result, type(sample_array[0]))

    # Cleans up
    sma.destroy()


def test_array_context_manager(int_array):
    """Verifies the functionality of the SharedMemoryArray class array() context manager.

    Tested configurations:
        - 0: Accessing the array with locking
        - 1: Accessing the array without locking
        - 2: Modifying the array through the context manager
    """
    # Creates a SharedMemoryArray instance
    sma = SharedMemoryArray.create_array("test_array_cm", int_array)
    sma.connect()

    # Tests reading with lock
    with sma.array(with_lock=True) as arr:
        np.testing.assert_array_equal(arr, int_array)
        assert isinstance(arr, np.ndarray)

    # Tests reading without the lock
    with sma.array(with_lock=False) as arr:
        np.testing.assert_array_equal(arr, int_array)

    # Tests modification through context manager
    with sma.array(with_lock=True) as arr:
        arr[0] = 100

    # Verifies the modification persisted
    assert sma[0] == 100

    # Cleans up
    sma.destroy()


def test_disconnect_connect(int_array):
    """Verifies the functionality of the SharedMemoryArray class disconnect() and connect() methods.

    Tested configurations:
        - 0: Disconnecting from a connected SharedMemoryArray
        - 1: Reconnecting to a disconnected SharedMemoryArray
        - 2: Verifying data integrity after reconnection
    """
    # Creates two arrays to handle Windows garbage collection behavior
    smu = SharedMemoryArray.create_array("test_connect", int_array)
    sma = SharedMemoryArray.create_array("test_disconnect", int_array)

    # Connects to tested arrays
    sma.connect()
    smu.connect()

    # Tests disconnection
    sma.disconnect()
    assert not sma.is_connected

    # Tests reconnection
    smu.connect()
    assert smu.is_connected

    # Verifies data integrity after reconnection
    with smu.array(with_lock=False) as arr:
        np.testing.assert_array_equal(arr, int_array)

    # Cleans up
    smu.destroy()


def test_enable_buffer_destruction(int_array):
    """Verifies the functionality of the enable_buffer_destruction() method.

    Tested configurations:
        - 0: Enabling buffer destruction flag
        - 1: Verifying the flag is set correctly
    """
    # Creates a SharedMemoryArray instance
    sma = SharedMemoryArray.create_array("test_destruction", int_array)
    sma.connect()

    # Enables buffer destruction
    sma.enable_buffer_destruction()
    assert sma._destroy_buffer is True

    # Manually cleans up (to prevent automatic destruction during test)
    sma._destroy_buffer = False
    sma.destroy()


def test_create_array_errors():
    """Verifies error handling in the SharedMemoryArray class create_array() method.

    Tested configurations:
        - 0: Attempting to create an array with an invalid prototype (list instead of the numpy array)
        - 1: Attempting to create an array with a name that already exists
    """
    # Tests with an invalid prototype type
    message = (
        f"Invalid 'prototype' argument type encountered when creating SharedMemoryArray object 'test_error'. "
        f"Expected a flat (one-dimensional) NumPy array but instead encountered {type([1, 2, 3]).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        SharedMemoryArray.create_array(name="test_error", prototype=[1, 2, 3])

    # Tests with existing name
    # Maintains reference to prevent Windows garbage collection
    _existing = SharedMemoryArray.create_array(name="existing_array", prototype=np.array([1, 2, 3]))
    _existing.connect()
    message = (
        f"Unable to create the 'existing_array' SharedMemoryArray object, as an object with this name already "
        f"exists. If this method is called from a child process, use the connect() method instead "
        f"to connect to the existing buffer. To clean-up the buffer left over from a previous "
        f"runtime, run this method with the 'exists_ok' flag set to True."
    )
    with pytest.raises(FileExistsError, match=error_format(message)):
        SharedMemoryArray.create_array(name="existing_array", prototype=np.array([4, 5, 6]))


def test_getitem_errors(int_array):
    """Verifies error handling in the SharedMemoryArray class __getitem__() method.

    Tested configurations:
        - 0: Attempting to read from a disconnected array
    """
    # Creates the array without connecting
    sma = SharedMemoryArray.create_array("test_getitem_error", int_array)

    # Tests reading from the disconnected array
    message = (
        f"Unable to access the data stored in the test_getitem_error SharedMemoryArray instance, as the instance is "
        f"not connected to the shared memory buffer. Call the connect() method prior to accessing the array's "
        f"data."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        _ = sma[0]


def test_setitem_errors(int_array):
    """Verifies error handling in the SharedMemoryArray class __setitem__() method.

    Tested configurations:
        - 0: Attempting to write to a disconnected array
    """
    # Creates the array without connecting
    sma = SharedMemoryArray.create_array("test_setitem_error", int_array)

    # Tests writing to the disconnected array
    message = (
        f"Unable to modify the data stored in the test_setitem_error SharedMemoryArray instance, as the instance is "
        f"not connected to the shared memory buffer. Call the connect() method prior to modifying the array's "
        f"data."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        sma[0] = 10


def test_array_context_manager_errors(int_array):
    """Verifies error handling in the SharedMemoryArray class array() context manager.

    Tested configurations:
        - 0: Attempting to use array() on a disconnected instance
    """
    # Creates the array without connecting
    sma = SharedMemoryArray.create_array("test_array_error", int_array)

    # Tests using array() on disconnected instance
    message = (
        f"Unable to access the data stored in the test_array_error SharedMemoryArray instance, as the it is not "
        f"connected to the shared memory buffer. Call the connect() method prior to calling the array() method."
    )
    with pytest.raises(ConnectionError, match=error_format(message)):
        with sma.array() as arr:
            pass


def read_write_worker(sma: SharedMemoryArray):
    """Worker function for cross-process read/write testing.

    This worker connects to a shared array, writes a test value, verifies the write operation, and then disconnects.
    It is used to verify that SharedMemoryArray can be accessed from multiple processes as intended.

    Args:
        sma: The SharedMemoryArray instance to test.
    """
    # Connects to the input array
    sma.connect()

    # Writes and verifies that the test payload has been written
    sma[2] = 42
    assert sma[2] == 42

    # Disconnects from the array and terminates the process
    sma.disconnect()


def concurrent_worker(sma: SharedMemoryArray, index: int):
    """Worker function for concurrent access testing.

    This worker repeatedly reads, increments, and writes back a value at a specific
    index to test that locking prevents race conditions during concurrent access.

    Args:
        sma: The SharedMemoryArray instance to test.
        index: The array index to repeatedly increment.
    """
    # Connects to the array
    sma.connect()

    # Performs repeated increment operations
    for _ in range(100):
        # Reads data from the input index
        value = sma[index]
        # Increments the value by one and writes it back to the array
        sma[index] = value + 1

    # Disconnects and terminates the process
    sma.disconnect()


@pytest.mark.xdist_group("cross_process")
def test_cross_process_read_write():
    """Verifies the ability of the SharedMemoryArray class to share data across processes.

    Tested configurations:
        - 0: Writing data from a child process
        - 1: Reading the written data from the parent process
    """
    # Instantiates the SMA object
    sma = SharedMemoryArray.create_array("test_cross_process", np.array([1, 2, 3, 4, 5], dtype=np.int32))

    # Writes (and reads) to the SMA from a different process
    p = Process(target=read_write_worker, args=(sma,))
    p.start()
    p.join()

    # Finish setting up the array in the local process
    sma.connect()
    sma.enable_buffer_destruction()

    # Verifies that the data written by the other process is accessible from the main process
    assert sma[2] == 42

    # Cleans up
    sma.destroy()


@pytest.mark.xdist_group("cross_process")
def test_cross_process_concurrent_access():
    """Verifies the ability of the SharedMemoryArray class to handle concurrent access from multiple processes.

    Tested configurations:
        - 0: Multiple processes (5) incrementing values in the shared array concurrently
        - 1: Verifying the final value of each array element after concurrent incrementing
    """
    # Instantiates the SMA object
    sma = SharedMemoryArray.create_array("test_concurrent", np.zeros(5, dtype=np.int32))

    # Generates multiple processes and uses each to repeatedly increment different indices
    processes = [Process(target=concurrent_worker, args=(sma, i)) for i in range(5)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # Finish setting up the array in the local process
    sma.connect()
    sma.enable_buffer_destruction()

    # Verifies all indices were incremented to the expected value
    with sma.array(with_lock=False) as arr:
        assert np.all(arr == 100)

    # Cleans up
    sma.destroy()
