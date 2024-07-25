"""Contains tests for SharedMemoryArray class and related methods, stored in the shared_memory package."""

import re
import textwrap
from multiprocessing import Process

import numpy as np
import pytest

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


def error_format(message: str) -> str:
    """Formats the input message to match the default Console format and escapes it using re, so that it can be used to
    verify raised exceptions.

    This method is used to set up pytest 'match' clauses to verify raised exceptions.

    Args:
        message: The message to format and escape, according to standard Ataraxis testing parameters.

    Returns:
        Formatted and escape message that can be used as the 'match' argument of pytest.raises() method.
    """
    return re.escape(textwrap.fill(message, width=120, break_long_words=False, break_on_hyphens=False))


def test_create_array(int_array):
    """Verifies the functionality of the SharedMemoryArray class create_array() method.

    Tested configurations:
        - 0: Creating a SharedMemoryArray with a valid numpy array
        - 1: Verifying the name, shape, datatype, and connection status of the created array
        - 2: Verifying the data integrity of the created array
    """
    sma = SharedMemoryArray.create_array("test_create_array", int_array)
    assert sma.name == "test_create_array"
    assert sma.shape == int_array.shape
    assert sma.datatype == int_array.dtype
    assert sma.is_connected
    np.testing.assert_array_equal(sma.read_data((0, 5)), int_array)

    # Destroys the array, freeing up the buffer name to be used by other SMA instances
    sma.disconnect()
    sma.destroy()

    # Verifies that the buffer has been freed up
    sma = SharedMemoryArray.create_array("test_create_array", int_array)

    # Cleans up after the runtime
    sma.disconnect()
    sma.destroy()


def test_repr(int_array):
    """Verifies the functionality of the SharedMemoryArray class __repr__() method.

    Tested configurations:
        - 0: Creating a SharedMemoryArray and verifying its string representation
    """
    sma = SharedMemoryArray.create_array("test_repr", int_array)
    expected_repr = (
        f"SharedMemoryArray(name='test_repr', shape={int_array.shape}, " f"datatype={int_array.dtype}, connected=True)"
    )
    assert repr(sma) == expected_repr


@pytest.mark.parametrize(
    "array_fixture, buffer_name, index, convert, expected, expected_type",
    [
        # Integer array tests
        ("int_array", "test_read_data_int_1", 0, True, 1, int),
        ("int_array", "test_read_data_int_2", -1, True, 5, int),
        ("int_array", "test_read_data_int_3", (0, 3), True, [1, 2, 3], list),
        ("int_array", "test_read_data_int_4", (1,), True, [2, 3, 4, 5], list),
        ("int_array", "test_read_data_int_5", (-3, -1), True, [3, 4], list),
        ("int_array", "test_read_data_int_6", 0, False, 1, np.int32),
        ("int_array", "test_read_data_int_7", -1, False, 5, np.int32),
        ("int_array", "test_read_data_int_8", (0, 3), False, np.array([1, 2, 3]), np.ndarray),
        ("int_array", "test_read_data_int_9", (1,), False, np.array([2, 3, 4, 5]), np.ndarray),
        ("int_array", "test_read_data_int_10", (-3, -1), False, np.array([3, 4]), np.ndarray),
        # Float array tests
        ("float_array", "test_read_data_float_1", 0, True, 1.1, float),
        ("float_array", "test_read_data_float_2", -1, True, 5.5, float),
        ("float_array", "test_read_data_float_3", (0, 3), True, [1.1, 2.2, 3.3], list),
        ("float_array", "test_read_data_float_4", (1,), True, [2.2, 3.3, 4.4, 5.5], list),
        ("float_array", "test_read_data_float_5", (-3, -1), True, [3.3, 4.4], list),
        ("float_array", "test_read_data_float_6", 0, False, 1.1, np.float64),
        ("float_array", "test_read_data_float_7", -1, False, 5.5, np.float64),
        ("float_array", "test_read_data_float_8", (0, 3), False, np.array([1.1, 2.2, 3.3]), np.ndarray),
        ("float_array", "test_read_data_float_9", (1,), False, np.array([2.2, 3.3, 4.4, 5.5]), np.ndarray),
        ("float_array", "test_read_data_float_10", (-3, -1), False, np.array([3.3, 4.4]), np.ndarray),
        # Boolean array tests
        ("bool_array", "test_read_data_bool_1", 0, True, True, bool),
        ("bool_array", "test_read_data_bool_2", -1, True, True, bool),
        ("bool_array", "test_read_data_bool_3", (0, 3), True, [True, False, True], list),
        ("bool_array", "test_read_data_bool_4", (1,), True, [False, True, False, True], list),
        ("bool_array", "test_read_data_bool_5", (-3, -1), True, [True, False], list),
        ("bool_array", "test_read_data_bool_6", 0, False, True, np.bool_),
        ("bool_array", "test_read_data_bool_7", -1, False, True, np.bool_),
        ("bool_array", "test_read_data_bool_8", (0, 3), False, np.array([True, False, True]), np.ndarray),
        ("bool_array", "test_read_data_bool_9", (1,), False, np.array([False, True, False, True]), np.ndarray),
        ("bool_array", "test_read_data_bool_10", (-3, -1), False, np.array([True, False]), np.ndarray),
        # String array tests
        ("string_array", "test_read_data_string_1", 0, True, "a", str),
        ("string_array", "test_read_data_string_2", -1, True, "e", str),
        ("string_array", "test_read_data_string_3", (0, 3), True, ["a", "b", "c"], list),
        ("string_array", "test_read_data_string_4", (1,), True, ["b", "c", "d", "e"], list),
        ("string_array", "test_read_data_string_5", (-3, -1), True, ["c", "d"], list),
        ("string_array", "test_read_data_string_6", 0, False, "a", np.str_),
        ("string_array", "test_read_data_string_7", -1, False, "e", np.str_),
        ("string_array", "test_read_data_string_8", (0, 3), False, np.array(["a", "b", "c"]), np.ndarray),
        ("string_array", "test_read_data_string_9", (1,), False, np.array(["b", "c", "d", "e"]), np.ndarray),
        ("string_array", "test_read_data_string_10", (-3, -1), False, np.array(["c", "d"]), np.ndarray),
    ],
)
def test_read_data(request, array_fixture, buffer_name, index, convert, expected, expected_type):
    """Verifies the functionality of the SharedMemoryArray class read_data() method.

    Notes:
        Uses separate buffer names to prevent name collisions when tests are spread over multiple cores during
        pytest-xdist runtime.

    Tested configurations:
        - Reading data at various indices (positive, negative, single, slices)
        - Reading from different data types (int32, float64, bool, string)
        - Testing both converted and non-converted outputs
        - Verifying correct return types for all scenarios
    """

    # Uses the test-specific fixture to get the prototype array and instantiate the SMA instance
    sample_array = request.getfixturevalue(array_fixture)
    sma = SharedMemoryArray.create_array(buffer_name, sample_array)

    # Reads data using test-specific parameters for the index and conversion flag
    result = sma.read_data(index=index, convert_output=convert, with_lock=False)

    # Verifies that the value returned by the test matches expectation
    if isinstance(expected, list):
        assert result == expected
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(result, expected)
    else:
        assert result == expected

    # Verifies that the type returned by the test matches expectation
    assert isinstance(result, expected_type)

    # Additional check for numpy scalar types
    if isinstance(result, (np.int32, np.float64, np.bool_, np.str_)):
        assert result.dtype == sample_array.dtype


@pytest.mark.parametrize(
    "array_fixture, buffer_name, index, data, expected",
    [
        # Integer array tests
        ("int_array", "test_write_data_int_1", 0, 10, 10),
        ("int_array", "test_write_data_int_2", -1, 50, 50),
        ("int_array", "test_write_data_int_3", (0, 3), [10, 20, 30], [10, 20, 30]),
        ("int_array", "test_write_data_int_4", (1,), [20, 30, 40, 50], [20, 30, 40, 50]),
        ("int_array", "test_write_data_int_5", (-3, -1), [30, 40], [30, 40]),
        ("int_array", "test_write_data_int_6", 0, np.int32(15), 15),
        ("int_array", "test_write_data_int_7", (0, 5), np.array([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5]),
        # Float array tests
        ("float_array", "test_write_data_float_1", 0, 10.5, 10.5),
        ("float_array", "test_write_data_float_2", -1, 50.5, 50.5),
        ("float_array", "test_write_data_float_3", (0, 3), [10.1, 20.2, 30.3], [10.1, 20.2, 30.3]),
        ("float_array", "test_write_data_float_4", (1,), [20.2, 30.3, 40.4, 50.5], [20.2, 30.3, 40.4, 50.5]),
        ("float_array", "test_write_data_float_5", (-3, -1), [30.3, 40.4], [30.3, 40.4]),
        ("float_array", "test_write_data_float_6", 0, np.float64(15.5), 15.5),
        (
            "float_array",
            "test_write_data_float_7",
            (0, 5),
            np.array([1.1, 2.2, 3.3, 4.4, 5.5]),
            [1.1, 2.2, 3.3, 4.4, 5.5],
        ),
        # Boolean array tests
        ("bool_array", "test_write_data_bool_1", 0, False, False),
        ("bool_array", "test_write_data_bool_2", -1, True, True),
        ("bool_array", "test_write_data_bool_3", (0, 3), [True, False, True], [True, False, True]),
        ("bool_array", "test_write_data_bool_4", (1,), [False, True, False, True], [False, True, False, True]),
        ("bool_array", "test_write_data_bool_5", (-3, -1), [False, True], [False, True]),
        ("bool_array", "test_write_data_bool_6", 0, np.bool_(True), True),
        (
            "bool_array",
            "test_write_data_bool_7",
            (0, 5),
            np.array([False, True, False, True, False]),
            [False, True, False, True, False],
        ),
        # String array tests
        ("string_array", "test_write_data_string_1", 0, "x", "x"),
        ("string_array", "test_write_data_string_2", -1, "z", "z"),
        ("string_array", "test_write_data_string_3", (0, 3), ["x", "y", "z"], ["x", "y", "z"]),
        ("string_array", "test_write_data_string_4", (1,), ["w", "x", "y", "z"], ["w", "x", "y", "z"]),
        ("string_array", "test_write_data_string_5", (-3, -1), ["y", "z"], ["y", "z"]),
        ("string_array", "test_write_data_string_6", 0, np.str_("m"), "m"),
        (
            "string_array",
            "test_write_data_string_7",
            (0, 5),
            np.array(["v", "w", "x", "y", "z"]),
            ["v", "w", "x", "y", "z"],
        ),
    ],
)
def test_write_data(request, array_fixture, buffer_name, index, data, expected):
    """Verifies the functionality of the SharedMemoryArray class write_data() method.

    Notes:
        Uses separate buffer names to prevent name collisions when tests are spread over multiple cores during
        pytest-xdist runtime.

    Tested configurations:
        - Writing data at various indices (positive, negative, single, slices)
        - Writing to different data types (int32, float64, bool, string)
        - Writing single values and lists/arrays of values
        - Writing using Python native types and numpy types
        - Verifying correct data writing for all scenarios
    """
    # Uses the test-specific fixture to get the prototype array and instantiate the SMA object.
    sample_array = request.getfixturevalue(array_fixture)
    sma = SharedMemoryArray.create_array(buffer_name, sample_array)

    # Writes test data using the tested combination of index and input data
    sma.write_data(index, data)
    result = sma.read_data(index)  # Reads the (supposedly) modified data back

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


def test_disconnect_connect(int_array):
    """Verifies the functionality of the SharedMemoryArray class disconnect() and connect() methods.

    Tested configurations:
        - 0: Disconnecting from a connected SharedMemoryArray
        - 1: Reconnecting to a disconnected SharedMemoryArray
        - 2: Verifying data integrity after reconnection
    """
    # Need to use 2 arrays on Windows. Once sma is disconnected, Windows garbage-collects the buffer. SMU does not have
    # this issue and, therefore, it is used to verify connection method.
    smu = SharedMemoryArray.create_array("test_connect", int_array)
    sma = SharedMemoryArray.create_array("test_disconnect", int_array)
    sma.disconnect()
    assert not sma.is_connected
    smu.connect()
    assert smu.is_connected
    np.testing.assert_array_equal(smu.read_data((0, 5)), int_array)


def test_create_array_error():
    """Verifies error handling in the SharedMemoryArray class create_array() method.

    Tested configurations:
        - 0: Attempting to create an array with an invalid prototype (list instead of a numpy array)
        - 1: Attempting to create an array with a multidimensional numpy array
        - 2: Attempting to create an array with a name that already exists
    """
    # Tests with an invalid prototype type
    message = (
        f"Invalid 'prototype' argument type encountered when creating SharedMemoryArray object 'test_error'. "
        f"Expected a flat (one-dimensional) numpy ndarray but instead encountered {type([1, 2, 3]).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        SharedMemoryArray.create_array(name="test_error", prototype=[1, 2, 3])

    # Tests with a multidimensional array
    multi_dim_array = np.array([[1, 2], [3, 4]])
    message = (
        f"Invalid 'prototype' array shape encountered when creating SharedMemoryArray object 'test_error_2'. "
        f"Expected a flat (one-dimensional) numpy ndarray but instead encountered prototype with shape "
        f"{multi_dim_array.shape} and dimensions number {multi_dim_array.ndim}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        SharedMemoryArray.create_array(name="test_error_2", prototype=multi_dim_array)

    # Tests with existing name
    # The array has to be saved to an object, otherwise it is not properly maintained on Windows that automatically
    # cleans up unreferenced objects.
    _x = SharedMemoryArray.create_array(name="existing_array", prototype=np.array([1, 2, 3]))
    message = (
        f"Unable to create SharedMemoryArray object using name 'existing_array', as object with this name already "
        f"exists. This is likely due to calling create_array() method from a child process. "
        f"Use connect() method to connect to the SharedMemoryArray from a child process."
    )
    with pytest.raises(FileExistsError, match=error_format(message)):
        SharedMemoryArray.create_array(name="existing_array", prototype=np.array([4, 5, 6]))


def test_read_data_error(int_array):
    """Verifies error handling in the SharedMemoryArray class read_data() method.

    Tested configurations:
        - 0: Attempting to read with an index greater than array length
        - 1: Attempting to read with a negative index that translates to a position before the array start
        - 2: Attempting to read with a stop index greater than array length
        - 3: Attempting to read with a start index greater than stop index
        - 4: Attempting to read from a disconnected array
        - 5: Attempting to read with an invalid index type
    """
    sma = SharedMemoryArray.create_array("test_read_error", int_array)

    # Tests index out of bounds (positive)
    message = (
        f"Unable to retrieve the data from test_read_error SharedMemoryArray class instance using start index "
        f"5. The index is outside the valid start index range (0:4)."
    )
    with pytest.raises(IndexError, match=error_format(message)):
        sma.read_data(5)

    # Tests index out of bounds (negative)
    message = (
        f"Unable to retrieve the data from test_read_error SharedMemoryArray class instance using start index "
        f"-6. The index is outside the valid start index range (0:4)."
    )
    with pytest.raises(IndexError, match=error_format(message)):
        sma.read_data(-6)

    # Tests stop index out of bounds
    message = (
        f"Unable to retrieve the data from test_read_error SharedMemoryArray class instance using stop index "
        f"6. The index is outside the valid stop index range (1:5)."
    )
    with pytest.raises(IndexError, match=error_format(message)):
        sma.read_data((0, 6))

    # Tests start index greater than stop index
    message = (
        f"Invalid pair of slice indices encountered when manipulating data of the test_read_error "
        f"SharedMemoryArray class instance. The start index (3) is greater than the end index (2), which is not "
        f"allowed."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        sma.read_data((3, 2))

    # Tests reading from a disconnected array
    sma.disconnect()
    message = (
        f"Unable to access the data stored in the test_read_error SharedMemoryArray instance, as the class is not "
        f"connected to the shared memory buffer. Use connect() method prior to calling other class methods."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        sma.read_data(0)

    # Tests invalid index type input
    # Recreates the array, as it is garbage-collected on Windows after the only instance of the array isz
    # disconnected.
    sma = SharedMemoryArray.create_array("test_read_error2", int_array)
    message = (
        f"Unable to read data from test_read_error2 SharedMemoryArray class instance. Expected an integer index or "
        f"a tuple of two integers, but encountered 'invalid' of type {type('invalid').__name__} instead."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        sma.read_data("invalid")

    # Tests invalid slice tuple input format
    message = (
        f"Unable to convert the index tuple into slice arguments for test_read_error2 SharedMemoryArray "
        f"instance. Expected a tuple with 1 or 2 elements (start and stop), but instead encountered "
        f"a tuple with {4} elements."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        sma.read_data((1, 2, 3, 4))


def test_write_data_error(int_array):
    """Verifies error handling in the SharedMemoryArray class write_data() method.

    Tested configurations:
        - 0: Attempting to write with an index greater than array length
        - 1: Attempting to write with a negative index that translates to a position before the array start
        - 2: Attempting to write with a stop index greater than array length
        - 3: Attempting to write with a start index greater than stop index
        - 4: Attempting to write data of an invalid type
        - 5: Attempting to write to a disconnected array
        - 6: Attempting to write with an invalid index type
    """
    sma = SharedMemoryArray.create_array("test_write_error", int_array)

    # Tests index out of bounds (positive)
    message = (
        f"Unable to retrieve the data from test_write_error SharedMemoryArray class instance using start index "
        f"5. The index is outside the valid start index range (0:4)."
    )
    with pytest.raises(IndexError, match=error_format(message)):
        sma.write_data(5, 10)

    # Tests index out of bounds (negative)
    message = (
        f"Unable to retrieve the data from test_write_error SharedMemoryArray class instance using start index "
        f"-6. The index is outside the valid start index range (0:4)."
    )
    with pytest.raises(IndexError, match=error_format(message)):
        sma.write_data(-6, 10)

    # Tests stop index out of bounds
    message = (
        f"Unable to retrieve the data from test_write_error SharedMemoryArray class instance using stop index "
        f"6. The index is outside the valid stop index range (1:5)."
    )
    with pytest.raises(IndexError, match=error_format(message)):
        sma.write_data((0, 6), [1, 2, 3, 4, 5, 6])

    # Tests start index greater than stop index
    message = (
        f"Invalid pair of slice indices encountered when manipulating data of the test_write_error "
        f"SharedMemoryArray class instance. The start index (3) is greater than the end index (2), which is not "
        f"allowed."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        sma.write_data((3, 2), [1, 2])

    # Tests writing an invalid data type
    message = (
        f"Unable write data to test_write_error SharedMemoryArray class instance with index 0. Encountered "
        f"the following error when converting the data to the array datatype (int32) and writing it "
        f"to the array: invalid literal for int() with base 10: 'invalid_data'."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        sma.write_data(0, "invalid_data")

    # Tests writing to a disconnected array
    sma.disconnect()
    message = (
        f"Unable to access the data stored in the test_write_error SharedMemoryArray instance, as the class is not "
        f"connected to the shared memory buffer. Use connect() method prior to calling other class methods."
    )
    with pytest.raises(RuntimeError, match=error_format(message)):
        sma.write_data(0, 10)

    # Tests invalid an index type
    sma = SharedMemoryArray.create_array("test_write_error2", int_array)
    sma.connect()
    message = (
        f"Unable to write data to test_write_error2 SharedMemoryArray class instance. Expected an integer index or "
        f"a tuple of two integers, but encountered 'invalid' of type str instead."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        sma.write_data(index="invalid", data=10)

    # Tests invalid slice tuple input format
    message = (
        f"Unable to convert the index tuple into slice arguments for test_write_error2 SharedMemoryArray "
        f"instance. Expected a tuple with 1 or 2 elements (start and stop), but instead encountered "
        f"a tuple with {4} elements."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        sma.read_data((1, 2, 3, 4))


def read_write_worker(sma: SharedMemoryArray):
    """This worker is used to verify that SharedMemoryArray class can be used from multiple processes as intended.

    To do so, it has to be defined outside the main test scope (to deal with how multiprocessing distributes the data
    across workers).

    Specifically, it is used as part of the cross_process_read_write() test task. THe test is written to NOT be
    concurrent, so it specifically tests that the data written from one process can be accessed from the other,
    regardless of the status of that other process. There is no potential that this task will clash with the main
    process.
    """
    # Connects to the input array
    sma.connect()
    # Writes and verifies that the test payload has been written
    sma.write_data(index=2, data=42, with_lock=False)
    assert sma.read_data(index=2, with_lock=False) == 42
    # Disconnects from the array and terminates the process
    sma.disconnect()


def concurrent_worker(sma: SharedMemoryArray, index: int):
    """This worker is used to verify that SharedMemoryArray is process-safe (when used with default locking flags).

    To do so, it has to be defined outside the main test scope (to deal with how multiprocessing distributes the data
    across workers).

    Specifically, it is used as part of the cross_process_concurrent_access() test task. This task evaluates whether
    locking is effective at preventing multiple processes from colliding when reading and writing to the same SMA
    instance.
    """
    # Connects to the array
    sma.connect()
    for _ in range(100):
        # Reads data from the input index
        value = sma.read_data(index)
        # Increments the value by one and writes it back to the array
        sma.write_data(index, value + 1)
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

    # Verifies that the data written by the other process is accessible from the main process
    assert sma.read_data(2) == 42


@pytest.mark.xdist_group("cross_process")
def test_cross_process_concurrent_access():
    """Verifies the ability of the SharedMemoryArray class to handle concurrent access from multiple processes.

    Tested configurations:
        - 0: Multiple processes (5) incrementing values in the shared array concurrently
        - 1: Verifying the final value of each array element after concurrent incrementing
    """
    # Instantiates the SMA object
    sma = SharedMemoryArray.create_array("test_concurrent", np.zeros(5, dtype=np.int32))

    # Generates multiple processes and uses each to repeatedly write and read data from different indices of the same
    # array.
    processes = [Process(target=concurrent_worker, args=(sma, i)) for i in range(5)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    assert np.all(sma.read_data((0, 5)) == 100)
