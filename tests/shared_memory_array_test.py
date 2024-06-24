import sys
import pytest
import numpy as np
from multiprocessing import Process
from ataraxis_data_structures.shared_memory import SharedMemoryArray


def write_to_shared_array(shared_array: SharedMemoryArray, data_to_write: np.ndarray):
    """A simple function used as a multiprocessing Process target to test SharedMemoryArray class.

    This function connects to the shared memory array and writes the input data to the slice(0, 2) of the array.
    """
    shared_array.connect()
    shared_array.write_data(slice(0, 2), data_to_write)


def read_from_shared_array(shared_array: SharedMemoryArray, expected_data: np.ndarray):
    """A simple function used as a multiprocessing Process target to test SharedMemoryArray class.

    This function connects to the shared memory array and reads the array data from slice(0, 2). After reading the data,
    it ensures that the data matches the expected_data
    """
    shared_array.connect()
    data = shared_array.read_data(slice(0, 2))
    assert np.array_equal(data, expected_data)


def test_shared_memory_array():
    """Tests SharedMemoryArray initialization and runtime behavior."""
    # Creates a prototype array and uses it to instantiate a SharedMemoryArray class.
    prototype = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shared_memory_array = SharedMemoryArray.create_array("test_array", prototype)

    # Checks if the shared memory array is connected and has the correct properties post-initialization.
    assert shared_memory_array._is_connected
    assert shared_memory_array.name == "test_array"
    assert shared_memory_array.shape == prototype.shape
    assert shared_memory_array.datatype == prototype.dtype

    # Tests reading data from the array.
    data = shared_memory_array.read_data(slice(0, 2))
    assert np.array_equal(data, prototype)

    # Tests writing data to the array.
    new_data = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.int32)
    shared_memory_array.write_data(slice(0, 2), new_data)
    updated_data = shared_memory_array.read_data(slice(0, 2))
    assert np.array_equal(updated_data, new_data)

    # Tests reading data from a disconnected array, which is expected to fail.
    disconnected_array = SharedMemoryArray("test_array", prototype.shape, prototype.dtype, None)
    with pytest.raises(RuntimeError, match="Cannot read data as the class is not connected to a shared memory array."):
        disconnected_array.read_data(0)

    # Test writing data to a disconnected array, which is expected to fail.
    with pytest.raises(RuntimeError, match="Cannot write data as the class is not connected to a shared memory array."):
        disconnected_array.write_data(0, np.array([1]))

    # Test reading data with an invalid index, which is expected to fail.
    with pytest.raises(ValueError,
                       match="Invalid index or slice when attempting to read the data from shared memory array."):
        shared_memory_array.read_data(10)

    # Test writing data with an invalid datatype, which is expected to fail.
    invalid_datatype = np.array([1.0, 2.0], dtype=np.float32)
    with pytest.raises(ValueError,
                       match=f"Input data must have the same datatype as the shared memory array: {prototype.dtype}."):
        shared_memory_array.write_data(0, invalid_datatype)

    # Test writing data that cannot fit in the shared array, which is expected to fail.
    invalid_shape = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    with pytest.raises(ValueError,
                       match="Input data cannot fit inside the shared memory array at the specified index or slice."):
        shared_memory_array.write_data(slice(0, 2), invalid_shape)

    # Test accessing properties of a disconnected array, which is expected to fail.
    with pytest.raises(RuntimeError,
                       match="Cannot retrieve array datatype as the class is not connected to a shared memory array."):
        # noinspection PyStatementEffect
        disconnected_array.datatype
    with pytest.raises(RuntimeError,
                       match="Cannot retrieve shared memory buffer name as the class is not connected to a shared memory array."):
        # noinspection PyStatementEffect
        disconnected_array.name
    with pytest.raises(RuntimeError,
                       match="Cannot retrieve shared memory array shape as the class is not connected to a shared memory array."):
        # noinspection PyStatementEffect
        disconnected_array.shape

    # Tests accessing the shared array from a different process.
    process = Process(target=read_from_shared_array, args=(shared_memory_array, new_data))
    process.start()
    process.join()

    # Test sharing data between processes, using a new array instance.
    prototype2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shared_array_2 = SharedMemoryArray.create_array("test_array_2", prototype2)

    # Create a process to write data to the shared array
    data_to_write = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int32)
    write_process = Process(target=write_to_shared_array,
                            args=(shared_array_2, data_to_write))
    write_process.start()
    write_process.join()

    # Create a process to read data from the shared array
    read_process = Process(target=read_from_shared_array,
                           args=(shared_array_2, data_to_write))
    read_process.start()
    read_process.join()

    # Verify that the data in the shared array matches the data written by the write process
    data = shared_array_2.read_data(slice(0, 2))
    assert np.array_equal(data, data_to_write)


def test_shared_memory_array_disconnect():
    """Tests the disconnect() method fo the SharedMemoryArray class."""
    # Creates a prototype array and uses it to initialize a SharedMemoryArray
    prototype = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    shared_array = SharedMemoryArray.create_array("test_array_disconnect", prototype)

    # Checks if the shared array is connected
    assert shared_array._is_connected

    # Disconnects from the shared array
    shared_array.disconnect()

    # Checks if the shared array is disconnected
    assert not shared_array._is_connected
