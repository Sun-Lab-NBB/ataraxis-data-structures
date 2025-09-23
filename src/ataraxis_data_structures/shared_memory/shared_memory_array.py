"""This module provides the SharedMemoryArray class that allows moving data between multiple Python processes through
a shared one-dimensional NumPy array memory buffer.
"""

from copy import copy
from typing import Any
from contextlib import contextmanager
from collections.abc import Iterable, Generator
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console


class SharedMemoryArray:
    """Wraps a one-dimensional NumPy array object and exposes methods for accessing the array's data from multiple
    different Python processes.

    During initialization, this class creates a persistent memory buffer to which it connects from different Python
    processes. The data inside the buffer is accessed via a one-dimensional NumPy array with optional locking to prevent
    race conditions.

    Notes:
        This class should only be instantiated inside the main runtime thread via the create_array() method. Do not
        attempt to instantiate the class manually. All child processes working with this class should use the connect()
        method to connect to the shared array wrapped by the class before calling any other method.

        Shared memory buffers are garbage-collected differently depending on the host Operating System. On Windows,
        garbage collection is handed off to the OS and cannot be enforced manually. On Unix (macOS and Linux), the
        buffer can be garbage-collected by calling the destroy() method.

    Args:
        name: The unique name to use for the shared memory buffer.
        shape: The shape of the NumPy array used to access the data in the shared memory buffer.
        datatype: The datatype of the NumPy array used to access the data in the shared memory buffer.
        buffer: The shared memory buffer that stores the data managed by the instance.

    Attributes:
        _name: Stores the name of the shared memory buffer.
        _shape: Stores the shape of the NumPy array used to access the buffered data.
        _datatype: Stores the datatype of the NumPy array used to access the buffered data.
        _buffer: Stores the Shared Memory buffer object.
        _lock: Stores the Lock object used to prevent multiple processes from working with the shared data at the same
            time.
        _array: Stores the NumPy array used to interface with the data stored in the shared memory buffer.
        _connected: Tracks whether the instance is connected to the shared memory buffer.
        _termination_guard: Tracks whether the shared memory buffer should be destroyed if this instance is
            garbage-collected.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        datatype: np.dtype[Any],
        buffer: SharedMemory,
    ) -> None:
        # Initialization method only saves input data into attributes. The method that actually sets up the class is
        # the create_array() class method.
        self._name: str = name
        self._shape: tuple[int, ...] = shape
        self._datatype: np.dtype[Any] = datatype
        self._buffer: SharedMemory = buffer
        self._lock = Lock()
        self._array: NDArray[Any] = np.zeros(shape=shape, dtype=datatype)
        self._connected: bool = False
        self._termination_guard: bool = False

    def __repr__(self) -> str:
        """Returns the string representation of the SharedMemoryArray instance."""
        return (
            f"SharedMemoryArray(name='{self._name}', shape={self._shape}, datatype={self._datatype}, "
            f"connected={self.is_connected})"
        )

    def __del__(self) -> None:
        """Ensures that the shared memory buffer is released when the instance is garbage-collected."""

        # If the termination guard is set, attempts to disconnect AND destroy the shared memory buffer as part of the
        # method's shutdown sequence.
        if self._termination_guard:
            self.destroy()
        else:
            # Otherwise, only disconnects the shared memory buffer.
            self.disconnect()

    @classmethod
    def create_array(
        cls,
        name: str,
        prototype: NDArray[Any],
        exists_ok: bool = False,
    ) -> "SharedMemoryArray":
        """Creates a SharedMemoryArray instance using the input prototype NumPy array.

        This method uses the input prototype to generate the shared memory buffer to store the prototype's data and
        fills the buffer with the data in the prototype array. All further interactions with the returned
        SharedMemoryArray instance interface with this shared memory buffer.

        Notes:
            This method should only be called when the array is first created in the main process (scope). All
            child processes should use the connect() method to connect to the existing array.

        Args:
            name: The name to give to the created SharedMemory object. Note, this name has to be unique across all
                processes using the array.
            prototype: The numpy ndarray instance to serve as the prototype for the created SharedMemoryArray.
                Currently, this class only works with flat (one-dimensional) numpy arrays. If you want to use it for
                multidimensional arrays, consider using np.ravel() or np.ndarray.flatten() methods to flatten your
                array.
            exists_ok: Determines how the method handles the case where the Sharedmemory buffer with the same name
                already exists. If the flag is False, the class will raise an exception and abort SharedMemoryArray
                creation. If the flag is True, the class will destroy the old buffer and recreate the new buffer using
                the vacated name.

        Returns:
            The configured SharedMemoryArray class instance. This instance should be passed to each of the processes
            that needs to access the wrapped array data.

        Raises:
            TypeError: If the input prototype is not a numpy ndarray.
            FileExistsError: If a shared memory object with the same name as the input 'name' argument value already
                exists.
        """
        # Ensures prototype is a numpy ndarray
        if not isinstance(prototype, NDArray[Any]):
            message = (
                f"Invalid 'prototype' argument type encountered when creating SharedMemoryArray object '{name}'. "
                f"Expected a flat (one-dimensional) NumPy array but instead encountered {type(prototype).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Also ensures that the prototype is flat
        if prototype.ndim != 1:
            message = (
                f"Invalid 'prototype' array shape encountered when creating SharedMemoryArray object '{name}'. "
                f"Expected a flat (one-dimensional) NumPy array but instead encountered prototype with "
                f"{prototype.ndim} dimensions."
            )
            console.error(message=message, error=ValueError)

        # Creates the shared memory buffer. This process raises FileExistsError if an object with this name
        # already exists.
        try:
            buffer: SharedMemory = SharedMemory(name=name, create=True, size=prototype.nbytes)
        except FileExistsError:
            # If the buffer already exists, but the method is configured to recreate the buffer, destroys the old buffer
            if exists_ok:
                # Destroys the existing shared memory buffer
                SharedMemory(name=name, create=False).unlink()

                # Recreates the shared memory buffer using the freed buffer name
                buffer = SharedMemory(name=name, create=True, size=prototype.nbytes)

            # Otherwise, raises an exception
            else:
                message = (
                    f"Unable to create SharedMemoryArray object using the name '{name}', as the object with this name "
                    f"already exists. If this method is called from a child process, use the connect() method instead "
                    f"to connect to the existing buffer. To clean-up the buffer left over from a previous "
                    f"runtime, run this method with the 'exists_ok' flag set to True."
                )
                console.error(message=message, error=FileExistsError)

        # Instantiates a numpy array using the shared memory buffer and copies the prototype array data into the shared
        # array instance.
        # noinspection PyUnboundLocalVariable
        shared_array: NDArray[Any] = np.ndarray(shape=prototype.shape, dtype=prototype.dtype, buffer=buffer.buf)
        shared_array[:] = prototype[:]

        # Packages the data necessary to connect to the shared array into the class object.
        shared_memory_array = cls(
            name=name,
            shape=shared_array.shape,
            datatype=shared_array.dtype,
            buffer=buffer,
        )

        # Connects the internal _array of the class object to the shared memory buffer.
        shared_memory_array.connect()

        # Returns the instantiated and connected class object to caller.
        return shared_memory_array

    def connect(self) -> None:
        """Connects to the shared memory buffer, allowing the instance to access and manipulate the shared data.

        This method should be called once for each Python process that uses this instance, before calling any other
        methods.
        """
        if not self._connected:
            # Connects to the shared memory buffer
            self._buffer = SharedMemory(name=self._name, create=False)
            # Re-initializes the internal _array with the data from the shared memory buffer.
            self._array = np.ndarray(shape=self._shape, dtype=self._datatype, buffer=self._buffer.buf)
            self._connected = True

    def disconnect(self) -> None:
        """Disconnects from the shared memory buffer, preventing the instance from accessing and manipulating the
        shared data.

        This method should be called by each Python process that no longer requires shared buffer access or as part
        of its shutdown sequence.

        Notes:
            This method does not destroy the shared memory buffer. It only releases the local reference to the shared
            memory buffer, potentially enabling it to be garbage-collected by the Operating System. Use the destroy()
            method on Unix-based Operating Systems to destroy the buffer.
        """
        if self._connected:
            self._buffer.close()
            self._connected = False

    def destroy(self) -> None:
        """Requests the instance's shared memory buffer to be destroyed.

        This method should only be called once from the highest runtime scope. Calling this method while having
        SharedMemoryArray instances connected to the buffer leads to undefined behavior.

        Notes:
            This method does not do anything on Windows. Windows automatically garbage-collects the buffers as long as
            they are no longer connected to by any SharedMemoryArray instances.
        """
        # If the instance is connected to the buffer, first disconnects it from the buffer
        self.disconnect()

        # Requests the shared memory buffer to be destroyed
        self._buffer.unlink()

    def _convert_to_slice(self, index: tuple[int, ...]) -> tuple[int, int | None]:
        """Converts the input tuple into start and stop arguments compatible with numpy slice operation.

        Args:
            index: The tuple of integers to parse as slice arguments. Has to contain a minimum of 1 element (start) and
                a maximum of 2 elements (start and stop).

        Returns:
            A 2-element tuple. The first element is start, and it is expected to always be an integer. The second
            element is stop, and it can be an integer or None.

        Raises:
            ValueError: If the input tuple contains an invalid number of elements.
        """
        # Parses the index tuple into slicing arguments (start and stop)
        start: int
        stop: int | None
        if isinstance(index, tuple):
            # Start only
            if len(index) == 1:
                start = int(index[0])
                return start, None
            # Start and stop
            if len(index) == 2:
                start = int(index[0])
                stop = int(index[1])
                return start, stop

        # Invalid input
        message: str = (
            f"Unable to convert the index tuple into slice arguments for {self.name} SharedMemoryArray "
            f"instance. Expected a tuple with 1 or 2 elements (start and stop), but instead encountered "
            f"a tuple with {len(index)} elements."
        )
        console.error(message=message, error=ValueError)
        # Fallback to appease mypy, should not be reachable
        raise ValueError(message)  # pragma: no cover

    @contextmanager
    def _optional_lock(self, with_lock: bool) -> Generator[Any, Any, None]:
        """Conditionally acquires the lock if the caller instructs the manager to do so.

        This is used to make locking optional for all data manipulation methods, improving class flexibility.

        Args:
            with_lock: Determine if the context should be run with or without the multiprocessing lock object.

        Returns:
              The context that has acquired the lock or an empty context if lock is not required.

        """
        if with_lock:
            with self._lock:
                yield
        else:
            yield

    def _verify_indices(self, start: int, stop: int | None) -> tuple[int, int | None]:
        """Converts start and stop indices used to slice the shared numpy array to positive values (if needed) and
        verifies them against array boundaries.

        This function handles both positive and negative indices, as well as None values.

        Args:
            start: The starting index of the slice. Can be positive or negative.
            stop: The ending index of the slice. Can be positive, negative, or None.

        Returns:
            A tuple of (start, stop) indices, where start is always an int and stop can be int or None.

        Raises:
            ValueError: If start index is larger than the stop index after both are converted to positive numbers
            IndexError: If either of the two indices is outside the array boundaries.
        """
        array_length = self.shape[0]

        # Saves initial start and stop index values to be used in error messages
        initial_start = copy(start)
        initial_stop = copy(stop)

        # Converts negative start index to positive
        if start < 0:
            start += array_length

        # Converts negative stop index to positive if it's not None. Uses < 1 here because stop normally cannot be 0
        # when it is valid. When stop is 0, this likely means it was produced from a single -1 (-1 + 1 = 0).
        if stop is not None and stop < 1:
            stop += array_length

        # Checks if start is within bounds
        if start < 0 or start >= array_length:
            message = (
                f"Unable to retrieve the data from {self.name} SharedMemoryArray class instance using start index "
                f"{initial_start}. The index is outside the valid start index range ({0}:{array_length - 1})."
            )
            console.error(message=message, error=IndexError)

        # Checks if stop is within bounds if it's not None
        if stop is not None and (stop < 1 or stop > array_length):
            message = (
                f"Unable to retrieve the data from {self.name} SharedMemoryArray class instance using stop index "
                f"{initial_stop}. The index is outside the valid stop index range ({1}:{array_length})."
            )
            console.error(message=message, error=IndexError)

        # Ensures start is not greater than stop (if stop is not None)
        if stop is not None and start > stop:
            message = (
                f"Invalid pair of slice indices encountered when manipulating data of the {self.name} "
                f"SharedMemoryArray class instance. The start index ({initial_start}) is greater than the end index "
                f"({initial_stop}), which is not allowed."
            )
            console.error(message=message, error=ValueError)

        return start, stop

    def read_data(self, index: int | tuple[int, ...], *, convert_output: bool = False, with_lock: bool = True) -> Any:
        """Reads data from the shared memory array at the specified slice or index.

        This method allows flexibly extracting slices and single values from the shared memory array wrapped by the
        class. The extracted data can be returned using numpy datatype or converted to Python datatype, if requested.
        Reading from the array will behave exactly like reading from a regular numpy array.

        Args:
            index: An integer index to read, when reading scalar data points. A tuple of up to 2 integers in the
                format (start, stop), when reading slices. A minimum of one integer (start) in the tuple is
                required. Stop index is excluded from the returned data slice (last returned index is stop-1).
            convert_output: Determines whether to convert the retrieved data into the closest Python datatype or to
                return it as the numpy datatype.
            with_lock: Determines whether to acquire the multiprocessing Lock before reading the data. Acquiring the
                lock prevents collisions with other python processes, but this may not be necessary for some use cases.

        Returns:
            The data at the specified index or slice. When a single data-value is extracted, it is returned as a
            scalar. When multiple data-values are extracted, they are returned as iterable (list, tuple, or numpy
            array).

        Raises:
            RuntimeError: If the class instance is not connected to a shared memory buffer.
            ValueError: If the input index tuple contains an invalid number of elements to parse it as slice start and
                stop values. If using slice indices and start index is greater than stop index after indices are
                converted to positive numbers (this is done internally, input indices can be negative).
            IndexError: If the input index or slice is outside the array boundaries.
        """
        # Ensures the class is connected to the shared memory buffer
        if not self._connected or self._array is None:
            message: str = (
                f"Unable to access the data stored in the {self.name} SharedMemoryArray instance, as the class is not "
                f"connected to the shared memory buffer. Use connect() method prior to calling other class methods."
            )
            console.error(message=message, error=RuntimeError)

        # If index is a tuple, decomposes it into slice operands to use on the array
        start: int = 0
        stop: int | None = None
        if isinstance(index, tuple):
            # noinspection PyTypeChecker
            start, stop = self._convert_to_slice(index=index)
        # To optimize variable use, also converts single indices to start / stop notation
        elif isinstance(index, int):
            start = index
            stop = index + 1
        else:
            message = (
                f"Unable to read data from {self.name} SharedMemoryArray class instance. Expected an integer index or "
                f"a tuple of two integers, but encountered '{index}' of type {type(index).__name__} instead."
            )
            console.error(message=message, error=ValueError)

        # Converts both indices to be positive and verifies that they are within the array boundaries and not malformed
        start, stop = self._verify_indices(start, stop)

        # Extracts the requested data. The data is copied locally to prevent any modifications to the underlying
        # array object.
        data: NDArray[Any]
        # Depending on the value of the 'with_lock' argument, this either acquires a Lock or runs without a lock.
        with self._optional_lock(with_lock=with_lock):
            # noinspection PyTestUnpassedFixture
            data = self._array[start:stop].copy()

        # Determines whether the data can be returned as a scalar or iterable and whether it needs to be converted to
        # Python datatype or returned as numpy datatype.
        if convert_output:
            if data.size != 1:
                return data.tolist()
            return data.item()
        if data.size != 1:
            return data
        return data[0]

    def write_data(
        self,
        index: int | tuple[int, ...],
        data: Any,
        with_lock: bool = True,
    ) -> None:
        """Writes data to the shared memory array at the specified index or indices (via slice).

        This method allows flexibly writing data to the shared memory array wrapped by the class. Before it is written,
        the input data is converted to the datatype of the array. Writing to the array will behave exactly like writing
        to a regular numpy array.

        Args:
             index: An integer index to write to, when writing scalar data points. A tuple of up to 2 integers in the
                format (start, stop), when writing slices. A minimum of one integer (start) in the tuple is required.
                Stop index is excluded from the modified array slice (last modified (overwritten) index is stop-1).
            data: The data to write to the shared numpy array. It in the format that is compatible with (convertible
                to) the size (shape) and the datatype of the array wrapped by the class.
            with_lock: Determines whether to acquire the multiprocessing Lock before writing the data. Acquiring the
                lock prevents collisions with other python processes, but this may not be necessary for some use cases.

        Raises:
            RuntimeError: If the class instance is not connected to a shared memory buffer.
            ValueError: If the input index tuple contains an invalid number of elements to parse it as slice start and
                stop values. If the method is unable to convert the input data into the array format, or if writing data
                to the array fails. If using slice indices and start index is greater than stop index after indices are
                converted to positive numbers (this is done internally, input indices can be negative).
            IndexError: If the input index or slice is outside the array boundaries.
        """
        # Ensures the class is connected to the shared memory buffer
        if not self._connected or self._array is None:
            message: str = (
                f"Unable to access the data stored in the {self.name} SharedMemoryArray instance, as the class is not "
                f"connected to the shared memory buffer. Use connect() method prior to calling other class methods."
            )
            console.error(message=message, error=RuntimeError)

        # If index is a tuple, decomposes it into slice operands to use on the array
        start: int = 0
        stop: int | None = None
        if isinstance(index, tuple):
            # noinspection PyTypeChecker
            start, stop = self._convert_to_slice(index=index)
        # To optimize variable use, also converts single indices to start / stop notation
        elif isinstance(index, int):
            start = index
            stop = index + 1
        else:
            message = (
                f"Unable to write data to {self.name} SharedMemoryArray class instance. Expected an integer index or "
                f"a tuple of two integers, but encountered '{index}' of type {type(index).__name__} instead."
            )
            console.error(message=message, error=ValueError)

        # Converts both indices to be positive and verifies that they are within the array boundaries and not malformed
        start, stop = self._verify_indices(start, stop)

        # If the input data is not a numpy array, converts it to the numpy array using the same datatype as the one
        # used by the shared memory array
        try:
            if not isinstance(data, np.ndarray):
                # The only difference between iterable and scalar is that scalars are first cast as a list
                if isinstance(data, Iterable):
                    data = np.array(object=data, dtype=self.datatype)
                else:
                    data = np.array(object=[data], dtype=self.datatype)

            # Writes the data to the array, optionally using the lock.
            with self._optional_lock(with_lock=with_lock):
                self._array[start:stop] = data
        # Catches and redirects ValueErrors, which is used by numpy to indicate conversion errors.
        except ValueError as e:
            message = (
                f"Unable write data to {self.name} SharedMemoryArray class instance with index {index}. Encountered "
                f"the following error when converting the data to the array datatype ({self.datatype}) and writing it "
                f"to the array: {e}."
            )
            console.error(message=message, error=ValueError)

    @property
    def datatype(
        self,
    ) -> np.dtype[Any]:
        """Returns the datatype used by the shared memory array."""
        return self._datatype

    @property
    def name(self) -> str:
        """Returns the name of the shared memory buffer."""
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the shared memory array."""
        return self._shape

    @property
    def is_connected(self) -> bool:
        """Returns True if the class is connected to the shared memory buffer that stores the shared array data.

        Connection to the shared memory buffer is required for most class methods to work.
        """
        return self._connected
