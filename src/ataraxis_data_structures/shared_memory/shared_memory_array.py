"""This module contains the SharedMemoryArray class that allows moving data between multiple Python processes through
a shared one-dimensional numpy array.

SharedMemoryArray works by creating multiple numpy array instances, one per each process, that share the same data
buffer. It is equipped with the necessary mechanisms to ensure thread- and process-safe data manipulation and functions
as an alternative to Queue objects.
"""

from copy import copy
from typing import Any, Union, Iterable, Optional, Generator
from contextlib import contextmanager
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console


class SharedMemoryArray:
    """Wraps a one-dimensional numpy array object and exposes methods for accessing the array data from multiple Python
    processes.

    This class enables sharing data between multiple Python processes in a way that compliments the functionality of
    the multiprocessing Queue class. Like Queue, this class provides the means for sharing data between multiple
    processes by implementing a shared memory buffer. Unlike Queue, which behaves like a buffered stream, the
    SharedMemoryArray class behaves like a numpy array. Unlike with Queue, the size and datatype of the array are fixed
    after initialization and cannot be changed. However, the elements of the array can be randomly accessed and
    modified, unlike the elements of the Queue, that can only be processed serially.

    This class should only be instantiated inside the main process via its create_array() method. Do not attempt to
    instantiate the class manually. All child processes working with this class should use the connect() method to
    connect to the shared array wrapped by the class before calling any other method.

    Notes:
        Shared memory objects are garbage-collected differently depending on the host OS. On Windows, garbage collection
        is handed off to the OS and cannot be enforced manually. On Unix (OSx and Linux), the buffer can be
        garbage-collected via appropriate commands.

    Args:
        name: The descriptive name to use for the shared memory array. The OS uses names to identify shared
            memory objects and have to be unique.
        shape: The shape of the shared numpy array.
        datatype: The datatype of the shared numpy array.
        buffer: The shared memory buffer that stores the data of the array and enables accessing the same data from
            different Python processes.

    Attributes:
        _name: Stores the name of the shared memory object.
        _shape: Stores the shape of the numpy array used to represent the buffered data.
        _datatype: Stores the datatype of the numpy array used to represent the buffered data.
        _buffer: The Shared Memory buffer object used to store the shared array data.
        _lock: A Lock object used to prevent multiple processes from working with the shared array data at the same
            time.
        _array: Stores the connected shared numpy array.
        _is_connected: Tracks whether the shared memory array wrapped by this class has been connected to.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        datatype: np.dtype[Any],
        buffer: Optional[SharedMemory],
    ):
        # Initialization method only saves input data into attributes. The method that actually sets up the class is
        # the create_array() class method.
        self._name: str = name
        self._shape: tuple[int, ...] = shape
        self._datatype: np.dtype[Any] = datatype
        self._buffer: Optional[SharedMemory] = buffer
        self._lock = Lock()
        self._array: Optional[NDArray[Any]] = None
        self._is_connected: bool = False

    def __repr__(self) -> str:
        """Generates and returns a class representation string."""
        return (
            f"SharedMemoryArray(name='{self._name}', shape={self._shape}, datatype={self._datatype}, "
            f"connected={self.is_connected})"
        )

    @classmethod
    def create_array(
        cls,
        name: str,
        prototype: NDArray[Any],
    ) -> "SharedMemoryArray":
        """Creates a SharedMemoryArray class instance using the input one-dimensional prototype array.

        This method uses the input prototype to generate a new numpy array that uses a shared memory buffer to store
        its data. It then extracts the information required to connect to the buffer and reinitialize the array in
        a different Python process and saves it to class attributes.

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

        Returns:
            The configured SharedMemoryArray class instance. This instance should be passed to each of the processes
            that needs to access the wrapped array data.

        Raises:
            TypeError: If the input prototype is not a numpy ndarray.
            FileExistsError: If a shared memory object with the same name as the input 'name' argument value already
                exists.
        """
        # Ensures prototype is a numpy ndarray
        if not isinstance(prototype, np.ndarray):
            message = (
                f"Invalid 'prototype' argument type encountered when creating SharedMemoryArray object '{name}'. "
                f"Expected a flat (one-dimensional) numpy ndarray but instead encountered {type(prototype).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Also ensures that the prototype is flat
        if prototype.ndim != 1:
            message = (
                f"Invalid 'prototype' array shape encountered when creating SharedMemoryArray object '{name}'. "
                f"Expected a flat (one-dimensional) numpy ndarray but instead encountered prototype with shape "
                f"{prototype.shape} and dimensions number {prototype.ndim}."
            )
            console.error(message=message, error=ValueError)

        # Creates the shared memory object. This process will raise FileExistsError if an object with this name
        # already exists. The shared memory object is used as a buffer to store the array data.
        try:
            buffer: SharedMemory = SharedMemory(name=name, create=True, size=prototype.nbytes)
        except FileExistsError:
            message = (
                f"Unable to create SharedMemoryArray object using name '{name}', as object with this name already "
                f"exists. This is likely due to calling create_array() method from a child process. "
                f"Use connect() method to connect to the SharedMemoryArray from a child process."
            )
            console.error(message=message, error=FileExistsError)

        # Instantiates a numpy ndarray using the shared memory buffer and copies prototype array data into the shared
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
        """Connects to the shared memory buffer that stores the array data, allowing to access and manipulate the data
        through this class.

        This method should be called once for each Python process that uses this class, before calling any other
        methods. It is called automatically as part of the create_array() method runtime.
        """
        self._buffer = SharedMemory(name=self._name)  # Connects to the buffer
        # Re-initializes the internal _array with the data from the shared memory buffer.
        self._array = np.ndarray(shape=self._shape, dtype=self._datatype, buffer=self._buffer.buf)
        self._is_connected = True

    def disconnect(self) -> None:
        """Disconnects the class from the shared memory buffer.

        This method should be called whenever the process no longer requires shared buffer access.

        Notes:
            This method does not destroy the shared memory buffer. It only releases the local reference to the shared
            memory buffer, potentially enabling it to be garbage-collected.
        """
        if self._is_connected and self._buffer is not None:
            self._buffer.close()
            self._is_connected = False

    def destroy(self) -> None:
        """Requests the underlying shared memory buffer to be destroyed.

        This method should only be called once from the highest runtime scope. Typically, this is done as part of a
        global runtime shutdown procedure to ensure all resources are released. Calling this method while having
        SharedMemoryArray instances connected to the buffer will lead to undefined behavior.

        This method will only work if the current instance is NOT connected to the buffer.

        Notes:
            This method does not do anything on Windows. Windows automatically garbage-collects the buffers as long as
            they are no longer connected to by any SharedMemoryArray instances.
        """
        if not self._is_connected and self._buffer is not None:
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
        stop: Optional[int]
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
            else:
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

    def _verify_indices(self, start: int, stop: Optional[int]) -> tuple[int, Optional[int]]:
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
            scalar. When multiple data-values are extracted, they are returned as an iterable.

        Raises:
            RuntimeError: If the class instance is not connected to a shared memory buffer.
            ValueError: If the input index tuple contains an invalid number of elements to parse it as slice start and
                stop values. If using slice indices and start index is greater than stop index after indices are
                converted to positive numbers (this is done internally, input indices can be negative).
            IndexError: If the input index or slice is outside the array boundaries.
        """

        # Ensures the class is connected to the shared memory buffer
        if not self._is_connected or self._array is None:
            message: str = (
                f"Unable to access the data stored in the {self.name} SharedMemoryArray instance, as the class is not "
                f"connected to the shared memory buffer. Use connect() method prior to calling other class methods."
            )
            console.error(message=message, error=RuntimeError)

        # If index is a tuple, decomposes it into slice operands to use on the array
        start: int = 0
        stop: Optional[int] = None
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
            data = self._array[start:stop].copy()  # type: ignore

        # Determines whether the data can be returned as a scalar or iterable and whether it needs to be converted to
        # Python datatype or returned as numpy datatype.
        if convert_output:
            if data.size != 1:
                return data.tolist()
            else:
                return data.item()
        elif data.size != 1:
            return data
        else:
            return data[0]

    def write_data(
        self,
        index: int | tuple[int, ...],
        data: Union[
            NDArray[Any],
            list[Any],
            tuple[Any],
            np.unsignedinteger[Any],
            np.signedinteger[Any],
            np.floating[Any],
            int,
            float,
            bool,
            str,
            None,
        ],
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
        if not self._is_connected or self._array is None:
            message: str = (
                f"Unable to access the data stored in the {self.name} SharedMemoryArray instance, as the class is not "
                f"connected to the shared memory buffer. Use connect() method prior to calling other class methods."
            )
            console.error(message=message, error=RuntimeError)

        # If index is a tuple, decomposes it into slice operands to use on the array
        start: int = 0
        stop: Optional[int] = None
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
                self._array[start:stop] = data  # type: ignore
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
        return self._is_connected
