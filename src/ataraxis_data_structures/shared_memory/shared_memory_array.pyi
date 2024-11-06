from typing import Any, Generator
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

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

    _name: Incomplete
    _shape: Incomplete
    _datatype: Incomplete
    _buffer: Incomplete
    _lock: Incomplete
    _array: Incomplete
    _is_connected: bool
    def __init__(
        self, name: str, shape: tuple[int, ...], datatype: np.dtype[Any], buffer: SharedMemory | None
    ) -> None: ...
    def __repr__(self) -> str:
        """Generates and returns a class representation string."""
    @classmethod
    def create_array(cls, name: str, prototype: NDArray[Any]) -> SharedMemoryArray:
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
    def connect(self) -> None:
        """Connects to the shared memory buffer that stores the array data, allowing to access and manipulate the data
        through this class.

        This method should be called once for each Python process that uses this class, before calling any other
        methods. It is called automatically as part of the create_array() method runtime.
        """
    def disconnect(self) -> None:
        """Disconnects the class from the shared memory buffer.

        This method should be called whenever the process no longer requires shared buffer access.

        Notes:
            This method does not destroy the shared memory buffer. It only releases the local reference to the shared
            memory buffer, potentially enabling it to be garbage-collected.
        """
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
    def _optional_lock(self, with_lock: bool) -> Generator[Any, Any, None]:
        """Conditionally acquires the lock if the caller instructs the manager to do so.

        This is used to make locking optional for all data manipulation methods, improving class flexibility.

        Args:
            with_lock: Determine if the context should be run with or without the multiprocessing lock object.

        Returns:
              The context that has acquired the lock or an empty context if lock is not required.

        """
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
    def write_data(
        self,
        index: int | tuple[int, ...],
        data: NDArray[Any]
        | list[Any]
        | tuple[Any]
        | np.unsignedinteger[Any]
        | np.signedinteger[Any]
        | np.floating[Any]
        | int
        | float
        | bool
        | str
        | None,
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
    @property
    def datatype(self) -> np.dtype[Any]:
        """Returns the datatype used by the shared memory array."""
    @property
    def name(self) -> str:
        """Returns the name of the shared memory buffer."""
    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the shared memory array."""
    @property
    def is_connected(self) -> bool:
        """Returns True if the class is connected to the shared memory buffer that stores the shared array data.

        Connection to the shared memory buffer is required for most class methods to work.
        """
