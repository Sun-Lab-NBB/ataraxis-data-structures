import textwrap
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple, Union

import numpy as np
from ataraxis_automation.utilities import format_message


class SharedMemoryArray:
    """A wrapper around an n-dimensional numpy array object that exposes methods for accessing the array buffer from
    multiple processes.

    This class is designed to compliment the Queue-based method for sharing data between multiple python processes.
    Similar to Queues, this class instantiates a shared memory buffer, to which all process-specific instance of this
    class link when their 'connect' property is called. Unlike Queue, however, this shared memory buffer is static
    post-initialization and represents a numpy array with all associated limitations (fixed datatypes, static size,
    etc.).

    This class should only be instantiated inside the main process via the create_array() method. Do not attempt to
    instantiate the class manually. All children processes should get an instance of this class as an argument and
    use the connect() method to connect to the buffer created by the founder instance inside the main scope.

    Notes:
        Shared memory objects are garbage-collected differently depending on the host-platform OS. On Windows-based
        systems, garbage collection is handed off to the OS and cannot be enforced manually. On Unix systems, the
        buffer can be garbage-collected via appropriate de-allocation commands.

        All data accessors from this class use multiprocessing Lock instance to ensure process- and thread-safety. This
        make this class less optimized for use-cases that rely on multiple processes simultaneously reading the same
        data for increased performance. In this case, it is advised to use a custom implementation of the shared
        memory system.

    Args:
        name: The descriptive name to use for the shared memory array. Names are used by the host system to identify
            shared memory objects and, therefore, have to be unique.
        shape: The shape of the numpy array, for which the shared memory buffer would be instantiated. Note, the shape
            cannot be changed post-instantiation.
        datatype: The datatype to be used by the numpy array. Note, the datatype cannot be changed post-instantiation.
        buffer: The memory buffer shared between all instances of this class across all processes (and threads).

    Attributes:
        _name: The descriptive name of the shared memory array. The name is sued to connect to the same shared memory
            buffer from different processes.
        _shape: The shape of the numpy array that uses the shared memory buffer. This is used to properly format the
            data available through the buffer.
        _datatype: The datatype of the numpy array that uses the shared memory buffer. This is also used to properly
            format the data available through the buffer.
        _buffer: The shared memory buffer that stores the array data. Has to be connected to vai connect() method
            before the class can be used.
        _lock: A Lock object used to ensure only one process is allowed to access (read or write) the array data at any
            point in time.
        _array: The inner object used to store the connected shared numpy array.
        _is_connected: A flag that tracks whether the shared memory array manged by this class has been connected to.
            This is a prerequisite for most other methods of the class to work.
    """

    def __init__(
        self,
        name: str,
        shape: tuple,
        datatype: np.dtype,
        buffer: Optional[SharedMemory],
    ):
        self._name: str = name
        self._shape: tuple = shape
        self._datatype: np.dtype = datatype
        self._buffer: Optional[SharedMemory] = buffer
        self._lock = Lock()
        self._array: Optional[np.ndarray] = None
        self._is_connected: bool = False

    @classmethod
    def create_array(cls, name: str, prototype: np.ndarray) -> "SharedMemoryArray":
        """Uses the input prototype numpy array to create an instance of this class.

        Specifically, this method first creates a shared bytes buffer that is sufficiently large to hold the data of the
        prototype array and then uses it to create a new numpy array with the same shape and datatype as the prototype.
        Subsequently, it copies all data from the prototype aray to the new shared memory array, enabling to access and
        manipulate the data from different processes (using returned class instance methods).

        Notes:
            This method should only be used once, when the array is first created in the root (main) process. All
            child processes should use the connect() method to connect to an existing array.

        Args:
            name: The name to give to the created SharedMemory object. Note, this name has to be unique across all
                scopes using the array.
            prototype: The numpy ndarray instance to serve as the prototype for the created shared memory object.

        Returns:
            The instance of the SharedMemoryArray class. This class exposes methods that allow connecting to the shared
            memory aray from different processes and thread-safe methods for reading and writing data to the array.

        Raises:
            FileExistsError: If a shared memory object with the same name as the input 'name' argument value already
                exists.
        """

        # Creates shared memory object. This process will raise a FileExistsError if an object with this name already
        # exists. The shared memory object is used as a buffer to store the array data.
        buffer = SharedMemory(create=True, size=prototype.nbytes, name=name)

        # Instantiates a numpy array using the shared memory buffer and copies prototype array data into the shared
        # array instance
        shared_arr = np.ndarray(shape=prototype.shape, dtype=prototype.dtype, buffer=buffer.buf)
        shared_arr[:] = prototype[:]

        # Packages the data necessary to connect to the shared array into the class object.
        shared_memory_array = cls(
            name=name,
            shape=shared_arr.shape,
            datatype=shared_arr.dtype,
            buffer=buffer,
        )

        # Connects the internal array of the class object to the shared memory buffer.
        shared_memory_array.connect()

        # Returns the instantiated and connected class object to caller.
        return shared_memory_array

    def connect(self):
        """Connects to the shared memory buffer that stores the array data, allowing to manipulate access and manipulate
         the data through this class.

        This method should be called once for each process that receives an instance of this class as input, before any
        other method of this class. This method is called automatically as part of the create_array() method runtime for
        the founding array.
        """
        self._buffer = SharedMemory(name=self._name)  # Connects to the buffer
        # Connects to the buffer using a numpy array
        self._array = np.ndarray(shape=self._shape, dtype=self._datatype, buffer=self._buffer.buf)
        self._is_connected = True  # Sets the connection flag

    def disconnect(self):
        """Disconnects the class from the shared memory buffer.

        This method should be called whenever the process no longer requires shared buffer access.
        """
        if self._is_connected:
            self._buffer.close()
            self._is_connected = False

    def read_data(self, index: int | slice) -> np.ndarray:
        """Reads data from the shared memory array at the specified index or slice.

        Args:
            index: The integer index to read a specific value or slice to read multiple values from the underlying
                shared numpy array.

        Returns:
            The data at the specified index or slice as a numpy array. The data will use the same datatype as the
            source array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
            ValueError: If the input index or slice is invalid.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot read data as the class is not connected to a shared memory array. Use connect() method to "
                "connect to the shared memory array."
            )
            raise RuntimeError(format_message(custom_error_message))

        with self._lock:
            try:
                return np.array(self._array[index])
            except IndexError:
                custom_error_message = (
                    "Invalid index or slice when attempting to read the data from shared memory array."
                )
                raise ValueError(format_message(custom_error_message))

    def write_data(self, index: int | slice, data: np.ndarray) -> None:
        """Writes data to the shared memory array at the specified index or slice.

        Args:
            index: The index or slice to write data to.
            data: The data to write to the shared memory array. Must be a numpy array with the same datatype as the
                shared memory array bound by the class.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
            ValueError: If the input data is not a numpy array, if the datatype of the input data does not match the
                datatype of the shared memory array, or if the data cannot fit inside the shared memory array at the
                specified index or slice.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot write data as the class is not connected to a shared memory array. Use connect() method to "
                "connect to the shared memory array."
            )
            raise RuntimeError(format_message(custom_error_message))

        if not isinstance(data, np.ndarray):
            custom_error_message = "Input data must be a numpy array."
            raise ValueError(format_message(custom_error_message))

        if data.dtype != self._datatype:
            custom_error_message = (
                f"Input data must have the same datatype as the shared memory array: {self._datatype}."
            )
            raise ValueError(format_message(custom_error_message))

        with self._lock:
            try:
                self._array[index] = data
            except ValueError:
                custom_error_message = (
                    "Input data cannot fit inside the shared memory array at the specified index or slice."
                )
                raise ValueError(format_message(custom_error_message))

    @property
    def datatype(self) -> np.dtype:
        """Returns the datatype of the shared memory array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot retrieve array datatype as the class is not connected to a shared memory array. Use connect() "
                "method to connect to the shared memory array."
            )
            raise RuntimeError(format_message(custom_error_message))
        return self._datatype

    @property
    def name(self) -> str:
        """Returns the name of the shared memory buffer.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot retrieve shared memory buffer name as the class is not connected to a shared memory array. "
                "Use connect() method to connect to the shared memory array."
            )
            raise RuntimeError(format_message(custom_error_message))
        return self._name

    @property
    def shape(self) -> tuple:
        """Returns the shape of the shared memory array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            custom_error_message = (
                "Cannot retrieve shared memory array shape as the class is not connected to a shared memory array. "
                "Use connect() method to connect to the shared memory array."
            )
            raise RuntimeError(format_message(custom_error_message))
        return self._shape

    @property
    def is_connected(self) -> bool:
        """Returns True if the shared memory array is connected to the shared buffer.

        Connection to the shared buffer is required from most class methods to work.
        """
        return self._is_connected
