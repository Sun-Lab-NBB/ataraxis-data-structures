"""This module contains the SharedMemoryArray class that provides methods for sharing data between multiple Python
processes through a shared c-buffer.
"""

from typing import Union, Optional
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console


class SharedMemoryArray:
    """Wraps an n-dimensional numpy array object and exposes methods for accessing the array buffer from multiple
    Python processes.

    This class is designed to compliment the Queue-based method for sharing data between multiple Python processes.
    Similar to Queue class, this class instantiates a shared memory buffer, to which all process-specific instances of
    this class link when their 'connect' method is called. Unlike Queue, however, the shared memory buffer from this
    class is static post-initialization and represents a numpy array with all associated limitations
    (fixed datatype, size, etc.). It also has the advantages of a numpy array, including the access to fast numpy
    functions and the ability to randomly access and manipulate any portion of the buffer.

    This class should only be instantiated inside the main process via the create_array() method. Do not attempt to
    instantiate the class manually. All children processes should get an instance of this class as an argument and
    use the class connect() method to connect to the buffer created by the founder instance inside the main scope.

    Notes:
        Shared memory objects are garbage-collected differently depending on the host-platform OS. On Windows, garbage
        collection is handed off to the OS and cannot be enforced manually. On Unix (OSx and Linux), the buffer can be
        garbage-collected via appropriate deallocation commands.

        All data accessors from this class use multiprocessing Lock instance to ensure process- and thread-safety. This
        makes this class less optimized for use-cases that rely on multiple processes simultaneously reading the same
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
        shape: tuple[int, ...],
        datatype: np.dtype[
            Union[
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            ]
        ],
        buffer: Optional[SharedMemory],
    ):
        self._name: str = name
        self._shape: tuple[int, ...] = shape
        self._datatype: np.dtype[
            Union[
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            ]
        ] = datatype
        self._buffer: Optional[SharedMemory] = buffer
        self._lock = Lock()
        self._array: Optional[
            NDArray[
                Union[
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                    np.float16,
                    np.float32,
                    np.float64,
                ]
            ]
        ] = None
        self._is_connected: bool = False

    def __repr__(self):
        return (
            f"SharedMemoryArray(name='{self._name}', shape={self._shape}, datatype={self._datatype}, "
            f"connected={self.is_connected})"
        )

    @classmethod
    def create_array(
        cls,
        name: str,
        prototype: NDArray[
            Union[
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            ]
        ],
    ) -> "SharedMemoryArray":
        """Creates a SharedMemoryArray class instance using the input prototype numpy array.

        This method extracts the datatype and shape from the input prototype array and uses them to create a new
        array object that uses a shared memory buffer. The buffer parameters are then stored in class attributes,
        enabling the class to reconnect to the buffer from different Python processes.

        Notes:
            This method should only be called when the array is first created in the main process (scope). All
            child processes should use the connect() method to connect to the existing array.

        Args:
            name: The name to give to the created SharedMemory object. Note, this name has to be unique across all
                processes using the array.
            prototype: The numpy ndarray instance to serve as the prototype for the created shared memory object.

        Returns:
            The configured SharedMemoryArray class instance.

        Raises:
            TypeError: If the input prototype is not a numpy ndarray.
            FileExistsError: If a shared memory object with the same name as the input 'name' argument value already
                exists.
        """
        # Ensures prototype is a numpy ndarray
        if not isinstance(prototype, np.ndarray):
            message = (
                f"Invalid 'prototype' argument type encountered when creating SharedMemoryArray object using name "
                f"'{name}'. Expected a numpy ndarray with a signed_integer, unsigned_integer or floating "
                f"datatype, but instead encountered {type(prototype)}."
            )
            console.error(message=message, error=TypeError)

        # Creates the shared memory object. This process will raise a FileExistsError if an object with this name
        # already exists. The shared memory object is used as a buffer to store the array data.
        try:
            buffer: SharedMemory = SharedMemory(create=True, size=prototype.nbytes, name=name)
        except FileExistsError:
            message = (
                f"Unable to create SharedMemoryArray object using name '{name}' as object with this name already "
                f"exists. This is likely due to calling create_array() method from a child process. "
                f"Use connect() method to connect to the SharedMemoryArray from a child process and ensure "
                f"__main__ guard is present to avoid unwanted array recreation attempts."
            )
            console.error(message=message, error=FileExistsError)

        # Instantiates a numpy ndarray using the shared memory buffer and copies prototype array data into the shared
        # array instance.
        # noinspection PyUnboundLocalVariable
        shared_array: NDArray[
            Union[
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            ]
        ] = np.ndarray(shape=prototype.shape, dtype=prototype.dtype, buffer=buffer.buf)
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
            memory buffer, potentially enabling it to be garbage-collected if this also happened to be the last
            reference.
        """
        if self._is_connected and self._buffer is not None:
            self._buffer.close()
            self._is_connected = False

    def read_data(
        self, index: int | slice | tuple[int, ...], *, convert_output: bool = False
    ) -> Union[
        NDArray[
            Union[
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            ]
        ],
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        int,
        float,
        list[int | float],
    ]:
        """Reads data from the shared memory array at the specified slice or index.

        This method allows flexibly extracting slices and single values from shallow and multidimensional numpy
        arrays. The extracted data can be returned using numpy datatype or converted to Python datatype, if requested.

        Args:
            index: The integer index to read a specific value or slice to read multiple values from the shared memory
                array. When reading single values from n-dimensional arrays, provide a tuple of integer
                indices to access the desired element (one index per dimension).
            convert_output: Determines whether to convert the retrieved data into the closest Python datatype or to
                return it as the numpy datatype.

        Returns:
            The data at the specified index or slice. When a single data-value is extracted, it is returned as a
            scalar. When multiple data-values are extracted, they are returned as an iterable.

        Raises:
            RuntimeError: If the class instance is not connected to a shared memory buffer.
            ValueError: If the input index or slice is invalid.
        """
        if not self._is_connected or self._array is None:
            message = (
                "Unable to read data from SharedMemoryArray class instance, as the class is not connected to the "
                "shared memory buffer. Use connect() method prior to calling other class methods."
            )
            console.error(message=message, error=RuntimeError)

        # Extracts the requested data. To minimize interference with other Python processes, only holds the Lock until
        # the data is copied locally.
        with self._lock:
            try:
                data: np.ndarray = self._array[index].copy()
            except IndexError:
                message = (
                    f"Invalid index or slice {index} of type {type(index)} encountered when attempting to read data "
                    f"from SharedMemoryArray class instance."
                )
                console.error(message=message, error=ValueError)

        # Determines whether the data can be returned as a scalar or iterable and whether it needs to be converted to
        # Python datatype or returned as numpy datatype.
        if convert_output:
            if isinstance(data, np.ndarray):
                return data.tolist()
            else:
                return data.item()
        elif data.size == 1:
            return data.astype(dtype=data.dtype)
        else:
            return data

    def write_data(
        self,
        index: int | slice,
        data: NDArray[
            Union[
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            ]
        ],
    ) -> None:
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
        if not self._is_connected or self._array is None:
            message = (
                "Cannot write data as the class is not connected to a shared memory array. Use connect() method to "
                "connect to the shared memory array."
            )
            console.error(message=message, error=RuntimeError)

        if not isinstance(data, np.ndarray):
            message = "Input data must be a numpy array."
            console.error(message=message, error=ValueError)

        if data.dtype != self._datatype:
            message = f"Input data must have the same datatype as the shared memory array: {self._datatype}."
            console.error(message=message, error=ValueError)

        with self._lock:
            try:
                self._array[index] = data
            except ValueError:
                message = "Input data cannot fit inside the shared memory array at the specified index or slice."
                console.error(message=message, error=ValueError)

    @property
    def datatype(
        self,
    ) -> np.dtype[
        np.int8
        | np.int16
        | np.int32
        | np.int64
        | np.uint8
        | np.uint16
        | np.uint32
        | np.uint64
        | np.float16
        | np.float32
        | np.float64
    ]:
        """Returns the datatype of the shared memory array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            message = (
                "Cannot retrieve array datatype as the class is not connected to a shared memory array. Use connect() "
                "method to connect to the shared memory array."
            )
            console.error(message=message, error=RuntimeError)
        return self._datatype

    @property
    def name(self) -> str:
        """Returns the name of the shared memory buffer.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            message = (
                "Cannot retrieve shared memory buffer name as the class is not connected to a shared memory array. "
                "Use connect() method to connect to the shared memory array."
            )
            console.error(message=message, error=RuntimeError)
        return self._name

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the shared memory array.

        Raises:
            RuntimeError: If the shared memory array has not been connected to by this class instance.
        """
        if not self._is_connected:
            message = (
                "Cannot retrieve shared memory buffer name as the class is not connected to a shared memory array. "
                "Use connect() method to connect to the shared memory array."
            )
            console.error(message=message, error=RuntimeError)
        return self._shape

    @property
    def is_connected(self) -> bool:
        """Returns True if the shared memory array is connected to the shared buffer.

        Connection to the shared buffer is required from most class methods to work.
        """
        return self._is_connected
