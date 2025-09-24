"""This module provides the SharedMemoryArray class that allows moving data between multiple Python processes through
a shared one-dimensional NumPy array memory buffer.
"""

from typing import Any
from contextlib import contextmanager
from collections.abc import Generator
from multiprocessing import Lock
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console


class SharedMemoryArray:
    """Wraps a NumPy n-dimensional array object and exposes methods for accessing the array's data from multiple
    different Python processes.

    During initialization, this class creates a persistent memory buffer to which it connects from different Python
    processes. The data inside the buffer is accessed via an n-dimensional NumPy array with optional locking to prevent
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

    Attributes:
        _name: Stores the name of the shared memory buffer.
        _shape: Stores the shape of the NumPy array used to access the buffered data.
        _datatype: Stores the datatype of the NumPy array used to access the buffered data.
        _buffer: Stores the Shared Memory buffer object.
        _lock: Stores the Lock object used to prevent multiple processes from working with the shared data at the same
            time.
        _array: Stores the NumPy array used to interface with the data stored in the shared memory buffer.
        _connected: Tracks whether the instance is connected to the shared memory buffer.
        _destroy_buffer: Tracks whether the shared memory buffer should be destroyed if this instance is
            garbage-collected.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        datatype: np.dtype[Any]
    ) -> None:
        # Initialization method only saves input data into attributes. The method that actually sets up the class is
        # the create_array() class method.
        self._name: str = name
        self._shape: tuple[int, ...] = shape
        self._datatype: np.dtype[Any] = datatype
        self._buffer: SharedMemory | None = None
        self._lock = Lock()
        self._array: NDArray[Any] = np.zeros(shape=shape, dtype=datatype)
        self._connected: bool = False
        self._destroy_buffer: bool = False

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
        if self._destroy_buffer:
            self.destroy()
        else:
            # Otherwise, only disconnects the shared memory buffer.
            self.disconnect()

    def __getitem__(self, index: int | slice) -> Any:
        """Gets value(s) at the specified array index or slice with automatic locking.

        This method allows retrieving the data from the SharedMemoryArray instance, without manually accessing the
        underlying array object. It is designed for simple access operations, such as reading a boolean flag value.

        Notes:
            This method always acquires the lock for thread-safe access. Use the array() method with appropriate
            locking configuration to read or write the data without locking.

        Args:
            index: The array index or slice to access.

        Returns:
            The value or array slice at the specified index. The returned data always uses an appropriate NumPy
            array or scalar datatype.

        Raises:
            ConnectionError: If the instance is not connected to the shared memory buffer.
            IndexError: If the requested index is out of bounds.
        """
        if not self._connected or self._array is None:
            message = (
                f"Unable to access the data stored in the {self.name} SharedMemoryArray instance, as the instance is "
                f"not connected to the shared memory buffer. Call the connect() method prior to accessing the array's "
                f"data."
            )
            console.error(message=message, error=ConnectionError)

            # Fallback to appease mypy, should not be reachable.
            raise ConnectionError(message)  # pragma: no cover

        with self.array(with_lock=True) as arr:
            # Returns a copy to prevent external modifications to the returned data from affecting the shared array
            # without going through __setitem__
            result = arr[index]
            if isinstance(result, np.ndarray):
                return result.copy()
            return result

    def __setitem__(self, index: int | slice, value: Any) -> None:
        """Sets value(s) at the specified array index or slice with automatic locking.

        This method allows modifying the data of the SharedMemoryArray instance, without manually accessing the
        underlying array object. It is designed for simple modification operations, such as writing a boolean flag
        value.

        Notes:
            The input values are saved in the underlying NumPy n-dimensional array. If the values are not
            compatible with the array's datatype, they are converted to the array's datatype before being written.

            This method always acquires the lock for thread-safe access. Use the array() method with appropriate
            locking configuration to read or write the data without locking.

        Args:
            index: The array index or slice to set.
            value: The value(s) to set at the specified index or slice.

        Raises:
            ConnectionError: If the instance is not connected to the shared memory buffer.
            IndexError: If the requested index is out of bounds.
            ValueError: If value's shape does not match the slice shape.
        """
        if not self._connected or self._array is None:
            message = (
                f"Unable to modify the data stored in the {self.name} SharedMemoryArray instance, as the instance is "
                f"not connected to the shared memory buffer. Call the connect() method prior to modifying the array's "
                f"data."
            )
            console.error(message=message, error=ConnectionError)

            # Fallback to appease mypy, should not be reachable.
            raise ConnectionError(message)  # pragma: no cover

        # Writes the input values at the specified index
        with self.array(with_lock=True) as arr:
            arr[index] = value

    def enable_buffer_destruction(self) -> None:
        """Configures the instance to destroy the shared memory buffer when it is garbage-collected.

        Enabling this option ensures that all shared memory buffer objects are properly cleaned up on Unix systems
        when the program runtime ends.

        Notes:
            This method should only be called in the main runtime thread, after setting up all child processes. Calling
            this method before starting the child processes may result in unexpected behavior due to child processes
            destroying the buffer as part of their shutdown sequence.
        """
        self._destroy_buffer = True

    @classmethod
    def create_array(
        cls,
        name: str,
        prototype: NDArray[Any],
        *,
        exists_ok: bool = False,
    ) -> "SharedMemoryArray":
        """Creates a SharedMemoryArray instance using the input prototype NumPy array.

        This method uses the input prototype to generate the shared memory buffer to store the prototype's data and
        fills the buffer with the data from the prototype array. All further interactions with the returned
        SharedMemoryArray instance manipulate the data stored in the shared memory buffer.

        Notes:
            This method should only be called when the array is first created in the main runtime thread (scope). All
            child processes should use the connect() method to connect to the existing array.

            After passing the returned instance to all child processes, call the instance's
            enable_buffer_destruction() method to ensure that the shared memory buffer is properly cleaned up when the
            program runtime ends.

        Args:
            name: The unique name to use for the shared memory buffer.
            prototype: The prototype NumPy array instance for the created SharedMemoryArray.
            exists_ok: Determines how the method handles the case where the shared memory buffer with the same name
                already exists. If False, the method raises an exception. If True, the method destroys the existing
                buffer and creates a new buffer using the input name and prototype data.

        Returns:
            The created SharedMemoryArray instance.

        Raises:
            TypeError: If the input prototype is not a NumPy array.
            FileExistsError: If a shared memory object with the same name as the input 'name' argument value already
                exists and the 'exists_ok' flag is False.
        """
        # Ensures prototype is a numpy ndarray
        if not isinstance(prototype, np.ndarray):
            message = (
                f"Invalid 'prototype' argument type encountered when creating SharedMemoryArray object '{name}'. "
                f"Expected a flat (one-dimensional) NumPy array but instead encountered {type(prototype).__name__}."
            )
            console.error(message=message, error=TypeError)

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
                    f"Unable to create the '{name}' SharedMemoryArray object, as an object with this name already "
                    f"exists. If this method is called from a child process, use the connect() method instead "
                    f"to connect to the existing buffer. To clean-up the buffer left over from a previous "
                    f"runtime, run this method with the 'exists_ok' flag set to True."
                )
                console.error(message=message, error=FileExistsError)

                # Fallback to appease mypy, should not be reachable.
                raise FileExistsError(message) from None  # pragma: no cover

        # Instantiates a numpy array using the shared memory buffer and copies the prototype array data into the shared
        # array instance.
        shared_array: NDArray[Any] = np.ndarray(shape=prototype.shape, dtype=prototype.dtype, buffer=buffer.buf)
        shared_array[:] = prototype[:]

        # Packages the data necessary to connect to the shared array into the class instance.
        shared_memory_array = cls(
            name=name,
            shape=shared_array.shape,
            datatype=shared_array.dtype,
        )

        # Connects the created instance to the shared memory buffer.
        shared_memory_array.connect()

        # Returns the instantiated and connected class object to caller.
        return shared_memory_array

    def connect(self) -> None:
        """Connects to the shared memory buffer, allowing the instance to access and manipulate the shared data.

        This method should be called once for each Python process that uses this instance, before calling any other
        methods.
        """
        if not self._connected and self._buffer is None:
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
        if self._connected and self._buffer is not None:
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
        if self._buffer is not None:
            # If the instance is connected to the buffer, first disconnects it from the buffer
            self.disconnect()

            # Requests the shared memory buffer to be destroyed
            self._buffer.unlink()

            # Releases the reference to the shared memory buffer
            self._buffer = None

    @contextmanager
    def array(self, *, with_lock: bool = True) -> Generator[NDArray[Any], None, None]:
        """Returns a context manager for accessing the managed shared memory array with optional locking.

        This method provides direct access to the underlying NumPy array through a context manager, with optional
        multiprocessing lock acquisition. It is recommended to call this method from a 'with' statement to ensure
        the proper lock acquisition and release.

        Args:
            with_lock: Determines whether to acquire the multiprocessing Lock before accessing the array. Acquiring
                the lock prevents collisions with other Python processes trying to simultaneously access the array's
                data.

        Notes:
            When with_lock=True (default), the lock is held for the entire duration of the context. Keep operations
            concise to avoid blocking other processes. When with_lock=False, ensure no other processes are writing to
            avoid race conditions and data corruption.

            The returned array is the actual shared array, not a copy. All modifications to the array are immediately
            visible to other processes.

        Yields:
            The shared NumPy array that can be directly manipulated using any NumPy operations. Changes made to this
            array directly affect the data stores in the shared memory buffer.

        Raises:
            ConnectionError: If the class instance is not connected to the shared memory buffer.
        """
        # Ensures the class is connected to the shared memory buffer
        if not self._connected or self._array is None:
            message = (
                f"Unable to access the data stored in the {self.name} SharedMemoryArray instance, as the it is not "
                f"connected to the shared memory buffer. Call the connect() method prior to calling the array() method."
            )
            console.error(message=message, error=ConnectionError)
            # This line shouldn't be reached due to console.error, but included for type checking
            raise ConnectionError(message)  # pragma: no cover

        # Conditionally acquire lock based on with_lock parameter
        if with_lock:
            with self._lock:
                yield self._array
        else:
            yield self._array

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
        """Returns True if the instance is connected to the shared memory buffer that stores the array data."""
        return self._connected
