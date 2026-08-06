"""Provides the SharedMemoryArray class that allows moving data between multiple Python processes through a shared
n-dimensional NumPy array memory buffer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from weakref import WeakValueDictionary
from contextlib import suppress, contextmanager
from multiprocessing import get_context
from multiprocessing.context import get_spawning_popen
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing import synchronize
    from multiprocessing.context import SpawnContext

    from numpy.typing import NDArray


_MULTIPROCESSING_CONTEXT: SpawnContext = get_context("spawn")
"""The spawn-based multiprocessing context used to create all synchronization primitives for SharedMemoryArray
instances, ensuring identical cross-process behavior on all supported platforms."""

_BUFFER_OWNERS: WeakValueDictionary[str, SharedMemoryArray] = WeakValueDictionary()
"""Maps each shared memory buffer name to the instance responsible for destroying that buffer in this process.

Destroying a buffer addresses it by name rather than by identity, so an instance whose name was rebound by a later
create_array() call would otherwise unlink the replacement buffer instead of its own. This mapping is what lets that
call strip the destruction right from the previous owner. The references are weak, so membership here never keeps an
instance alive past its last strong reference.
"""


class SharedMemoryArray:
    """Wraps a NumPy n-dimensional array object and exposes methods for accessing the array's data from multiple
    different Python processes.

    During initialization, this class creates a persistent memory buffer to which it connects from different Python
    processes. The data inside the buffer is accessed via an n-dimensional NumPy array with optional locking to prevent
    race conditions.

    Notes:
        Supports instantiation inside the main runtime thread via the create_array() method alone. Do not attempt to
        instantiate the class manually. The create_array() method returns an instance already connected to the shared
        memory buffer. Each process that receives the instance connects as part of the transfer, so no process calls
        connect() to reach the array data.

        The creating process destroys the shared memory buffer when its instance is garbage-collected. Every process
        that receives the instance only disconnects from the buffer, which leaves the creating process solely
        responsible for the buffer's lifetime.

        Shared memory buffers are garbage-collected differently depending on the host Operating System. On Windows,
        garbage collection is handed off to the OS and cannot be enforced manually. On Unix (macOS and Linux), the
        buffer can be garbage-collected by calling the destroy() method.

    Args:
        name: The unique name to use for the shared memory buffer.
        shape: The shape of the NumPy array used to access the data in the shared memory buffer.
        datatype: The datatype of the NumPy array used to access the data in the shared memory buffer.
        buffer: The SharedMemory buffer that stores the shared data.
        auto_connect: Determines whether the processes that receive this instance connect to the shared memory buffer
            as part of unpickling it.

    Attributes:
        _name: Stores the name of the shared memory buffer.
        _shape: Stores the shape of the NumPy array used to access the buffered data.
        _datatype: Stores the datatype of the NumPy array used to access the buffered data.
        _buffer: Stores the Shared Memory buffer object.
        _lock: Stores the Lock object used to prevent multiple processes from working with the shared data at the same
            time.
        _array: Stores the NumPy array used to interface with the data stored in the shared memory buffer.
        _connected: Determines whether the instance is connected to the shared memory buffer.
        _destroy_buffer: Determines whether the shared memory buffer is destroyed when this instance is
            garbage-collected.
        _auto_connect: Determines whether the processes that receive this instance connect to the shared memory buffer
            automatically.
    """

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        datatype: np.dtype[Any],
        buffer: SharedMemory,
        *,
        auto_connect: bool = True,
    ) -> None:
        """Initializes the SharedMemoryArray instance from data prepared by the create_array() method."""
        # The create_array() class method is the actual constructor, so __init__ only stores the precomputed values.
        self._name: str = name
        self._shape: tuple[int, ...] = shape
        self._datatype: np.dtype[Any] = datatype
        self._buffer: SharedMemory | None = buffer
        self._lock: synchronize.Lock = _MULTIPROCESSING_CONTEXT.Lock()
        self._array: NDArray[Any] | None = None
        self._connected: bool = False
        # Only create_array() reaches this path, so the instance built here is always the creating process's own.
        # __getstate__ clears the guard on every transferred copy, which leaves this instance the sole destroyer.
        self._destroy_buffer: bool = True
        self._auto_connect: bool = auto_connect

        # Claims the buffer name, which lets a later create_array() call over the same name find this instance and
        # strip its destruction right before the name comes to refer to a different buffer.
        _BUFFER_OWNERS[name] = self

        # The buffer handed in by create_array() is already open, so the view is bound here rather than through
        # connect(). Connecting would open a second handle to the same buffer and leave the first one for the garbage
        # collector to close at an arbitrary later point. This is what makes create_array() return a connected
        # instance. Unpickling restores the attributes through __setstate__ instead, so it never reaches this path.
        self._bind_array()

    def __repr__(self) -> str:
        """Returns a string representation of the SharedMemoryArray instance."""
        return (
            f"SharedMemoryArray(name='{self._name}', shape={self._shape}, datatype={self._datatype}, "
            f"connected={self.is_connected})"
        )

    def __del__(self) -> None:
        """Ensures that the shared memory buffer is released when the instance is garbage-collected."""
        if self._destroy_buffer:
            self.destroy()
        else:
            self.disconnect()

    def __getstate__(self) -> dict[str, Any]:
        """Returns a picklable representation of the instance for transfer to other Python processes.

        Excludes the live shared memory handle, which each process rebuilds by connecting to the shared buffer. The
        state always reports the instance as disconnected, ensuring every process establishes its own connection
        regardless of the originating instance's connection state.

        Raises:
            RuntimeError: If the instance is serialized outside of starting a Python process, which is what sending
                it through a Queue or a Pipe does.
        """
        # The instance's lock reaches another process only by inheritance, which multiprocessing arranges while it
        # starts that process. Serializing at any other point produces a copy whose lock synchronizes nothing, so the
        # attempt is refused here. Leaving it to the lock reports a failure that names neither this class nor the
        # transfer that works, and a Queue defers that report to its feeder thread, where it strands the consumer.
        if get_spawning_popen() is None:
            message = (
                f"Unable to transfer the '{self._name}' SharedMemoryArray instance to another process, as the "
                f"transfer is not part of starting that process. Pass the instance through the 'args' argument of a "
                f"Process or the 'initargs' argument of a process pool, both of which start a process and therefore "
                f"inherit the instance's lock. A Queue or a Pipe cannot carry the instance, and it does not need to, "
                f"since every process connected to the array already shares its data through the memory buffer."
            )
            console.error(message=message, error=RuntimeError)

        state = self.__dict__.copy()
        # The NumPy view into the buffer is bound to the originating process. Excluding it also avoids copying the
        # entire array payload during transfer.
        state["_buffer"] = None
        state["_array"] = None
        state["_connected"] = False
        state["_destroy_buffer"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restores the instance state after it is transferred to another Python process.

        Raises:
            FileNotFoundError: If the instance was created with the ``auto_connect`` flag and the shared memory buffer
                with the instance's name does not exist.
        """
        self.__dict__.update(state)

        # Connects while the instance is still being unpickled, which is the earliest point at which the receiving
        # process is able to reach the shared buffer. Connecting here also covers the processes that never run
        # library code of their own, such as the workers a process pool starts from an initializer.
        if self._auto_connect:
            self.connect()

    def __getitem__(self, index: int | slice) -> Any:
        """Gets value(s) at the specified array index or slice with automatic locking.

        Notes:
            Always acquires the lock for thread-safe access. Use the array() method with the appropriate locking
            configuration to read or write the data without locking.

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

        # Takes the lock directly rather than through the array() context manager, whose generator, wrapper object,
        # and repeated connection guard dominate the cost of a single element access. The lock acquisition, the
        # returned values, and the guard above are the same either way.
        with self._lock:
            # Returns a copy to prevent external modifications to the returned data from affecting the shared array
            # without going through __setitem__.
            result = self._array[index]
            if isinstance(result, np.ndarray):
                return result.copy()
            return result

    def __setitem__(self, index: int | slice, value: Any) -> None:
        """Sets value(s) at the specified array index or slice with automatic locking.

        Notes:
            The input values are saved in the underlying NumPy n-dimensional array. If the values are not
            compatible with the array's datatype, they are converted to the array's datatype before being written.

            Always acquires the lock for thread-safe access. Use the array() method with the appropriate locking
            configuration to read or write the data without locking.

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

        # Takes the lock directly, for the reason given in __getitem__.
        with self._lock:
            self._array[index] = value

    @classmethod
    def create_array(
        cls,
        name: str,
        prototype: NDArray[Any],
        *,
        exists_ok: bool = False,
        auto_connect: bool = True,
    ) -> SharedMemoryArray:
        """Creates a SharedMemoryArray instance using the input prototype NumPy array.

        Notes:
            Applies only when the array is first created in the main runtime thread (scope). The returned instance is
            already connected to the shared memory buffer, and every process it is passed to connects as part of the
            transfer, so no process calls connect() to reach the array data.

            The calling process destroys the shared memory buffer when the returned instance is garbage-collected. The
            instance therefore has to stay referenced for as long as any process still uses the buffer, and calling
            destroy() ends the buffer's lifetime ahead of that point.

        Args:
            name: The unique name to use for the shared memory buffer.
            prototype: The prototype NumPy array instance for the created SharedMemoryArray.
            exists_ok: Determines how the method handles the case where the shared memory buffer with the same name
                already exists. If False, the method raises an exception. If True, the method unlinks the existing
                buffer and creates a new buffer using the input name and prototype data.
            auto_connect: Determines whether the processes that receive the created instance connect to the shared
                memory buffer as part of unpickling it. Disabling this defers the connection to each receiving
                process's own connect() call, which keeps a buffer that is destroyed between the spawn and the
                worker's first access from failing the worker's startup.

        Returns:
            The created and connected SharedMemoryArray instance.

        Raises:
            TypeError: If the input prototype is not a NumPy array.
            ValueError: If the input prototype is an empty NumPy array, as a shared memory buffer cannot be created
                with a size of zero bytes.
            FileExistsError: If a shared memory object with the same name as the input ``name`` argument value already
                exists and the ``exists_ok`` flag is False. Also raised if the ``exists_ok`` flag is True and the
                buffer name is still claimed by an open handle after the existing buffer is unlinked, which is the
                state Windows leaves behind while any handle to the buffer remains open.
            IndexError: If the input prototype is a zero-dimensional NumPy array, which passes the type check above
                and then fails the slice assignment that fills the buffer. The buffer is released before the error
                propagates, so the buffer name is left free for a later call to claim.
        """
        if not isinstance(prototype, np.ndarray):
            message = (
                f"Unable to create the '{name}' SharedMemoryArray object using the provided prototype. The "
                f"'prototype' argument must be a NumPy array, but got {type(prototype).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Creates the shared memory buffer. Raises FileExistsError if an object with this name already exists.
        try:
            buffer: SharedMemory = SharedMemory(name=name, create=True, size=prototype.nbytes)
        except FileExistsError:
            if exists_ok:
                # Strips the destruction right from the instance that holds this name in this process, if any. The
                # unlink below leaves that instance pointing at a buffer that no longer exists, so letting it keep
                # the right would have it unlink the replacement buffer when it is garbage-collected.
                cls._revoke_buffer_ownership(name=name)

                SharedMemory(name=name, create=False).unlink()

                # Recreates the shared memory buffer using the freed buffer name. Unlinking frees the name outright on
                # Unix, while Windows keeps it claimed until the last handle to the buffer closes, so the recreation
                # is the step that reports an outstanding handle.
                try:
                    buffer = SharedMemory(name=name, create=True, size=prototype.nbytes)
                except FileExistsError:
                    message = (
                        f"Unable to recreate the '{name}' SharedMemoryArray object, as the shared memory buffer with "
                        f"this name is still held by an open handle. Windows destroys a buffer only once every handle "
                        f"to it is closed, so unlinking one that this runtime or another process still holds leaves "
                        f"the name claimed. Disconnect every SharedMemoryArray instance connected to this buffer, "
                        f"then call this method again."
                    )
                    console.error(message=message, error=FileExistsError)

            else:
                message = (
                    f"Unable to create the '{name}' SharedMemoryArray object, as an object with this name already "
                    f"exists. If this method is called from a child process, use the connect() method instead "
                    f"to connect to the existing buffer. To clean-up the buffer left over from a previous "
                    f"runtime, run this method with the 'exists_ok' flag set to True."
                )
                console.error(message=message, error=FileExistsError)

        # Instantiates a NumPy array using the shared memory buffer and copies the prototype array data into the shared
        # array instance. Releases the buffer if either step fails, since no instance exists yet to release it through
        # destroy() and the name would otherwise stay claimed for the rest of the runtime.
        try:
            shared_array: NDArray[Any] = np.ndarray(shape=prototype.shape, dtype=prototype.dtype, buffer=buffer.buf)
            shared_array[:] = prototype[:]
        except BaseException:
            buffer.close()
            buffer.unlink()
            raise

        # Packages the data necessary to work with the shared array into the class instance. Initialization binds the
        # instance to the buffer created above, so the calling process is able to access the data straight away.
        return cls(
            name=name,
            shape=shared_array.shape,
            datatype=shared_array.dtype,
            buffer=buffer,
            auto_connect=auto_connect,
        )

    def connect(self) -> None:
        """Guarantees that the instance is connected to the shared memory buffer by the time the call returns.

        Calling this method is optional for an array created through create_array(), as the creating process and every
        process that receives the instance are connected already, which makes the call a no-op. Calling it anyway
        establishes the connection at a point the caller chooses rather than leaving it implicit, and it reconnects an
        instance that called disconnect().

        Notes:
            An array created with the ``auto_connect`` flag disabled relies on each receiving process calling this
            method before it accesses the array data.

        Raises:
            FileNotFoundError: If the shared memory buffer with the instance's name does not exist. This typically
                indicates that the buffer has already been destroyed via the destroy() method.
        """
        if self._connected:
            return

        self._buffer = SharedMemory(name=self._name, create=False)
        self._bind_array()

    def disconnect(self) -> None:
        """Disconnects from the shared memory buffer, preventing the instance from accessing and manipulating the
        shared data.

        Applies to each Python process that no longer requires shared buffer access, and to a process's shutdown
        sequence. A process that exits releases its handle regardless, so the call bounds the handle's lifetime to the
        work that needs it rather than to the process.

        Notes:
            Releases the local reference to the shared memory buffer without destroying it, potentially enabling the
            buffer to be garbage-collected by the Operating System. Use the destroy() method on Unix-based Operating
            Systems to destroy the buffer.
        """
        if self._connected and self._buffer is not None:
            # Releases the NumPy view before closing the handle. The view maps the buffer's memory directly, so a view
            # left bound after the mapping is torn down reads freed memory and segfaults the interpreter.
            self._array = None
            self._connected = False
            self._buffer.close()

    def destroy(self) -> None:
        """Requests the instance's shared memory buffer to be destroyed.

        Applies only to a single call issued from the highest runtime scope. Calling this method while having
        SharedMemoryArray instances connected to the buffer leads to undefined behavior.

        Notes:
            On Windows, the underlying unlink() call has no effect, as the Operating System destroys the buffer only
            after every handle to it is closed. The method still disconnects this instance from the buffer and
            releases its local handle on all platforms, so the instance cannot access the array data until connect()
            is called again.

            Unlinking addresses the buffer by name. Recreating a name through the ``exists_ok`` flag therefore strips
            the destruction right from the instance that held it, which confines this method to the buffer the
            instance created. Two processes creating the same name concurrently fall outside that protection, so
            buffer names have to stay unique across every process that runs at the same time.
        """
        if self._buffer is not None:
            self.disconnect()

            # A missing segment means the buffer already reached the state this method exists to produce, which
            # happens when the interpreter's resource tracker reclaims it during an unclean shutdown. Reporting it
            # would add noise to an already correct outcome, and __del__ reaches this call on every creating instance.
            with suppress(FileNotFoundError):
                self._buffer.unlink()

            self._buffer = None

    @contextmanager
    def array(self, *, with_lock: bool = True) -> Generator[NDArray[Any], None, None]:
        """Returns a context manager for accessing the managed shared memory array with optional locking.

        Notes:
            When ``with_lock`` is True (default), the lock is held for the entire duration of the context. Keep
            operations concise to avoid blocking other processes. When ``with_lock`` is False, ensure no other
            processes are writing to avoid race conditions and data corruption.

            The returned array is the actual shared array, not a copy. All modifications to the array are immediately
            visible to other processes.

        Args:
            with_lock: Determines whether to acquire the multiprocessing Lock before accessing the array. Acquiring
                the lock prevents collisions with other Python processes trying to simultaneously access the array's
                data.

        Yields:
            The shared NumPy array that can be directly manipulated using any NumPy operations. Changes made to this
            array directly affect the data stored in the shared memory buffer.

        Raises:
            ConnectionError: If the class instance is not connected to the shared memory buffer.
        """
        if not self._connected or self._array is None:
            message = (
                f"Unable to access the data stored in the {self.name} SharedMemoryArray instance, as it is not "
                f"connected to the shared memory buffer. Call the connect() method prior to calling the array() method."
            )
            console.error(message=message, error=ConnectionError)

        if with_lock:
            with self._lock:
                yield self._array
        else:
            yield self._array

    @property
    def datatype(self) -> np.dtype[Any]:
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

    @classmethod
    def _revoke_buffer_ownership(cls, name: str) -> None:
        """Strips the destruction right from the instance that currently owns the input shared memory buffer name.

        Args:
            name: The name of the shared memory buffer whose owner is releasing its destruction right.
        """
        owner = _BUFFER_OWNERS.pop(name, None)
        if owner is not None:
            owner._destroy_buffer = False  # noqa: SLF001 - Revokes the right on a sibling instance of this class.

    def _bind_array(self) -> None:
        """Binds the internal NumPy view to the data stored in the instance's open shared memory buffer and marks the
        instance as connected.
        """
        # The callers of this method establish the open handle, so the guard only narrows the type for mypy.
        if self._buffer is None:  # pragma: no cover
            return

        self._array = np.ndarray(shape=self._shape, dtype=self._datatype, buffer=self._buffer.buf)
        self._connected = True
