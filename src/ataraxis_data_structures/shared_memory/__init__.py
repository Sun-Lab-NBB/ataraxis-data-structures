"""This package provides the SharedMemoryArray class that exposes methods for transferring data between multiple Python
processes via a shared numpy array.

See shared_memory_array.py for more details on the class and its methods.
"""

from .shared_memory_array import SharedMemoryArray

__all__ = ["SharedMemoryArray"]
