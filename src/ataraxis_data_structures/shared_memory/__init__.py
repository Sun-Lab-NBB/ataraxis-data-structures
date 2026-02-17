"""Provides assets for transferring data between multiple Python processes via a shared NumPy array memory buffer."""

from .shared_memory_array import SharedMemoryArray

__all__ = ["SharedMemoryArray"]
