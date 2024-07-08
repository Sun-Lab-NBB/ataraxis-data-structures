"""This package provides the SharedMemoryArray class that exposes methods for sharing data between multiple Python
processes through a shared c-buffer.

SharedMemoryArray works by creating multiple numpy array instances, one per each process, that share the same low-level
buffer. It is equipped with the necessary mechanisms to ensure thread- and process-safe data manipulation and
functions as an alternative to Queue objects.
"""

# Exposes SharedMemoryArray class for use in other modules
from .shared_memory_array import SharedMemoryArray

__all__ = ["SharedMemoryArray"]
