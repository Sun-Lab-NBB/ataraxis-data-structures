import numpy as np
from ataraxis_data_structures import SharedMemoryArray

# Initializes a SharedMemoryArray
prototype = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
buffer_name = "unique_buffer"
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)
sma.connect()

# The SharedMemoryArray data can be accessed directly using indexing or slicing, just like any regular NumPy array or
# Python iterable:

# Index
assert sma[2] == np.uint64(3)
assert isinstance(sma[2], np.uint64)
sma[2] = 123  # Written data must be convertible to the datatype of the underlying NumPy array
assert sma[2] == np.uint64(123)

# Slice
assert np.array_equal(sma[:4], np.array([1, 2, 123, 4], dtype=np.uint64))
assert isinstance(sma[:4], np.ndarray)

# It is also possible to directly access the underlying NumPy array, which allows using the full range of NumPy
# operations. The accessor method can be used from within a context manager to enforce exclusive access to the array's
# data via an internal multiprocessing lock mechanism:
with sma.array(with_lock=True) as array:
    print(f"Before clipping: {array}")

    # Clipping replaces the out-of-bounds value '123' with '10'.
    array = np.clip(array, 0, 10)

    print(f"After clipping: {array}")

# Cleans up the array buffer
sma.disconnect()
sma.destroy()
