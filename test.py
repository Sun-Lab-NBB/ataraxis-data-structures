import numpy as np
import numpy.typing as npt
from src.ataraxis_data_structures.shared_memory import SharedMemoryArray

prototype = np.empty(shape=(10, 10), dtype=np.uint64)
x = SharedMemoryArray.create_array(name="test_array", prototype=prototype)
print(x)

ret = x.read_data(index=slice(1, 2), convert_output=True)
print(type(ret))
print(ret)
