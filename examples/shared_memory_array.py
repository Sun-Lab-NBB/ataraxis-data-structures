# This example demonstrates the use of SharedMemoryArray in a multiprocessing context.

from multiprocessing import Process

import numpy as np

from ataraxis_data_structures import SharedMemoryArray


def concurrent_worker(shared_memory_object: SharedMemoryArray, index: int) -> None:
    """This worker will run in a different process.

    It increments a shared memory array variable by 1 if the variable is even. Since each increment will
    shift it to be odd, to work as intended, this process has to work together with a different process that
    increments odd values. The process shuts down once the value reaches 200.

    Args:
        shared_memory_object: The SharedMemoryArray instance to work with.
        index: The index inside the array to increment
    """
    # Connects to the array
    shared_memory_object.connect()

    # Runs until the value becomes 200
    while shared_memory_object.read_data(index) < 200:
        # Reads data from the input index
        shared_value = shared_memory_object.read_data(index)

        # Checks if the value is even and below 200
        if shared_value % 2 == 0 and shared_value < 200:
            # Increments the value by one and writes it back to the array
            shared_memory_object.write_data(index, shared_value + 1)

    # Disconnects and terminates the process
    shared_memory_object.disconnect()


if __name__ == "__main__":
    # Initializes a SharedMemoryArray
    sma = SharedMemoryArray.create_array("test_concurrent", np.zeros(5, dtype=np.int32))

    # Generates multiple processes and uses each to repeatedly write and read data from different indices of the same
    # array.
    processes = [Process(target=concurrent_worker, args=(sma, i)) for i in range(5)]
    for p in processes:
        p.start()

    # For each of the array indices, increments the value of the index if it is odd. Child processes increment even
    # values and ignore odd ones, so the only way for this code to finish is if children and parent process take turns
    # incrementing shared values until they reach 200
    while np.any(sma.read_data((0, 5)) < 200):  # Runs as long as any value is below 200
        # Loops over addressable indices
        for i in range(5):
            value = sma.read_data(i)
            if value % 2 != 0 and value < 200:  # If the value is odd and below 200, increments the value by 1
                sma.write_data(i, value + 1)

    # Waits for the processes to join
    for p in processes:
        p.join()

    # Verifies that all processes ran as expected and incremented their respective variable
    assert np.all(sma.read_data((0, 5)) == 200)

    # Cleans up the shared memory array after all processes are terminated
    sma.disconnect()
    sma.destroy()
