# This example demonstrates the use of SharedMemoryArray in a multiprocessing context.

from multiprocessing import Process
from ataraxis_base_utilities import console
from ataraxis_time import PrecisionTimer
import numpy as np
from ataraxis_data_structures import SharedMemoryArray


def concurrent_worker(shared_memory_object: SharedMemoryArray, index: int) -> None:
    """This worker runs in a remote process.

    It increments the shared memory array variable by 1 if the variable is even. Since each increment shifts it to be
    odd, to work as intended, this process has to work together with a different process that increments odd values.
    The process shuts down once the value reaches 200.

    Args:
        shared_memory_object: The SharedMemoryArray instance to work with.
        index: The index inside the array to increment
    """
    # Connects to the array
    shared_memory_object.connect()

    # Runs until the value becomes 200
    while shared_memory_object[index] < 200:

        # Reads data from the input index
        shared_value = shared_memory_object[index]

        # Checks if the value is even and below 200
        if shared_value % 2 == 0 and shared_value < 200:

            # Increments the value by one and writes it back to the array
            shared_memory_object[index] = shared_value + 1

    # Disconnects and terminates the process
    shared_memory_object.disconnect()


if __name__ == "__main__":

    console.enable()  # Enables terminal printouts

    # Initializes a SharedMemoryArray
    sma = SharedMemoryArray.create_array("test_concurrent", np.zeros(5, dtype=np.int32))

    # Generates multiple processes and uses each to repeatedly write and read data from different indices of the same
    # array.
    processes = [Process(target=concurrent_worker, args=(sma, i)) for i in range(5)]
    for p in processes:
        p.start()

    # Finishes setting up the local array instance by connecting to the shared memory buffer and enabling the shared
    # memory buffer cleanup when the instance is garbage-collected (a safety feature).
    sma.connect()
    sma.enable_buffer_destruction()

    # Marks the beginning of the test runtime
    console.echo(f"Running the multiprocessing example on {len(processes)} processes...")
    timer = PrecisionTimer("ms")
    timer.reset()

    # For each of the array indices, increments the value of the index if it is odd. Child processes increment even
    # values and ignore odd ones, so the only way for this code to finish is if children and parent process take
    # turns incrementing shared values until they reach 200
    while np.any(sma[0:5] < 200):  # Runs as long as any value is below 200

        # Note, while it is possible to index the data from the SharedMemoryArray, it is also possible to retrieve
        # and manipulate the underlying NumPy array directly. This allows using the full range of NumPy operations
        # on the shared memory data:
        with sma.array(with_lock=True) as arr:
            mask = (arr % 2 != 0) & (arr < 200)  # Uses a boolean mask to discover odd values below 200
            arr[mask] += 1  # Increments only the values that meet the condition above

    # Waits for the processes to join
    for p in processes:
        p.join()

    # Verifies that all processes ran as expected and incremented their respective variable
    assert np.all(sma[0:5] == 200)

    # Marks the end of the test runtime.
    time_taken = timer.elapsed
    console.echo(f"Example runtime: complete. Time taken: {time_taken/1000:.2f} seconds.")

    # Cleans up the shared memory array after all processes are terminated
    sma.disconnect()
    sma.destroy()
