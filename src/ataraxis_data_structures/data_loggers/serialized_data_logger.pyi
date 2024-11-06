from pathlib import Path
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

from ..shared_memory import SharedMemoryArray as SharedMemoryArray

class DataLogger:
    """Saves input data as uncompressed byte numpy array (.npy) files using the requested number of cores and threads.

    This class instantiates and manages the runtime of a logger spread over the requested number of cores and threads.
    The class exposes a shared multiprocessing Queue via the 'input_queue' property, which can be used to buffer and
    pipe the data to the processes used by the logger. The class expects the data to consist of 4 elements: the ID
    code of the source (integer), the count of the data object for that source (integer), the acquisition timestamp
    (integer) and the serialized data to log (the array of bytes (np.uint8)).

    Initializing the class does not start the logger processes! Call start() method to initialize the logger processes.

    Notes:
        Do not instantiate more than a single instance of DataLogger class at a time, as the second initialization will
        fail! Instead, tweak the number of processes and threads as necessary to comply with the load and share the
        input_queue of the initialized DataLogger with all other classes that need to log serialized data. For most use
        cases, using a single process (core) with 5-10 threads will be enough to prevent the buffer from filling up.
        For demanding runtimes, you can increase the number of cores as necessary to comply with the demand.

        This class will log data into the same directory to allow for the most efficient post-runtime compression.
        Since all arrays are saved using the source_id as part of the filename, it is possible to demix the data based
        on its source. Additionally, the order in which the data came is also preserved in resultant file names,
        allowing to demix the data based on the time it was logged (not to mention that all timestamps are also
        preserved inside the array).

    Args:
        output_directory: The directory where the log folder will be created.
        process_count: The number of processes to use for logging data.
        thread_count: The number of threads to use for logging data. Note, this number of threads will be created for
            each process.
        sleep_timer: The time in microseconds to delay between polling the queue. This parameter may help with managing
            the power and thermal load of the cores assigned to the data logger by temporarily suspending their
            activity. It is likely that delays below 1 millisecond (1000 microseconds) will not produce a measurable
            impact, as the cores execute a 'busy' wait sequence for very short delay periods. Set this argument to 0 to
            disable delays entirely.

    Attributes:
        _process_count: The number of processes to use for logging data.
        _thread_count: The number of threads to use for logging data. Note, this number of threads will be created for
            each process.
        _sleep_timer: The time in microseconds to delay between polling the queue.
        _output_directory: The directory where the log folder will be created.
        _mp_manager: A manager object used to instantiate and manage the multiprocessing Queue.
        _input_queue: The multiprocessing Queue used to buffer and pipe the data to the logger processes.
        _terminator_array: A shared memory array used to terminate (shut down) the logger processes.
        _logger_processes: A tuple of Process objects, each representing a logger process.
        _started: A boolean flag used to track whether Logger processes are running.
        _expired: This flag is used to ensure that start-shutdown cycle can only be performed once.
    """

    _process_count: Incomplete
    _thread_count: Incomplete
    _sleep_timer: Incomplete
    _output_directory: Incomplete
    _mp_manager: Incomplete
    _input_queue: Incomplete
    _terminator_array: Incomplete
    _logger_processes: Incomplete
    _started: bool
    def __init__(
        self, output_directory: Path, process_count: int = 1, thread_count: int = 5, sleep_timer: int = 5000
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the DataLogger instance."""
    def __del__(self) -> None:
        """Ensures that logger resources are properly released when the class is garbage collected."""
    def start(self) -> None:
        """Starts the logger processes."""
    def shutdown(self) -> None:
        """Stops the logger processes once they save all buffered data and releases reserved resources."""
    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:
        """Thread worker function that saves the data.

        Since data saving is primarily IO-bound, using multiple threads per each Process is likely to achieve the best
        saving performance.
        """
    @staticmethod
    def _log_cycle(
        input_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        output_directory: Path,
        thread_count: int,
        sleep_time: int = 1000,
    ) -> None:
        """The function passed to Process classes to log the data.

        This function sets up the necessary assets (threads and queues) to accept, preprocess, and save the input data
        as .npy files.

        Args:
            input_queue: The multiprocessing Queue object used to buffer and pipe the data to the logger processes.
            terminator_array: A shared memory array used to terminate (shut down) the logger processes.
            output_directory: The path to the directory where to save the data.
            thread_count: The number of threads to use for logging.
            sleep_time: The time in microseconds to delay between polling the queue once it has been emptied. If the
                queue is not empty, this process will not sleep.
        """
    def compress_logs(self, remove_sources: bool = False, verbose: bool = False) -> None:
        """Consolidates all .npy files in the log directory into a single compressed .npz file.

        Individual .npy files are grouped by source_id and acquisition number.

        Args:
            remove_sources: Determines whether to remove the individual .npy files after they have been consolidated
                into .npz archives.
            verbose: Determines whether to print processed arrays to console. This option is mostly useful for debugging
                other Ataraxis libraries and should be disabled by default.
        """
    @property
    def input_queue(self) -> MPQueue:
        """Returns the multiprocessing Queue used to buffer and pipe the data to the logger processes.

        Share this queue with all source processes that need to log data.
        """
