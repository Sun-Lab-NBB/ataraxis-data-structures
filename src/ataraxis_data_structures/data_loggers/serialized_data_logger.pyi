from pathlib import Path
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray
from ataraxis_base_utilities import LogLevel as LogLevel

from ..shared_memory import SharedMemoryArray as SharedMemoryArray

class DataLogger:
    """Saves input data as uncompressed byte numpy array (.npy) files using the requested number of cores and threads.

    This class instantiates and manages the runtime of a logger distributed over the requested number of cores and
    threads. The class exposes a shared multiprocessing Queue via the 'input_queue' property, which can be used to
    buffer and pipe the data to the logger from other Processes. The class expects the data to consist of 4 elements:
    the ID code of the source (integer), the sequence position of the data object for that source (integer), the
    acquisition timestamp (integer) and the serialized data to log (the array of bytes (np.uint8)).

    Notes:
        Initializing the class does not start the logger processes! Call start() method to initialize the logger
        processes.

        Do not instantiate more than a single instance of DataLogger class at a time! Since this class uses
        SharedMemoryArray with a fixed name to terminate the controller Processes, only one DataLogger instance can
        exist at a given time. Delete the class instance if you need to recreate it for any reason.

        Tweak the number of processes and threads as necessary to comply with the load and share the input_queue of the
        initialized DataLogger with all other classes that need to log serialized data. For most use cases, using a
        single process (core) with 5-10 threads will be enough to prevent the buffer from filling up.
        For demanding runtimes, you can increase the number of cores as necessary to comply with the demand.

        This class will log data from all sources and Processes into the same directory to allow for the most efficient
        post-runtime compression. Since all arrays are saved using the source_id as part of the filename, it is possible
        to demix the data based on its source during post-processing. Additionally, the sequence numbers of logged
        arrays are also used in file names to aid sorting saved data.

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
        _process_count: The number of processes to use for data saving.
        _thread_count: The number of threads to use for data saving. Note, this number of threads will be created for
            each process.
        _sleep_timer: The time in microseconds to delay between polling the queue.
        _output_directory: The directory where the log folder will be created.
        _mp_manager: A manager object used to instantiate and manage the multiprocessing Queue.
        _input_queue: The multiprocessing Queue used to buffer and pipe the data to the logger processes.
        _terminator_array: A shared memory array used to terminate (shut down) the logger processes.
        _logger_processes: A tuple of Process objects, each representing a logger process.
        _started: A boolean flag used to track whether Logger processes are running.
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
    @staticmethod
    def _vacate_shared_memory_buffer() -> None:
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed when DataLogger used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer, preventing the DataLogger from being
        reinitialized. This method allows manually removing that buffer to reset the system.
        """
    def start(self) -> None:
        """Starts the logger processes.

        Once this method is called, data submitted to the 'input_queue' of the class instance will be saved to disk via
        the started Processes.
        """
    def shutdown(self) -> None:
        """Stops the logger processes once they save all buffered data and releases reserved resources."""
    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:
        """Thread worker function that saves the data.

        Args:
            filename: The name of the file to save the data to. Note, the name has to be suffix-less, as '.npy' suffix
                will be appended automatically.
            data: The data to be saved, packaged into one-dimensional bytes array.

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
        """Consolidates all .npy files in the log directory into a single compressed .npz archive for each source_id.

        Individual .npy files are grouped by acquisition number before being compressed. Sources are demixed to allow
        for more efficient data processing and reduce the RAM requirements when compressing sizeable chunks of data.

        Notes:
            This method requires all data from the same source to be loaded into RAM before it is added to the .npz
            archive. While this should not be a problem for most runtimes, you can modify this method to use memory
            mapping if your specific use circumstance runs into RAM issues.

            If 'verbose' flag is set to True, the method will enable the Console class to print data to the terminal.
            Overall, this flag should not be enabled together with other 'verbose' ataraxis runtimes, when possible.

        Args:
            remove_sources: Determines whether to remove the individual .npy files after they have been consolidated
                into .npz archives. Usually, this is a safe option that saves disk space.
            verbose: Determines whether to print processed arrays to console. This option is mostly useful for debugging
                other Ataraxis libraries and should be disabled by default.
        """
    @property
    def input_queue(self) -> MPQueue:
        """Returns the multiprocessing Queue used to buffer and pipe the data to the logger processes.

        Share this queue with all source processes that need to log data. Note, the queue expects the input data to be
        a 4-element tuple in the following order:
        -1 the ID code of the source (integer). Has to be unique across all systems that send data to be logged!
        -2 the acquisition number of the data object (integer). This helps track the order in which data was acquired.
        -3 the acquisition timestamp (integer). Tracks when the data was originally acquired.
        -4 the serialized data (A uint8 NumPy array). This has to be a one-dimensional array.

        """
