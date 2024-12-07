from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

from ..shared_memory import SharedMemoryArray as SharedMemoryArray

@dataclass(frozen=True)
class LogPackage:
    """Stores the data and ID information to be logged by the DataLogger class and exposes methods for packaging this
    data into the format expected by the logger.

    This class collects, preprocesses, and stores the data to be logged by the DataLogger instance. To be logged,
    entries have to be packed into this class instance and submitted (put) into the input_queue exposes by the
    DataLogger class. All data to be logged has to come wrapped into this class instance!
    """

    source_id: int
    time_stamp: int
    serialized_data: NDArray[np.uint8]
    def get_data(self) -> tuple[str, NDArray[np.uint8]]:
        """Constructs and returns the filename and the serialized data package to be logged.

        Returns:
            A tuple of two elements. The first element is the name to use for the log file, which consists of
            zero-padded source id and zero-padded time stamp, separated by an underscore. The second element is the
            data to be logged as a one-dimensional bytes numpy array. THe logged data includes the original data object
            and the pre-pended source id and time stamp.
        """

class DataLogger:
    """Saves input data as an uncompressed byte numpy array (.npy) files using the requested number of cores and
    threads.

    This class instantiates and manages the runtime of a logger distributed over the requested number of cores and
    threads. The class exposes a shared multiprocessing Queue via the 'input_queue' property, which can be used to
    buffer and pipe the data to the logger from other Processes. The class expects the data to be first packaged into
    LogPackage class instance also available from this library, before it is sent to the logger via the queue object.

    Notes:
        Initializing the class does not start the logger processes! Call start() method to initialize the logger
        processes.

        Once the logger process(es) have been started, the class also initializes and maintains a watchdog thread that
        monitors the runtime status of the processes. If a process shuts down, the thread will detect this and raise
        the appropriate error to notify the user. Make sure the main process periodically releases GIL to allow the
        thread to assess the state of the remote process!

        This class is designed to only be instantiated once. However, for particularly demanding use cases with many
        data producers, the shared Queue may become the bottleneck. In this case, you can initialize multiple
        DataLogger instances, each using a unique instance_name argument.

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
        instance_name: The name of the data logger instance. Critically, this is the name used to initialize the
            SharedMemory buffer used to control the child processes, so it has to be unique across all other
            Ataraxis codebase instances that also use shared memory.
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
        _name: Stores the name of the data logger instance.
        _output_directory: The directory where the log folder will be created.
        _started: A boolean flag used to track whether Logger processes are running.
        _mp_manager: A manager object used to instantiate and manage the multiprocessing Queue.
        _input_queue: The multiprocessing Queue used to buffer and pipe the data to the logger processes.
        _logger_processes: A tuple of Process objects, each representing a logger process.
        _terminator_array: A shared memory array used to terminate (shut down) the logger processes.
        _watchdog_thread: A thread used to monitor the runtime status of remote logger processes.
    """

    _process_count: Incomplete
    _thread_count: Incomplete
    _sleep_timer: Incomplete
    _name: Incomplete
    _output_directory: Incomplete
    _started: bool
    _mp_manager: Incomplete
    _input_queue: Incomplete
    _terminator_array: Incomplete
    _logger_processes: Incomplete
    _watchdog_thread: Incomplete
    def __init__(
        self,
        output_directory: Path,
        instance_name: str = "data_logger",
        process_count: int = 1,
        thread_count: int = 5,
        sleep_timer: int = 5000,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the DataLogger instance."""
    def __del__(self) -> None:
        """Ensures that logger resources are properly released when the class is garbage collected."""
    def start(self) -> None:
        """Starts the logger processes and the assets used to control and ensure the processes are alive.

        Once this method is called, data submitted to the 'input_queue' of the class instance will be saved to disk via
        the started Processes.
        """
    def stop(self) -> None:
        """Stops the logger processes once they save all buffered data and releases reserved resources."""
    def _watchdog(self) -> None:
        """This function should be used by the watchdog thread to ensure the logger processes are alive during runtime.

        This function will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """
    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:
        """Thread worker function that saves the data.

        Args:
            filename: The name of the file to save the data to. Note, the name has to be suffix-less, as '.npy' suffix
                will be appended automatically.
            data: The data to be saved, packaged into a one-dimensional bytes array.

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
        for more efficient data processing and reduce the RAM requirements when compressing sizable chunks of data.

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

        Share this queue with all source processes that need to log data. To ensure correct data packaging, package the
        data using the LogPackage class exposed by this library before putting it into the queue.
        """
    def _vacate_shared_memory_buffer(self) -> None:
        """Clears the SharedMemory buffer that uses instance-specific name.

        While this method should not be needed when DataLogger used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer, preventing the DataLogger from being
        reinitialized. This method allows manually removing that buffer to reset the system.

        The method will not do anything if the shared memory buffer does not exist.
        """
    @property
    def name(self) -> str:
        """Returns the name of the DataLogger instance."""
    @property
    def started(self) -> bool:
        """Returns True if the DataLogger has been started and is actively logging data."""
