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
    entries have to be packed into this class instance and submitted (put) into the logger input queue exposed by the
    DataLogger class.

    Notes:
        This class is optimized for working with other Ataraxis libraries. It expects the time to come from
        ataraxis-time (PrecisionTimer) and other data from Ataraxis libraries designed to interface with various
        hardware.
    """

    source_id: np.uint8
    time_stamp: np.uint64
    serialized_data: NDArray[np.uint8]
    def get_data(self) -> tuple[str, NDArray[np.uint8]]:
        """Constructs and returns the filename and the serialized data package to be logged.

        Returns:
            A tuple of two elements. The first element is the name to use for the log file, which consists of
            zero-padded source id and zero-padded time stamp, separated by an underscore. The second element is the
            data to be logged as a one-dimensional bytes numpy array. The logged data includes the original data object
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

        Tweak the number of processes and threads as necessary to keep up with the load and share the input_queue of the
        initialized DataLogger with all classes that need to log serialized data. For most use cases, using a
        single process (core) with 5-10 threads will be enough to prevent the buffer from filling up.
        For demanding runtimes, you can increase the number of cores as necessary to keep up with the demand.

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
        exist_ok: Determines how the class behaves if a SharedMemory buffer with the same name as the one used by the
            class already exists. If this argument is set to True, the class will destroy the existing buffer and
            make a new buffer for itself. If the class is used correctly, the only case where a buffer would already
            exist is if the class ran into an error during the previous runtime, so setting this to True should be
            safe for most runtimes.

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
        _exist_ok: Determines how the class handles already existing shared memory buffer errors.
    """

    _started: bool
    _process_count: Incomplete
    _thread_count: Incomplete
    _sleep_timer: Incomplete
    _name: Incomplete
    _exist_ok: Incomplete
    _output_directory: Incomplete
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
        exist_ok: bool = False,
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
    def _compress_source(self, source_id: int, files: list[Path], remove_sources: bool, mem_map: bool) -> int:
        """Compresses all log entries for a single source into a single .npz archive.

        This helper function is used by the compress_log() method to compress all available sources in-parallel to
        improve runtime efficiency.

        Notes:
            If this function is instructed to remove source files, deletes individual .npy files after compressing them
            as .npz archive. The method ensures that all compressed entries match source entries, before deleting source
            .npy files. Therefore, it is always safe to delete source files.

        Args:
            source_id: The ID-code for the source, whose logs are compressed by this function.
            files: The list of paths to the .npy log files of the processed source.
            remove_sources: Determines whether to remove original .npy files after generating the compressed .npz
                archive.
            mem_map: Determines whether to load all original entries into memory (RAM) before compression or whether to
                load them via memory-mapping. Setting this to False is faster, but may lead to error when processing
                many large datasets in-parallel.

        Raises:
            ValueError: If the function is instructed to delete source files and one of the compressed entries does not
                match the original source entry. This indicates that compression altered the original data.
        """
    def compress_logs(
        self,
        remove_sources: bool = False,
        memory_mapping: bool = True,
        verbose: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """Consolidates all .npy files in the log directory into a single compressed .npz archive for each source_id.

        All entries within each source are grouped by their acquisition timestamp value before compression. The
        compressed archive names include the ID code of the source that generated original log entries

        Notes:
            To improve runtime efficiency, the method processes all log sources in-parallel, using multithreading.
            The exact number of parallel threads used by the method depends on the number of available CPU cores. This
            number can be further adjusting by modifying the max_workers argument.

            This method requires all data from the same source to be loaded into RAM before it is added to the .npz
            archive. While this should not be a problem for most use cases, it may lead to out-of-memory errors. To
            avoid this, the method uses memory-mapping by default, to reduce the RAM requirements. If your machine
            has sufficient RAM, disable this by setting the memory_mapping argument to False.

        Args:
            remove_sources: Determines whether to remove the individual .npy files after they have been consolidated
                into .npz archives. The method ensures that all compressed entries match the original entries before
                deleting source files, so this option is safe for all use cases.
            memory_mapping: Determines whether the method uses memory-mapping (disk) to stage the data before
                compression or loads all data into RAM. Disabling this option makes the method considerably faster, but
                may lead to out-of-memory errors in certain use cases. Note, due to collisions with Windows not
                releasing memory-mapped files, this argument does not do anything on Windows.
            verbose: Determines whether to print compression progress to terminal. Due to a generally fast compression
                time, this option is generally not needed for most runtimes.
            max_workers: Determines the number of threads use to process logs entries from different sources
                in-parallel. If set to None, the method uses the number of CPU cores - 4 threads.
        """
    @property
    def input_queue(self) -> MPQueue:
        """Returns the multiprocessing Queue used to buffer and pipe the data to the logger processes.

        Share this queue with all source processes that need to log data. To ensure correct data packaging, package the
        data using the LogPackage class exposed by this library before putting it into the queue.
        """
    @property
    def name(self) -> str:
        """Returns the name of the DataLogger instance."""
    @property
    def started(self) -> bool:
        """Returns True if the DataLogger has been started and is actively logging data."""
    @property
    def output_directory(self) -> Path:
        """Returns the path to the directory where the data is saved."""
