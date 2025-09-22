"""This module provides the DataLogger class to efficiently save (log) serialized data collected from different
Processes to disk.
"""

import os
import sys
from queue import Empty
from typing import TYPE_CHECKING, Any
from pathlib import Path
import platform
from functools import partial
from threading import Thread
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import console, ensure_directory_exists

from ..shared_memory import SharedMemoryArray

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager


def _load_numpy_files(
    file_paths: tuple[Path, ...], *, memory_map: bool = False
) -> tuple[tuple[str, ...], tuple[NDArray[Any], ...]]:
    """Loads multiple .npy files either into memory or as memory-mapped arrays.

    This service function is used during log compression to load all raw log files into memory in-parallel for faster
    processing.

    Args:
        file_paths: The paths to the .npy files to load.
        memory_map: Determines whether to memory-map the files or load them into memory (RAM).

    Returns:
        A tuple of two elements. The first element is a tuple of loaded file names (without extension). The second
        element is a tuple of loaded or memory-mapped data arrays.
    """
    mmap_mode = "r" if memory_map else None
    results = [(fp.stem, np.load(fp, mmap_mode=mmap_mode)) for fp in file_paths]
    return tuple(zip(*results, strict=False)) if results else ((), ())


def _load_numpy_archive(file_path: Path) -> dict[str, NDArray[Any]]:
    """Loads a numpy .npz archive containing multiple arrays as a dictionary.

    This service function is used during compressed log verification to load all entries from a compressed log archive
    into memory in-parallel.

    Args:
        file_path: The path to the .npz log archive to load.

    Returns:
        A dictionary that uses log entry names as keys and serialized log entry data (stored in NumPy arrays) as values.
    """
    # Loads the data and formats it as a dictionary before returning it to caller
    with np.load(file_path) as npz_data:
        return {key: npz_data[key] for key in npz_data.files}


def _assemble_archive(
    output_directory: Path,
    source_id: int,
    source_data: dict[str, NDArray[Any]],
) -> tuple[int, Path]:
    """Assembles all log entries for a single source (producer) into a single .npz archive.

    This helper function is used during log archive generation to assemble multiple archives in parallel.

    Args:
        output_directory: The path to the directory where to create the log archive.
        source_id: The ID-code of the source whose data is assembled into an archive.
        source_data: A dictionary that uses log-entries (entry names) as keys and stores the source data as NumPy
            array values.

    Returns:
        A tuple of two elements. The first element is the archive file stem (file name without extension). The second
        element is the path to the compressed log file.
    """
    # Computes the output path
    output_path = output_directory.joinpath(f"{source_id}_log.npz")

    # Assembles all source data into an uncompressed .npz archive.
    np.savez(output_path, **source_data)

    # Returns the source id and the path to the compressed log file to caller.
    return source_id, output_path


def _compare_arrays(source_id: int, stem: str, original_array: NDArray[Any], archived_array: NDArray[Any]) -> None:
    """Compares a pair of NumPy arrays for exact equality.

    This service function is used during log verification to compare source and archived log entry data in-parallel.

    Args:
        source_id: The ID-code of the source whose data is verified by this function.
        stem: The file name of the archived log entry being verified.
        original_array: The log entry data from the source .npy file.
        archived_array: The log entry data array from the .npz archive.

    Raises:
        ValueError: If the arrays do not match.
    """
    if not np.array_equal(original_array, archived_array):
        message = f"Data integrity check failed for source {source_id}, entry {stem}."
        console.error(message=message, error=ValueError)
        # Fallback to appease mypy, should not be reachable.
        raise ValueError(message)  # pragma: no cover


def assemble_log_archives(
    log_directory: Path,
    max_workers: int | None = None,
    *,
    remove_sources: bool = True,
    memory_mapping: bool = True,
    verbose: bool = False,
    verify_integrity: bool = False,
) -> None:
    """Consolidates all .npy files in the target log directory into .npz archives, one for each unique source.

    This function is designed to post-process the directories filled by DataLogger instances during runtime.

    Notes:
        All log entries inside each archive are grouped by their acquisition timestamp value before consolidation. The
        consolidated archive names include the ID code of the source that generated the original log entries.

    Args:
        log_directory: The path to the directory that stores the log entries as .npy files.
        max_workers: Determines the number of threads used to process the data in parallel. If set to None, the
            function uses the number of CPU cores - 2 threads.
        remove_sources: Determines whether to remove the .npy files after consolidating their data into .npz archives.
        memory_mapping: Determines whether to memory-map or loads the processed data into RAM during processing. Due to
            Windows not releasing memory-mapped file handles, this function always loads the data into RAM when running
            on Windows.
        verbose: Determines whether to communicate the log assembly progress via the terminal.
        verify_integrity: Determines whether to verify the integrity of the created archives against the original log
            entries before removing sources.
    """
    # Resolves the number of threads and processes to use during runtime.
    max_workers = max(1, (os.cpu_count() - 2) if max_workers is None else max_workers)

    # Due to erratic interaction between memory mapping and Windows (as always), disables memory mapping on
    # Windows. Use max_workers to avoid out-of-memory errors on Windows.
    memory_mapping = memory_mapping and platform.system() != "Windows"

    # Collects all .npy files and groups them by source_id
    source_files: dict[int, list[Path]] = defaultdict(list)
    for file_path in log_directory.rglob("*.npy"):
        source_id = int(file_path.stem.split("_")[0])
        source_files[source_id].append(file_path)

    # Sorts files within each source_id group by their integer-convertible timestamp
    for files in source_files.values():
        files.sort(key=lambda x: int(x.stem.split("_")[1]))

    # Initiates log processing. Since some steps of log processes are more efficiently executed via multithreading and
    # others via multiprocessing, uses both process and thread pool executors to efficiently process the data.
    with (
        ProcessPoolExecutor(max_workers=max_workers) as p_executor,
        ThreadPoolExecutor(max_workers=max_workers) as t_executor,
    ):
        # PHASE 1: Loads source files in parallel batches.
        total_files = sum(len(files) for files in source_files.values())
        loaded_data: dict[int, dict[str, NDArray[Any]]] = {source_id: {} for source_id in source_files}

        # Over-batches the data with the scale number of 4 to optimize runtime performance.
        load_numpy = partial(_load_numpy_files, memory_map=memory_mapping)
        batch_size = int(np.ceil(total_files / max_workers * 4))

        # Groups files into batches across all sources.
        load_futures = [
            (source_id, p_executor.submit(load_numpy, file_batch))
            for source_id, files in source_files.items()
            for i in range(0, len(files), batch_size)
            for file_batch in [tuple(files[i : i + batch_size])]
        ]

        # Waits for all log entries to be loaded
        with tqdm(
            total=total_files,
            desc="Loading log entry data into memory",
            unit="entries",
            disable=not verbose,
        ) as pbar:
            for source_id, future in load_futures:
                stems, arrays = future.result()
                for stem, array in zip(stems, arrays, strict=False):
                    loaded_data[source_id][stem] = array
                    pbar.update(1)

        # PHASE 2: Assembles archives. Here, each archive is processed in-parallel, but all archive log entries for
        # each archive are processed sequentially.
        assemble = partial(_assemble_archive, log_directory)
        # noinspection PyTypeChecker
        archive_futures = {
            p_executor.submit(assemble, source_id, loaded_data[source_id]): source_id for source_id in source_files
        }

        # Waits for all archives to be assembled.
        archives = {}
        with tqdm(
            total=len(source_files),
            desc="Generating archives for all unique sources",
            unit="sources",
            disable=not verbose,
        ) as pbar:
            for future in as_completed(archive_futures):
                archive_id, archive_path = future.result()
                archives[archive_id] = archive_path
                pbar.update(1)

        # PHASE 3: Verifies archived data integrity against the original data if this is requested.
        if verify_integrity:
            # Loads assembled archives into memory.
            archived_futures = {
                source_id: p_executor.submit(_load_numpy_archive, path) for source_id, path in archives.items()
            }

            # Waits for the data to be loaded.
            archive_data = {}
            with tqdm(
                total=len(archives), desc="Loading archive data into memory", unit="archives", disable=not verbose
            ) as pbar:
                for source_id, future in archived_futures.items():
                    archive_data[source_id] = future.result()
                    pbar.update(1)

            # Verifies the integrity of each archive data against the original data.
            # noinspection PyTypeChecker
            verification_futures = [
                t_executor.submit(
                    partial(_compare_arrays, source_id), stem, original_array, archive_data[source_id][stem]
                )
                for source_id, source_data in loaded_data.items()
                for stem, original_array in source_data.items()
            ]

            # Tracks verification progress.
            with tqdm(
                total=len(verification_futures),
                desc="Verifying archived data integrity",
                unit="entries",
                disable=not verbose,
            ) as pbar:
                for future in as_completed(verification_futures):
                    future.result()  # Propagates errors if comparison fails
                    pbar.update(1)

        # PHASE 4: Removes source files if requested.
        if remove_sources:
            all_files = [f for files in source_files.values() for f in files]
            removal_futures = [t_executor.submit(f.unlink) for f in all_files]

            with tqdm(
                total=len(all_files), desc="Removing processed source files", unit="files", disable=not verbose
            ) as pbar:
                for future in as_completed(removal_futures):
                    future.result()
                    pbar.update(1)


@dataclass(frozen=True)
class LogPackage:
    """Stores the data and ID information to be logged by the DataLogger class and exposes methods for packaging this
    data into the format expected by the logger.

    Notes:
        During runtime, the DataLogger class expects all data sent for logging via the input Queue object to be
        packaged into an instance of this class.
    """

    source_id: np.uint8
    """The ID code of the source that produced the data. Has to be unique across all systems that send data
    to the same DataLogger instance during runtime."""

    acquisition_time: np.uint64
    """Tracks when the data was acquired. This value typically communicates the number of microseconds elapsed since 
    the onset of the data acquisition runtime."""

    serialized_data: NDArray[np.uint8]
    """The serialized data to be logged, stored as a one-dimensional bytes' NumPy array."""

    def get_data(self) -> tuple[str, NDArray[np.uint8]]:  # pragma: no cover
        """Constructs and returns the filename and the serialized data package to be logged.

        Returns:
            A tuple of two elements. The first element is the name to use for the log file, which consists of
            zero-padded source id and zero-padded time stamp, separated by an underscore. The second element is the
            data to be logged as a one-dimensional bytes numpy array. The logged data includes the original data object
            and the pre-pended source id and time stamp.
        """
        # Prepares the data by converting zero-dimensional numpy inputs to arrays and concatenating all data into one
        # array
        serialized_time_stamp = np.frombuffer(buffer=self.acquisition_time, dtype=np.uint8).copy()
        serialized_source = np.frombuffer(buffer=self.source_id, dtype=np.uint8).copy()

        # Note, it is assumed that each source produces the data sequentially and that timestamps are acquired with
        # high enough resolution to resolve the order of data acquisition.
        data = np.concatenate([serialized_source, serialized_time_stamp, self.serialized_data], dtype=np.uint8).copy()

        # Zero-pads ID and timestamp. Uses the correct number of zeroes to represent the number of digits that fits into
        # each datatype (uint8 and uint64).
        log_name = f"{self.source_id:03d}_{self.acquisition_time:020d}.npy"

        return log_name, data


class DataLogger:
    """Saves input data as uncompressed byte numpy array (.npy) files using the requested number of cores and
    threads.

    This class instantiates and manages the runtime of a logger distributed over the requested number of cores and
    threads. The class exposes a shared multiprocessing Queue via the 'input_queue' property, which can be used to
    buffer and pipe the data to the logger from other Processes. The class expects the data to be first packaged into
    LogPackage class instance also available from this library, before it is sent to the logger via the queue object.

    Notes:
        Initializing the class does not start the logger processes! Call the start() method to initialize the logger
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

    def __init__(
        self,
        output_directory: Path,
        instance_name: str = "data_logger",
        process_count: int = 1,
        thread_count: int = 5,
        sleep_timer: int = 5000,
        exist_ok: bool = False,
    ) -> None:
        # Initializes a variable that tracks whether the class has been started.
        self._started: bool = False
        # Since __del__ now terminates the manager, it is a good idea to keep it high on the initialization order
        self._mp_manager: SyncManager = Manager()

        # Ensures numeric inputs are not negative.
        self._process_count: int = max(1, process_count)
        self._thread_count: int = max(1, thread_count)
        self._sleep_timer: int = max(0, sleep_timer)
        self._name = str(instance_name)
        self._exist_ok = exist_ok

        # If necessary, ensures that the output directory tree exists. This involves creating an additional folder
        # 'data_log', to which the data will be saved in an uncompressed format. The folder also includes the logger
        # instance name
        self._output_directory: Path = output_directory.joinpath(f"{self._name}_data_log")
        ensure_directory_exists(self._output_directory)  # This also ensures the input is a valid Path object

        # Sets up the multiprocessing Queue to be shared by all logger and data source processes.
        self._input_queue: MPQueue = self._mp_manager.Queue()

        self._terminator_array: SharedMemoryArray | None = None
        self._logger_processes: tuple[Process, ...] = tuple()
        self._watchdog_thread: Thread | None = None

    def __repr__(self) -> str:
        """Returns a string representation of the DataLogger instance."""
        message = (
            f"DataLogger(name={self._name}, output_directory={self._output_directory}, "
            f"process_count={self._process_count}, thread_count={self._thread_count}, "
            f"sleep_timer={self._sleep_timer} microseconds, started={self._started})"
        )
        return message

    def __del__(self) -> None:
        """Ensures that logger resources are properly released when the class is garbage collected."""
        self.stop()
        self._mp_manager.shutdown()  # Destroys the queue buffers used by the object

    def start(self) -> None:
        """Starts the logger processes and the assets used to control and ensure the processes are alive.

        Once this method is called, data submitted to the 'input_queue' of the class instance will be saved to disk via
        the started Processes.
        """
        # Prevents re-starting an already started process
        if self._started:
            return

        # Initializes the terminator array, used to control the logger process(es)
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator", prototype=np.zeros(shape=1, dtype=np.uint8), exist_ok=self._exist_ok
        )  # Instantiation automatically connects the main process to the array.

        # Creates and pacakge processes into the tuple
        self._logger_processes = tuple(
            [
                Process(
                    target=self._log_cycle,
                    args=(
                        self._input_queue,
                        self._terminator_array,
                        self._output_directory,
                        self._thread_count,
                        self._sleep_timer,
                    ),
                    daemon=True,
                )
                for _ in range(self._process_count)
            ]
        )

        # Creates the watchdog thread.
        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)

        # Ensures that the terminator array is set appropriately to prevent processes from terminating
        if self._terminator_array is not None:
            self._terminator_array.write_data(index=0, data=np.uint8(0))

        # Starts logger processes
        for process in self._logger_processes:
            process.start()

        # Starts the process watchdog thread
        self._watchdog_thread.start()

        # Sets the tracker flag. Among other things, this actually activates the watchdog thread.
        self._started = True

    def stop(self) -> None:
        """Stops the logger processes once they save all buffered data and releases reserved resources."""
        if not self._started:
            return

        # Amongst other things this soft-inactivates the watchdog thread.
        self._started = False

        # Issues the shutdown command to the remote processes and the watchdog thread
        if self._terminator_array is not None:
            self._terminator_array.write_data(index=0, data=np.uint8(1))

        # Waits until the process(es) shut down.
        for process in self._logger_processes:
            process.join()

        # Waits for the watchdog thread to shut down.
        if self._watchdog_thread is not None:
            self._watchdog_thread.join()

        # Ensures the shared memory array is destroyed when the class is garbage-collected
        if self._terminator_array is not None:
            self._terminator_array.disconnect()
            self._terminator_array.destroy()

    def _watchdog(self) -> None:
        """This function should be used by the watchdog thread to ensure the logger processes are alive during runtime.

        This function will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """
        timer = PrecisionTimer(precision="ms")

        # The watchdog function will run until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay(delay=20, allow_sleep=True, block=False)

            if not self._started:
                continue

            # Only checks that processes are alive if they are started. The shutdown() flips the started tracker
            # before actually shutting down the processes, so there should be no collisions here.
            process_number = 0
            error = False
            for num, process in enumerate(self._logger_processes, start=1):  # pragma: no cover
                # If a started process is not alive, it has encountered an error forcing it to shut down.
                if not process.is_alive():
                    error = True
                    process_number = num

            # The error is raised outside the checking context to allow gracefully shutting down all assets before
            # terminating runtime with an error message.
            if error:
                message = (
                    f"DataLogger process {process_number} out of {len(self._logger_processes)} has been prematurely "
                    f"shut down. This likely indicates that the process has encountered a runtime error that "
                    f"terminated the process."
                )
                # Since the raised error terminates class runtime, cleans up all resources, just like how stop() does it
                self._terminator_array.write_data(index=0, data=np.uint8(1))
                for process in self._logger_processes:
                    process.join()
                self._terminator_array.disconnect()
                self._terminator_array.destroy()
                self._started = False  # Prevents stop() from running via __del__
                console.error(message=message, error=RuntimeError)

    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:  # pragma: no cover
        """Thread worker function that saves the data.

        Args:
            filename: The name of the file to save the data to. Note, the name has to be suffix-less, as the '.npy'
                suffix will be appended automatically.
            data: The data to be saved, packaged into a one-dimensional bytes array.

        Since data saving is primarily IO-bound, using multiple threads per each Process is likely to achieve the best
        saving performance.
        """
        np.save(file=filename, arr=data, allow_pickle=False)

    @staticmethod
    def _log_cycle(
        input_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        output_directory: Path,
        thread_count: int,
        sleep_time: int = 1000,
    ) -> None:  # pragma: no cover
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
        # Connects to the shared memory array
        terminator_array.connect()

        # Creates a thread pool for this process. It will manage the local saving threads
        executor = ThreadPoolExecutor(max_workers=thread_count)

        # Initializes the timer instance used to temporarily pause the execution when there is no data to process
        sleep_timer = PrecisionTimer(precision="us")

        # Main process loop. This loop will run until BOTH the terminator flag is passed and the input queue is empty.
        while not terminator_array.read_data(index=0, convert_output=False) or not input_queue.empty():
            try:
                # Gets data from the input queue with timeout. The data is expected to be packaged into the LogPackage
                # class.
                package: LogPackage = input_queue.get_nowait()

                # Pre-processes the data
                file_name, data = package.get_data()

                # Generates the full name for the output log file by merging the name of the specific file with the
                # path to the output directory
                filename = output_directory.joinpath(file_name)

                # Submits the task to the thread pool to be executed
                executor.submit(DataLogger._save_data, filename, data)

            # If the queue is empty, invokes the sleep timer to reduce CPU load.
            except (Empty, KeyError):
                sleep_timer.delay(delay=sleep_time, allow_sleep=True, block=False)

            # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
            # before re-raising the exception to terminate the process.
            except Exception as e:
                sys.stderr.write(str(e))
                sys.stderr.flush()

                # If the class runs into a runtime error, ensures proper termination of remote process resources.
                executor.shutdown(wait=True)
                terminator_array.disconnect()

                raise e

        # If the process escapes the loop due to encountering the shutdown command, shuts the executor threads and
        # disconnects from the terminator array before ending the runtime.
        executor.shutdown(wait=True)
        terminator_array.disconnect()

    def compress_logs(
        self,
        remove_sources: bool = False,
        memory_mapping: bool = False,
        verbose: bool = False,
        compress: bool = True,
        verify_integrity: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """Consolidates all .npy files in the target log directory into a compressed .npz archive for each source_id.

        All entries within each source are grouped by their acquisition timestamp value before compression. The
        compressed archive names include the ID code of the source that generated original log entries. This function
        can compress any log directory generated by a DataLogger instance and can be used without an initialized
        DataLogger.

        Notes:
            Primarily, this method functions as a wrapper around the instance-independent 'compress_npy_logs' methods
            exposed by this library. It automatically resolves the path to the uncompressed log directory using instance
            attributes.

            To improve runtime efficiency, the function parallelizes all data processing steps. The exact number of
            parallel threads used by the function depends on the number of available CPU cores. This number can be
            further adjusting by modifying the max_workers argument.

            This function requires all data from the same source to be loaded into RAM before it is added to the .npz
            archive. While this should not be an issue for most runtimes and expected use patterns, this function can be
            configured to use memory-mapping instead of directly loading data into RAM. This has a noticeable processing
            speed reduction and is not recommended for most users.

            Since this function is intended to optimize how logs are stored on disk, it is statically configured to
            remove the source .npy files after generating compressed .npz entries. As an extra security measure, it is
            possible to request the function to verify the integrity of the compressed data against the sources before
            removing source files. It is heavily discouraged however, as this adds a noticeable performance
            (runtime speed) overhead and data corruption is generally extremely uncommon and unlikely.

            Additionally, it is possible to disable log compression and instead just aggregated the log entries into an
            uncompressed .npz file. This is not recommended, since compression speed is very fast and does not majorly
            affect the runtime speed, but may noticeably reduce disk usage. However, decompression takes a considerable
            time, so some processing runtimes may benefit from not compressing the generated log runtimes if fast
            decompression speed is a priority.

        Args:
            remove_sources: Determines whether to remove the individual .npy files after they have been consolidated
                into .npz archives.
            memory_mapping: Determines whether the function uses memory-mapping (disk) to stage the data before
                compression or loads all data into RAM. Disabling this option makes the function considerably faster,
                but may lead to out-of-memory errors in very rare use cases. Note, due to collisions with Windows not
                releasing memory-mapped files, this argument does not do anything on Windows.
            verbose: Determines whether to print compression progress to terminal.
            compress: Determines whether to compress the output .npz archive file for each source. While the intention
                behind this function is to compress archive data, it is possible to use the function to just aggregate
                the data into .npz files without compression.
            verify_integrity: Determines whether to verify the integrity of compressed data against the original log
                entries before removing sources. Since it is highly unlikely that compression alters the data, it is
                recommended to have this option disabled for most runtimes.
            max_workers: Determines the number of threads used to carry out various processing phases in-parallel. Note,
                some processing phases parallelize log source processing and other parallelize log entry processing.
                Therefore, it is generally desirable to use as many threads as possible. If set to None, the function
                uses the number of (logical) CPU cores - 2 threads.
        """
        assemble_log_archives(
            log_directory=self.output_directory,
            remove_sources=remove_sources,
            memory_mapping=memory_mapping,
            verbose=verbose,
            compress=compress,
            verify_integrity=verify_integrity,
            max_workers=max_workers,
        )

    @property
    def input_queue(self) -> MPQueue:
        """Returns the multiprocessing Queue used to buffer and pipe the data to the logger processes.

        Share this queue with all source processes that need to log data. To ensure correct data packaging, package the
        data using the LogPackage class exposed by this library before putting it into the queue.
        """
        return self._input_queue

    @property
    def name(self) -> str:
        """Returns the name of the DataLogger instance."""
        return self._name

    @property
    def started(self) -> bool:
        """Returns True if the DataLogger has been started and is actively logging data."""
        return self._started

    @property
    def output_directory(self) -> Path:
        """Returns the path to the directory where the data is saved."""
        return self._output_directory
