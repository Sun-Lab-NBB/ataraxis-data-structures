"""This module provides the DataLogger class to efficiently save (log) serialized data collected from different
Processes to disk.
"""

import os
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
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import console, ensure_directory_exists

from ..shared_memory import SharedMemoryArray

if TYPE_CHECKING:
    from multiprocessing.managers import SyncManager


def _load_numpy_files(
    file_paths: tuple[Path, ...], *, memory_map: bool = False
) -> tuple[tuple[str, ...], tuple[NDArray[Any], ...]]:  # pragma: no cover
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


def _load_numpy_archive(file_path: Path) -> dict[str, NDArray[Any]]:  # pragma: no cover
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
) -> tuple[int, Path]:  # pragma: no cover
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
    if not np.array_equal(original_array, archived_array):  # pragma: no cover
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

    @property
    def data(self) -> tuple[str, NDArray[np.uint8]]:  # pragma: no cover
        """Returns the filename and the serialized data package to be processed by a DataLogger instance.

        Note:
            This property is designed to be exclusively accessed by the DataLogger instance.
        """
        # Prepares the data by converting zero-dimensional numpy inputs to arrays and concatenating all data into one
        # array
        serialized_acquisition_time = np.frombuffer(buffer=self.acquisition_time, dtype=np.uint8).copy()
        serialized_source = np.frombuffer(buffer=self.source_id, dtype=np.uint8).copy()

        # Note, it is assumed that each source produces the data sequentially and that timestamps are acquired with
        # high enough resolution to resolve the order of data acquisition.
        data = np.concatenate(
            [serialized_source, serialized_acquisition_time, self.serialized_data], dtype=np.uint8
        ).copy()

        # Zero-pads ID and timestamp. Uses the correct number of zeroes to represent the number of digits that
        # fits into each datatype (uint8 and uint64).
        log_name = f"{self.source_id:03d}_{self.acquisition_time:020d}.npy"

        return log_name, data


class DataLogger:
    """Instantiates and manages the runtime of a data logger that saves serialized data collected from multiple
    concurrently active sources.

    This class manages the runtime of a data logger running in a separate process. The logger uses multiple concurrent
    threads to optimize the I/O operations associated with saving the data to disk, achieving high throughput under
    a wide range of scenarios.

    Notes:
        Initializing the class does not start the logger! Call the start() method to ensure that the logger is fully
        initialized before submitting data for logging.

        Use the multiprocessing Queue exposed via the 'input_queue' property, to send the data to the logger. The data
        must be packaged into the LogPackage class instance before it is submitted to the queue.

    Args:
        output_directory: The directory where to save the logged data. The data is saved under a subdirectory named
            after the logger instance.
        instance_name: The name of the logger instance. This name has to be unique across all concurrently active
            DataLogger instances.
        thread_count: The number of threads to use for saving the data to disk. It is recommended to use multiple
            threads to parallelize the I/O operations associated with writing the logged data to disk.
        poll_interval: The interval, in milliseconds, between polling the input queue. Primarily, this is designed to
            optimize the CPU usage during light workloads. Setting this to 0 disabled the polling delay mechanism.

    Attributes:
        _started: Tracks whether the logger process is running.
        _mp_manager: Stores the manager object used to instantiate and manage the multiprocessing Queue.
        _thread_count: Stores the number of concurrently active data saving threads.
        _poll_interval: Stores the data queue poll interval, in milliseconds.
        _name: Stores the name of the data logger instance.
        _output_directory: Stores the directory where the data is saved.
        _input_queue: Stores the multiprocessing Queue used to buffer and pipe the data to the logger process.
        _logger_process: Stores the Process object that runs the data logging cycle.
        _terminator_array: Stores the shared memory array used to terminate (shut down) the logger process.
        _watchdog_thread: Stores the thread used to monitor the runtime status of the remote logger process.
    """

    def __init__(
        self,
        output_directory: Path,
        instance_name: str,
        thread_count: int = 5,
        poll_interval: int = 5,
    ) -> None:
        # Initializes runtime control attributes
        self._started: bool = False
        self._mp_manager: SyncManager = Manager()

        # Ensures numeric inputs are not negative.
        self._thread_count: int = max(1, thread_count)
        self._poll_interval: int = max(0, poll_interval)
        self._name = str(instance_name)

        # If necessary, ensures that the output directory tree exists.
        self._output_directory: Path = output_directory.joinpath(f"{self._name}_data_log")
        ensure_directory_exists(self._output_directory)

        # Sets up the multiprocessing Queue to be shared by all logger and data source processes.
        self._input_queue: MPQueue = self._mp_manager.Queue()

        # Creates the infrastructure for running the logger.
        self._terminator_array: SharedMemoryArray | None = None
        self._logger_process: Process | None = None
        self._watchdog_thread: Thread | None = None

    def __repr__(self) -> str:
        """Returns the string representation of the DataLogger instance."""
        return (
            f"DataLogger(name={self._name}, output_directory={self._output_directory}, "
            f"thread_count={self._thread_count}, poll_interval={self._poll_interval} ms, started={self._started})"
        )

    def __del__(self) -> None:
        """Releases the reserved resources when the instance is garbage collected."""
        self.stop()
        self._mp_manager.shutdown()  # Destroys the queue buffers

    def start(self) -> None:
        """Starts the remote logger process and the assets used to control and monitor the logger's uptime."""
        # Prevents re-starting an already started process
        if self._started:
            return

        # Initializes the terminator array, used to control the logger process(es)
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator", prototype=np.zeros(shape=1, dtype=np.uint8), exists_ok=True
        )  # Instantiation automatically connects the main process to the array.

        # Creates and starts the logger process.
        self._logger_process = Process(
            target=self._log_cycle,
            args=(
                self._input_queue,
                self._terminator_array,
                self._output_directory,
                self._thread_count,
                self._poll_interval,
            ),
            daemon=True,
        )
        self._logger_process.start()

        # Creates and starts the watchdog thread.
        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

        # Sets the tracker flag.
        self._started = True

    def stop(self) -> None:
        """Stops the logger process once it saves all buffered data and releases reserved resources."""
        if not self._started:
            return

        # Soft-inactivates the watchdog thread.
        self._started = False

        # Issues the shutdown command to the remote process and the watchdog thread
        if self._terminator_array is not None:
            self._terminator_array.write_data(index=0, data=np.uint8(1))

        # Waits for the process to shut down.
        if self._logger_process is not None:
            self._logger_process.join()

        # Waits for the watchdog thread to shut down.
        if self._watchdog_thread is not None:
            self._watchdog_thread.join()

        # Destroys the shared memory array instance.
        if self._terminator_array is not None:
            self._terminator_array.disconnect()
            self._terminator_array.destroy()

    def _watchdog(self) -> None:
        """Raises a ChildProcessError if the logger process has prematurely shut down.

        This service function is the target for the watchdog thread.
        """
        timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # The watchdog function runs until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):
            # Repeats the check every 20 ms.
            timer.delay(delay=20, allow_sleep=True, block=False)

            if not self._started:
                continue

            # Only checks that the process is alive if it is started.
            if not self._logger_process.is_alive():  # pragma: no cover
                # Cleans up all resources, similar to the stop() method.
                self._terminator_array.write_data(index=0, data=np.uint8(1))
                self._logger_process.join()
                self._terminator_array.disconnect()
                self._terminator_array.destroy()
                self._started = False  # Prevents stop() from running via __del__

                # Raises the error
                message = (
                    f"Remote logger process for the {self._name} DataLogger has been prematurely shut down. This "
                    f"likely indicates that the process has encountered a runtime error."
                )
                console.error(message=message, error=ChildProcessError)

    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:  # pragma: no cover
        """Saves the input data as the specified .npy file.

        This worker function is the target for each data-saving thread.

        Args:
            filename: The name of the file to save the data to, without the suffix.
            data: The data to be saved, packaged into a one-dimensional bytes' array.
        """
        np.save(file=filename, arr=data, allow_pickle=False)

    @staticmethod
    def _log_cycle(
        input_queue: MPQueue,
        terminator_array: SharedMemoryArray,
        output_directory: Path,
        thread_count: int,
        poll_interval: int,
    ) -> None:  # pragma: no cover
        """Continuously queries and saves the data coming through the input_queue to disk as .npy files.

        This function sets up the necessary assets (threads and queues) to accept, preprocess, and save the input data
        as .npy files. It is the target for the DataLogger's remote process.

        Args:
            input_queue: The multiprocessing Queue object used to buffer and pipe the data to the logger process.
            terminator_array: A shared memory array used to terminate (shut down) the logger process.
            output_directory: The path to the directory where to save the data.
            thread_count: The number of threads to use for parallelizing I/O operations.
            poll_interval: The interval, in milliseconds, at which to poll the input queue for new data if the queue
                has been emptied.
        """
        # Connects to the shared memory array
        terminator_array.connect()

        # Creates a thread pool to manage the data-saving threads
        executor = ThreadPoolExecutor(max_workers=thread_count)

        # Initializes the timer instance to delay polling the queue during idle periods.
        sleep_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # Main process loop. This loop runs until BOTH the terminator flag is passed and the input queue is empty.
        try:
            while not terminator_array.read_data(index=0, convert_output=False) or not input_queue.empty():
                try:
                    # Gets the data from the input queue with timeout. The data is expected to be packaged into the
                    # LogPackage instance.
                    package: LogPackage = input_queue.get_nowait()

                    # Pre-processes the data
                    file_name, data = package.data

                    # Generates the full name for the output log file by merging the name of the specific file with the
                    # path to the output directory
                    filename = output_directory.joinpath(file_name)

                    # Submits the task to the thread pool to be executed
                    executor.submit(DataLogger._save_data, filename, data)

                # If the queue is empty, invokes the sleep timer to reduce CPU load.
                except (Empty, KeyError):
                    sleep_timer.delay(delay=poll_interval, allow_sleep=True, block=False)
        finally:
            # Ensures all remote assets are released before the process shutdown.
            executor.shutdown(wait=True)
            terminator_array.disconnect()

    @property
    def input_queue(self) -> MPQueue:
        """Returns the multiprocessing Queue used to buffer and pipe the data to the logger process."""
        return self._input_queue

    @property
    def name(self) -> str:
        """Returns the name of the instance."""
        return self._name

    @property
    def alive(self) -> bool:
        """Returns True if the instance's logger process is currently running."""
        return self._started

    @property
    def output_directory(self) -> Path:
        """Returns the path to the directory where the data is saved."""
        return self._output_directory
