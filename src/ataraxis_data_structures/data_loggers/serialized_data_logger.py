"""Provides the DataLogger class to efficiently save (log) serialized data collected from different processes to
disk, the LogPackage class used to submit data for logging, and the assemble_log_archives() function that
consolidates the logged .npy entries into per-source .npz archives.
"""

from __future__ import annotations

from queue import Empty
from typing import TYPE_CHECKING, Any, Literal
from operator import itemgetter
import platform
from functools import partial
from threading import Lock, Thread
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import (
    Queue as MultiprocessingQueue,
    get_context,
)
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import (
    LogLevel,
    console,
    resolve_worker_count,
    ensure_directory_exists,
)

from ..processing import limit_worker_threads
from ..shared_memory import SharedMemoryArray

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Generator
    from concurrent.futures import Future
    from multiprocessing.context import SpawnContext
    from multiprocessing.process import BaseProcess
    from multiprocessing.managers import SyncManager

    from numpy.typing import NDArray


LOG_DIRECTORY_SUFFIX: str = "_data_log"
"""The name suffix of the output directory each DataLogger instance creates for its log entries and archives.

Notes:
    Every directory is named ``{instance_name}{LOG_DIRECTORY_SUFFIX}`` after the logger instance that owns it, so a
    consumer resolves a logger's output directory from the logger's name alone.
"""

LOG_ARCHIVE_SUFFIX: str = "_log.npz"
"""The filename suffix of the .npz log archives ``assemble_log_archives()`` writes.

Notes:
    Every archive is named ``{source_id}{LOG_ARCHIVE_SUFFIX}`` after the source whose entries it holds, so a consumer
    resolves an archive from a source ID alone.
"""

_MULTIPROCESSING_CONTEXT: SpawnContext = get_context("spawn")
"""The spawn-based multiprocessing context used to create the process pool that assembles log archives, ensuring
identical cross-platform behavior on all supported platforms."""

_BATCH_OVERSCALE_FACTOR: int = 4
"""The multiplier applied to the per-worker share of log entries when sizing the batches used for parallel loading."""

_SOURCE_ID_BYTE_SIZE: int = 1
"""The number of bytes the source ID occupies at the start of each serialized log entry."""

_HEADER_BYTE_SIZE: int = 9
"""The number of bytes the source ID and the acquisition timestamp occupy together at the start of each serialized log
entry."""


@dataclass(frozen=True, slots=True)
class LogPackage:
    """Stores the data and ID information to be logged by the DataLogger class and exposes methods for packaging this
    data into the format expected by the logger.
    """

    source_id: np.uint8
    """The ID code of the source that produced the data. Has to be unique across all systems that send data
    to the same DataLogger instance during runtime."""

    acquisition_time: np.uint64
    """The timestamp of when the data was acquired. This value typically communicates the number of microseconds
    elapsed since the onset of the data acquisition runtime."""

    serialized_data: NDArray[np.uint8]
    """The serialized data to be logged, stored as a one-dimensional byte array."""

    @property
    def data(self) -> tuple[str, NDArray[np.uint8]]:
        """Returns the filename and the serialized data package to be processed by a DataLogger instance."""
        # Fills one preallocated buffer, so an entry costs a single allocation. The timestamp is serialized through
        # its own native-order bytes, which is the layout LogArchiveReader reads back. The buffer owns its memory and
        # aliases none of its inputs, so the result needs no defensive copy.
        data: NDArray[np.uint8] = np.empty(shape=_HEADER_BYTE_SIZE + self.serialized_data.size, dtype=np.uint8)
        data[0] = self.source_id
        data[_SOURCE_ID_BYTE_SIZE:_HEADER_BYTE_SIZE] = np.frombuffer(
            buffer=np.uint64(self.acquisition_time).tobytes(), dtype=np.uint8
        )

        # Copies the payload under same-kind casting, which rejects a signed, floating point, or complex payload
        # rather than reinterpreting its elements as bytes. A wider unsigned payload is narrowed element by element,
        # which is the rule np.concatenate applies to the same dtypes.
        np.copyto(dst=data[_HEADER_BYTE_SIZE:], src=self.serialized_data, casting="same_kind")

        # Zero-pads ID and timestamp. Uses the correct number of zeroes to represent the number of digits that
        # fit into each datatype (uint8 and uint64).
        log_name = f"{self.source_id:03d}_{self.acquisition_time:020d}.npy"

        return log_name, data


class DataLogger:
    """Manages the runtime of a data logger that saves serialized data collected from multiple concurrently active
    sources.

    The logger runs in a separate process and uses multiple concurrent threads to optimize the I/O operations
    associated with saving the data to disk.

    Notes:
        The start() method must complete before any data is submitted for logging.

        Use the multiprocessing Queue exposed via the ``input_queue`` property to send the data to the logger. The data
        must be packaged into the LogPackage class instance before it is submitted to the queue.

        Submitting data to the input queue does not confirm that the data reached the disk, since the logger process
        writes the entries asynchronously. A write that fails while the logger is running terminates the logger
        process, which the watchdog thread reports as a ChildProcessError. A write that fails during the shutdown
        sequence is reported by stop() as a warning.

    Args:
        output_directory: The directory in which to save the logged data. The data is saved under a subdirectory named
            '{instance_name}_data_log'.
        instance_name: The name of the logger instance. This name has to be unique across all concurrently active
            DataLogger instances.
        thread_count: The number of threads to use for saving the data to disk. It is recommended to use multiple
            threads to parallelize the I/O operations associated with writing the logged data to disk. Values below 1
            are clamped to 1.
        poll_interval: The interval, in milliseconds, between polling the input queue. Primarily, this is designed to
            optimize the CPU usage during light workloads. Setting this to 0 disables the polling delay mechanism.
            Negative values are clamped to 0.

    Attributes:
        _started: Tracks whether the logger process is running.
        _shutdown_lock: Stores the lock that serializes the shutdown sequence between stop() and the watchdog thread,
            so exactly one of the two retires the instance.
        _multiprocessing_context: Stores the spawn-based multiprocessing context used to create the manager and the
            logger process.
        _multiprocessing_manager: Stores the manager object used to instantiate and manage the multiprocessing Queue.
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
        self._started: bool = False

        # Serializes the shutdown sequence. stop() and the watchdog thread both clear the started flag and then
        # release the terminator array, and the flag alone cannot separate them, since a thread reads it several
        # statements before it touches the array. The array's own lock does not cover this, as it guards element
        # access while disconnect() and destroy() take no lock at all.
        self._shutdown_lock: Lock = Lock()

        self._multiprocessing_context: SpawnContext = get_context("spawn")
        self._multiprocessing_manager: SyncManager = self._multiprocessing_context.Manager()

        self._thread_count: int = max(1, thread_count)
        self._poll_interval: int = max(0, poll_interval)
        self._name: str = str(instance_name)

        # If necessary, ensures that the output directory tree exists. The path is declared as a directory, since an
        # instance name carrying a dot would otherwise leave the final component reading as a file suffix, which would
        # create the parent alone and leave the logger writing into a directory that does not exist.
        self._output_directory: Path = output_directory.joinpath(f"{self._name}{LOG_DIRECTORY_SUFFIX}")
        ensure_directory_exists(path=self._output_directory, is_file=False)

        # Sets up the multiprocessing Queue to be shared by all logger and data source processes.
        self._input_queue: MultiprocessingQueue = (  # type: ignore[type-arg]
            self._multiprocessing_manager.Queue()  # type: ignore[assignment]
        )

        self._terminator_array: SharedMemoryArray | None = None
        self._logger_process: BaseProcess | None = None
        self._watchdog_thread: Thread | None = None

    def __repr__(self) -> str:
        """Returns the string representation of the DataLogger instance."""
        return (
            f"DataLogger(name={self._name}, output_directory={self._output_directory}, "
            f"thread_count={self._thread_count}, poll_interval={self._poll_interval} ms, started={self._started})"
        )

    def __del__(self) -> None:
        """Releases the reserved resources when the instance is garbage-collected."""
        self.stop()
        self._multiprocessing_manager.shutdown()  # Destroys the queue buffers.

    def start(self) -> None:
        """Starts the remote logger process and the assets used to control and monitor the logger's uptime."""
        if self._started:
            return

        # Initializes the terminator array, used to control the logger process(es). Creating the array also connects
        # the main process to the shared memory buffer.
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator",
            prototype=np.zeros(shape=1, dtype=np.uint8),
            exists_ok=True,
        )

        # Creates and starts the logger process. The logger writes .npy files rather than performing numeric work, so
        # its process is spawned under the thread limit to keep the numeric backends from opening a pool it never uses.
        with limit_worker_threads():
            self._logger_process = self._multiprocessing_context.Process(
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

        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)
        self._watchdog_thread.start()

        self._started = True

    def stop(self) -> None:
        """Stops the logger process once it saves all buffered data and releases reserved resources.

        Notes:
            A logger process that failed to save one of its buffered entries is reported through a warning rather than
            an exception. This method only reaches that check during the shutdown sequence, where raising would mask
            the shutdown work of the caller that is already unwinding, and where the data is lost either way.

            The shutdown is claimed under a lock the watchdog thread takes as well, so exactly one of the two performs
            the teardown. The lock is released before this method joins that thread, since the watchdog acquires the
            same lock and holding it across the join would leave each side waiting on the other.
        """
        # Claims the shutdown. A watchdog that reached its own teardown first leaves the flag clear, which takes the
        # early return below and keeps this method away from the terminator array that thread has already released.
        with self._shutdown_lock:
            if not self._started:
                return

            # Soft-inactivates the watchdog thread.
            self._started = False

            # Issues the shutdown command to the remote process and the watchdog thread.
            if self._terminator_array is not None:
                self._terminator_array[0] = 1

        logger_exit_code: int | None = None
        if self._logger_process is not None:
            self._logger_process.join()
            logger_exit_code = self._logger_process.exitcode

        if self._watchdog_thread is not None:
            self._watchdog_thread.join()

        if self._terminator_array is not None:
            self._terminator_array.disconnect()
            self._terminator_array.destroy()

        # The logger process exits non-zero when a disk write fails, so the code is the only evidence the caller gets
        # that its data did not reach the disk. Every resource above is released first, so reporting the failure does
        # not also leak the shared memory buffer. The console is temporarily enabled, since a silent data loss report
        # would defeat the purpose of making the check at all.
        if logger_exit_code:
            message = (
                f"Unable to confirm that the {self._name} DataLogger saved all buffered data. The logger process "
                f"exited with code {logger_exit_code} instead of shutting down cleanly, which indicates that at least "
                f"one of the submitted log entries did not reach the disk. Treat the data this logger recorded as "
                f"incomplete."
            )
            with console.temporarily_enabled():
                console.echo(message=message, level=LogLevel.WARNING)

    @property
    def input_queue(self) -> MultiprocessingQueue:  # type: ignore[type-arg]
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

    def _watchdog(self) -> None:
        """Raises a ChildProcessError if the logger process has prematurely shut down."""
        timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # The watchdog function runs until the global shutdown command is issued.
        while self._terminator_array is not None and not self._terminator_array[0]:
            timer.delay(delay=20, allow_sleep=True, block=False)

            if not self._started:
                continue

            if self._logger_process is not None and not self._logger_process.is_alive():
                # Claims the shutdown under the lock stop() takes as well, so exactly one of the two retires the
                # instance. A concurrent stop() can claim it between the liveness check above and this acquisition,
                # and the claimant owns every step below. Re-reading the flag here keeps this thread from tearing the
                # instance down a second time and reporting a shutdown that already happened.
                with self._shutdown_lock:
                    if not self._started:
                        return

                    # Retires the instance in the same order stop() uses, clearing the started flag before releasing
                    # the terminator array.
                    self._started = False

                    # Cleans up all resources, similar to the stop() method.
                    self._terminator_array[0] = 1
                    self._logger_process.join()
                    self._terminator_array.disconnect()
                    self._terminator_array.destroy()

                message = (
                    f"Remote logger process for the {self._name} DataLogger has been prematurely shut down. This "
                    f"likely indicates that the process has encountered a runtime error."
                )
                console.error(message=message, error=ChildProcessError)

    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:
        """Saves the input data as the specified .npy file.

        Args:
            filename: The full path to the .npy file to save the data to. The name already includes the .npy suffix.
            data: The data to be saved, packaged into a one-dimensional byte array.
        """
        np.save(file=filename, arr=data, allow_pickle=False)

    @staticmethod
    def _retire_completed_writes(pending_writes: list[Future[None]]) -> list[Future[None]]:
        """Removes the finished disk writes from the input list and re-raises the error of the first failed write.

        Args:
            pending_writes: The futures of the disk writes submitted to the logger process thread pool.

        Returns:
            The futures of the disk writes that have not finished yet.

        Raises:
            OSError: If saving one of the finished log entries to disk failed.
        """
        running_writes = []
        for write in pending_writes:
            if not write.done():
                running_writes.append(write)
                continue

            # Propagates the error of a finished write. Discarding the result instead would allow the logger to report
            # a clean shutdown after silently losing the data of that log entry.
            write.result()

        return running_writes

    @staticmethod
    def _log_cycle(
        input_queue: MultiprocessingQueue,  # type: ignore[type-arg]
        terminator_array: SharedMemoryArray,
        output_directory: Path,
        thread_count: int,
        poll_interval: int,
    ) -> None:
        """Continuously queries and saves the data coming through the input_queue to disk as .npy files.

        Sets up the necessary assets (threads and queues) to accept, preprocess, and save the input data as .npy
        files.

        Args:
            input_queue: The multiprocessing Queue object used to buffer and pipe the data to the logger process.
            terminator_array: A shared memory array used to terminate (shut down) the logger process.
            output_directory: The path to the directory in which to save the data.
            thread_count: The number of threads to use for parallelizing I/O operations.
            poll_interval: The interval, in milliseconds, at which to poll the input queue for new data if the queue
                has been emptied.

        Raises:
            OSError: If saving any of the processed log entries to disk failed. Propagating the error terminates the
                logger process with a non-zero exit code, which is how the failure reaches the parent process.
        """
        # The terminator array connects to the shared memory buffer as part of being transferred into this process,
        # so it is ready to use here. The finally block below still disconnects it before the process shuts down.
        executor = ThreadPoolExecutor(max_workers=thread_count)

        # Initializes the timer instance to delay polling the queue during idle periods.
        sleep_timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

        # Tracks the writes still in flight. Retiring each one as it completes bounds the list by the writes that have
        # not finished yet, and surfaces a failed write while the logger is still running.
        pending_writes: list[Future[None]] = []

        try:
            while not terminator_array[0] or not input_queue.empty():
                try:
                    package: LogPackage = input_queue.get_nowait()

                    file_name, data = package.data

                    filename = output_directory.joinpath(file_name)

                    pending_writes.append(executor.submit(DataLogger._save_data, filename=filename, data=data))
                    pending_writes = DataLogger._retire_completed_writes(pending_writes=pending_writes)

                # If the queue is empty, invokes the sleep timer to reduce CPU load. Whether the consumer ever
                # outpaces the producer depends on runtime timing, so the branch stays outside the measured corpus.
                except (Empty, KeyError):  # pragma: no cover
                    sleep_timer.delay(delay=poll_interval, allow_sleep=True, block=False)
        finally:
            # Ensures all remote assets are released before the process shutdown.
            executor.shutdown(wait=True)
            terminator_array.disconnect()

        # Re-raises the error of the first failed write once the pool has drained. The error terminates this process
        # with a non-zero exit code, which is what stop() reads to determine that some data did not reach the disk.
        DataLogger._retire_completed_writes(pending_writes=pending_writes)


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

    Notes:
        Log entries are grouped into archives by their source, and the entries within each archive are sorted by their
        acquisition timestamp value before consolidation. The consolidated archive names include the ID code of the
        source that generated the original log entries.

        Discovery covers the target directory itself and does not descend into its subdirectories, since one
        DataLogger instance is the unit of serialization and owns exactly one log directory. Entry names carry the
        source ID and the acquisition timestamp alone, so entries from two logger instances that share a source ID
        would collide on name if a single call consolidated both.

    Args:
        log_directory: The path to the directory that stores the log entries of one DataLogger instance as .npy
            files, which is the directory the instance exposes through its ``output_directory`` property.
        max_workers: Determines the number of worker processes and threads used to process the data in parallel. A
            positive value is honored exactly, capped at the logical core count. If set to None, 0, or a negative
            value, the function uses the number of CPU cores minus 2, clamped to at least 1.
        remove_sources: Determines whether to remove the .npy files after consolidating their data into .npz archives.
        memory_mapping: Determines whether to memory-map or load the processed data into RAM during processing. Due to
            Windows not releasing memory-mapped file handles, this function always loads the data into RAM when running
            on Windows.
        verbose: Determines whether to communicate the log assembly progress via the terminal.
        verify_integrity: Determines whether to verify the integrity of the created archives against the original log
            entries before removing sources.
    """
    max_workers = resolve_worker_count(requested_workers=max_workers or 0)

    # Windows does not release memory-mapped file handles reliably, so memory mapping is disabled on that platform.
    # Callers cap RAM usage on Windows through the max_workers argument.
    memory_mapping = memory_mapping and platform.system() != "Windows"

    # Collects all .npy files and groups them by source_id, parsing each stem once into its source and timestamp
    # fields so the sort below reads the parsed timestamp rather than splitting every stem a second time.
    source_entries: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for file_path in log_directory.glob("*.npy"):
        source_field, timestamp_field = file_path.stem.split("_")[:2]
        source_entries[int(source_field)].append((int(timestamp_field), file_path))

    # Sorts entries within each source_id group by their acquisition timestamp and drops the parsed sort keys.
    source_files: dict[int, list[Path]] = {
        source_id: [file_path for _, file_path in sorted(entries, key=itemgetter(0))]
        for source_id, entries in source_entries.items()
    }

    # Initiates log processing. Since some steps of log processing are more efficiently executed via multithreading
    # and others via multiprocessing, uses both process and thread pool executors to efficiently process the data.
    with (
        _progress_display(enabled=verbose),
        limit_worker_threads(),
        console.temporarily_enabled(),
        ProcessPoolExecutor(max_workers=max_workers, mp_context=_MULTIPROCESSING_CONTEXT) as process_executor,
        ThreadPoolExecutor(max_workers=max_workers) as thread_executor,
    ):
        # PHASE 1: Loads source files in parallel batches.
        total_files = sum(len(files) for files in source_files.values())
        loaded_data: dict[int, dict[str, NDArray[Any]]] = {source_id: {} for source_id in source_files}

        # Sizes each batch at the over-scale factor multiple of the per-worker share of the log entries.
        load_numpy = partial(_load_numpy_files, memory_map=memory_mapping)
        batch_size = int(np.ceil(total_files / max_workers * _BATCH_OVERSCALE_FACTOR))

        load_futures = [
            (source_id, process_executor.submit(load_numpy, file_paths=file_batch))
            for source_id, files in source_files.items()
            for start_index in range(0, len(files), batch_size)
            for file_batch in [tuple(files[start_index : start_index + batch_size])]
        ]

        with console.progress(
            total=total_files,
            description="Loading log entry data into memory",
            unit="entries",
        ) as progress_bar:
            for source_id, load_future in load_futures:
                stems, arrays = load_future.result()
                for stem, array in zip(stems, arrays, strict=False):
                    loaded_data[source_id][stem] = array
                    progress_bar.update(n=1)

        # PHASE 2: Assembles archives. Here, each archive is processed in parallel, but all archive log entries for
        # each archive are processed sequentially.
        assemble = partial(_assemble_archive, output_directory=log_directory)
        archive_futures = {
            process_executor.submit(assemble, source_id=source_id, source_data=loaded_data[source_id]): source_id
            for source_id in source_files
        }

        archives = {}
        with console.progress(
            total=len(source_files),
            description="Generating archives for all unique sources",
            unit="sources",
        ) as progress_bar:
            for archive_future in as_completed(archive_futures):
                archive_id, archive_path = archive_future.result()
                archives[archive_id] = archive_path
                progress_bar.update(n=1)

        # PHASE 3: Verifies archived data integrity against the original data if this is requested.
        if verify_integrity:
            archived_futures = {
                source_id: process_executor.submit(_load_numpy_archive, file_path=path)
                for source_id, path in archives.items()
            }

            archive_data = {}
            with console.progress(
                total=len(archives),
                description="Loading archive data into memory",
                unit="archives",
            ) as progress_bar:
                for source_id, integrity_future in archived_futures.items():
                    archive_data[source_id] = integrity_future.result()
                    progress_bar.update(n=1)

            verification_futures = [
                thread_executor.submit(
                    _compare_arrays,
                    source_id=source_id,
                    stem=stem,
                    original_array=original_array,
                    archived_array=archive_data[source_id][stem],
                )
                for source_id, source_data in loaded_data.items()
                for stem, original_array in source_data.items()
            ]

            with console.progress(
                total=len(verification_futures),
                description="Verifying archived data integrity",
                unit="entries",
            ) as progress_bar:
                for verify_future in as_completed(verification_futures):
                    verify_future.result()  # Propagates errors if comparison fails.
                    progress_bar.update(n=1)

        # PHASE 4: Removes source files if requested.
        if remove_sources:
            all_files = [file_path for files in source_files.values() for file_path in files]
            removal_futures = [thread_executor.submit(file_path.unlink) for file_path in all_files]

            with console.progress(
                total=len(all_files),
                description="Removing processed source files",
                unit="files",
            ) as progress_bar:
                for remove_future in as_completed(removal_futures):
                    remove_future.result()
                    progress_bar.update(n=1)


@contextmanager
def _progress_display(*, enabled: bool) -> Generator[None, None, None]:
    """Sets the console progress display for the duration of the context and restores the previous state on exit.

    Notes:
        The console tracks the progress display separately from its enabled state, so ``temporarily_enabled()`` does
        not restore it. The restore runs on the exception path as well, which keeps a failed archive assembly from
        leaving the process-global display flag inverted for every later caller.

    Args:
        enabled: Determines whether the progress display is active inside the context.
    """
    previous_progress = console.progress_enabled
    if enabled:
        console.enable_progress()
    else:
        console.disable_progress()
    try:
        yield
    finally:
        if previous_progress:
            console.enable_progress()
        else:
            console.disable_progress()


def _load_numpy_files(
    file_paths: tuple[Path, ...],
    *,
    memory_map: bool = False,
) -> tuple[tuple[str, ...], tuple[NDArray[Any], ...]]:
    """Loads multiple .npy files either into memory or as memory-mapped arrays.

    Args:
        file_paths: The paths to the .npy files to load.
        memory_map: Determines whether to memory-map the files or load them into memory (RAM).

    Returns:
        The first element is the tuple of loaded file names, without their extension. The second is the tuple of
        loaded or memory-mapped data arrays.
    """
    mmap_mode: Literal["r"] | None = "r" if memory_map else None
    results = [(file_path.stem, np.load(file=file_path, mmap_mode=mmap_mode)) for file_path in file_paths]
    return tuple(zip(*results, strict=False)) if results else ((), ())  # type: ignore[return-value]


def _load_numpy_archive(file_path: Path) -> dict[str, NDArray[Any]]:
    """Loads a NumPy .npz archive containing multiple arrays as a dictionary.

    Args:
        file_path: The path to the .npz log archive to load.

    Returns:
        The data of every log entry in the archive, keyed by the entry name.
    """
    with np.load(file=file_path) as npz_data:
        return {key: npz_data[key] for key in npz_data.files}


def _assemble_archive(
    output_directory: Path,
    source_id: int,
    source_data: dict[str, NDArray[Any]],
) -> tuple[int, Path]:
    """Assembles all log entries for a single source (producer) into a single .npz archive.

    Args:
        output_directory: The path to the directory in which to create the log archive.
        source_id: The ID-code of the source whose data is assembled into an archive.
        source_data: A dictionary that uses log-entries (entry names) as keys and stores the source data as NumPy
            array values.

    Returns:
        The first element is the source ID code. The second is the path to the uncompressed .npz log archive.
    """
    output_path = output_directory.joinpath(f"{source_id}{LOG_ARCHIVE_SUFFIX}")

    np.savez(file=output_path, allow_pickle=False, **source_data)

    return source_id, output_path


def _compare_arrays(source_id: int, stem: str, original_array: NDArray[Any], archived_array: NDArray[Any]) -> None:
    """Compares a pair of NumPy arrays for exact equality.

    Args:
        source_id: The ID-code of the source whose data is verified by this function.
        stem: The file name of the archived log entry being verified.
        original_array: The log entry data from the source .npy file.
        archived_array: The log entry data array from the .npz archive.

    Raises:
        ValueError: If the arrays do not match.
    """
    if not np.array_equal(a1=original_array, a2=archived_array):
        message = (
            f"Unable to verify the integrity of the assembled log archive for source {source_id}. The archived data "
            f"for entry {stem} must exactly match the data of the original .npy log entry, but the two differ."
        )
        console.error(message=message, error=ValueError)
