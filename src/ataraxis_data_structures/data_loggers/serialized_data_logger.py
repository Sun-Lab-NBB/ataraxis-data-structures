"""This module contains the DataLogger class that allows efficiently saving serialized byte-array data collected from
different Processes.

DataLogger works by creating the requested number of multithreaded logger processes and exposing a single shared Queue
that is used to buffer and pipe the data to be logged to the saver processes. The class is optimized for working with
byte-serialized payloads stored in Numpy arrays.
"""

import os
import sys
from queue import Empty
from pathlib import Path
import platform
from threading import Thread
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.managers import SyncManager

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import console, ensure_directory_exists

from ..shared_memory import SharedMemoryArray


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
    """The ID code of the source that produced the data. Has to be unique across all systems that send data
    to same DataLogger instance during runtime, as this information is used to identify sources inside log files!"""

    time_stamp: np.uint64
    """The data acquisition time. Tracks when the data was originally acquired."""

    serialized_data: NDArray[np.uint8]
    """The data to be logged, stored as a one-dimensional bytes numpy array."""

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
        serialized_time_stamp = np.frombuffer(buffer=self.time_stamp, dtype=np.uint8).copy()
        serialized_source = np.frombuffer(buffer=self.source_id, dtype=np.uint8).copy()

        # Note, it is assumed that each source produces the data sequentially and that timestamps are acquired with
        # high enough resolution to resolve the order of data acquisition.
        data = np.concatenate([serialized_source, serialized_time_stamp, self.serialized_data], dtype=np.uint8).copy()

        # Zero-pads ID and timestamp. Uses the correct number of zeroes to represent the number of digits that fits into
        # each datatype (uint8 and uint64).
        log_name = f"{self.source_id:03d}_{self.time_stamp:020d}"

        return log_name, data


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
        ensure_directory_exists(self._output_directory)  # This also ensures input is a valid Path object

        # Sets up the multiprocessing Queue to be shared by all logger and data source processes.
        self._mp_manager: SyncManager = Manager()
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

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

        # Shuts down the multiprocessing manager, which automatically garbage-collects queue objects.
        self._mp_manager.shutdown()

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
        while not self._terminator_array.read_data(index=0):  # type: ignore
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay_noblock(delay=20, allow_sleep=True)

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

            # The error is raised outside teh checking context to allow gracefully shutting down all assets before
            # terminating runtime with an error message.
            if error:
                message = (
                    f"DataLogger process {process_number} out of {len(self._logger_processes)} has been prematurely "
                    f"shut down. This likely indicates that the process has encountered a runtime error that "
                    f"terminated the process."
                )
                # Since the raised error terminates class runtime, cleans up all resources, just like how stop() does it
                self._terminator_array.write_data(index=0, data=np.uint8(1))  # type: ignore
                for process in self._logger_processes:
                    process.join()
                self._mp_manager.shutdown()
                self._terminator_array.disconnect()  # type: ignore
                self._terminator_array.destroy()  # type: ignore
                self._started = False  # Prevents stop() from running via __del__
                console.error(message=message, error=RuntimeError)

    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:  # pragma: no cover
        """Thread worker function that saves the data.

        Args:
            filename: The name of the file to save the data to. Note, the name has to be suffix-less, as '.npy' suffix
                will be appended automatically.
            data: The data to be saved, packaged into a one-dimensional bytes array.

        Since data saving is primarily IO-bound, using multiple threads per each Process is likely to achieve the best
        saving performance.
        """
        np.save(file=filename, arr=data, allow_pickle=False)

    @staticmethod
    def _log_cycle(
        input_queue: MPQueue,  # type: ignore
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

        # Creates thread pool for this process. It will manage the local saving threads
        executor = ThreadPoolExecutor(max_workers=thread_count)

        # Initializes the timer instance used to temporarily pause the execution when there is no data to process
        sleep_timer = PrecisionTimer(precision="us")

        # Main process loop. This loop will run until BOTH the terminator flag is passed and the input queue is empty.
        while not terminator_array.read_data(index=0, convert_output=False) or not input_queue.empty():
            try:
                # Gets data from input queue with timeout. The data is expected to be packaged into the LogPackage
                # class.
                package: LogPackage = input_queue.get_nowait()

                # Pre-processes the data
                file_name, data = package.get_data()

                # Generates the full name for the output log file by merging the name of the specific file with the
                # path to the output directory
                filename = output_directory.joinpath(file_name)

                # Submits the task to thread pool to be executed
                executor.submit(DataLogger._save_data, filename, data)

            # If the queue is empty, invokes the sleep timer to reduce CPU load.
            except (Empty, KeyError):
                sleep_timer.delay_noblock(delay=sleep_time, allow_sleep=True)

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
        # Collects all .npy files for the processed source
        source_data = {}
        for file_path in files:
            stem = file_path.stem
            # If the function is not running in memory-mapping mode, directly loads the data from each array into
            # memory (RAM). This is faster than memory-mapping but, depending on the use case, may lead to out-of-memory
            # errors.
            if not mem_map:
                source_data[f"{stem}"] = np.load(file_path)

            # Alternatively, instead of loading data into memory, generates a memory-mapped view of each source array.
            # This is slower than using RAM, but may be necessary for some use cases.
            else:
                array_data = np.load(file_path, mmap_mode="r")
                source_data[f"{stem}"] = array_data

        # Compresses individual entries into a single .npz archive
        output_path = self._output_directory.joinpath(f"{source_id}_log.npz")
        np.savez_compressed(output_path, **source_data)

        # If requested, cleans up no longer necessary source files
        if remove_sources:
            # Verifies each file individually, keeping only one entry in memory at a time. Loading an .npz file
            # automatically uses memory mapping, unless a specific array is accessed.
            with np.load(output_path) as compressed_data:
                for file_path in files:
                    stem = file_path.stem

                    # Loads and verifies one original file at a time. This should have minimal RAM footprint.
                    original = source_data[f"{stem}"]
                    compressed = compressed_data[f"{stem}"]

                    # If verification is failed, triggers an error
                    if not np.array_equal(original, compressed):  # pragma: no cover
                        message = (
                            f"Unable to compress the log entries for the source {source_id} logged by DataLogger "
                            f"{self.name}. Data integrity check failed for entry {stem}. Compressed data does not "
                            f"match original, so cannot remove the original file."
                        )
                        console.error(message=message, error=ValueError)

            # If this point is reached, it is safe to remove all original files.
            for file in files:
                file.unlink()

        # Returns teh source id to, if necessary, update the progress bar
        return source_id

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
        # Resolves the number of threads to use
        if max_workers is None:
            max_workers = os.cpu_count() - 4  # type: ignore
        elif not isinstance(max_workers, int) or max_workers <= 0:
            max_workers = 1  # Minimum of 1 worker

        # Collects all .npy files and groups them by source_id
        source_files: dict[int, list[Path]] = defaultdict(list)
        for file_path in self._output_directory.glob("*.npy"):
            source_id = int(file_path.stem.split("_")[0])
            source_files[source_id].append(file_path)

        # Sorts files within each source_id group by their integer-convertible timestamp
        for source_id in source_files:
            source_files[source_id].sort(key=lambda x: int(x.stem.split("_")[1]))

        # Due to erratic interaction between memory mapping and Windows (as always), disables memory mapping on
        # Windows. Use max_workers to avoid out-of-memory errors on Windows.
        if memory_mapping and platform.system() == "Windows":
            memory_mapping = False

        # Processes sources in parallel using threads. Since compression is primarily done by numpy and I/O processors,
        # this is likely the most efficient way of processing multiple sources.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit each source for compression in-parallel
            future_to_source = {
                executor.submit(self._compress_source, source_id, files, remove_sources, memory_mapping): source_id
                for source_id, files in source_files.items()
            }

            # If the method is called in verbose mode, displays progress to user via the progress bar
            with tqdm(
                total=len(source_files),
                desc="Generating compressed archives for sources",
                unit="sources",
                disable=not verbose,
            ) as pbar:
                # Progress bar is updated with each processed source
                for future in as_completed(future_to_source):
                    future.result()
                    pbar.update(1)

    @property
    def input_queue(self) -> MPQueue:  # type: ignore
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
