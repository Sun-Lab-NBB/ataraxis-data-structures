"""This module contains the DataLogger class that allows efficiently saving serialized byte-array data collected from
different Processes.

DataLogger works by creating the requested number of multithreaded logger processes and exposing a single shared Queue
that is sued to buffer and pipe the data to be logged to the saver processes. The class is optimized for working with
byte-serialized payloads stored in Numpy arrays.
"""

import sys
from queue import Empty, Queue
from typing import Optional
from pathlib import Path
from operator import index
from threading import Thread
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from ..shared_memory import SharedMemoryArray


@dataclass(frozen=True)
class LogPackage:
    """Stores the data and ID information to be logged by the DataLogger class and exposes methods for packaging this
    data into the format expected by the logger.

    This class collects, preprocesses and stores the data to be logged by the DataLogger instance. To be logged, entries
    have to be packed into this class instance and submitted (put) into the input_queue exposes by the DataLogger class.
    All data to be logged has to come wrapped into this class instance!
    """

    source_id: int
    """The ID code of the source that produced the data. Has to be unique across all systems that send data
    to be logged during a single runtime!"""

    time_stamp: int
    """The data acquisition time. Tracks when the data was originally acquired."""

    serialized_data: NDArray[np.uint8]
    """The data to be logged, stored as a one-dimensional bytes numpy array."""

    def get_data(self) -> tuple[str, NDArray[np.uint8]]:
        """Constructs and returns the filename and the serialized data package to be logged.

        Returns:
            A tuple of two elements. The first element is the name to use for the log file, which consists of
            zero-padded source id and zero-padded time stamp, separated by an underscore. The second element is the
            data to be logged as a one-dimensional bytes numpy array. THe logged data includes the original data object
            and the pre-pended source id and time stamp.
        """

        # Prepares the data by serializing originally non-numpy inputs and concatenating all data into one array
        serialized_time_stamp = np.frombuffer(buffer=np.uint64(self.time_stamp), dtype=np.uint8).copy()
        serialized_source = np.frombuffer(buffer=np.uint8(self.source_id), dtype=np.uint8).copy()

        # Note, it is assumed that each source produces the data sequentially and that timestamps are acquired with
        # sufficiently high resolution to resolve the order of data acquisition.
        # noinspection PyArgumentList
        data = np.concatenate([serialized_source, serialized_time_stamp, self.serialized_data], dtype=np.uint8).copy()

        # Zero-pads the timestamps. Uint64 allows for 19 zeroes in total, so pads to 19 digits. Statically appends the
        # source id as the first number, using underscore to separate source and timestamps.
        log_name = f"{self.source_id}_{self.time_stamp:019d}"

        return log_name, data


class DataLogger:
    """Saves input data as uncompressed byte numpy array (.npy) files using the requested number of cores and threads.

    This class instantiates and manages the runtime of a logger distributed over the requested number of cores and
    threads. The class exposes a shared multiprocessing Queue via the 'input_queue' property, which can be used to
    buffer and pipe the data to the logger from other Processes. The class expects the data to be first packaged into
    LogPackage class instance also available from this library, before it is sent to the logger via the queue object.

    Notes:
        Initializing the class does not start the logger processes! Call start() method to initialize the logger
        processes.

        Once the logger process(es) have been started, the class also initializes and maintains a watchdog thread that
        monitors the runtime status of the processes. If a process shuts down, the thread will detect this and raise
        the appropriate error to notify the user. Make sure the main process periodically releases GIL to allows the
        thread to assess the state of the remote process!

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

    def __init__(
        self,
        output_directory: Path,
        process_count: int = 1,
        thread_count: int = 5,
        sleep_timer: int = 5000,
    ) -> None:
        # Ensures numeric inputs are not negative.
        self._process_count: int = process_count if process_count > 1 else 1
        self._thread_count: int = thread_count if thread_count > 1 else 1
        self._sleep_timer: int = sleep_timer if sleep_timer > 0 else 0

        # If necessary, ensures that the output directory tree exists. This involves creating an additional folder
        # 'data_log', to which the data will be saved in an uncompressed format.
        self._output_directory: Path = output_directory.joinpath("data_log")
        # noinspection PyProtectedMember
        console._ensure_directory_exists(self._output_directory)  # This also ensures input is a valid Path object

        # Initializes a variable that tracks whether the class has been started.
        self._started: bool = False

        # Sets up the multiprocessing Queue to be shared by all logger and data source processes.
        self._mp_manager: SyncManager = Manager()
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        self._terminator_array: Optional[SharedMemoryArray] = None
        self._logger_processes: tuple[Process, ...] = tuple()
        self._watchdog_thread: Optional[Thread] = None

    def __repr__(self) -> str:
        """Returns a string representation of the DataLogger instance."""
        message = (
            f"DataLogger(output_directory={self._output_directory}, process_count={self._process_count}, "
            f"thread_count={self._thread_count}, sleep_timer={self._sleep_timer} microseconds, started={self._started})"
        )
        return message

    def __del__(self) -> None:
        """Ensures that logger resources are properly released when the class is garbage collected."""
        self.shutdown()

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
            name=f"data_logger_terminator",  # Static name ensures only one logger is active at a time.
            prototype=np.zeros(shape=1, dtype=np.uint8),
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

    def shutdown(self) -> None:
        """Stops the logger processes once they save all buffered data and releases reserved resources."""
        if not self._started:
            return

        # Amongst other things this soft-inactivates the watchdog thread.
        self._started = False

        # Issues the shutdown command to the remote processes and the watchdog thread
        if self._terminator_array is not None:
            self._terminator_array.write_data(index=0, data=1)

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
        """This script should be used by the watchdog thread to ensure the logger processes are alive during runtime.

        This script will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """

        timer = PrecisionTimer(precision="ms")

        # The watchdog script will ru until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):  # type: ignore
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay_noblock(delay=20, allow_sleep=True)

            if not self._started:
                continue

            # Only checks that processes are alive if they are started. The shutdown() flips the started tracker
            # before actually shutting down the processes, so there should be no collisions here.
            for num, process in enumerate(self._logger_processes, start=1):
                # If a started process is not alive, it has encountered an error forcing it to shut down.
                if not process.is_alive():
                    message = (
                        f"Data logger process {num} out of {len(self._logger_processes)} has been prematurely shut "
                        f"down. This likely indicates that the process has encountered a runtime error that "
                        f"terminated the process."
                    )
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

                # Submits task to thread pool to be executed
                executor.submit(DataLogger._save_data, filename, data)

            # If the queue is empty, invokes the sleep timer to reduce CPU load.
            except (Empty, KeyError):
                sleep_timer.delay_noblock(delay=sleep_time, allow_sleep=True)

            # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
            # before re-raising the exception to terminate the process.
            except Exception as e:
                sys.stderr.write(str(e))
                sys.stderr.flush()
                raise e

        # If the process escapes the loop due to encountering the shutdown command, shuts the executor threads and
        # disconnects from the terminator array before ending the runtime.
        executor.shutdown(wait=True)
        terminator_array.disconnect()

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
        was_enabled = console.is_enabled  # Records the initial console status
        if verbose and not was_enabled:
            console.enable()  # Ensures Console is enabled if verbose mode is enabled.

        # Groups files by source_id
        source_files: dict[int, list[Path]] = defaultdict(list)

        # Collects all .npy files and groups them by source_id
        for file_path in self._output_directory.glob("*.npy"):
            source_id = int(file_path.stem.split("_")[0])
            source_files[source_id].append(file_path)

        # Sorts files within each source_id group by their integer-convertible timestamp
        for source_id in source_files:
            source_files[source_id].sort(key=lambda x: int(x.stem.split("_")[1]))

        # Compresses all .npy files for each source into a single source-specific compressed .npz file
        source_data = {}
        for source_id, files in source_files.items():
            # Loads and uses the array data to fill a temporary dictionary that will be used for .npz archive creation.
            for file_path in files:
                stem = file_path.stem
                source_data[f"{stem}"] = np.load(file_path)
                if verbose:
                    console.echo(
                        message=f"Compressing {stem} file with data {source_data[f'{stem}']}.", level=LogLevel.INFO
                    )

            # Compresses the data for each source into a separate .npz archive named after the source_id
            output_path = self._output_directory.joinpath(f"{source_id}_data_log.npz")
            np.savez_compressed(output_path, **source_data)

            # If source removal is requested, deletes all compressed .npy files
            if remove_sources:
                for file in files:
                    if verbose:
                        console.echo(message=f"Removing compressed file {file}.", level=LogLevel.INFO)
                    file.unlink()

        if not was_enabled and verbose:
            console.echo(message=f"Log compression complete.", level=LogLevel.SUCCESS)
            console.disable()  # Disables the Console if it was enabled by this runtime.

    @property
    def input_queue(self) -> MPQueue:  # type: ignore
        """Returns the multiprocessing Queue used to buffer and pipe the data to the logger processes.

        Share this queue with all source processes that need to log data. To ensure correct data packaging, package the
        data using the LogPackage class exposed by this library before putting it into the queue.
        """
        return self._input_queue

    @staticmethod
    def _vacate_shared_memory_buffer() -> None:  # pragma: no cover
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed when DataLogger used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer, preventing the DataLogger from being
        reinitialized. This method allows manually removing that buffer to reset the system.

        The method will not do anything if the shared memory buffer does not exist.
        """
        try:
            buffer = SharedMemory(name=f"data_logger_terminator", create=False)
            buffer.close()
            buffer.unlink()
        except Exception:
            pass
