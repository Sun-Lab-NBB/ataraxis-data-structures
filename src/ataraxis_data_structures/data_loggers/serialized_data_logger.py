import queue
from typing import Optional
from pathlib import Path
from threading import Thread
from collections import defaultdict
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.managers import SyncManager

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import console

from ..shared_memory import SharedMemoryArray


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

    def __init__(
        self,
        output_directory: Path,
        process_count: int = 1,
        thread_count: int = 5,
        sleep_timer: int = 5000,
    ) -> None:
        self._process_count: int = process_count
        self._thread_count: int = thread_count
        self._sleep_timer: int = sleep_timer if sleep_timer > 0 else 0  # Ensures sleep timer is not negative.

        # If necessary, ensures that the output directory tree exists. This involves creating an additional folder
        # 'data_log', to which the data will be saved in an uncompressed format.
        self._output_directory: Path = output_directory.joinpath("data_log")
        # noinspection PyProtectedMember
        console._ensure_directory_exists(self._output_directory)

        # Sets up the multiprocessing Queue to be shared by all logger and data source processes.
        self._mp_manager: SyncManager = Manager()
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"data_logger_terminator",  # This basically prevents two Loggers to be active at the same time.
            prototype=np.zeros(shape=1, dtype=np.uint8),
        )  # Instantiation automatically connects the main process to the array.

        # Pre-initializes assets used to store logger Processes and track logger active/inactive status
        self._logger_processes: Optional[tuple[Process, ...]] = None
        self._started: bool = False

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

        # Ensures the shared memory array is destroyed when the class is garbage-collected
        self._terminator_array.disconnect()
        self._terminator_array.destroy()

    def start(self) -> None:
        """Starts the logger processes."""
        if self._started:
            return

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

        for process in self._logger_processes:
            process.start()

        # Sets the tracker flag to enable shutdown()
        self._started = True

    def shutdown(self) -> None:
        """Stops the logger processes once they save all buffered data and releases reserved resources."""
        if not self._started:
            return

        self._terminator_array.write_data(0, 0)

        # If check is to appease mypy
        if self._logger_processes is not None:
            for process in self._logger_processes:
                process.join()

    @staticmethod
    def _save_data(filename: Path, data: NDArray[np.uint8]) -> None:  # pragma: no cover
        """Thread worker function that saves the data.

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
        shutdown = False  # Pre-initializes the shutdown flag

        # Creates thread pool for this process
        executor = ThreadPoolExecutor(max_workers=thread_count)

        # Instantiates a local queue for managing thread tasks
        local_queue = queue.Queue()  # type: ignore

        def process_local_queue() -> None:
            """Worker function that distributes local tasks to worker threads."""
            while True:
                try:
                    # Polls the queue without waiting
                    file_name, data_to_save = local_queue.get_nowait()
                    executor.submit(DataLogger._save_data, file_name, data_to_save)
                    local_queue.task_done()
                except queue.Empty:
                    # If the queue is empty, evaluates whether the shutdown flag is set. If so, terminates the thread
                    if shutdown:
                        break

        # Starts worker threads
        workers = []
        for _ in range(thread_count):
            worker = Thread(target=process_local_queue, daemon=True)
            worker.start()
            workers.append(worker)

        # Initializes the timer instance used to briefly inactivate the process when it consumes all data to be logged.
        sleep_timer = PrecisionTimer(precision="us")

        # Main process loop. This loop will run until BOTH the terminator flag is passed and the input queue is empty.
        while terminator_array.read_data(index=0, convert_output=False) or not input_queue.empty():
            data: NDArray[np.uint8]
            source_id: int
            object_count: int
            time_stamp: int
            try:
                # Gets data from input queue with timeout. Expects the data to be passed as a 4-element tuple.
                source_id, object_count, time_stamp, serialized_data = input_queue.get_nowait()

                # Prepares the data by serializing originally non-numpy inputs and concatenating all data into one array
                serialized_time_stamp = np.frombuffer(buffer=np.uint64(time_stamp).tobytes(), dtype=np.uint8)
                serialized_source = np.frombuffer(buffer=np.uint64(source_id).tobytes(), dtype=np.uint8)

                # Note, object_count is only needed to properly handle logging with multiple cores. It is assumed that
                # each source produces the data sequentially and, therefore, that timestamps can be used to resolve the
                # order of data acquisition.
                data = np.concatenate(
                    [serialized_source, serialized_time_stamp, serialized_data], dtype=np.uint8
                ).copy()

                # Generates the filename for the data. This includes the source id, the process number, and a
                # zero-padded object count. The object_count is padded to 20 zeroes, which should be enough for most
                # envisioned runtimes.
                filename = output_directory.joinpath(f"{source_id}_{object_count:020d}")

                # Adds the logging task to local queue for thread processing
                local_queue.put((filename, data))

            except queue.Empty:
                # If the queue becomes empty, blocks the process for the requested number of microseconds. With
                # sufficiently log delays, this can help with managing the power draw and the thermal load of the host
                # system.
                sleep_timer.delay_noblock(delay=sleep_time, allow_sleep=True)

        # If the loop above is escapes, the logger received the termination command.
        # Sets the flag used to terminate the threads
        shutdown = True

        # Waits for all thread tasks to complete
        local_queue.join()

        # Waits for worker threads to finish
        for worker in workers:
            worker.join()

        # Shuts the thread pool down
        executor.shutdown(wait=True)

        # Disconnects from the shared memory array
        terminator_array.disconnect()

    def compress_logs(self, remove_sources: bool = False, verbose: bool = False) -> None:
        """Consolidates all .npy files in the log directory into a single compressed .npz file.

        Individual .npy files are grouped by source_id and acquisition number.

        Args:
            remove_sources: Determines whether to remove the individual .npy files after they have been consolidated
                into .npz archives.
            verbose: Determines whether to print processed arrays to console. This option is mostly useful for debugging
                other Ataraxis libraries and should be disabled by default.
        """
        # Groups files by source_id
        source_files: dict[int, list[Path]] = defaultdict(list)

        # Collects all .npy files and groups them by source_id
        for file_path in self._output_directory.glob("*.npy"):
            source_id = int(file_path.stem.split("_")[0])
            source_files[source_id].append(file_path)

        # Sorts files within each source_id group by sequence (object) number
        for source_id in source_files:
            source_files[source_id].sort(key=lambda x: int(x.stem.split("_")[1]))

        # Compresses all .npy files for each source into a single compressed .npz file
        source_data = {}
        for source_id, files in source_files.items():
            # Loads and uses the array data to fill a temporary dictionary that will be used for .npz archive creation.
            for file_path in files:
                stem = file_path.stem
                source_data[f"{stem}"] = np.load(file_path)

                if verbose:
                    console.echo(message=f"Compressing {stem} file with data {source_data[f'{stem}']}")

            # Compresses the data for each source into a separate .npz archived named after the source_id
            output_path = self._output_directory.parent.joinpath(f"{source_id}.npz")
            np.savez_compressed(output_path, **source_data)

            # If source removal is requested, deletes all compressed .npy files
            if remove_sources:
                for file in files:
                    file.unlink()

    @property
    def input_queue(self) -> MPQueue:  # type: ignore
        """Returns the multiprocessing Queue used to buffer and pipe the data to the logger processes.

        Share this queue with all source processes that need to log data.
        """
        return self._input_queue
