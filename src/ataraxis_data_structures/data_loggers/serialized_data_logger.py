import numpy as np
from numpy.typing import NDArray
from ..data_structures import NestedDictionary
from ..shared_memory import SharedMemoryArray
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from multiprocessing.managers import SyncManager
from ataraxis_base_utilities import console


class DataLogger:
    def __init__(self, process_count: int = 1, thread_count: int = 5):
        self._source_map = NestedDictionary()

        # Sets up the multiprocessing Queue to be shared by all logger and source processes. This queue can be used to
        # share and buffer the data to be logged with the logger processes that will log it.
        self._mp_manager: SyncManager = multiprocessing.Manager()
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"logger_terminator_array",  # Uses class name to ensure the array buffer name is unique
            prototype=np.empty(shape=1, dtype=np.uint8),
        )  # Instantiation automatically connects the main process to the array.

    def add_source(self, source_code: np.uint8, source_name: str, output_directory: Path):
        self._source_map.write_nested_value(variable_path=f"{source_name}.code", value=source_code)

    @staticmethod
    def log_cycle(input_queue: MPQueue, terminator_array: SharedMemoryArray, output_directory: Path):
        terminator_array.connect()
        object_count = 0
        while terminator_array.read_data(index=1, convert_output=False) or not input_queue.empty():
            data: NDArray[np.uint8]
            timestamp: int
            source: int

            if not input_queue.empty():
                source, timestamp, data = input_queue.get()
                data = np.concat(arrays=(np.uint8(source), np.uint64(timestamp), data), dtype=np.uint8)
            else:
                continue

            object_count += 1
            filename = output_directory.joinpath(f"{source}_{object_count}")

            np.save(file=filename, arr=data, allow_pickle=False)
