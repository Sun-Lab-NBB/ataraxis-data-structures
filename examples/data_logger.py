import time as tm
from pathlib import Path
import tempfile

import numpy as np

from ataraxis_data_structures import DataLogger, LogPackage

# Due to the internal use of Process classes, the logger has to be protected by the __main__ guard.
if __name__ == "__main__":
    # As a minimum, each DataLogger has to be given the output folder and the name to use for the shared buffer. The
    # name has to be unique across all DataLogger instances used at the same time.
    tempdir = tempfile.TemporaryDirectory()  # A temporary directory for illustration purposes
    logger = DataLogger(output_directory=Path(tempdir.name), instance_name="my_name")

    # The DataLogger will create a new folder: 'tempdir/my_name_data_log' to store logged entries.

    # Before the DataLogger starts saving data, its saver processes need to be initialized.
    logger.start()

    # The data can be submitted to the DataLogger via its input_queue. This property returns a multiprocessing Queue
    # object.
    logger_queue = logger.input_queue

    # The data to be logged has to be packaged into a LogPackage dataclass before being submitted to the Queue.
    source_id = np.uint8(1)  # Has to be an unit8
    timestamp = np.uint64(tm.perf_counter_ns())  # Has to be an uint64
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)  # Has to be an uint8 numpy array
    logger_queue.put(LogPackage(source_id, timestamp, data))

    # The timer used to timestamp the log entries has to be precise enough to resolve two consecutive datapoints
    # (timestamps have to differ for the two consecutive datapoints, so nanosecond or microsecond timers are best).
    timestamp = np.uint64(tm.perf_counter_ns())
    data = np.array([6, 7, 8, 9, 10], dtype=np.uint8)
    logger_queue.put(LogPackage(source_id, timestamp, data))  # Same source id

    # Shutdown ensures all buffered data is saved before the logger is terminated. This prevents all further data
    # logging until the instance is started again.
    logger.stop()

    # Verifies two .npy files were created, one for each submitted LogPackage. Note, DataLogger exposes the path to the
    # log folder via its output_directory property.
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 2

    # The logger also provides a method for compressing all .npy files into .npz archives. This method is intended to be
    # called after the 'online' runtime is over to optimize the memory occupied by data. To achieve minimal disk space
    # usage, call the method wit the remove_sources argument. The method verifies compressed data against the original
    # entries before removing source files, so it is always safe to delete source files.
    logger.compress_logs(remove_sources=True)

    # The compression creates a single .npz file named after the source_id
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 0
    assert len(list(logger.output_directory.glob("**/*.npz"))) == 1
