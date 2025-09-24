import time as tm
from pathlib import Path
import tempfile
import numpy as np
from ataraxis_data_structures import DataLogger, LogPackage, assemble_log_archives
from ataraxis_time import get_timestamp, TimestampFormats

# Due to the internal use of Process classes, the logger has to be protected by the __main__ guard.
if __name__ == "__main__":
    # As a minimum, each DataLogger has to be given the output folder and a unique name to distinguish the instance from
    # any other concurrently active DataLogger instance.
    tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes
    logger = DataLogger(output_directory=Path(tempdir.name), instance_name="my_name")

    # The DataLogger initialized above creates a new folder: 'tempdir/my_name_data_log' to store logged entries.

    # Before the DataLogger starts saving data, its saver processes need to be initialized.
    logger.start()

    # The data can be submitted to the DataLogger via its input_queue. This property returns a multiprocessing Queue
    # object.
    logger_queue = logger.input_queue

    # The data to be logged has to be packaged into a LogPackage dataclass before being submitted to the Queue.
    source_id = np.uint8(1)  # Has to be an unit8 type
    timestamp = np.uint64(get_timestamp(output_format=TimestampFormats.INTEGER))  # Has to be an uint64 type
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)  # Has to be an uint8 numpy array
    logger_queue.put(LogPackage(source_id, timestamp, data))

    # The timer used to timestamp the log entries has to be precise enough to resolve two consecutive datapoints
    # (timestamps have to differ for the two consecutive datapoints, so nanosecond or microsecond timers are best).
    timestamp = np.uint64(get_timestamp(output_format=TimestampFormats.INTEGER))
    data = np.array([6, 7, 8, 9, 10], dtype=np.uint8)
    logger_queue.put(LogPackage(source_id, timestamp, data))  # Same source id

    # Stopping the logger ensures all buffered data is saved before the remote logger process is terminated. This
    # prevents all further data logging until the instance is started again.
    logger.stop()

    # Verifies that the two .npy files were created, one for each submitted LogPackage. Note, DataLogger exposes the
    # path to the log folder via its output_directory property.
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 2

    # The library also provides a function for aggregating all individual .npy files into .npz archives. This function
    # is intended to be called after the 'online' runtime is over to optimize how the data is stored on disk.
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, memory_mapping=True, verbose=True)

    # The archive assembly creates a single .npz file named after the source_id
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 0
    assert len(list(logger.output_directory.glob("**/*.npz"))) == 1
