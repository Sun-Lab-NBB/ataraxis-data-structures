from pathlib import Path
import tempfile
import numpy as np
from ataraxis_data_structures import DataLogger, LogPackage, assemble_log_archives
from ataraxis_time import get_timestamp, TimestampFormats

# Due to the internal use of the 'Process' class, each DataLogger call has to be protected by the __main__ guard at
# the highest level of the call hierarchy.
if __name__ == "__main__":
    # As a minimum, each DataLogger has to be given the path to the output directory and a unique name to distinguish
    # the instance from any other concurrently active DataLogger instance.
    tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes
    logger = DataLogger(output_directory=Path(tempdir.name), instance_name="my_name")

    # The DataLogger initialized above creates a new directory: 'tempdir/my_name_data_log' to store logged entries.

    # Before the DataLogger starts saving data, its saver process needs to be initialized via the start() method.
    # Until the saver is initialized, the instance buffers all incoming data in RAM (via the internal Queue object),
    # which may eventually exhaust the available memory.
    logger.start()

    # The DataLogger uses a multiprocessing Queue to buffer and pipe the incoming data to the saver process. The queue
    # is accessible via the 'input_queue' property of each logger instance.
    logger_queue = logger.input_queue

    # The DataLogger is explicitly designed to log serialized data. All data submitted to the logger must be packaged
    # into a LogPackage instance to ensure that it adheres to the proper format expected by the logger instance.
    source_id = np.uint8(1)  # Has to be an unit8 type
    timestamp = np.uint64(get_timestamp(output_format=TimestampFormats.INTEGER))  # Has to be an uint64 type
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)  # Has to be an uint8 NumPy array
    logger_queue.put(LogPackage(source_id, timestamp, data))

    # The timer used to timestamp the log entries has to be precise enough to resolve two consecutive data entries.
    # Due to these constraints, it is recommended to use a nanosecond or microsecond timer, such as the one offered
    # by the ataraxis-time library.
    timestamp = np.uint64(get_timestamp(output_format=TimestampFormats.INTEGER))
    data = np.array([6, 7, 8, 9, 10], dtype=np.uint8)
    logger_queue.put(LogPackage(source_id, timestamp, data))  # Same source id as the package above

    # Stops the data logger.
    logger.stop()

    # The DataLogger saves the input LogPackage instances as serialized NumPy byte array .npy files. The output
    # directory for the saved files can be queried from the DataLogger instance's 'output_directory' property.
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 2

    # Depending on the runtime context, a DataLogger instance can generate a large number of individual .npy files as
    # part of its runtime. While having advantages for real-time data logging, this format of storing the data is not
    # ideal for later data transfer and manipulation. Therefore, it is recommended to always use the
    # assemble_log_archives() function to aggregate the individual .npy files into one or more .npz archives.
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, memory_mapping=True, verbose=True)

    # The archive assembly creates a single .npz file named after the source_id (1_log.npz), using all available .npy
    # files. Generally, each unique data source is assembled into a separate .npz archive.
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 0
    assert len(list(logger.output_directory.glob("**/*.npz"))) == 1
