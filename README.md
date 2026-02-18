# ataraxis-data-structures

Provides classes and structures for storing, manipulating, and sharing data between Python processes.

![PyPI - Version](https://img.shields.io/pypi/v/ataraxis-data-structures)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ataraxis-data-structures)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/ataraxis-data-structures)
![PyPI - Status](https://img.shields.io/pypi/status/ataraxis-data-structures)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/ataraxis-data-structures)

___

## Detailed Description

This library aggregates the classes and methods used by other Ataraxis and Sun lab libraries for working with data.
This includes classes to manipulate the data, share (move) the data between different Python processes, and store the
data in non-volatile memory (on disk). Generally, these classes either implement novel functionality not available
through other popular libraries or extend existing functionality to match specific needs of other project Ataraxis
libraries.

## Features

- Supports Windows, Linux, and macOS.
- Provides a process- and thread-safe way of sharing data between multiple processes through a NumPy array structure.
- Extends the standard Python dataclass to support saving and loading its data to and from YAML files.
- Provides a fast and scalable data logger optimized for saving serialized data from multiple parallel processes in
  non-volatile memory.
- Offers efficient batch processing of log archives with support for parallel workflows.
- Includes a file-based processing pipeline tracker for coordinating multi-process and multi-host data processing jobs.
- Provides utilities for data integrity verification, directory transfer, and time-series interpolation.
- Apache 2.0 License.

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
  - [YamlConfig](#yamlconfig)
  - [SharedMemoryArray](#sharedmemoryarray)
  - [DataLogger](#datalogger)
  - [LogArchiveReader](#logarchivereader)
  - [ProcessingTracker](#processingtracker)
  - [Processing Utilities](#processing-utilities)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

___

## Dependencies

For users, all library dependencies are installed automatically by all supported installation methods. For developers,
see the [Developers](#developers) section for information on installing additional development dependencies.

___

## Installation

### Source

***Note,*** installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the
   [stable releases](https://github.com/Sun-Lab-NBB/ataraxis-data-structures/tags) that include precompiled binary
   and source code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Run `pip install .` to install the project and its dependencies.

### pip

Use the following command to install the library and all of its dependencies via
[pip](https://pip.pypa.io/en/stable/): `pip install ataraxis-data-structures`

___

## Usage

This section provides an overview of each component exposed by the library. For detailed information about method
signatures and parameters, consult the [API documentation](#api-documentation).

### YamlConfig

The YamlConfig class extends the functionality of the standard Python dataclass module by bundling the dataclass
instances with methods to save and load their data to and from .yaml files. Primarily, this functionality is
implemented to support storing runtime configuration data in a non-volatile, human-readable, and editable format.

The YamlConfig class is designed to be subclassed by custom dataclass instances to gain the .yaml saving and loading
functionality realized through the inherited `to_yaml()` and `from_yaml()` methods:

```python
from ataraxis_data_structures import YamlConfig
from dataclasses import dataclass
from pathlib import Path
import tempfile


# All YamlConfig functionality is accessed via subclassing.
@dataclass
class MyConfig(YamlConfig):
    integer: int = 0
    string: str = "random"


# Instantiates the test class using custom values that do not match the default initialization values.
config = MyConfig(integer=123, string="hello")

# Saves the instance data to a YAML file in a temporary directory. The saved data can be modified by directly
# editing the saved .yaml file.
tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes.
out_path = Path(tempdir.name).joinpath("my_config.yaml")  # Resolves the path to the output file.
config.to_yaml(file_path=out_path)

# Ensures that the cache file has been created.
assert out_path.exists()

# Creates a new MyConfig instance using the data inside the .yaml file.
loaded_config = MyConfig.from_yaml(file_path=out_path)

# Ensures that the loaded data matches the original MyConfig instance data.
assert loaded_config.integer == config.integer
assert loaded_config.string == config.string
```

### SharedMemoryArray

The SharedMemoryArray class supports sharing data between multiple Python processes in a thread- and process-safe way.
To do so, it implements a shared memory buffer accessed via an n-dimensional NumPy array instance, allowing different
processes to read and write any element(s) of the array.

#### SharedMemoryArray Creation

The SharedMemoryArray only needs to be instantiated once by the main runtime process (thread) and provided to all
children processes as an input. The initialization process uses the specified prototype NumPy array and unique buffer
name to generate a (new) NumPy array whose data is stored in a shared memory buffer accessible from any thread or
process. ***Note,*** the array dimensions and datatype cannot be changed after initialization.

```python
from ataraxis_data_structures import SharedMemoryArray
import numpy as np

# The prototype array and buffer name determine the layout of the SharedMemoryArray for its entire lifetime:
prototype = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
buffer_name = "unique_buffer_name"  # Has to be unique for all concurrently used SharedMemoryArray instances.

# To initialize the array, use the create_array() method. Do not call the class initialization method directly!
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)

# Ensures that the shared memory buffer is destroyed when the instance is garbage-collected.
sma.enable_buffer_destruction()

# The instantiated SharedMemoryArray object wraps an n-dimensional NumPy array with the same dimensions and data
# type as the prototype and uses the unique shared memory buffer name to identify the shared memory buffer to
# connect to from different processes.
assert sma.name == buffer_name
assert sma.shape == prototype.shape
assert sma.datatype == prototype.dtype

# Demonstrates the current values for the critical SharedMemoryArray parameters evaluated above:
print(sma)
```

#### SharedMemoryArray Connection, Disconnection, and Destruction

Each process using the SharedMemoryArray instance, including the process that created it, must use the `connect()`
method to connect to the array before reading or writing data. At the end of its runtime, each connected process must
call the `disconnect()` method to release the local reference to the shared buffer. The main process also needs to call
the `destroy()` method to destroy the shared memory buffer.

```python
import numpy as np
from ataraxis_data_structures import SharedMemoryArray

# Initializes a SharedMemoryArray
prototype = np.zeros(shape=6, dtype=np.uint64)
buffer_name = "unique_buffer"
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)

# This method has to be called before attempting to manipulate the data inside the array.
sma.connect()

# The connection status of the array can be verified at any time by using is_connected property:
assert sma.is_connected

# Each process that connected to the shared memory buffer must disconnect from it at the end of its runtime. On
# Windows platforms, when all processes are disconnected from the buffer, the buffer is automatically
# garbage-collected.
sma.disconnect()  # For each connect() call, there has to be a matching disconnect() call

assert not sma.is_connected

# On Unix platforms, the buffer persists even after being disconnected by all instances, unless it is explicitly
# destroyed.
sma.destroy()  # For each create_array() call, there has to be a matching destroy() call
```

#### Reading and Writing SharedMemoryArray Data

For routine data writing or reading operations, the SharedMemoryArray supports accessing its data via indexing or
slicing, just like a regular NumPy array. Critically, accessing the data in this way is process-safe, as the instance
first acquires an exclusive multiprocessing Lock before interfacing with the data. For more complex access scenarios,
it is possible to use the `array()` method to directly access and manipulate the underlying NumPy array object used by
the instance.

```python
import numpy as np
from ataraxis_data_structures import SharedMemoryArray

# Initializes a SharedMemoryArray
prototype = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
buffer_name = "unique_buffer"
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)
sma.connect()

# The SharedMemoryArray data can be accessed directly using indexing or slicing, just like any regular NumPy array
# or Python iterable:

# Index
assert sma[2] == np.uint64(3)
assert isinstance(sma[2], np.uint64)
sma[2] = 123  # Written data must be convertible to the datatype of the underlying NumPy array
assert sma[2] == np.uint64(123)

# Slice
assert np.array_equal(sma[:4], np.array([1, 2, 123, 4], dtype=np.uint64))
assert isinstance(sma[:4], np.ndarray)

# It is also possible to directly access the underlying NumPy array, which allows using the full range of NumPy
# operations. The accessor method can be used from within a context manager to enforce exclusive access to the
# array's data via an internal multiprocessing lock mechanism:
with sma.array(with_lock=True) as array:
    print(f"Before clipping: {array}")

    # Clipping replaces the out-of-bounds value '123' with '10'.
    array = np.clip(array, 0, 10)

    print(f"After clipping: {array}")

# Cleans up the array buffer
sma.disconnect()
sma.destroy()
```

#### Using SharedMemoryArray from Multiple Processes

While all methods showcased above run in the same process, the main advantage of the SharedMemoryArray class is that
it behaves the same way when used from different Python processes. The following example demonstrates using a
SharedMemoryArray across multiple concurrent processes:

```python
from multiprocessing import Process
from ataraxis_base_utilities import console
from ataraxis_time import PrecisionTimer
import numpy as np
from ataraxis_data_structures import SharedMemoryArray


def concurrent_worker(shared_memory_object: SharedMemoryArray, index: int) -> None:
    """This worker runs in a remote process.

    It increments the shared memory array variable by 1 if the variable is even. Since each increment shifts it to
    be odd, to work as intended, this process has to work together with a different process that increments odd
    values. The process shuts down once the value reaches 200.

    Args:
        shared_memory_object: The SharedMemoryArray instance to work with.
        index: The index inside the array to increment
    """
    # Connects to the array
    shared_memory_object.connect()

    # Runs until the value becomes 200
    while shared_memory_object[index] < 200:
        # Reads data from the input index
        shared_value = shared_memory_object[index]

        # Checks if the value is even and below 200
        if shared_value % 2 == 0 and shared_value < 200:
            # Increments the value by one and writes it back to the array
            shared_memory_object[index] = shared_value + 1

    # Disconnects and terminates the process
    shared_memory_object.disconnect()


if __name__ == "__main__":
    console.enable()  # Enables terminal printouts

    # Initializes a SharedMemoryArray
    sma = SharedMemoryArray.create_array("test_concurrent", np.zeros(5, dtype=np.int32))

    # Generates multiple processes and uses each to repeatedly write and read data from different indices of the
    # same array.
    processes = [Process(target=concurrent_worker, args=(sma, i)) for i in range(5)]
    for p in processes:
        p.start()

    # Finishes setting up the local array instance by connecting to the shared memory buffer and enabling the
    # shared memory buffer cleanup when the instance is garbage-collected (a safety feature).
    sma.connect()
    sma.enable_buffer_destruction()

    # Marks the beginning of the test runtime
    console.echo(f"Running the multiprocessing example on {len(processes)} processes...")
    timer = PrecisionTimer("ms")
    timer.reset()

    # For each of the array indices, increments the value of the index if it is odd. Child processes increment
    # even values and ignore odd ones, so the only way for this code to finish is if children and parent process
    # take turns incrementing shared values until they reach 200
    while np.any(sma[0:5] < 200):  # Runs as long as any value is below 200
        # Note, while it is possible to index the data from the SharedMemoryArray, it is also possible to retrieve
        # and manipulate the underlying NumPy array directly. This allows using the full range of NumPy operations
        # on the shared memory data:
        with sma.array(with_lock=True) as arr:
            mask = (arr % 2 != 0) & (arr < 200)  # Uses a boolean mask to discover odd values below 200
            arr[mask] += 1  # Increments only the values that meet the condition above

    # Waits for the processes to join
    for p in processes:
        p.join()

    # Verifies that all processes ran as expected and incremented their respective variable
    assert np.all(sma[0:5] == 200)

    # Marks the end of the test runtime.
    time_taken = timer.elapsed
    console.echo(f"Example runtime: complete. Time taken: {time_taken / 1000:.2f} seconds.")

    # Cleans up the shared memory array after all processes are terminated
    sma.disconnect()
    sma.destroy()
```

### DataLogger

The DataLogger class initializes and manages the runtime of a logger process running in an independent Process and
exposes a shared Queue object for buffering and piping data from any other Process to the logger. Currently, the class
is specifically designed for saving serialized byte arrays used by other Ataraxis libraries, most notably the
ataraxis-video-system and the ataraxis-transport-layer.

#### Creating and Starting the DataLogger

DataLogger is intended to only be initialized once in the main runtime thread (Process) and provided to all children
Processes as an input. ***Note,*** while a single DataLogger instance is typically enough for most use cases, it is
possible to use more than a single DataLogger instance at the same time.

```python
from pathlib import Path
import tempfile
from ataraxis_data_structures import DataLogger

# Due to the internal use of the 'Process' class, each DataLogger call has to be protected by the __main__ guard
# at the highest level of the call hierarchy.
if __name__ == "__main__":
    # As a minimum, each DataLogger has to be given the path to the output directory and a unique name to
    # distinguish the instance from any other concurrently active DataLogger instance.
    tempdir = tempfile.TemporaryDirectory()  # Creates a temporary directory for illustration purposes
    logger = DataLogger(output_directory=Path(tempdir.name), instance_name="my_name")

    # The DataLogger initialized above creates a new directory: 'tempdir/my_name_data_log' to store logged entries.

    # Before the DataLogger starts saving data, its saver process needs to be initialized via the start() method.
    # Until the saver is initialized, the instance buffers all incoming data in RAM (via the internal Queue
    # object), which may eventually exhaust the available memory.
    logger.start()

    # Each call to the start() method must be matched with a corresponding call to the stop() method. This method
    # shuts down the logger process and releases any resources held by the instance.
    logger.stop()
```

#### Data Logging

The DataLogger is explicitly designed to log serialized data of arbitrary size. To enforce the correct data
formatting, all data submitted to the logger must be packaged into a LogPackage class instance before it is put into
the DataLogger's input queue.

```python
from pathlib import Path
import tempfile
import numpy as np
from ataraxis_data_structures import DataLogger, LogPackage, assemble_log_archives
from ataraxis_time import get_timestamp, TimestampFormats

if __name__ == "__main__":
    # Initializes and starts the DataLogger.
    tempdir = tempfile.TemporaryDirectory()
    logger = DataLogger(output_directory=Path(tempdir.name), instance_name="my_name")
    logger.start()

    # The DataLogger uses a multiprocessing Queue to buffer and pipe the incoming data to the saver process. The
    # queue is accessible via the 'input_queue' property of each logger instance.
    logger_queue = logger.input_queue

    # The DataLogger is explicitly designed to log serialized data. All data submitted to the logger must be
    # packaged into a LogPackage instance to ensure that it adheres to the proper format expected by the logger
    # instance.
    source_id = np.uint8(1)  # Has to be an uint8 type
    timestamp = np.uint64(get_timestamp(output_format=TimestampFormats.INTEGER))  # Has to be an uint64 type
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint8)  # Has to be an uint8 NumPy array
    logger_queue.put(LogPackage(source_id, timestamp, data))

    # The timer used to timestamp the log entries has to be precise enough to resolve two consecutive data
    # entries. Due to these constraints, it is recommended to use a nanosecond or microsecond timer, such as the
    # one offered by the ataraxis-time library.
    timestamp = np.uint64(get_timestamp(output_format=TimestampFormats.INTEGER))
    data = np.array([6, 7, 8, 9, 10], dtype=np.uint8)
    logger_queue.put(LogPackage(source_id, timestamp, data))  # Same source id as the package above

    # Stops the data logger.
    logger.stop()

    # The DataLogger saves the input LogPackage instances as serialized NumPy byte array .npy files. The output
    # directory for the saved files can be queried from the DataLogger instance's 'output_directory' property.
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 2
```

#### Log Archive Assembly

To optimize the log writing speed and minimize the time the data sits in the volatile memory, all log entries are saved
to disk as separate NumPy array .npy files. While this format is efficient for time-critical runtimes, it is not
optimal for long-term storage and data transfer. To help with optimizing the post-runtime data storage, the library
offers the `assemble_log_archives()` function which aggregates .npy files from the same data source into an
(uncompressed) .npz archive.

```python
from pathlib import Path
import tempfile
import numpy as np
from ataraxis_data_structures import DataLogger, LogPackage, assemble_log_archives

if __name__ == "__main__":
    # Creates and starts the DataLogger instance.
    tempdir = tempfile.TemporaryDirectory()
    logger = DataLogger(output_directory=Path(tempdir.name), instance_name="my_name")
    logger.start()
    logger_queue = logger.input_queue

    # Generates and logs 255 data messages. This generates 255 unique .npy files under the logger's output
    # directory.
    for i in range(255):
        logger_queue.put(LogPackage(np.uint8(1), np.uint64(i), np.array([i, i, i], dtype=np.uint8)))

    # Stops the data logger.
    logger.stop()

    # Depending on the runtime context, a DataLogger instance can generate a large number of individual .npy files
    # as part of its runtime. While having advantages for real-time data logging, this format of storing the data
    # is not ideal for later data transfer and manipulation. Therefore, it is recommended to always use the
    # assemble_log_archives() function to aggregate the individual .npy files into one or more .npz archives.
    assemble_log_archives(
        log_directory=logger.output_directory, remove_sources=True, memory_mapping=True, verbose=True
    )

    # The archive assembly creates a single .npz file named after the source_id (1_log.npz), using all available
    # .npy files. Generally, each unique data source is assembled into a separate .npz archive.
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 0
    assert len(list(logger.output_directory.glob("**/*.npz"))) == 1
```

### LogArchiveReader

The LogArchiveReader class provides efficient access to .npz log archives generated by DataLogger instances. It
supports onset timestamp discovery, message iteration, and batch assignment for multiprocessing workflows.

#### Basic Usage

Each .npz archive contains messages from a single source (producer). The LogArchiveReader automatically discovers the
onset timestamp stored in the archive and converts elapsed timestamps to absolute UTC timestamps.

```python
from pathlib import Path
from ataraxis_data_structures import LogArchiveReader

# Creates a reader for an existing archive
archive_path = Path("/path/to/1_log.npz")
reader = LogArchiveReader(archive_path)

# The onset timestamp is automatically discovered (UTC epoch reference in microseconds)
print(f"Onset timestamp: {reader.onset_timestamp_us}")
print(f"Message count: {reader.message_count}")

# Iterates through all messages in the archive
for message in reader.iter_messages():
    print(f"Timestamp: {message.timestamp_us}, Payload size: {len(message.payload)}")
```

#### Batch Processing for Multiprocessing Workflows

The LogArchiveReader supports efficient batch processing for parallel workflows. The main process can generate batch
assignments, and worker processes can create lightweight reader instances by passing the pre-discovered onset timestamp
to skip redundant scanning.

```python
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from ataraxis_data_structures import LogArchiveReader
import numpy as np


def process_batch(archive_path: Path, onset_us: np.uint64, keys: list[str]) -> int:
    """Worker function that processes a batch of messages."""
    # Creates a lightweight reader with pre-discovered onset (skips onset discovery)
    reader = LogArchiveReader(archive_path, onset_us=onset_us)

    processed = 0
    for message in reader.iter_messages(keys=keys):
        # Process each message...
        processed += 1
    return processed


if __name__ == "__main__":
    archive_path = Path("/path/to/1_log.npz")

    # Main process discovers onset and generates batches
    reader = LogArchiveReader(archive_path)
    onset_us = reader.onset_timestamp_us
    batches = reader.get_batches(workers=4, batch_multiplier=4)

    # Distributes batches to worker processes
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_batch, archive_path, onset_us, batch) for batch in batches]
        total_processed = sum(f.result() for f in futures)

    print(f"Processed {total_processed} messages")
```

#### Reading All Messages at Once

For smaller archives, all messages can be read into memory at once:

```python
from pathlib import Path
from ataraxis_data_structures import LogArchiveReader

reader = LogArchiveReader(Path("/path/to/1_log.npz"))

# Returns a tuple of (timestamps_array, payloads_list)
timestamps, payloads = reader.read_all_messages()
print(f"Read {len(timestamps)} messages")
```

### ProcessingTracker

The ProcessingTracker class tracks the state of data processing pipelines and provides tools for communicating this
state between multiple processes and host machines. It uses a file-based approach with a .yaml file for state storage
and a .lock file for thread-safe access.

#### Creating and Initializing the Tracker

The ProcessingTracker extends YamlConfig and uses file locking to ensure safe concurrent access from multiple
processes.

```python
from pathlib import Path
from ataraxis_data_structures import ProcessingTracker

# Creates a tracker pointing to a .yaml file
tracker = ProcessingTracker(file_path=Path("/path/to/tracker.yaml"))

# Initializes jobs to be tracked (each job is a tuple of (job_name, specifier))
# Specifiers differentiate instances of the same job (e.g., different data batches)
job_ids = tracker.initialize_jobs([
    ("process_video", "session_001"),
    ("process_video", "session_002"),
    ("extract_frames", "session_001"),
    ("extract_frames", "session_002"),
    ("generate_report", ""),  # Empty specifier for jobs without batches
])

print(f"Initialized {len(job_ids)} jobs")
```

#### Managing Job Lifecycle

Jobs transition through states: SCHEDULED → RUNNING → SUCCEEDED or FAILED.

```python
from pathlib import Path
from ataraxis_data_structures import ProcessingTracker, ProcessingStatus

tracker = ProcessingTracker(file_path=Path("/path/to/tracker.yaml"))

# Generates a job ID using the same name and specifier used during initialization
job_id = ProcessingTracker.generate_job_id("process_video", "session_001")

# Marks the job as started (optionally with an executor ID like a SLURM job ID)
tracker.start_job(job_id, executor_id="slurm_12345")

# Queries the current status
status = tracker.get_job_status(job_id)
print(f"Job status: {status}")  # ProcessingStatus.RUNNING

# Marks the job as completed successfully
tracker.complete_job(job_id)

# Or, if the job failed:
# tracker.fail_job(job_id, error_message="Out of memory")
```

#### Querying Pipeline State

The ProcessingTracker provides methods for querying the overall pipeline state and individual job information.

```python
from pathlib import Path
from ataraxis_data_structures import ProcessingTracker, ProcessingStatus

tracker = ProcessingTracker(file_path=Path("/path/to/tracker.yaml"))

# Checks if all jobs have completed successfully
if tracker.complete:
    print("Pipeline completed successfully!")

# Checks if any job has failed
if tracker.encountered_error:
    print("Pipeline encountered errors!")

# Gets a summary of job counts by status
summary = tracker.get_summary()
for status, count in summary.items():
    print(f"{status.name}: {count}")

# Gets all job IDs with a specific status
failed_jobs = tracker.get_jobs_by_status(ProcessingStatus.FAILED)
scheduled_jobs = tracker.get_jobs_by_status("SCHEDULED")  # String names also work

# Searches for jobs by name or specifier patterns
matches = tracker.find_jobs(job_name="process", specifier="session_001")
for job_id, (name, spec) in matches.items():
    print(f"Found job: {name} ({spec})")

# Gets detailed job information
job_info = tracker.get_job_info(job_id)
print(f"Job: {job_info.job_name}, Status: {job_info.status}")
print(f"Started at: {job_info.started_at}, Completed at: {job_info.completed_at}")
```

#### Retrying Failed Jobs

Failed jobs can be reset for retry:

```python
from pathlib import Path
from ataraxis_data_structures import ProcessingTracker

tracker = ProcessingTracker(file_path=Path("/path/to/tracker.yaml"))

# Resets all failed jobs back to SCHEDULED status
retried_ids = tracker.retry_failed_jobs()
print(f"Reset {len(retried_ids)} failed jobs for retry")

# Or reset the entire tracker
tracker.reset()
```

### Processing Utilities

The library provides several utility functions for common data processing tasks.

#### Directory Checksum Calculation

The `calculate_directory_checksum()` function computes an xxHash3-128 checksum for an entire directory, accounting for
both file contents and directory structure.

```python
from pathlib import Path
from ataraxis_data_structures import calculate_directory_checksum

# Calculates checksum with progress tracking
checksum = calculate_directory_checksum(
    directory=Path("/path/to/data"),
    num_processes=None,  # Uses all available CPU cores
    progress=True,       # Shows progress bar
    save_checksum=True,  # Saves to ax_checksum.txt in the directory
)
print(f"Directory checksum: {checksum}")

# Calculates checksum without saving or progress tracking (for batch processing)
checksum = calculate_directory_checksum(
    directory=Path("/path/to/data"),
    progress=False,
    save_checksum=False,
    excluded_files={"ax_checksum.txt", ".gitignore"},  # Excludes specific files
)
```

#### Directory Transfer

The `transfer_directory()` function copies directories with optional integrity verification and parallel processing.

```python
from pathlib import Path
from ataraxis_data_structures import transfer_directory

# Transfers with integrity verification
transfer_directory(
    source=Path("/path/to/source"),
    destination=Path("/path/to/destination"),
    num_threads=4,           # Uses 4 threads for parallel copy
    verify_integrity=True,   # Verifies checksum after transfer
    remove_source=False,     # Keeps source after transfer
    progress=True,           # Shows progress bar
)

# Moves data (transfers and removes source)
transfer_directory(
    source=Path("/path/to/source"),
    destination=Path("/path/to/destination"),
    num_threads=0,           # Uses all available CPU cores
    verify_integrity=True,
    remove_source=True,      # Removes source after successful transfer
)
```

#### Directory Deletion

The `delete_directory()` function removes directories using parallel file deletion for improved performance.

```python
from pathlib import Path
from ataraxis_data_structures import delete_directory

# Deletes a directory and all its contents
delete_directory(Path("/path/to/directory"))
```

#### Data Interpolation

The `interpolate_data()` function aligns time-series data to target coordinates using linear interpolation (for
continuous data) or last-known-value interpolation (for discrete data).

```python
import numpy as np
from ataraxis_data_structures import interpolate_data

# Source data with timestamps and values
source_timestamps = np.array([0, 100, 200, 300, 400], dtype=np.uint64)
source_values = np.array([10.0, 20.0, 15.0, 25.0, 30.0], dtype=np.float64)

# Target timestamps for interpolation
target_timestamps = np.array([50, 150, 250, 350], dtype=np.uint64)

# Continuous interpolation (linear)
interpolated_continuous = interpolate_data(
    source_coordinates=source_timestamps,
    source_values=source_values,
    target_coordinates=target_timestamps,
    is_discrete=False,
)
print(f"Continuous: {interpolated_continuous}")  # [15.0, 17.5, 20.0, 27.5]

# Discrete interpolation (last known value)
discrete_values = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
interpolated_discrete = interpolate_data(
    source_coordinates=source_timestamps,
    source_values=discrete_values,
    target_coordinates=target_timestamps,
    is_discrete=True,
)
print(f"Discrete: {interpolated_discrete}")  # [1, 2, 3, 4]
```

___

## API Documentation

See the [API documentation](https://ataraxis-data-structures-api-docs.netlify.app/) for the detailed description of the
methods and classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to modify
the source code of this library.

### Installing the Project

***Note,*** this installation method requires **mamba version 2.3.2 or above**. Currently, all Sun lab automation
pipelines require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. `cd` to the root directory of the prepared project distribution.
4. Install the core Sun lab development dependencies into the ***base*** mamba environment via the
   `mamba install tox uv tox-uv` command.
5. Use the `tox -e create` command to create the project-specific development environment followed by `tox -e install`
   command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project.
   Currently, this library supports the three latest stable versions. It is recommended to use a tool like
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project uses `tox` for development automation. The following tox environments are available:

| Environment          | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `lint`               | Runs ruff formatting, ruff linting, and mypy type checking   |
| `stubs`              | Generates py.typed marker and .pyi stub files                |
| `{py312,...}-test`   | Runs the test suite via pytest for each supported Python     |
| `coverage`           | Aggregates test coverage into an HTML report                 |
| `docs`               | Builds the API documentation via Sphinx                      |
| `build`              | Builds sdist and wheel distributions                         |
| `upload`             | Uploads distributions to PyPI via twine                      |
| `install`            | Builds and installs the project into its mamba environment   |
| `uninstall`          | Uninstalls the project from its mamba environment            |
| `create`             | Creates the project's mamba development environment          |
| `remove`             | Removes the project's mamba development environment          |
| `provision`          | Recreates the mamba environment from scratch                 |
| `export`             | Exports the mamba environment as .yml and spec.txt files     |
| `import`             | Creates or updates the mamba environment from a .yml file    |

Run any environment using `tox -e ENVIRONMENT`. For example, `tox -e lint`.

***Note,*** all pull requests for this project have to successfully complete the `tox` task before being merged. To
expedite the task's runtime, use the `tox --parallel` command to run some tasks in parallel.

### Automation Troubleshooting

Many packages used in `tox` automation pipelines (uv, mypy, ruff) and `tox` itself may experience runtime failures. In
most cases, this is related to their caching behavior. If an unintelligible error is encountered with any of the
automation components, deleting the corresponding cache directories (`.tox`, `.ruff_cache`, `.mypy_cache`, etc.)
manually or via a CLI command typically resolves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-data-structures/tags) for the available project
releases.

___

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))

___

## License

This project is licensed under the Apache 2.0 License: see the [LICENSE](LICENSE) file for details.

___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.
