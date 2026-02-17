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

___

## Features

- Supports Windows, Linux, and macOS.
- Provides a Process- and Thread-safe way of sharing data between multiple processes through a NumPy array structure.
- Extends the standard Python dataclass to support saving and loading its data to / from YAML files.
- Provides a fast and scalable data logger optimized for saving serialized data from multiple parallel processes in 
  non-volatile memory.
- Apache 2.0 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)

___

## Dependencies

For users, all library dependencies are installed automatically by all supported installation methods 
(see the [Installation](#installation) section).

***Note!*** Developers should see the [Developers](#developers) section for information on installing additional 
development dependencies.

___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the 
   [stable releases](https://github.com/Sun-Lab-NBB/ataraxis-data-structures/releases) that include precompiled binary 
   and source code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
   binaries, use ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file.

### pip

Use the following command to install the library using pip: ```pip install ataraxis-data-structures```

___

## Usage

This section is broken into subsections for each exposed utility class or module. For each, it only provides the 
minimalistic (quickstart) functionality overview, which does not reflect the nuances of using each asset. To learn 
about the nuances, consult the [API documentation](#api-documentation) or see the [example implementations](examples).

### YamlConfig
The YamlConfig class extends the functionality of the standard Python dataclass module by bundling the dataclass 
instances with methods to save and load their data to / from .yaml files. Primarily, this functionality is implemented 
to support storing runtime configuration data in a non-volatile, human-readable, and editable format.

The YamlConfig class is designed to be **subclassed** by custom dataclass instances to gain the .yaml saving and 
loading functionality realized through the inherited **to_yaml()** and **from_yaml()** methods:
```
from ataraxis_data_structures import YamlConfig
from dataclasses import dataclass
from pathlib import Path
import tempfile


# All YamlConfig functionality is accessed via subclassing.
@dataclass
class MyConfig(YamlConfig):
    integer: int = 0
    string: str = 'random'


# Instantiates the test class using custom values that do not match the default initialization values.
config = MyConfig(integer=123, string='hello')

# Saves the instance data to a YAML file in a temporary directory. The saved data can be modified by directly editing
# the saved .yaml file.
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
The SharedMemoryArray only needs to be instantiated __once__ by the main runtime process (thread) and provided to all 
children processes as an input. The initialization process uses the specified prototype NumPy array and unique buffer 
name to generate a (new) NumPy array whose data is stored in a shared memory buffer accessible from any thread or 
process. *__Note!__* The array dimensions and datatype cannot be changed after initialization.
```
from ataraxis_data_structures import SharedMemoryArray
import numpy as np

# The prototype array and buffer name determine the layout of the SharedMemoryArray for its entire lifetime:
prototype = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
buffer_name = 'unique_buffer_name'  # Has to be unique for all concurrently used SharedMemoryArray instances.

# To initialize the array, use the create_array() method. Do not call the class initialization method directly!
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)

# Ensures that the shared memory buffer is destroyed when the instance is garbage-collected.
sma.enable_buffer_destruction()

# The instantiated SharedMemoryArray object wraps an n-dimensional NumPy array with the same dimensions and data type
# as the prototype and uses the unique shared memory buffer name to identify the shared memory buffer to connect to from
# different processes.
assert sma.name == buffer_name
assert sma.shape == prototype.shape
assert sma.datatype == prototype.dtype

# Demonstrates the current values for the critical SharedMemoryArray parameters evaluated above:
print(sma)
```

#### SharedMemoryArray Connection, Disconnection, and Destruction
Each process using the SharedMemoryArray instance, __including__ the process that created it, must use the 
__connect()__ method to connect to the array before reading or writing data. At the end of its runtime, each connected
process must call the __disconnect()__ method to release the local reference to the shared buffer. The **main** process 
also needs to call the __destroy()__ method to destroy the shared memory buffer.
```
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

# Each process that connected to the shared memory buffer must disconnect from it at the end of its runtime. On Windows
# platforms, when all processes are disconnected from the buffer, the buffer is automatically garbage-collected.
sma.disconnect()  # For each connect() call, there has to be a matching disconnect() call

assert not sma.is_connected

# On Unix platforms, the buffer persists even after being disconnected by all instances, unless it is explicitly
# destroyed.
sma.destroy()  # For each create_array() call, there has to be a matching destroy() call
```

#### Reading and Writing SharedMemoryArray Data
For routine data writing or reading operations, the SharedMemoryArray supports accessing its data via __indexing__ or 
__slicing__, just like a regular NumPy array. Critically, accessing the data in this way is process-safe, as the 
instance first acquires an exclusive multiprocessing Lock before interfacing with the data. For more complex access 
scenarios, it is possible to use the __array()__ method to directly access and manipulate the underlying NumPy array 
object used by the instance.
```
import numpy as np
from ataraxis_data_structures import SharedMemoryArray

# Initializes a SharedMemoryArray
prototype = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
buffer_name = "unique_buffer"
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)
sma.connect()

# The SharedMemoryArray data can be accessed directly using indexing or slicing, just like any regular NumPy array or
# Python iterable:

# Index
assert sma[2] == np.uint64(3)
assert isinstance(sma[2], np.uint64)
sma[2] = 123  # Written data must be convertible to the datatype of the underlying NumPy array
assert sma[2] == np.uint64(123)

# Slice
assert np.array_equal(sma[:4], np.array([1, 2, 123, 4], dtype=np.uint64))
assert isinstance(sma[:4], np.ndarray)

# It is also possible to directly access the underlying NumPy array, which allows using the full range of NumPy
# operations. The accessor method can be used from within a context manager to enforce exclusive access to the array's
# data via an internal multiprocessing lock mechanism:
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
it behaves the same way when used from different Python processes. See the [example](examples/shared_memory_array.py) 
script for more details.

### DataLogger
The DataLogger class initializes and manages the runtime of a logger process running in an independent Process and 
exposes a shared Queue object for buffering and piping data from any other Process to the logger. Currently, the 
class is specifically designed for saving serialized byte arrays used by other Ataraxis libraries, most notably the
__ataraxis-video-system__ and the __ataraxis-transport-layer__.

The sections below break down various aspects of working with the DataLogger instance. The individual sections can also
be seen as a combined [example](examples/data_logger.py) script.

#### Creating and Starting the DataLogger
DataLogger is intended to only be initialized ***once*** in the main runtime thread (Process) and provided to all 
children Processes as an input. ***Note!*** While a single DataLogger instance is typically enough for most use cases, 
it is possible to use more than a single DataLogger instance at the same time.
```
from pathlib import Path
import tempfile
from ataraxis_data_structures import DataLogger

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

    # Each call to the start() method must be matched with a corresponding call to the stop() method. This method shuts
    # down the logger process and releases any resources held by the instance.
    logger.stop()
```

#### Data Logging
The DataLogger is explicitly designed to log serialized data of arbitrary size. To enforce the correct data formatting, 
all data submitted to the logger ***must*** be packaged into a __LogPackage__ class instance before it is put into the
DataLoger’s input queue.
```
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
```

#### Log Archive Assembly
To optimize the log writing speed and minimize the time the data sits in the volatile memory, all log entries are saved 
to disk as separate NumPy array .npy files. While this format is efficient for time-critical runtimes, it is not 
optimal for long-term storage and data transfer. To help with optimizing the post-runtime data storage, the library 
offers the __assemble_log_archives()__ function which aggregates .npy files from the same data source into an
(uncompressed) .npz archive.

```
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

    # Generates and logs 255 data messages. This generates 255 unique .npy files under the logger's output directory.
    for i in range(255):
        logger_queue.put(LogPackage(np.uint8(1), np.uint64(i), np.array([i, i, i], dtype=np.uint8)))

    # Stops the data logger.
    logger.stop()

    # Depending on the runtime context, a DataLogger instance can generate a large number of individual .npy files as
    # part of its runtime. While having advantages for real-time data logging, this format of storing the data is not
    # ideal for later data transfer and manipulation. Therefore, it is recommended to always use the
    # assemble_log_archives() function to aggregate the individual .npy files into one or more .npz archives.
    assemble_log_archives(log_directory=logger.output_directory, remove_sources=True, memory_mapping=True, verbose=True)

    # The archive assembly creates a single .npz file named after the source_id (1_log.npz), using all available .npy
    # files. Generally, each unique data source is assembled into a separate .npz archive.
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 0
    assert len(list(logger.output_directory.glob("**/*.npz"))) == 1
```

___

## API Documentation

See the [API documentation](https://ataraxis-data-structures-api-docs.netlify.app/) for the detailed description of the 
methods and classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for project developers.

### Installing the Project

***Note!*** This installation method requires **mamba version 2.3.2 or above**. Currently, all Sun lab automation 
pipelines require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Install the core Sun lab development dependencies into the ***base*** mamba environment via the 
   ```mamba install tox uv tox-uv``` command.
5. Use the ```tox -e create``` command to create the project-specific development environment followed by 
   ```tox -e install``` command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project. 
   Currently, this library supports the three latest stable versions. It is recommended to use a tool like 
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check the [tox.ini file](tox.ini) for details about the 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All pull requests for this project have to successfully complete the ```tox``` task before being merged. 
To expedite the task’s runtime, use the ```tox --parallel``` command to run some tasks in-parallel.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself may experience runtime failures. In 
most cases, this is related to their caching behavior. If an unintelligible error is encountered with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a CLI command typically solves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-data-structures/tags) for the available project 
releases.

---

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

---
