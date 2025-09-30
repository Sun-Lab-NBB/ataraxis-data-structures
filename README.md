# ataraxis-data-structures

A Python library that provides classes and structures for storing, manipulating, and sharing data between Python 
processes.

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
- GPL 3 License.

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
2. Unpack the downloaded distribution archive and ```cd``` to the root directory of the project.
3. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
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
to support storing runtime configuration data in non-volatile and human-readable (and editable!) format.

Any dataclass that subclasses the base 'YamlConfig' class exposed by this library inherits two methods: **to_yaml()** 
and **from_yaml** that jointly enable caching dataclass instance data to .yaml files.
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
The SharedMemoryArray class allows sharing data between multiple Python processes in a thread- and process-safe way.
To do so, it implements a shared memory buffer accessed via an n-dimensional NumPy array instance, allowing different 
processes to read and write any element(s) of the array.

#### Array creation
The SharedMemoryArray only needs to be instantiated __once__ by the main runtime process (thread) and provided to all 
children processes as an input. The initialization process uses the specified prototype NumPy array and unique buffer 
name to generate a new . 

*__Note!__* The array dimensions and datatype cannot be changed after initialization, the resultant SharedMemoryArray
will always use the same shape and datatype.
```
from ataraxis_data_structures import SharedMemoryArray
import numpy as np

# The prototype array and buffer name determine the layout of the SharedMemoryArray for its entire lifetime:
prototype = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
buffer_name = 'unique_buffer'

# To initialize the array, use create_array() method. DO NOT use class initialization method directly! This example
# is configured to recreate the buffer, it already exists.
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype, exist_ok=True)

# The instantiated SharedMemoryArray object wraps an array with the same dimensions and data type as the prototype
# and uses the unique buffer name to identify the shared memory buffer to connect from different processes.
assert sma.name == buffer_name
assert sma.shape == prototype.shape
assert sma.datatype == prototype.dtype

# Remember to clean up at the end. If this si not done, the shared memory buffer may be left hogging computer resources
# after the runtime is over (Only on Unix platforms).
sma.disconnect()
sma.destroy()
```

#### Array connection, disconnection and destruction
Each __child__ process has to use the __connect()__ method to connect to the array before reading or writing data. 
The parent process that has created the array connects to the array automatically during creation and does not need to 
be reconnected. At the end of each connected process runtime, you need to call the __disconnect()__ method to remove 
the reference to the shared buffer:
```
import numpy as np

from ataraxis_data_structures import SharedMemoryArray

# Initializes a SharedMemoryArray
prototype = np.zeros(shape=6, dtype=np.uint64)
buffer_name = "unique_buffer"
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)

# This method has to be called before any child process that received the array can manipulate its data. While the
# process that creates the array is connected automatically, calling the connect() method again does not have negative
# consequences.
sma.connect()

# You can verify the connection status of the array by using is_connected property:
assert sma.is_connected

# This disconnects the array from shared buffer. On Windows platforms, when all instances are disconnected from the
# buffer, the buffer is automatically garbage-collected. Therefore, it is important to make sure the array has at least
# one connected instance at all times, unless you no longer intend to use the class. On Unix platforms, the buffer may
# persist even after being disconnected by all instances.
sma.disconnect()  # For each connect(), there has to be a matching disconnect() statement

assert not sma.is_connected
```

#### Reading array data
To read from the array wrapped by the class, you can use the __read_data()__ method. The method allows reading
individual values and array slices and return data as NumPy or Python values:
```
import numpy as np
from ataraxis_data_structures import SharedMemoryArray

# Initializes a SharedMemoryArray
prototype = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
buffer_name = "unique_buffer"
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)
sma.connect()

# The method can be used to read individual elements from the array. By default, the data is read as the numpy datatype
# used by the array.
output = sma.read_data(index=2)
assert output == np.uint64(3)
assert isinstance(output, np.uint64)

# To read a slice of the array, provide a tuple of two indices (for closed range) or a tuple of one index (for open
# range).
output = sma.read_data(index=(1, 4), convert_output=False, with_lock=False)
assert np.array_equal(output, np.array([2, 3, 4], dtype=np.uint64))
assert isinstance(output, np.ndarray)

```

#### Writing array data
To write data to the array wrapped by the class, use the __write_data()__ method. Its API is deliberately kept very 
similar to the read method:
```
import numpy as np
from ataraxis_data_structures import SharedMemoryArray

# Initializes a SharedMemoryArray
prototype = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)
buffer_name = "unique_buffer"
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)
sma.connect()

# Data writing method has a similar API to data reading method. It can write scalars and slices to the shared memory
# array. It tries to automatically convert the input into the type used by the array as needed:
sma.write_data(index=1, data=7, with_lock=True)
assert sma.read_data(index=1, convert_output=True) == 7

# Writing by slice is also supported
sma.write_data(index=(1, 3), data=[10, 11], with_lock=False)
assert sma.read_data(index=(0,), convert_output=True) == [1, 10, 11, 4, 5, 6]
```

#### Using the array from multiple processes
While all methods showcased above run from the same process, the main advantage of the class is that they work
just as well when used from different Python processes. See the [example](examples/shared_memory_array.py) script for 
more details.

### DataLogger
The DataLogger class sets up data logger instances running on isolated cores (Processes) and exposes a shared Queue 
object for buffering and piping data from any other Process to the logger cores. Currently, the logger is only intended 
for saving serialized byte arrays used by other Ataraxis libraries (notably: ataraxis-video-system and 
ataraxis-transport-layer).

#### Logger creation and use
DataLogger is intended to only be initialized once and used by many input processes, which should be enough for most 
use cases. However, it is possible to initialize multiple DataLogger instances by overriding the default 'instance_name'
argument value. The example showcased below is also available as a [script](examples/data_logger.py):
```
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
    # usage, call the method with the remove_sources argument.
    logger.compress_logs(remove_sources=True)

    # The compression creates a single .npz file named after the source_id
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 0
    assert len(list(logger.output_directory.glob("**/*.npz"))) == 1
```

#### Log compression
To optimize runtime performance (log writing speed), all log entries are saved to disk as serialized NumPy arrays, each
stored in a separate .npy file. While this format is adequate during time-critical runtimes, it is not optimal for 
long-term storage and data transfer.

To facilitate long-term log storage, the library exposes a global, multiprocessing-safe, and instance-independent 
function `compress_npy_logs()`. This function behaves exactly like the instance-bound log compression method does, but 
can be used to compress log entries without the need to have an initialized DataLogger instance. You can
use the `output_directory` property of an initialized DataLogger instance to get the path to the directory that stores 
uncompressed log entries, which is a required argument for the instance-independent log compression function.

Alternatively, you can also use the `compress_logs` method exposed by the DataLogger instance to compress the logs 
immediately after runtime. Overall, it is highly encouraged to compress the logs as soon as possible.

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
2. Unpack the downloaded distribution archive and ```cd``` to the root project directory.
3. Install the core Sun lab development dependencies into the ***base*** mamba environment via the 
   ```mamba install tox uv tox-uv``` command.
4. Use the ```tox -e create``` command to create the project-specific development environment followed by 
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

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.

___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.

---
