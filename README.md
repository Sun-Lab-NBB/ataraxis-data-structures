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

This library aggregates the classes and methods that broadly help working with data. This includes 
classes to manipulate the data, share (move) the data between different Python processes and save and load the 
data from storage. Generally, these classes either implement novel functionality not available through other popular 
libraries or extend existing functionality to match specific needs of other project Ataraxis libraries.
___

## Features

- Supports Windows, Linux, and macOS.
- Provides a Process- and Thread-safe way of sharing data between Python processes through a NumPy array structure.
- Provides tools for working with complex nested dictionaries using a path-like API.
- Extends standard Python dataclass to enable it to save and load itself to / from YAML files.
- Provides a massively scalable data logger optimized for saving byte-serialized data from multiple input Processes to
  disk during real-time acquisition.
- Pure-python API.
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

For users, all library dependencies are installed automatically for all supported installation methods 
(see [Installation](#installation) section). 

For developers, see the [Developers](#developers) section for information on installing additional development 
dependencies.
___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.
Developers should see the [Developers](#Developers) section for more details on installing from source. The instructions
below assume you are ***not*** a developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/ataraxis-data-structures/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file, to install the 
   wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install ataraxis-data-structures```.
___

## Usage

This section is broken into subsections for each exposed utility class or module. For each, it only provides the 
minimalistic (quickstart) functionality overview, which does not reflect the nuances of using each method. To learn 
about the nuances, consult the [API documentation](#api-documentation).

### NestedDictionary
The NestedDictionary class wraps and manages a Python dictionary object. It exposes methods for evaluating the layout 
of the wrapped dictionary and manipulating values and sub-dictionaries in the hierarchy using a path-like API.

#### Reading, Writing and Removing values
The class provides an easy-to-use API for managing deeply nested values inside the wrapped dictionary. It consists of
three principal methods __write_nested_value()__, __read_nested_value()__ and __delete_nested_value()__
```
from ataraxis_data_structures import NestedDictionary

# By default, the class initializes as an empty dictionary object
nested_dictionary = NestedDictionary()

# The class is designed to work with nested dictonary paths, which are one-dimensional iterables of keys. Note! Key
# datatypes are important, the class respects input key datatype where possible.
path = ['level1', 'sublevel2', 'value1']  # This is the same as dict['level1']['sublevel2']['value1']

# To write into the dictionary, you can use a path-like API:
nested_dictionary.write_nested_value(variable_path=path, value=111)

# To read from the nested dictionary, you can use the same path-like API:
assert nested_dictionary.read_nested_value(variable_path=path) == 111

# Both methods can be used to read and write individual values and whole dictionary sections:
path = ['level2']
nested_dictionary.write_nested_value(variable_path=path, value={'sublevel2': {'subsublevel1': {'value': 3}}})
assert nested_dictionary.read_nested_value(variable_path=path) == {'sublevel2': {'subsublevel1': {'value': 3}}}

# Finally, delete_nested_value() can be used to remove values from the dictionary. Attempting to read a non-existent
# value raises keyError, just like in a regular dictionary:
nested_dictionary.delete_nested_value(variable_path=path)
try:
    nested_dictionary.read_nested_value(variable_path=path)
except KeyError:
    print('Deleted')
```

#### Path API
The class supports two formats when specifying paths to desired values and sub-dictionaries: an iterable of
keys and a delimited string.
```
from ataraxis_data_structures import NestedDictionary

# Python dictionaries are very flexible with the datatypes that can be used for dictionary keys.
seed_dict = {11: {'11': {True: False}}}
nested_dictionary = NestedDictionary(seed_dict)

# When working with dictionaries that mix multiple different types for keys, you have to use the 'iterable' path format.
# This is the only format that reliably preserves and accounts for key datatypes:
assert nested_dictionary.read_nested_value([11, '11', True]) is False

# However, when all dictionary keys are of the same datatype, you can use the second format of delimiter-delimited
# strings. This format does not preserve key datatype information, but it is more human-friendly and mimics the
# path API commonly used in file systems:
seed_dict = {'11': {'11': {'True': False}}}
nested_dictionary = NestedDictionary(seed_dict, path_delimiter='/')

assert nested_dictionary.read_nested_value('11/11/True') is False

# You can always modify the 'delimiter' character via set_path_delimiter() method:
nested_dictionary.set_path_delimiter('.')
assert nested_dictionary.read_nested_value('11.11.True') is False
```

#### Key datatype methods
The class comes with a set of methods that can be used to discover and potentially modify dictionary key datatypes.
Primarily, these methods are designed to convert the dictionary to use the same datatype for all keys, where possible, 
to enable using the 'delimited string' path API.
```
from ataraxis_data_structures import NestedDictionary

# Instantiates a dictionary with mixed datatypes.
seed_dict = {11: {'11': {True: False}}}
nested_dictionary = NestedDictionary(seed_dict)

# The 'key_datatypes' property returns the datatypes used by the dictionary as keys as a sorted list of strings.
# This property will NOT reflect any manual changes to the wrapped dictionary after NestedDictionary instantiation, but
# will be updated when the dictionary is modified via the NestedDictionary API.
assert nested_dictionary.key_datatypes == ('bool', 'int', 'str')

# Use the convert_all_keys_to_datatype method to convert all keys to the desired type.
nested_dictionary.convert_all_keys_to_datatype(datatype='int')
assert nested_dictionary.key_datatypes == ('int',)  # All keys have been converted to integers
```

#### Extracting variable paths
The class is equipped with methods for mapping dictionaries with unknown topologies. Specifically, the class
can find the paths to all terminal values or to specific terminal (value), intermediate (sub-dictionary) or both 
(all) dictionary elements:
```
from ataraxis_data_structures import NestedDictionary

# Instantiates a dictionary with mixed datatypes complex nesting
seed_dict = {"11": {"11": {"11": False}}, "key2": {"key2": 123}}
nested_dictionary = NestedDictionary(seed_dict)

# Extracts the paths to all values stored in the dictionary and returns them using iterable path API format (internally,
# it is referred to as 'raw').
value_paths = nested_dictionary.extract_nested_variable_paths(return_raw=True)

# The method has extracted the path to the two terminal values in the dictionary
assert len(value_paths) == 2
assert value_paths[0] == ("11", "11", "11")
assert value_paths[1] == ("key2", "key2")

# If you need to find the path to a specific variable or section, you can use the find_nested_variable_path() to search
# for the desired path:

# The search can be customized to only evaluate dictionary section keys (intermediate_only), which allows searching for
# specific sections:
intermediate_paths = nested_dictionary.find_nested_variable_path(
    target_key="key2", search_mode="intermediate_only", return_raw=True
)

# There is only one 'section' key2 in the dictionary, and this key is found inside the highest scope of the dictionary:
assert intermediate_paths == ('key2',)

# Alternatively, you can search for terminal keys (value keys) only:
terminal_paths = nested_dictionary.find_nested_variable_path(
    target_key="11", search_mode="terminal_only", return_raw=True
)

# There is exactly one path that satisfies those search requirements
assert terminal_paths == ("11", "11", "11")
```

### YamlConfig
The YamlConfig class extends the functionality of standard Python dataclasses by bundling them with methods to save and
load class data to / from .yaml files. Primarily, this is helpful for classes that store configuration data for other
runtimes so that they can be stored between runtimes and edited (.yaml is human-readable).

#### Saving and loading config data
This class is intentionally kept as minimalistic as possible. It does not do any input data validation and relies on the
user manually implementing that functionality, if necessary. The class is designed to be used as a parent for custom
dataclasses. 

All class 'yaml' functionality is realized through to_yaml() and from_yaml() methods:
```
from ataraxis_data_structures import YamlConfig
from dataclasses import dataclass
from pathlib import Path
import tempfile


# First, the class needs to be subclassed as a custom dataclass
@dataclass
class MyConfig(YamlConfig):
    # Note the 'base' class initialization values. This ensures that if the class data is not loaded from manual
    # storage, the example below will not work.
    integer: int = 0
    string: str = 'random'


# Instantiates the class using custom values
config = MyConfig(integer=123, string='hello')

# Uses temporary directory to generate the path that will be used to store the file
temp_dir = tempfile.mkdtemp()
out_path = Path(temp_dir).joinpath("my_config.yaml")

# Saves the class as a .yaml file. If you want to see / edit the file manually, replace the example 'temporary'
# directory with a custom directory
config.to_yaml(file_path=out_path)

# Ensures the file has been written
assert out_path.exists()

# Loads and re-instantiates the config as a dataclass using the data inside the .yaml file
loaded_config = MyConfig.from_yaml(file_path=out_path)

# Ensures that the loaded config data matches the original config
assert loaded_config.integer == config.integer
assert loaded_config.string == config.string
```

### SharedMemoryArray
The SharedMemoryArray class allows sharing data between multiple Python processes in a thread- and process-safe way.
It is designed to compliment other common data-sharing methods, such as multiprocessing and multithreading Queue 
classes. The class implements a shared one-dimensional numpy array, allowing different processes to dynamically write 
and read any elements of the array independent of order and without mandatory 'consumption' of manipulated elements.

#### Array creation
The SharedMemoryArray only needs to be initialized __once__ by the highest scope process. That is, only the parent 
process should create the SharedMemoryArray instance and provide it as an argument to all children processes during
their instantiation. The initialization process uses the input prototype numpy array and unique buffer name to generate 
a shared memory buffer and fill it with input array data. 

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
    # usage, call the method wit the remove_sources argument. The method verifies compressed data against the original
    # entries before removing source files, so it is always safe to delete source files.
    logger.compress_logs(remove_sources=True)

    # The compression creates a single .npz file named after the source_id
    assert len(list(logger.output_directory.glob("**/*.npy"))) == 0
    assert len(list(logger.output_directory.glob("**/*.npz"))) == 1
```
___

## API Documentation

See the [API documentation](https://ataraxis-data-structures-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library.

### Installing the library

The easiest way to ensure you have most recent development dependencies and library source files is to install the 
python environment for your OS (see below). All environments used during development are exported as .yml files and as 
spec.txt files to the [envs](envs) folder. The environment snapshots were taken on each of the three explicitly 
supported OS families: Windows 11, OSx Darwin, and GNU Linux.

**Note!** Since the OSx environment was built for the Darwin platform (Apple Silicon), it may not work on Intel-based 
Apple devices.

1. If you do not already have it installed, install [tox](https://tox.wiki/en/latest/user_guide.html) into the active
   python environment. The rest of this installation guide relies on the interaction of local tox installation with the
   configuration files included in with this library.
2. Download this repository to your local machine using your preferred method, such as git-cloning. If necessary, unpack
   and move the project directory to the appropriate location on your system.
3. ```cd``` to the root directory of the project using your command line interface of choice. Make sure it contains
   the `tox.ini` and `pyproject.toml` files.
4. Run ```tox -e import``` to automatically import the os-specific development environment included with the source 
   distribution. Alternatively, you can use ```tox -e create``` to create the environment from scratch and automatically
   install the necessary dependencies using pyproject.toml file. 
5. If either step 4 command fails, use ```tox -e provision``` to fix a partially installed environment.

**Hint:** while only the platforms mentioned above were explicitly evaluated, this project will likely work on any 
common OS, but may require additional configurations steps.

### Additional Dependencies

In addition to installing the required python packages, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. These 
   versions will be installed in-addition to the main Python version installed in the development environment.
   The easiest way to get tox to work as intended is to have separate python distributions, but using 
   [pyenv](https://github.com/pyenv/pyenv) is a good alternative. This is needed for the 'test' task to work as 
   intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check [tox.ini file](tox.ini) for details about 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All commits to this project have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime for this task, use ```tox --parallel```.

For more information, you can also see the 'Usage' section of the 
[ataraxis-automation project](https://github.com/Sun-Lab-NBB/ataraxis-automation#Usage) documentation.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself are prone to various failures. In 
most cases, this is related to their caching behavior. Despite a considerable effort to disable caching behavior known 
to be problematic, in some cases it cannot or should not be eliminated. If you run into an unintelligible error with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a cli command is very likely to fix the issue.
___

## Versioning

We use [semantic versioning](https://semver.org/) for this project. For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-data-structures/tags).

---

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Edwin Chen

___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- The creators of all other projects used in our development automation pipelines [see pyproject.toml](pyproject.toml).

---
