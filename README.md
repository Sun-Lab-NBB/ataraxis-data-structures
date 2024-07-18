# ataraxis-data-structures

Provides a wide range of non-standard data structures and related data-manipulation methods.

![PyPI - Version](https://img.shields.io/pypi/v/ataraxis-data-structures)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ataraxis-data-structures)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/ataraxis-data-structures)
![PyPI - Status](https://img.shields.io/pypi/status/ataraxis-data-structures)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/ataraxis-data-structures)
___

## Detailed Description

This library provides data structures not readily available from standard Python libraries, such as nested dictionaries 
and shared memory arrays. In addition to these datastructures, it exposes helper-methods to manipulate the data that 
are also not readily available from standard or common Python libraries.

Unlike many other Ataraxis modules, this library does not have a very well-defined specialization beyond abstractly 
dealing with data storage, representation and manipulation. More or less anything data-related not found 
inside standard or popular Python libraries like numpy, scipy, pandas, etc. is a good candidate to be added to this 
library. It is designed to be updated frequently to scale with the needs of other Ataraxis modules, but it can also be 
used as a repository of helpful datastructures and methods to use in non-Ataraxis projects.
___

## Features

- Supports Windows, Linux, and OSx.
- Supports Multiprocessing.
- Supports 
- Pure-python API.
- GPL 3 License.

___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
___

## Dependencies

For users, all library dependencies are installed automatically for all supported installation methods 
(see [Installation](#installation) section). For developers, see the [Developers](#developers) section for 
information on installing additional development dependencies.
___

## Installation

### Source

1. Download this repository to your local machine using your preferred method, such as git-cloning. Optionally, use one
   of the stable releases that include precompiled binary wheels in addition to source code.
2. ```cd``` to the root directory of the project using your CLI of choice.
3. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
   binaries, use ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file.

### PIP

Use the following command to install the library using PIP: ```pip install ataraxis-data-structures```

### Conda / Mamba

**_Note. Due to conda-forge contributing process being more nuanced than pip uploads, conda versions may lag behind
pip and source code distributions._**

Use the following command to install the library using Conda or Mamba: ```conda install ataraxis-data-structures```
___

## Usage

Add minimal examples on how the end-user can use your library. This section is not to be an in-depth guide on using the
library. Instead, it should provide enough information to start using the library with the expectation that the user
can then study the API documentation and code-hints to figure out how to master the library.

### Quickstart

#### PythonDataConverter
The `PythonDataConverter` is a class that validates and cross converts data from one python type to another.
The core of the class functionality is the `validate_value` method. To properly use the classs one must initialize an
instance of the PythonDataConverter class with accepts one required positional argument `validator` which much be an 
instance of `BoolConverter`, `NoneConverter`, `NumericConverter`, or `StringConverter`. Each of these validator classes has
their own configurations and must be initalized and passed into the `PythonDataConverter`class. Some arguments have
default values. Here is an example of creating a `PythonDataConverter` that utilizes a `NumericConverter` with default
parameters.
```
converter = PythonDataConverter(validator=NumericConverter())
converter.validate_value("7.1")  # Returns the float 7.1
```

#### NumpyDataConverter
The `NumpyConverter` class is a converter and validator is is able to convert python datatypes to numpy datatypes. The
class extends the functionality of the `PythonDataConverter` to support numpy datatype conversion for only a limited set of
numpy datatypes. Numpy strings are not supported. A requirement of the `NumpyDataConverter` is for the `filter_failed`
argument of the `PythonDataConverter` to be true, the defaulted false is not allowed. Here is an example of a numeric
`NumpyDataConverter`. Note, `NumericConverter` cannot have both fields `allow_int` and `allow_float` being true when passed
into the `NumpyDataConverter`. Also, the NumpyDataConverter will automatically optimize the bit-width and sign (only 
integers) of numeric data types is no arguemnt is passed for `bit_width` or `signed`
```
validator = PythonDataConverter(validator=NumericConverter(allow_float=False), filter_failed=True)
converter = NumpyDataConverter(validator)
converter.python_to_numpy_converter("7.1")   # Returns 7.1 with type np.uint8
```
This can also convert from numpy datatypes to python natives. Using the same validator and converter:
```
converter.numpy_to_python_converter(np.uint8(7))   # Returns 7 with type int
```

### Config
#### Config: BoolConverter, NoneConverter, NumericConverter, or StringConverter
The following are examples to initilize each converter class with the full suite of configurations. Class fields that
require assignment have default values.
```
# BoolConverter
bool_convert = BoolConverter(
   parse_bool_equivalents = True # Default true, allows true equivalents ("True", "true", 1, "1", 1.0)  and false equivalents
                                 # ("False", "false, 0, "0", 0.0) bool conversion
)

# NonConverter
none_convert = NoneConverter(
   pare_none_equivalents = True  # Default true, allows none equivalent ("None", "none", "Null", "null") none conversion
)

# NumericConverter
num_convert = NumericConverter(
   parse_number_strings = True   # Default true, converts numbers in string format
   allow_int = True              # Default true, allow int conversion
   allow_float = True            # Default true, allow float conversion
   number_lower_limit = 7        # Default None, rejects numbers lower than this threshold
   number_upper_limit = 17       # Default None, rejects numbers greater than this threshold
)

# StringConverter
string_convert = StringConverter(
   allow_string_conversion: bool = False,          # Default false, allows non-string inputs to convert to strings 
   string_options = ['Bobby', 'Dobby', 'Poppy'],   # Default None, rejects string inputs not in this list/tuple
   string_force_lower: bool = False,               # Default false, force string output to lowercase
)

```

#### Config: PythonDataConverter
```
python_convert = PythonDataConverter(
   NumericConverter(),              # Required, StringConverter not supported
   iterable_output_type = 'tuple',  # Default None, pass in tuple/list for shallow array conversion support
   filter_failed: bool = False,     # Default False, omits the values in a array that failed conversion   
)

```
#### Configh: NumpyDataConverter
```
numpy_convert = NumpyDataConverter(
   PythonDataConverter(NumericConverter()),  # Required
   output_bit_width = "auto",                # Default 'auto', forces numeric inputs to a specific bit width (8, 16, 32, 64)
                                             # Replaced with inf if too large
   signed = True,                            # Default true, optimize the signed or unsigned int based on which saves memory.
)
```
___

## API Documentation

See the [API documentation](https://ataraxis-data-structures-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library. The documentation also 
covers any available cli/gui-interfaces (such as benchmarks).
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library. Additionally, it contains instructions for recreating the conda environments
that were used during development from the included .yml files.

### Installing the library

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` to the root directory of the project using your CLI of choice.
3. Install development dependencies. You have multiple options of satisfying this requirement:
    1. **_Preferred Method:_** Use conda or pip to install
       [tox](https://tox.wiki/en/latest/config.html#provision_tox_env) or use an environment that has it installed and
       call ```tox -e import-env``` to automatically import the os-specific development environment included with the
       source code in your local conda distribution. Alternatively, see [environments](#environments) section for other
       environment installation methods.
    2. Run ```python -m pip install .'[dev]'``` command to install development dependencies and the library using 
       pip. On some systems, you may need to use a slightly modified version of this command: 
       ```python -m pip install .[dev]```.
    3. As long as you have an environment with [tox](https://tox.wiki/en/latest/config.html#provision_tox_env) installed
       and do not intend to run any code outside the predefined project automation pipelines, tox will automatically
       install all required dependencies for each task.

**Note:** When using tox automation, having a local version of the library may interfere with tox methods that attempt
to build the library using an isolated environment. It is advised to remove the library from your test environment, or
disconnect from the environment, prior to running any tox tasks. This problem is rarely observed with the latest version
of the automation pipeline, but is worth mentioning.

### Additional Dependencies

In addition to installing the required python packages, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. 
  Currently, this library supports version 3.10 and above. The easiest way to get tox to work as intended is to have 
  separate python distributions, but using [pyenv](https://github.com/pyenv/pyenv) is a good alternative too. 
  This is needed for the 'test' task to work as intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/config.html#provision_tox_env). 
Check [tox.ini file](tox.ini) for details about available pipelines and their implementation.

**Note!** All commits to this library have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime task for this task, use ```tox --parallel```.

### Environments

All environments used during development are exported as .yml files and as spec.txt files to the [envs](envs) folder.
The environment snapshots were taken on each of the three supported OS families: Windows 11, OSx 14.5 and
Ubuntu 22.04 LTS.

To install the development environment for your OS:

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` into the [envs](envs) folder.
3. Use one of the installation methods below:
    1. **_Preferred Method_**: Install [tox](https://tox.wiki/en/latest/config.html#provision_tox_env) or use another
       environment with already installed tox and call ```tox -e import-env```.
    2. **_Alternative Method_**: Run ```conda env create -f ENVNAME.yml``` or ```mamba env create -f ENVNAME.yml```. 
       Replace 'ENVNAME.yml' with the name of the environment you want to install (axds_dev_osx for OSx, 
       axds_dev_win for Windows and axds_dev_lin for Linux).

**Note:** the OSx environment was built against M1 (Apple Silicon) platform and may not work on Intel-based Apple 
devices.

___

## Authors

- Ivan Kondratyev.
___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun Lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
