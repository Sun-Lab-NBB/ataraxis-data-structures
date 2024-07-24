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

This library aggregates the classes and methods that are broadly help working with data. This includes 
classes to manipulate the data, share (move) the data between different Python processes and save and load the 
data from storage. 

Generally, these classes either implement novel functionality not available through other popular libraries or extend 
existing functionality to match specific needs of other project Ataraxis module. That said, the library is written in a
way that it can be used as a standalone module with minimum dependency on other Ataraxis modules.
___

## Features

- Supports Windows, Linux, and OSx.
- Provides a Process- and Thread-safe way of sharing data between Python processes through a NumPy array structure.
- Provides tools for working with complex nested dictionaries using a path-like API.
- Provides a set of classes for converting between a wide range of Python and NumPy scalar and iterable datatypes.
- Extends standard Python dataclass to enable it to save and load itself to / from YAML files.
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
(see [Installation](#installation) section). For developers, see the [Developers](#developers) section for 
information on installing additional development dependencies.
___

## Installation

### Source

1. Download this repository to your local machine using your preferred method, such as git-cloning. Optionally, use one
   of the stable releases that include precompiled binary wheels in addition to source code.
2. ```cd``` to the root directory of the project using your command line interface of choice.
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

This section is broken into subsections for each exposed utility class or module. For each, it progresses from a 
minimalistic example and / or 'quickstart' to detailed notes on nuanced class functionality 
(if the class has such functionality).

### Data Converters
Generally, Data Converters are designed to in some way mimic the functionality of 
[pydantic](https://docs.pydantic.dev/latest/) project, but in a different scope. Unlike pydantic, which is primarily 
a data validator, our Converters are primarily designed ot be flexible data converters.

#### Base Converters
To assist converting to specific Python scalar types, we provide 4 'Base' converters: NumericConverter, 
BooleanConverter, StringConverter, and NoneConverter. After initial configuration, each converter takes in any input 
and conditionally converts it to the specific Python scalar datatype using __validate_value()__ class method.

__NumericConverter:__ Converts inputs to integers, floats, or both:
```
from ataraxis_data_structures.data_converters import NumericConverter

# NumericConverter is used to convert inputs into integers, floats or both. By default, it is configured to return
# both types. Depending on configuration, the class can be constrained to one type of outputs:
num_converter = NumericConverter(allow_integer_output=False, allow_float_output=True)
assert num_converter.validate_value(3) == 3.0

# When converting floats to integers, the class will only carry out the conversion if doing so does not require
# rounding or otherwise altering the value.
num_converter = NumericConverter(allow_integer_output=True, allow_float_output=False)
assert num_converter.validate_value(3.0) == 3

# The class can convert number-equivalents to numeric types depending on configuration. When possible, it prefers
# floating-point numbers over integers:
num_converter = NumericConverter(allow_integer_output=True, allow_float_output=True, parse_number_strings=True)
assert num_converter.validate_value('3.0') == 3.0

# NumericConverter can also filter input values based on a specified range. If the value fails validation, the method 
# returns None.
num_converter = NumericConverter(number_lower_limit=1, number_upper_limit=2, allow_float_output=False)
assert num_converter.validate_value('3.0') is None
```

__BooleanConverter:__ Converts inputs to booleans:
```
from ataraxis_data_structures.data_converters import BooleanConverter

# Boolean converter only has one additional parameter: whether to convert boolean-equivalents.
bool_converter = BooleanConverter(parse_boolean_equivalents=True)

assert bool_converter.validate_value(1) is True
assert bool_converter.validate_value(True) is True
assert bool_converter.validate_value('true') is True

assert bool_converter.validate_value(0) is False
assert bool_converter.validate_value(False) is False
assert bool_converter.validate_value('false') is False

# If valdiation fails for any input, the emthod returns None
bool_converter = BooleanConverter(parse_boolean_equivalents=False)
assert bool_converter.validate_value(1) is None
```

__NoneConverter:__ Converts inputs to None:
```
from ataraxis_data_structures.data_converters import NoneConverter

# None converter only has one additional parameter: whether to convert None equivalents.
bool_converter = NoneConverter(parse_none_equivalents=True)

assert bool_converter.validate_value('Null') is None
assert bool_converter.validate_value(None) is None
assert bool_converter.validate_value('none') is None

# If the method is not able to convert or validate the input, it returns string "None":
assert bool_converter.validate_value("Not an equivalent") == 'None'
```

__StringConverter:__ Converts inputs to strings. Since most Python scalar types are string-convertible, the default 
class configuration is to NOT convert inputs:
```
from ataraxis_data_structures.data_converters import StringConverter

# By default, string converter is configured to only validate, but not convert inputs:
str_converter = StringConverter()
assert str_converter.validate_value("True") == 'True'
assert str_converter.validate_value(1) is None  # Conversion failed

# To enable conversion, set the appropriate class initialization argument:
str_converter = StringConverter(allow_string_conversion=True)
assert str_converter.validate_value(1) == '1'

# Additionally, the class can be sued to filter inputs based on a predefined list and force strings to be lower-case.
# Note, filtering is NOT case-sensitive:
str_converter = StringConverter(allow_string_conversion=True, string_force_lower=True, string_options=['1', 'ok'])
assert str_converter.validate_value(1) == '1'
assert str_converter.validate_value('OK') == 'ok'  # Valid option, converted to the lower case
assert str_converter.validate_value('2') is None  # Not a valid option
```

#### PythonDataConverter
The PythonDataConverter class expands upon the functionality of the 'Base' Converter classes. To do so, it accepts 
pre-configured instances of the 'Base' Converter classes and applies them to inputs to its' __validate_value()__ method.

__PythonDataConverter__ extends converter functionality to one-dimensional iterable inputs and outputs by applying 
a 'Base' converter to each element of the iterable. It also works with scalars:
```
from ataraxis_data_structures.data_converters import NumericConverter, PythonDataConverter

# Each input converter has to be preconfigured
numeric_converter = NumericConverter(allow_integer_output=True, allow_float_output=False, parse_number_strings=True)

# PythonDataConverter has arguments that allow providing the class with an instance for each of the 'Base' converters.
# By default, all 'Converter' arguments are set to None, indicating they are not in use. The class requires at least one
# converter to work.
python_converter = PythonDataConverter(numeric_converter=numeric_converter)

# PythonDataConverter class extends wrapped 'Base' converter functionality to iterables:
assert python_converter.validate_value("33") == 33

# Defaults to tuple outputs. Unlike 'Base' Converters, the class uses a long 'Validation/ConversionError' string to
# denote outputs that failed to be converted
assert python_converter.validate_value(["33", 11, 14.0, 3.32]) == (33, 11, 14, "Validation/ConversionError")

# Optionally, the class can be configured to filter 'failed' iterable elements out and return a list instead of a tuple
python_converter = PythonDataConverter(
    numeric_converter=numeric_converter, filter_failed_elements=True, iterable_output_type="list"
)
assert python_converter.validate_value(["33", 11, 14.0, 3.32]) == [33, 11, 14]
```

__PythonDataConverter__ also allows combining multiple 'Base' converters to allow multiple output types. 
*__Note__*: The outputs are preferentially converted in this order float > integer > boolean > None > string"
```

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
___

## API Documentation

See the [API documentation](https://ataraxis-data-structures-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library. Additionally, it contains instructions for recreating the conda environments
that were used during development from the included .yml files.

### Installing the library

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` to the root directory of the project using your command line interface of choice.
3. Install development dependencies. You have multiple options of satisfying this requirement:
    1. **_Preferred Method:_** Use conda or pip to install
       [tox](https://tox.wiki/en/latest/user_guide.html) or use an environment that has it installed and
       call ```tox -e import``` to automatically import the os-specific development environment included with the
       source code in your local conda distribution. Alternatively, you can use ```tox -e create``` to create the 
       environment from scratch and automatically install the necessary dependencies using pyproject.toml file. See 
       [environments](#environments) section for other environment installation methods.
    2. Run ```python -m pip install .'[dev]'``` command to install development dependencies and the library using 
       pip. On some systems, you may need to use a slightly modified version of this command: 
       ```python -m pip install .[dev]```.
    3. As long as you have an environment with [tox](https://tox.wiki/en/latest/user_guide.html) installed
       and do not intend to run any code outside the predefined project automation pipelines, tox will automatically
       install all required dependencies for each task.

**Note:** When using tox automation, having a local version of the library may interfere with tox tasks that attempt
to build the library using an isolated environment. While the problem is rare, our 'tox' pipelines automatically 
install and uninstall the project from its' conda environment. This relies on a static tox configuration and will only 
target the project-specific environment, so it is advised to always ```tox -e import``` or ```tox -e create``` the 
project environment using 'tox' before running other tox commands.

### Additional Dependencies

In addition to installing the required python packages, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. 
  Currently, this library supports version 3.10 and above. The easiest way to get tox to work as intended is to have 
  separate python distributions, but using [pyenv](https://github.com/pyenv/pyenv) is a good alternative too. 
  This is needed for the 'test' task to work as intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check [tox.ini file](tox.ini) for details about 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All commits to this project have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime task for this task, use ```tox --parallel```.

For more information, you can also see the 'Usage' section of the 
[ataraxis-automation project](https://github.com/Sun-Lab-NBB/ataraxis-automation) documentation.

### Environments

All environments used during development are exported as .yml files and as spec.txt files to the [envs](envs) folder.
The environment snapshots were taken on each of the three explicitly supported OS families: Windows 11, OSx (M1) 14.5
and Linux Ubuntu 22.04 LTS.

**Note!** Since the OSx environment was built for an M1 (Apple Silicon) platform, it may not work on Intel-based 
Apple devices.

To install the development environment for your OS:

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` into the [envs](envs) folder.
3. Use one of the installation methods below:
    1. **_Preferred Method_**: Install [tox](https://tox.wiki/en/latest/user_guide.html) or use another
       environment with already installed tox and call ```tox -e import```.
    2. **_Alternative Method_**: Run ```conda env create -f ENVNAME.yml``` or ```mamba env create -f ENVNAME.yml```. 
       Replace 'ENVNAME.yml' with the name of the environment you want to install (axbu_dev_osx for OSx, 
       axbu_dev_win for Windows, and axbu_dev_lin for Linux).

**Hint:** while only the platforms mentioned above were explicitly evaluated, this project is likely to work on any 
common OS, but may require additional configurations steps.

Since the release of [ataraxis-automation](https://github.com/Sun-Lab-NBB/ataraxis-automation) version 2.0.0 you can 
also create the development environment from scratch via pyproject.toml dependencies. To do this, use 
```tox -e create``` from project root directory.

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

- All Sun Lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- [numpy](https://github.com/numpy/numpy) project for providing low-level functionality for many of the 
  classes exposed through this library.
- [dacite](https://github.com/konradhalas/dacite) and [pyyaml](https://github.com/yaml/pyyaml/) for jointly providing
  the low-level functionality to read and write dataclasses to / from .yaml files.
- The creators of all other projects used in our development automation pipelines [see pyproject.toml](pyproject.toml).

---
