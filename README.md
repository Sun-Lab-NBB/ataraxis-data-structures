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

This library aggregates the classes and methods that broadly help working with data. This includes 
classes to manipulate the data, share (move) the data between different Python processes and save and load the 
data from storage. 

Generally, these classes either implement novel functionality not available through other popular libraries or extend 
existing functionality to match specific needs of other project Ataraxis modules. That said, the library is written 
in a way that it can be used as a standalone module with minimum dependency on other Ataraxis modules.
___

## Features

- Supports Windows, Linux, and macOS.
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
Generally, Data Converters are designed to in some way mimic the functionality of the
[pydantic](https://docs.pydantic.dev/latest/) project. Unlike pydantic, which is primarily a data validator, 
our Converters are designed specifically for flexible data conversion. While pydantic provides a fairly 
inflexible 'coercion' mechanism to cast input data to desired types, Converter classes offer a flexible and 
nuanced mechanism for casting Python variables between different types.

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
class configuration is to NOT convert inputs (to validate them without a conversion):
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
pre-configured instances of the 'Base' Converter classes and applies them to inputs via its' __validate_value()__ 
method.

__PythonDataConverter__ extends converter functionality to __one-dimensional iterable inputs and outputs__ by applying 
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

__PythonDataConverter__ also allows combining __multiple 'Base' converters__ to allow multiple output types. 
*__Note:__* The outputs are preferentially converted in this order float > integer > boolean > None > string:
```
from ataraxis_data_structures.data_converters import (
    NumericConverter,
    BooleanConverter,
    StringConverter,
    PythonDataConverter,
)

# Configured converters to be combined through PythonDataConverter
numeric_converter = NumericConverter(allow_integer_output=True, allow_float_output=False, parse_number_strings=True)
bool_converter = BooleanConverter(parse_boolean_equivalents=True)
string_converter = StringConverter(allow_string_conversion=True)

# When provided with multiple converters, they are applied in this order: Numeric > Boolean > None > String
python_converter = PythonDataConverter(
    numeric_converter=numeric_converter, boolean_converter=bool_converter, string_converter=string_converter
)

# Output depends on the application hierarchy and the configuration of each 'Base' converter. If at least one converter
# 'validates' the value successfully, the 'highest' success value is returned.
assert python_converter.validate_value('33') == 33  # Parses integer-convertible string as integer

assert python_converter.validate_value('True') is True  # Parses boolean-equivalent string as boolean

# Since numeric converter cannot output floats and the input is not boolean-equivalent, it is processed by
# string-converter as a string
assert python_converter.validate_value(14.123) == '14.123'

# The principles showcased above are iteratively applied to each element of iterable inputs:
assert python_converter.validate_value(["22", False, 11.0, 3.32]) == (22, False, 11, '3.32')
```

__PythonDataConverter__ can be configured to raise exceptions instead of returning string error types:
```
from ataraxis_data_structures.data_converters import (
    NumericConverter,
    BooleanConverter,
    StringConverter,
    PythonDataConverter,
)

# Configures base converters to make sure input floating values will fail validation.
numeric_converter = NumericConverter(allow_float_output=False)
bool_converter = BooleanConverter(parse_boolean_equivalents=False)
string_converter = StringConverter(allow_string_conversion=False)

# By default, PythonDataConverter is configured to return 'Validation/ConversionError' string for any input(s) that
# fails conversion:
python_converter = PythonDataConverter(
    numeric_converter=numeric_converter, boolean_converter=bool_converter, string_converter=string_converter
)
assert python_converter.validate_value([3.124, 1.213]) == ("Validation/ConversionError", "Validation/ConversionError")

# However, the class can be configured to raise errors instead:
python_converter = PythonDataConverter(
    numeric_converter=numeric_converter,
    boolean_converter=bool_converter,
    string_converter=string_converter,
    raise_errors=True,
)
try:
    python_converter.validate_value([3.124, 1.213])  # This raises value error
except ValueError as e:
    print(f'Encountered error: {e}')
```

#### NumpyDataConverter
The NumpyDataConverter class extends the functionality of PythonDataConverter class to support converting to and from
NumPy datatypes. The fundamental difference between Python and NumPy data is that NumPy uses c-extensions and, 
therefore, requires input and output data to be strictly typed before it is processed. In the context of data 
conversion, this typically means that there is a single NumPy datatype into which we need to 'funnel' one or more 
Python types.

*__Note!__* At this time, NumpyDataConverter only supports integer, floating-point, and boolean conversion. Support 
for strings may be added in the future, but currently it is not planned.

__NumpyDataConverter__ works by wrapping an instance of PythonDataConverter class configured in a way that it outputs
a single Python datatype. After initial configuration, use __convert_value_to_numpy()__ method to convert input 
Python values to NumPy values.
```
from ataraxis_data_structures.data_converters import (
    NumericConverter,
    PythonDataConverter,
    NumpyDataConverter
)
import numpy as np

# NumpyDataConverter requires a PythonDataConverter instance configured to return a single type:
numeric_converter = NumericConverter(allow_float_output=False, allow_integer_output=True)  # Only integers are allowed

# PythonDataConverter has to use only one Base converter to satisfy he conditions mentioned above. Additionally, the
# class has to be configured to raise errors instead of returning error-strings:
python_converter = PythonDataConverter(numeric_converter=numeric_converter, raise_errors=True)

numpy_converter = NumpyDataConverter(python_converter=python_converter)

# By default, NumpyDataConverter prefers signed integers to unsigned integers and automatically uses the smallest
# bit-width sufficient to represent the data. This is in contrast to the 'standard' numpy behavior that defaults 
# to 32 or 64 bit-widths depending on the output type.
assert numpy_converter.convert_value_to_numpy('3') == np.int8(3)
assert isinstance(numpy_converter.convert_value_to_numpy('3'), np.int8)
```

__NumpyDataConverter__ can be additionally configured to produce outputs of specific bit-widths and, for integers,
signed or unsigned type:
```
from ataraxis_data_structures.data_converters import (
    NumericConverter,
    PythonDataConverter,
    NumpyDataConverter
)
import numpy as np

# Specifically, configures the converter to produce unsigned integers using 64 bit-widths.
numeric_converter = NumericConverter(allow_float_output=False, allow_integer_output=True)
python_converter = PythonDataConverter(numeric_converter=numeric_converter, raise_errors=True)
numpy_converter = NumpyDataConverter(python_converter=python_converter, output_bit_width=64, signed=False)

# Although the number would have automatically been converted to an 8-bit signed integer, our configuration ensures
# it is a 64-bit unsigned integer.
assert numpy_converter.convert_value_to_numpy('11') == np.uint64(11)
assert isinstance(numpy_converter.convert_value_to_numpy('11'), np.uint64)

# This works for iterables as well:
output = numpy_converter.convert_value_to_numpy([11, 341, 67481])
expected = np.array([11, 341, 67481], dtype=np.uint64)
assert np.array_equal(output, expected)
assert output.dtype == np.uint64
```

__NumpyDataConverter__ can be used to convert numpy datatypes back to Python types using __convert_value_from_numpy()__
method:
```
from ataraxis_data_structures.data_converters import (
    NumericConverter,
    PythonDataConverter,
    NumpyDataConverter
)
import numpy as np

# Configures the converter to work with floating-point numbers
numeric_converter = NumericConverter(allow_float_output=True, allow_integer_output=False)
python_converter = PythonDataConverter(numeric_converter=numeric_converter, raise_errors=True)
numpy_converter = NumpyDataConverter(python_converter=python_converter)

# Converts scalar floating types to python types
assert numpy_converter.convert_value_from_numpy(np.float64(1.23456789)) == 1.23456789
assert isinstance(numpy_converter.convert_value_from_numpy(np.float64(1.23456789)), float)

# Also works for iterables
input_array = np.array([1.234, 5.671, 6.978], dtype=np.float16)
output = numpy_converter.convert_value_from_numpy(input_array)
assert np.allclose(output, (1.234, 5.671, 6.978), atol=0.01, rtol=0)  # Fuzzy comparison due to rounding
assert isinstance(output, tuple)
```

### NestedDictionary
The NestedDictionary class wraps and manages a Python dictionary object. It exposes methods for evaluating the layout 
of the wrapped dictionary and manipulating values and sub-dictionaries in the hierarchy using a path-like API.

#### Reading and Writing values
The class contains two principal methods likely to be helpful for most users: __write_nested_value()__ and 
__read_nested_value()__ which can be used together with a Path-like API to work with dictionary values:
```
from ataraxis_data_structures import NestedDictionary

# By default, the class initializes as an empty dictionary object
nested_dictionary = NestedDictionary()

# The class is designed to work with nested paths, which are one-dimensional iterables of keys. The class always
# crawls the dictionary from the highest hierarchy, sequentially indexing sublevels of the dictionary using the
# provided keys. Note! Key datatypes are important, the class respects input key datatype where possible.
path = ['level1', 'sublevel2', 'value1']  # This is the same as nested_dict['level1']['sublevel2']['value1']

# To write into the dictionary, you can use a path-like API:
nested_dictionary.write_nested_value(variable_path=path, value=111)

# To read from the nested dictionary, you can use the same path-like API:
assert nested_dictionary.read_nested_value(variable_path=path) == 111

# Both methods can be used to read and write individual values and whole dictionary sections:
path = ['level2']
nested_dictionary.write_nested_value(variable_path=path, value={'sublevel2': {'subsublevel1': {'value': 3}}})
assert nested_dictionary.read_nested_value(variable_path=path) == {'sublevel2': {'subsublevel1': {'value': 3}}}
```

#### Wrapping existing dictionaries
The class can wrap pre-created dictionaries to extend class functionality to almost any Python dictionary object:
```
from ataraxis_data_structures import NestedDictionary

# The class can be initialized with a pre-created dictionary to manage that dictionary
seed_dict = {'key1': {'key2': {'key3': 10}}, 12: 'value1'}
nested_dictionary = NestedDictionary(seed_dict)

assert nested_dictionary.read_nested_value(['key1', 'key2', 'key3']) == 10
assert nested_dictionary.read_nested_value([12]) == 'value1'
```

#### Path API
The class generally supports two formats used to specify paths to desired values and sub-dictionaries: an iterable of
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

# If you do not know the datatypes of your dictionary, you can access them via the 'key_datatypes' property, which
# returns them as a sorted list of strings. The property is updated during class initialization and when using methods
# that modify the dictionary, but it references a static set under-the-hood and will NOT reflect any manual changes to
# the dictionary.
assert nested_dictionary.key_datatypes == ('bool', 'int', 'str')

# You can use the convert_all_keys_to_datatype method to convert all keys to the desired type. By default, the method
# modifies the wrapped dictionary in-place, but it can be optionally configured to return a new NestedDictionary class
# instance that wraps the modified dictionary
new_nested_dict = nested_dictionary.convert_all_keys_to_datatype(datatype='str', modify_class_dictionary=False)
assert new_nested_dict.key_datatypes == ('str',)  # All keys have been converted to strings
assert nested_dictionary.key_datatypes == ('bool', 'int', 'str')  # Conversion did not affect original dictionary

# This showcases the default behavior of in-place conversion
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

# Finally, you can evaluate all keys: terminal and intermediate.
all_paths = nested_dictionary.find_nested_variable_path(
    target_key="11", search_mode="all", return_raw=True
)

# Here, 3 tuples are returned as a tuple of tuples. In the examples above, the algorithm automatically optimized
# returned data by returning it as a single tuple, since each search discovered a single path.
assert len(all_paths) == 3
assert all_paths[0] == ("11",)
assert all_paths[1] == ("11", "11",)
assert all_paths[2] == ("11", "11", "11")
```

#### Overwriting and deleting values
In addition to reading and adding new values to the dictionary, the class offers methods for overwriting and removing
existing dictionary sections and values. These methods can be flexibly configured to carry out a wide range of 
potentially destructive dictionary operations:
```
from ataraxis_data_structures import NestedDictionary

# Instantiates a dictionary with mixed datatypes complex nesting
seed_dict = {"11": {"11": {"11": False}}, "key2": {"key2": 123}}
nested_dictionary = NestedDictionary(seed_dict)

# By default, the write function is configured to allow overwriting dictionary values
value_path = "11.11.11"
modified_dictionary = nested_dictionary.write_nested_value(
    value_path, value=True, allow_terminal_overwrite=True, modify_class_dictionary=False
)

# Ensures that 'False' is overwritten with true in the modified dictionary
assert modified_dictionary.read_nested_value(value_path) is True
assert nested_dictionary.read_nested_value(value_path) is False

# You can also overwrite dictionary sections, which is not enabled by default:
value_path = "11.11"
modified_dictionary = nested_dictionary.write_nested_value(
    value_path, value={"12": "not bool"}, allow_intermediate_overwrite=True, modify_class_dictionary=False
)

# This time, the whole intermediate section has been overwritten with the provided dictionary
assert modified_dictionary.read_nested_value(value_path) == {"12": "not bool"}
assert nested_dictionary.read_nested_value(value_path) == {"11": False}

# Similarly, you can also delete dictionary values and sections by using the dedicated deletion method. By default, it
# is designed to remove all dictionary sections that are empty after the deletion has been carried out
value_path = "11.11.11"
modified_dictionary = nested_dictionary.delete_nested_value(
    variable_path=value_path, modify_class_dictionary=False, delete_empty_sections=True
)

# Ensures the whole branch of '11' keys has been removed from the dictionary
assert '11.11.11' not in modified_dictionary.extract_nested_variable_paths()

# When empty section deletion is disabled, the branch should remain despite no longer having the deleted key:value pair
modified_dictionary = nested_dictionary.delete_nested_value(
    variable_path=value_path, modify_class_dictionary=False, delete_empty_sections=False,
)

# This path now points to an empty dictionary section, but it exists
assert '11.11' in modified_dictionary.extract_nested_variable_paths()
assert modified_dictionary.read_nested_value('11.11') == {}
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
config.to_yaml(config_path=out_path)

# Ensures the file has been written
assert out_path.exists()

# Loads and re-instantiates the config as a dataclass using the data inside the .yaml file
loaded_config = MyConfig.from_yaml(config_path=out_path)

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

# To initialize the array, use create_array() method. DO NOT use class initialization method directly!
sma = SharedMemoryArray.create_array(name=buffer_name, prototype=prototype)

# The instantiated SharedMemoryArray object wraps an array with the same dimensions and data type as the prototype
# and uses the unique buffer name to identify the shared memory buffer to connect from different processes.
assert sma.name == buffer_name
assert sma.shape == prototype.shape
assert sma.datatype == prototype.dtype
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
# process that creates the array is connected automatically, calling the connect() method does not have negative
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

# On Unix platforms, you may need to manually destroy the array by calling the destroy() method. This has no effect on
# Windows (see above):
sma.destroy()  # While not strictly necessary, for each create_array(), there should be a matching destroy() call.
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
# used by the array
output = sma.read_data(index=2)
assert output == np.uint64(3)
assert isinstance(output, np.uint64)

# You can use 'convert_output' flag to force the method to us ePython datatypes for the returned data:
output = sma.read_data(index=2, convert_output=True)
assert output == 3
assert isinstance(output, int)

# By default, the method acquires a Lock object before reading data, preventing multiple processes from working with
# the array at the same time. For some use cases this can be detrimental (for example, when you are using the array to
# share the data between multiple read-only processes). In this case, you can read the data without locking:
output = sma.read_data(index=2, convert_output=True, with_lock=False)
assert output == 3
assert isinstance(output, int)

# To read a slice of the array, provide a tuple of two indices (for closed range) or a tuple of one index (start, open
# range).
output = sma.read_data(index=(0,), convert_output=True, with_lock=False)
assert output == [1, 2, 3, 4, 5, 6]
assert isinstance(output, list)

# Closed range end-index is excluded from sliced data
output = sma.read_data(index=(1, 4), convert_output=False, with_lock=False)
assert np.array_equal(output, np.array([2, 3, 4], dtype=np.uint64))
assert isinstance(output, np.ndarray)
```

#### Writing array data
To write data to the array wrapped by the class, use the __write_data()__ method. It's API is deliberately kept very 
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

# Numpy inputs are automatically converted to the correct datatype if possible
sma.write_data(index=1, data=np.uint8(9), with_lock=True)
assert sma.read_data(index=1, convert_output=False) == np.uint8(9)

# Writing by slice is also supported
sma.write_data(index=(1, 3), data=[10, 11], with_lock=False)
assert sma.read_data(index=(0,), convert_output=True) == [1, 10, 11, 4, 5, 6]
```

#### Using the array from multiple processes
While all methods showcased above run from the same process, the main advantage of the class is that they work
just as well when used from different Python processes:
```
import numpy as np
from ataraxis_data_structures import SharedMemoryArray
from multiprocessing import Process


def concurrent_worker(shared_memory_object: SharedMemoryArray, index: int):
    """This worker will run in a different process.

    It increments a shared memory array variable by 1 if the variable is even. Since each increment will
    shift it to be odd, to work as intended, this process has to work together with a different process that
    increments odd values. The process shuts down once the value reaches 200.

    Args:
        shared_memory_object: The SharedMemoryArray instance to work with.
        index: The index inside the array to increment
    """
    # Connects to the array
    shared_memory_object.connect()

    # Runs until the value becomes 200
    while shared_memory_object.read_data(index) < 200:
        # Reads data from the input index
        shared_value = shared_memory_object.read_data(index)

        # Checks if the value is even and below 200
        if shared_value % 2 == 0 and shared_value < 200:
            # Increments the value by one and writes it back to the array
            shared_memory_object.write_data(index, shared_value + 1)

    # Disconnects and terminates the process
    shared_memory_object.disconnect()


if __name__ == "__main__":
    # Initializes a SharedMemoryArray
    sma = SharedMemoryArray.create_array("test_concurrent", np.zeros(5, dtype=np.int32))

    # Generates multiple processes and uses each to repeatedly write and read data from different indices of the same
    # array.
    processes = [Process(target=concurrent_worker, args=(sma, i)) for i in range(5)]
    for p in processes:
        p.start()

    # For each of the array indices, increments the value of the index if it is odd. Child processes increment even
    # values and ignore odd ones, so the only way for this code to finish is if children and parent process take turns
    # incrementing shared values until they reach 200
    while np.any(sma.read_data((0, 5)) < 200):  # Runs as long as any value is below 200
        # Loops over addressable indices
        for i in range(5):
            value = sma.read_data(i)
            if value % 2 != 0 and value < 200:  # If the value is odd and below 200, increments the value by 1
                sma.write_data(i, value + 1)

    # Waits for the processes to join
    for p in processes:
        p.join()

    # Verifies that all processes ran as expected and incremented their respective variable
    assert np.all(sma.read_data((0, 5)) == 200)

    # Cleans up the shared memory array after all processes are terminated
    sma.disconnect()
    sma.destroy()
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
To minimize the runtime for this task, use ```tox --parallel```.

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

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- [numpy](https://github.com/numpy/numpy) project for providing low-level functionality for many of the 
  classes exposed through this library.
- [dacite](https://github.com/konradhalas/dacite) and [pyyaml](https://github.com/yaml/pyyaml/) for jointly providing
  the low-level functionality to read and write dataclasses to / from .yaml files.
- The creators of all other projects used in our development automation pipelines [see pyproject.toml](pyproject.toml).

---
