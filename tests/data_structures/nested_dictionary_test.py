"""Contains tests for classes and methods stored inside the nested_dictionary module of the data_structures pacakge."""

import re
import textwrap
import pytest
import numpy as np
from typing import Optional, Any, Literal, Union
from ataraxis_data_structures.data_structures import NestedDictionary


def error_format(message: str) -> str:
    """Formats the input message to match the default Console format and escapes it using re, so that it can be used to
    verify raised exceptions.

    This method is used to set up pytest 'match' clauses to verify raised exceptions.

    Args:
        message: The message to format and escape, according to standard Ataraxis testing parameters.

    Returns:
        Formatted and escape message that can be used as the 'match' argument of pytest.raises() method.
    """
    return re.escape(textwrap.fill(message, width=120, break_long_words=False, break_on_hyphens=False))


@pytest.mark.parametrize(
    "seed_dictionary, path_delimiter, expected_key_datatypes",
    [
        (None, ".", set()),
        ({}, ".", set()),
        ({"a": 1, "b": {"c": 2}}, ".", {"str"}),
        ({"a": 1, "b": {"c": 2}}, "/", {"str"}),
        ({1: "a", 2: 3}, ".", {"int"}),
        ({1.0: "a", 2.0: 3}, ".", {"float"}),
        ({None: "a", "b": {None: 2}}, ".", {"NoneType", "str"}),
        ({"a": 1, 2: {"c": 3.0}, None: {4.0: "d"}}, ".", {"str", "int", "NoneType", "float"}),
        ({"a.b": 1, "c": {"d.e": 2}}, "|", {"str"}),
        ({(1, 2): "a", "b": {(3, 4): "c"}}, ".", {"str", "tuple"}),
        ({"a": [], "b": {}, "c": set()}, ".", {"str"}),
        ({"a": 1, "b": {"c": {"d": {"e": 5}}}}, ".", {"str"}),
        ({}, "::", set()),
        ({1: {2: {3: {4: {5: 6}}}}, "a": {"b": {"c": {"d": "e"}}}}, ".", {"int", "str"}),
    ],
)
def test_nested_dictionary_init(seed_dictionary: Optional[dict], path_delimiter: str, expected_key_datatypes: set):
    """
    Verifies the functionality of NestedDictionary class initialization.

    This test covers the following scenarios:
    0. Initialization with None (default empty dictionary)
    1. Initialization with an empty dictionary
    2. Simple nested dictionary with string keys and default delimiter
    3. Simple nested dictionary with string keys and custom delimiter
    4. Nested dictionary with integer keys
    5. Nested dictionary with float keys
    6. Nested dictionary with None keys
    7. Nested dictionary with mixed key types
    8. Dictionary with keys containing delimiter character
    9. Dictionary with tuple keys (not in _valid_datatypes, but should still work)
    10. Dictionary with various value types (list, dict, set)
    11. Deeply nested dictionary
    12. Empty dictionary with custom delimiter
    13. Complex nested dictionary with multiple branches and key types

    These tests ensure that the NestedDictionary class initializes correctly with various
    input combinations, properly sets the path delimiter, and correctly identifies the
    key datatypes used in the dictionary.
    """
    nd = NestedDictionary(seed_dictionary, path_delimiter)

    # Tests instance creation
    assert isinstance(nd, NestedDictionary)

    # Tests path delimiter setting
    assert nd._path_delimiter == path_delimiter

    # Tests nested dictionary content
    if seed_dictionary is None:
        assert nd._nested_dictionary == {}
    else:
        assert nd._nested_dictionary == seed_dictionary

    # Tests key datatypes extraction
    assert nd._key_datatypes == expected_key_datatypes

    # Tests valid datatypes
    assert set(nd._valid_datatypes) == {"int", "str", "float", "NoneType"}

    # Additional assertions to verify the structure remains intact
    if seed_dictionary:

        def assert_dict_structure(d1: dict, d2: dict):
            assert set(d1.keys()) == set(d2.keys())
            for k, v in d1.items():
                if isinstance(v, dict):
                    assert_dict_structure(v, d2[k])
                else:
                    assert v == d2[k]

        assert_dict_structure(seed_dictionary, nd._nested_dictionary)


def test_nested_dictionary_properties_and_delimiter():
    """
    Verifies the functioning of NestedDictionary class properties (key_datatypes, path_delimiter)
    and the set_path_delimiter method.

    This test covers the following scenarios:
    1. Initialization and retrieval of key_datatypes
    2. Initialization and retrieval of path_delimiter
    3. Successful setting of new path_delimiter
    4. Error handling when setting invalid path_delimiter

    These tests ensure that the properties correctly return the expected values
    and that the set_path_delimiter method functions as expected in both success and error cases.
    """
    # Test initialization and key_datatypes property
    nd = NestedDictionary({"a": 1, 2: "b", 3.0: True})
    assert nd.key_datatypes == ("float", "int", "str")

    # Test path_delimiter property
    assert nd.path_delimiter == "."

    # Test successful set_path_delimiter
    nd.set_path_delimiter("::")
    assert nd.path_delimiter == "::"

    # Test set_path_delimiter with various valid inputs
    valid_delimiters = ["/", "->", "#", ""]
    for delimiter in valid_delimiters:
        nd.set_path_delimiter(delimiter)
        assert nd.path_delimiter == delimiter

    # Test error handling in set_path_delimiter
    invalid_delimiters = [123, 3.14, True, None, [1, 2, 3], {"a": 1}]
    for invalid_delimiter in invalid_delimiters:
        message = (
            f"A string 'new_delimiter' expected when setting the path delimiter, but "
            f"encountered '{type(invalid_delimiter).__name__}' instead."
        )
        with pytest.raises(TypeError, match=error_format(message)):
            # noinspection PyTypeChecker
            nd.set_path_delimiter(invalid_delimiter)

    # Verify that the last valid delimiter is still set
    assert nd.path_delimiter == ""

    # Test key_datatypes with empty dictionary
    empty_nd = NestedDictionary()
    assert empty_nd.key_datatypes == tuple()

    # Test key_datatypes with single datatype
    single_type_nd = NestedDictionary({1: "a", 2: "b", 3: "c"})
    assert single_type_nd.key_datatypes == ("int",)

    # Test key_datatypes with all supported types
    all_types_nd = NestedDictionary({1: "a", "b": 2, 3.0: True, None: "d"})
    assert all_types_nd.key_datatypes == ('NoneType', 'float', 'int', 'str')


def test_nested_dictionary_init_error():
    """Verifies the error-handling behavior of NestedDictionary class initialization."""

    # Verifies invalid seed_dictionary input type handling
    invalid_seed = "not valid"
    message = (
        f"A dictionary or None 'nested_dict' expected when initializing NestedDictionary class instance, but "
        f"encountered '{type(invalid_seed).__name__}' instead."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NestedDictionary(invalid_seed, path_delimiter=".")

    # Verifies invalid path_delimiter input type handling
    invalid_delimiter = [1, 2, 3]
    message = (
        f"A string 'path_delimiter' expected when initializing NestedDictionary class instance, but "
        f"encountered '{type(invalid_delimiter).__name__}' instead."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NestedDictionary(seed_dictionary=None, path_delimiter=invalid_delimiter)


def test_nested_dictionary_repr():
    """Verifies the functionality of NestedDictionary class __repr__ method.

    This test creates a single, comprehensive instance of NestedDictionary
    with mixed key types, nested structure, and a custom delimiter.
    It then checks if the __repr__ method correctly represents this instance.
    """
    # Creates a comprehensive nested dictionary
    seed_dictionary = {
        "str_key": 1,
        2: {"nested_str": "value"},
        3.14: [1, 2, 3],
        None: {"a": 1, "b": 2},
        "nested": {"deeply": {"nested": None}},
    }
    path_delimiter = "::"

    # Creates NestedDictionary instance
    nd = NestedDictionary(seed_dictionary, path_delimiter)

    # Expected string representation
    expected_repr = (
        "NestedDictionary(key_datatypes=NoneType, float, int, str, "
        f"path_delimiter='{path_delimiter}', "
        "data={'str_key': 1, 2: {'nested_str': 'value'}, 3.14: [1, 2, 3], "
        "None: {'a': 1, 'b': 2}, 'nested': {'deeply': {'nested': None}}})"
    )

    # Test the string representation
    assert repr(nd) == expected_repr


@pytest.mark.parametrize(
    "nested_dict, expected_types",
    [
        ({"a": 1, "b": {"c": 2}}, {"str"}),
        ({1: "a", "b": {2.0: True}}, {"int", "str", "float"}),
        ({None: 1, "c": 3}, {"NoneType", "str"}),
        ({1: "a", 2: "b", 3: "c"}, {"int"}),
        ({"a": 1, "b": 2, "c": 3}, {"str"}),
        ({1.0: "a", 2.0: "b", 3.0: "c"}, {"float"}),
        ({None: "a"}, {"NoneType"}),
        ({1: {2: {3: "a"}}, "b": {"c": {4.0: True}}}, {"int", "str", "float"}),
        ({1: "a", "b": 2, 3.0: True, None: False}, {"int", "str", "float", "NoneType"}),
        ({}, set()),  # Empty dictionary should return an empty set
    ],
)
def test_extract_key_datatypes(nested_dict: dict, expected_types: set):
    """Verifies the functionality of NestedDictionary class extract_key_datatypes() method.

    This test covers the following scenarios:
    1. Simple nested dictionary with only string keys
    2. Mixed key types (int, str, float) in a nested structure
    3. Multiple valid datatypes (int, str, NoneType) in a flat structure
    4. Homogeneous key types: int only
    5. Homogeneous key types: str only
    6. Homogeneous key types: float only
    7. Homogeneous key types: NoneType only
    8. Deeply nested structure with mixed key types
    9. Flat structure with all valid datatypes (except bool)
    10. Empty dictionary

    These tests ensure that the method correctly identifies and returns all unique
    key datatypes used in the dictionary, regardless of nesting level or combination.
    """

    # Since the method returns a set which has no inherent order, the equivalence check has to be carrier out
    # element-wise
    nd = NestedDictionary(nested_dict)
    extracted_types = nd.extract_key_datatypes()
    assert extracted_types == expected_types
    assert len(extracted_types) == len(expected_types)


@pytest.mark.parametrize(
    "nested_dict, return_raw, expected_output",
    [
        # Simple nested dictionary
        ({"a": 1, "b": {"c": 2}}, False, ("a", "b.c")),
        ({"a": 1, "b": {"c": 2}}, True, (("a",), ("b", "c"))),
        # Multiple levels of nesting
        ({"x": {"y": {"z": 1}}}, False, ("x.y.z",)),
        ({"x": {"y": {"z": 1}}}, True, (("x", "y", "z"),)),
        # Mixed key types
        ({1: "a", "b": {2.0: True}}, False, ("1", "b.2.0")),
        ({1: "a", "b": {2.0: True}}, True, ((1,), ("b", 2.0))),
        # Multiple branches
        ({"a": 1, "b": 2, "c": {"d": 3, "e": 4}}, False, ("a", "b", "c.d", "c.e")),
        ({"a": 1, "b": 2, "c": {"d": 3, "e": 4}}, True, (("a",), ("b",), ("c", "d"), ("c", "e"))),
        # Empty nested dictionaries
        ({"a": {}, "b": 1}, False, ("b",)),
        ({"a": {}, "b": 1}, True, (("b",),)),
        # None as a key
        ({None: 1, "a": {None: 2}}, False, ("None", "a.None")),
        ({None: 1, "a": {None: 2}}, True, ((None,), ("a", None))),
        # Mixed types and deep nesting
        ({1: {"a": 2}, 3.14: {None: {"b": True}}}, False, ("1.a", "3.14.None.b")),
        ({1: {"a": 2}, 3.14: {None: {"b": True}}}, True, ((1, "a"), (3.14, None, "b"))),
        # Empty dictionary
        ({}, False, ()),
        ({}, True, ()),
        # Dictionary with only nested empty dictionaries
        ({"a": {}, "b": {"c": {}}}, False, ()),
        ({"a": {}, "b": {"c": {}}}, True, ()),
        # Repeated keys at different levels
        ({"a": 1, "b": {"a": 2}}, False, ("a", "b.a")),
        ({"a": 1, "b": {"a": 2}}, True, (("a",), ("b", "a"))),
        # Very long path
        ({"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}, False, ("a.b.c.d.e.f",)),
        ({"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}, True, (("a", "b", "c", "d", "e", "f"),)),
    ],
)
def test_extract_nested_variable_paths(nested_dict: dict, return_raw: bool, expected_output: tuple):
    """Verifies the functionality of NestedDictionary class extract_nested_variable_paths() method.

    This test covers the following scenarios:
    0. Simple nested dictionary with string keys
    1. Multiple levels of nesting
    2. Mixed key types (int, str, float)
    3. Multiple branches in the dictionary
    4. Empty nested dictionaries
    5. None as a key
    6. Mixed types and deep nesting
    7. Empty dictionary
    8. Dictionary with only nested empty dictionaries
    9. Repeated keys at different levels
    10. Very long path

    Each scenario is tested with both return_raw=False and return_raw=True to verify
    both string and raw tuple output formats.

    These tests ensure that the method correctly extracts all paths in the nested dictionary,
    regardless of nesting level, key types, or dictionary structure.
    """
    nd = NestedDictionary(nested_dict)
    result = nd.extract_nested_variable_paths(return_raw=return_raw)
    assert set(result) == set(expected_output)  # Uses set to ignore order
    assert len(result) == len(expected_output)  # Ensures no extra or missing paths


@pytest.mark.parametrize(
    "key, datatype, expected_result",
    [
        # String to int
        ("123", "int", 123),
        ("-456", "int", -456),
        ("0", "int", 0),

        # Int to string
        (123, "str", "123"),
        (-456, "str", "-456"),
        (0, "str", "0"),

        # String to float
        ("3.14", "float", 3.14),
        ("-2.718", "float", -2.718),
        ("0.0", "float", 0.0),
        ("1e-3", "float", 0.001),

        # Float to int (truncation)
        (3.14, "int", 3),
        (-2.718, "int", -2),
        (0.0, "int", 0),

        # Float to string
        (3.14, "str", "3.14"),
        (-2.718, "str", "-2.718"),
        (0.0, "str", "0.0"),

        # Boolean to string
        (True, "str", "True"),
        (False, "str", "False"),

        # None to string
        (None, "str", "None"),

        # Various types to NoneType
        ("any_string", "NoneType", None),
        (123, "NoneType", None),
        (3.14, "NoneType", None),
        (True, "NoneType", None),
        (None, "NoneType", None),

        # Edge cases
        ("inf", "float", float('inf')),
        ("-inf", "float", float('-inf')),
        ("nan", "float", float('nan')),
        (float('inf'), "str", "inf"),
        (float('-inf'), "str", "-inf"),
        (float('nan'), "str", "nan"),
    ],
)
def test_convert_key_to_datatype_success(
        key: Any, datatype: Literal["int", "str", "float", "NoneType"], expected_result: Any
):
    """Verifies the functioning of the NestedDictionary class _convert_key_to_datatype() method.

    This test covers the following scenarios:
    0. String to int (positive, negative, zero)
    1. Int to string (positive, negative, zero)
    2. String to float (positive, negative, zero, scientific notation)
    3. Float to int (truncation for positive and negative numbers)
    4. Float to string (positive, negative, zero)
    5. Boolean to string
    6. None to string
    7. Various types to NoneType
    8. Edge cases (infinity and NaN)

    These tests ensure that the method correctly converts keys to the specified datatypes
    for all possible valid input combinations.
    """
    nd = NestedDictionary()  # Create an instance of NestedDictionary
    result = nd._convert_key_to_datatype(key, datatype)

    if isinstance(expected_result, float) and datatype == "float":
        if expected_result != expected_result:  # NaN check
            assert result != result
        elif expected_result in (float('inf'), float('-inf')):
            assert result == expected_result
        else:
            assert result == pytest.approx(expected_result)
    else:
        assert result == expected_result

    assert isinstance(result, type(expected_result))


def test_convert_key_to_datatype_error():
    """Verifies the error-handling behavior of NestedDictionary class _convert_key_to_datatype() method."""

    nd = NestedDictionary()  # Create an instance of NestedDictionary

    # Verifies handling of unsupported datatype
    invalid_datatype = "unsupported_type"
    key = "test_key"
    message = (
        f"Unexpected datatype '{invalid_datatype}' encountered when converting key '{key}' to the requested "
        f"datatype. Select one of the supported datatypes: {nd._valid_datatypes}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        nd._convert_key_to_datatype(key, invalid_datatype)

    # Verifies handling of invalid conversion to int
    invalid_key = "not_an_int"
    datatype = "int"
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        nd._convert_key_to_datatype(invalid_key, datatype)

    # Verifies handling of invalid conversion to float
    invalid_key = "not_a_float"
    datatype = "float"
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        nd._convert_key_to_datatype(invalid_key, datatype)


@pytest.mark.parametrize(
    "variable_path, key_datatypes, expected_result",
    [
        # String paths
        ("a.b.c", {"str"}, ("a", "b", "c")),
        ("1.2.3", {"int"}, (1, 2, 3)),
        ("1.2.3.4", {"float"}, (1.0, 2.0, 3.0, 4.0)),
        ("None.None", {"NoneType"}, (None, None)),

        # Lists
        (["a", "b", "c"], {"str"}, ("a", "b", "c")),
        ([1, 2, 3], {"int"}, (1, 2, 3)),
        ([1.0, 2.0, 3.0], {"float"}, (1.0, 2.0, 3.0)),
        ([None, None], {"NoneType"}, (None, None)),

        # Tuples
        (("a", "b", "c"), {"str"}, ("a", "b", "c")),
        ((1, 2, 3), {"int"}, (1, 2, 3)),
        ((1.0, 2.0, 3.0), {"float"}, (1.0, 2.0, 3.0)),
        ((None, None), {"NoneType"}, (None, None)),

        # Numpy arrays
        (np.array(["a", "b", "c"]), {"str"}, ("a", "b", "c")),
        (np.array([1, 2, 3]), {"int"}, (1, 2, 3)),
        (np.array([1.0, 2.0, 3.0]), {"float"}, (1.0, 2.0, 3.0)),

        # Mixed key types (only for non-string paths)
        (["a", 1, 2.0, None], {"str", "int", "float", "NoneType"}, ("a", 1, 2.0, None)),
        (("a", 1, 2.0, None), {"str", "int", "float", "NoneType"}, ("a", 1, 2.0, None)),

        # Edge cases
        ("", {"str"}, tuple([""])),  # Empty string
        ([], {"str"}, tuple()),  # Empty list
        ((), {"str"}, tuple()),  # Empty tuple
        (np.array([]), {"int"}, tuple()),  # Empty numpy array
    ],
)
def test_convert_variable_path_to_keys_success(
        variable_path: Union[str, list, tuple, np.ndarray],
        key_datatypes: set,
        expected_result: tuple,
):
    """Verifies the functioning of the NestedDictionary class _convert_variable_path_to_keys() method.

    This test covers the following scenarios:
    0. String paths with different key types (str, int, float, NoneType)
    1. Lists with different key types
    2. Tuples with different key types
    3. Numpy arrays with different key types
    4. Mixed key types for non-string paths
    5. Edge cases (empty inputs)

    These tests ensure that the method correctly converts various input formats to a tuple of keys,
    handling all supported datatypes and input formats.
    """
    nd = NestedDictionary()  # Create an instance of NestedDictionary
    nd._key_datatypes = key_datatypes  # Set the key_datatypes for the test
    nd._valid_datatypes = ("int", "str", "float", "NoneType")  # Set valid datatypes
    result = nd._convert_variable_path_to_keys(variable_path)
    assert result == expected_result
    assert isinstance(result, tuple)


def test_convert_variable_path_to_keys_error():
    """Verifies the error-handling behavior of NestedDictionary class _convert_variable_path_to_keys() method."""

    nd = NestedDictionary()  # Create an instance of NestedDictionary

    # Verifies handling of a string path ending with delimiter
    nd._key_datatypes = {"str"}
    invalid_path = "a.b.c."
    message = (
        f"A delimiter-ending variable_path string '{invalid_path}' encountered when converting "
        f"variable path to a sequence of keys, which is not allowed. Make sure the variable path ends "
        f"with a valid key."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        nd._convert_variable_path_to_keys(invalid_path)

    # Verifies handling of the string path with mixed key types
    nd._key_datatypes = {"str", "int"}
    invalid_path = "a.b.c"
    message = (
        f"An unsupported delimited string variable_path '{invalid_path}' encountered when converting "
        f"variable path to a sequence of keys. To support delimited string inputs, the dictionary has to "
        f"use a single key datatype, but found {len(nd.key_datatypes)} ({nd.key_datatypes}) "
        f"instead. Provide a list or tuple of keys with each key using one of the supported datatypes "
        f"({nd._valid_datatypes})."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        nd._convert_variable_path_to_keys(invalid_path)

    # Verifies handling of a numpy array with multiple dimensions
    nd._key_datatypes = {"int"}
    invalid_path = np.array([[1, 2], [3, 4]])
    message = (
        f"Unable to convert the input variable path numpy array to a tuple of datatype-specific "
        f"keys when converting variable path to a sequence of keys. Expected a one-dimensional array "
        f"as input, but encountered an array with unsupported shape ({invalid_path.shape}) and "
        f"dimensionality {invalid_path.ndim}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        nd._convert_variable_path_to_keys(invalid_path)

    # Verifies handling of a numpy array with mixed key types
    nd._key_datatypes = {"str", "int"}
    invalid_path = np.array([1, 2, 3])
    message = (
        f"An unsupported numpy array variable_path '{invalid_path}' encountered when converting "
        f"variable path to a sequence of keys. To support numpy array inputs, the dictionary has to "
        f"use a single key datatype, but found {len(nd.key_datatypes)} ({nd.key_datatypes}) "
        f"instead. Provide a list or tuple of keys with each key using one of the supported datatypes "
        f"({nd._valid_datatypes})."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        nd._convert_variable_path_to_keys(invalid_path)

    # Verifies handling of an unsupported input type
    invalid_path = {1, 2, 3}  # A set, which is not a supported input type
    message = (
        f"A string, tuple, list or one-dimensional numpy array variable_path expected when "
        f"converting variable path to a sequence of keys. Instead, encountered "
        f"'{type(invalid_path).__name__}'. Use one fo the supported variable_path formats."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        nd._convert_variable_path_to_keys(invalid_path)


@pytest.fixture
def nested_dict():
    """This fixture generates a nested dictionary used by the dictionary manipulation tests below."""
    return NestedDictionary({
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3,
                'f': [4, 5, 6]
            }
        },
        1: 'int_key',
        2.5: 'float_key',
        None: 'none_key',
        'g': {
            1: {
                2.5: {
                    None: 'nested_mixed_keys'
                }
            }
        }
    })


@pytest.mark.parametrize(
    "variable_path, expected_result",
    [
        ("a", 1),
        ("b.c", 2),
        ("b.d.e", 3),
        ("b.d.f", [4, 5, 6]),
        (["b", "d", "f"], [4, 5, 6]),
        (("b", "d", "f"), [4, 5, 6]),
        (np.array(["b", "d", "f"]), [4, 5, 6]),
        ("1", "int_key"),
        ("2.5", "float_key"),
        ("None", "none_key"),
        ([1], "int_key"),
        ([2.5], "float_key"),
        ([None], "none_key"),
        ("g.1.2.5.None", "nested_mixed_keys"),
        (["g", 1, 2.5, None], "nested_mixed_keys"),
        ("b", {"c": 2, "d": {"e": 3, "f": [4, 5, 6]}}),
        ("b.d", {"e": 3, "f": [4, 5, 6]}),
    ]
)
def test_read_nested_value_success(nested_dict: NestedDictionary, variable_path: Union[str, list, tuple, np.ndarray], expected_result: Any):
    """
    Verifies the successful functioning of the NestedDictionary class read_nested_value() method.

    This test covers the following scenarios:
    1. Reading simple key-value pairs
    2. Reading nested values using string paths
    3. Reading nested values using list, tuple, and numpy array paths
    4. Reading values with integer, float, and None keys
    5. Reading nested values with mixed key types
    6. Reading dictionary sections
    7. Reading list values
    8. Various combinations of nesting levels and key types

    These tests ensure that the method correctly retrieves values from the nested dictionary
    for all valid input combinations and data types.
    """
    result = nested_dict.read_nested_value(variable_path)
    assert result == expected_result


def test_read_nested_value_error(nested_dict: NestedDictionary):
    """
    Verifies the error-handling behavior of the NestedDictionary class read_nested_value() method.

    This test covers the following error scenarios:
    1. Accessing a non-existent key
    2. Attempting to access nested keys on a non-dictionary value
    3. Using an incorrect key type
    4. Using an incorrect key type in a nested path
    """
    # Test for non-existent key
    non_existent_key = "non_existent_key"
    message = (
        f"Key '{non_existent_key}' of type 'str' not found when reading nested value from "
        f"dictionary using path '{non_existent_key}'. Make sure the requested key is of the correct "
        f"datatype. Available keys (and their datatypes) at this level: "
        f"{[[k, type(k).__name__] for k in nested_dict._nested_dictionary.keys()]}."
    )
    with pytest.raises(KeyError, match=error_format(message)):
        nested_dict.read_nested_value(non_existent_key)

    # Test for attempting to access nested keys on a non-dictionary value
    invalid_nested_path = "a.nested"
    message = (
        f"Unable to fully crawl the path '{invalid_nested_path}', when reading nested value from "
        f"dictionary. The last used key 'a' returned '1' of type "
        f"'int' instead of the expected dictionary type."
    )
    with pytest.raises(KeyError, match=error_format(message)):
        nested_dict.read_nested_value(invalid_nested_path)

    # Test for incorrect key type
    incorrect_key_type = ["1"]  # Using string "1" instead of int 1
    message = (
        f"Key '1' of type 'str' not found when reading nested value from "
        f"dictionary using path '{incorrect_key_type}'. Make sure the requested key is of the correct "
        f"datatype. Available keys (and their datatypes) at this level: "
        f"{[[k, type(k).__name__] for k in nested_dict._nested_dictionary.keys()]}."
    )
    with pytest.raises(KeyError, match=error_format(message)):
        nested_dict.read_nested_value(incorrect_key_type)

    # Test for incorrect key type in a nested path
    incorrect_nested_key_type = ["g", "1", 2.5, None]  # Using string "1" instead of int 1
    message = (
        f"Key '1' of type 'str' not found when reading nested value from "
        f"dictionary using path '{incorrect_nested_key_type}'. Make sure the requested key is of the correct "
        f"datatype. Available keys (and their datatypes) at this level: "
        f"{[[k, type(k).__name__] for k in nested_dict._nested_dictionary['g'].keys()]}."
    )
    with pytest.raises(KeyError, match=error_format(message)):
        nested_dict.read_nested_value(incorrect_nested_key_type)
