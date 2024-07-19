import re
import textwrap

import numpy as np
import pytest

from ataraxis_data_structures.data_converters import (
    BooleanConverter,
    NoneConverter,
    StringConverter,
    NumericConverter,
    PythonDataConverter,
    NumpyDataConverter,
)


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
    "config,input_value,expected",
    [
        ({}, 5, 5),
        ({}, 5.5, 5.5),
        ({}, True, 1.0),
        ({"allow_integer_output": False}, True, 1.0),
        ({"allow_integer_output": False}, 5, 5.0),
        ({"allow_float_output": False}, 5.0, 5),
        ({"parse_number_strings": True}, "5.5", 5.5),
        ({"parse_number_strings": True}, "5", 5.0),
        ({"number_lower_limit": 0, "number_upper_limit": 10}, 5, 5),
        (
            {
                "allow_integer_output": False,
                "allow_float_output": True,
                "number_lower_limit": 0,
                "number_upper_limit": 10,
            },
            5,
            5.0,
        ),
        (
            {
                "allow_integer_output": True,
                "allow_float_output": False,
                "number_lower_limit": 0,
                "number_upper_limit": 10,
            },
            5.0,
            5,
        ),
        ({}, "not a number", None),
        ({}, [1, 2, 3], None),
        ({"parse_number_strings": False}, "5.5", None),
        ({"number_lower_limit": 0}, -5, None),
        ({"number_upper_limit": 10}, 15, None),
        ({"allow_float_output": False}, 5.5, None),
    ],
)
def test_numeric_converter_validate_value(config, input_value, expected) -> None:
    """Verifies the functionality of the NumericConverter class validate_value() method.

    Evaluates the following scenarios:
        0 - Validation of an integer input with integers allowed.
        1 - Validation of a float input with floats allowed.
        2 - Conversion of a boolean input to integer output.
        3 - Conversion of a boolean input to float output, with integers not allowed.
        4 - Conversion of an integer input into a float, with integers not allowed.
        5 - Conversion of an integer-convertible float input into an integer, with float outputs not allowed.
        6 - Conversion of a string into a float.
        7 - Conversion of a string into an integer.
        8 - Validation of a number within the minimum and maximum limits.
        9 - Conversion of an integer into float, with floats not allowed and limits enforced.
        10 - Conversion of an integer-convertible float into an integer, with integers not allowed and limits enforced.
        11 - Failure for a non-number-convertible string.
        12 - Failure for a non-supported input value (list).
        14 - Failure for a string input with string parsing disabled
        15 - Failure for a number below the lower limit.
        16 - Failure for a number above the upper limit
        17 - Failure for a float input with floats not allowed and the input not integer-convertible.
    """
    converter = NumericConverter(**config)
    output = converter.validate_value(input_value)
    assert output == expected
    assert isinstance(output, type(expected))


def test_numeric_converter_repr() -> None:
    """Verifies the functionality of NumericConverter class __repr__() method."""
    converter = NumericConverter()
    representation_string = (
        f"NumericConverter(parse_number_strings={True}, allow_integer_output={True}, allow_float_output={True}, "
        f"number_lower_limit={None}, number_upper_limit={None})"
    )
    assert repr(converter) == representation_string


def test_numeric_converter_init_errors() -> None:
    """Verifies the error-handling behavior of the NumericConverter class initialization method."""

    # Tests invalid initialization argument types:
    invalid_input = "invalid"

    # String parsing
    message = (
        f"Unable to initialize NumericConverter class instance. Expected a boolean parse_number_strings "
        f"argument value, but encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NumericConverter(parse_number_strings=invalid_input)

    # Allow integer
    message = (
        f"Unable to initialize NumericConverter class instance. Expected a boolean allow_integer_output "
        f"argument value, but encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NumericConverter(allow_integer_output=invalid_input)

    # Allow float
    message = (
        f"Unable to initialize NumericConverter class instance. Expected a boolean allow_float_output "
        f"argument value, but encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NumericConverter(allow_float_output=invalid_input)

    # Lower Bound
    message = (
        f"Unable to initialize NumericConverter class instance. Expected an integer, float or NoneType "
        f"number_lower_limit argument value, but encountered {invalid_input} of "
        f"type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NumericConverter(number_lower_limit=invalid_input)

    # Upper Bound
    message = (
        f"Unable to initialize NumericConverter class instance. Expected an integer, float or NoneType "
        f"number_upper_limit argument value, but encountered {invalid_input} of "
        f"type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NumericConverter(number_upper_limit=invalid_input)

    # Lower bound equal to upper bound
    upper_bound = 10
    lower_bound = 10
    message = (
        f"Unable to initialize NumericConverter class instance. Expected a number_lower_limit that is less than "
        f"the number_upper_limit, but encountered a lower limit of {lower_bound} and an upper limit of "
        f"{upper_bound}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        NumericConverter(number_lower_limit=lower_bound, number_upper_limit=upper_bound)

    # Lower bound greater than upper bound
    upper_bound = 10
    lower_bound = 15
    message = (
        f"Unable to initialize NumericConverter class instance. Expected a number_lower_limit that is less than "
        f"the number_upper_limit, but encountered a lower limit of {lower_bound} and an upper limit of "
        f"{upper_bound}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        NumericConverter(number_lower_limit=lower_bound, number_upper_limit=upper_bound)

    # Both allow_int and allow_float are disabled
    message = (
        f"Unable to initialize NumericConverter class instance. Expected allow_integer_output, "
        f"allow_float_output or both to be True, but both are set to False. At least one output type must be "
        f"enabled to instantiate a class."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        NumericConverter(allow_integer_output=False, allow_float_output=False)


def test_numeric_converter_properties() -> None:
    """Verifies the functionality NumericConverter class accessor properties."""
    converter = NumericConverter(
        parse_number_strings=True,
        allow_integer_output=True,
        allow_float_output=True,
        number_lower_limit=0,
        number_upper_limit=10,
    )

    assert converter.parse_number_strings
    assert converter.allow_integer_output
    assert converter.allow_float_output
    assert converter.number_lower_limit == 0
    assert converter.number_upper_limit == 10


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"parse_number_strings": False},
        {"allow_integer_output": False, "allow_float_output": True},
        {"allow_integer_output": True, "allow_float_output": False},
        {"number_lower_limit": -10, "number_upper_limit": 10},
    ],
)
def test_numeric_converter_config(config) -> None:
    """Verifies that initializing NumericConverter class using **kwargs config works as expected."""
    converter = NumericConverter(**config)
    for key, value in config.items():
        if key == "parse_number_strings":
            assert converter.parse_number_strings == value
        elif key == "allow_integer_output":
            assert converter.allow_integer_output == value
        elif key == "allow_float_output":
            assert converter.allow_float_output == value
        elif key == "number_lower_limit":
            assert converter.number_lower_limit == value
        elif key == "number_upper_limit":
            assert converter.number_upper_limit == value


@pytest.mark.parametrize(
    "config,input_value,expected",
    [
        ({}, True, True),
        ({}, False, False),
        ({}, "True", True),
        ({}, "False", False),
        ({}, "true", True),
        ({}, "false", False),
        ({}, 1, True),
        ({}, 0, False),
        ({}, "1", True),
        ({}, "0", False),
        ({}, 1.0, True),
        ({}, 0.0, False),
        ({"parse_boolean_equivalents": False}, "True", None),
        ({"parse_boolean_equivalents": False}, "False", None),
        ({"parse_boolean_equivalents": False}, "true", None),
        ({"parse_boolean_equivalents": False}, "false", None),
        ({"parse_boolean_equivalents": False}, 1, None),
        ({"parse_boolean_equivalents": False}, 0, None),
        ({"parse_boolean_equivalents": False}, "1", None),
        ({"parse_boolean_equivalents": False}, "0", None),
        ({"parse_boolean_equivalents": False}, 1.0, None),
        ({"parse_boolean_equivalents": False}, 0.0, None),
    ],
)
def test_boolean_converter_validate_value(config, input_value, expected) -> None:
    """Verifies the functionality of the BooleanConverter class validate_value() method.

    Evaluates the following scenarios:
        0 - Conversion of a boolean input to a boolean output, with boolean equivalents disabled.
        1 - Conversion of a boolean input to a boolean output, with boolean equivalents disabled.
        2 - Conversion of a string input to a boolean output, with boolean equivalents enabled.
        3 - Conversion of a string input to a boolean output, with boolean equivalents enabled.
        4 - Conversion of a string input to a boolean output, with boolean equivalents enabled.
        5 - Conversion of a string input to a boolean output, with boolean equivalents enabled.
        6 - Conversion of an integer input to a boolean output, with boolean equivalents enabled.
        7 - Conversion of an integer input to a boolean output, with boolean equivalents enabled.
        8 - Conversion of a string input to a boolean output, with boolean equivalents enabled.
        9 - Conversion of a string input to a boolean output, with boolean equivalents enabled.
        10 - Conversion of a float input to a boolean output, with boolean equivalents enabled.
        11 - Conversion of a float input to a boolean output, with boolean equivalents enabled.
        12 - Failure for a string input with boolean equivalents disabled.
        13 - Failure for a string input with boolean equivalents disabled.
        14 - Failure for a string input with boolean equivalents disabled.
        15 - Failure for a string input with boolean equivalents disabled.
        16 - Failure for an integer input with boolean equivalents disabled.
        17 - Failure for an integer input with boolean equivalents disabled.
        18 - Failure for a string input with boolean equivalents disabled.
        19 - Failure for a string input with boolean equivalents disabled.
        20 - Failure for a float input with boolean equivalents disabled.
        21 - Failure for a float input with boolean equivalents disabled.
    """
    converter = BooleanConverter(**config)
    output = converter.validate_value(input_value)
    assert output == expected
    assert isinstance(output, type(expected))


def test_boolean_converter_repr() -> None:
    """Verifies the functionality of the BooleanConverter class __repr__() method."""
    converter = BooleanConverter()
    representation_string = f"BooleanConverter(parse_boolean_equivalents={True})"
    assert repr(converter) == representation_string


def test_boolean_converter_init_errors() -> None:
    """Verifies the error-handling behavior of the BooleanConverter class initialization method."""

    # Verifies that an invalid parse_boolean_equivalents type is handled correctly
    invalid_input = "invalid"
    message = (
        f"Unable to initialize BooleanConverter class instance. Expected a boolean parse_boolean_equivalents "
        f"argument value, but encountered {invalid_input} of "
        f"type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        BooleanConverter(parse_boolean_equivalents=invalid_input)


def test_boolean_converter_properties() -> None:
    """Verifies the functionality BooleanConverter class accessor properties."""
    converter = BooleanConverter(parse_boolean_equivalents=False)

    assert not converter.parse_boolean_equivalents


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"parse_boolean_equivalents": False},
    ],
)
def test_boolean_converter_config(config) -> None:
    """Verifies that initializing BooleanConverter class using **kwargs config works as expected."""
    converter = BooleanConverter(**config)
    for key, value in config.items():
        if key == "parse_boolean_equivalents":
            assert converter.parse_boolean_equivalents == value


@pytest.mark.parametrize(
    "config,input_value,expected",
    [
        ({}, None, None),
        ({}, "None", None),
        ({}, "none", None),
        ({}, "null", None),
        ({}, "Null", None),
        ({}, "nil", "None"),
        ({}, 5, "None"),
        ({}, 5.5, "None"),
        ({}, True, "None"),
        ({}, False, "None"),
        ({"parse_none_equivalents": False}, "None", "None"),
        ({"parse_none_equivalents": False}, "none", "None"),
        ({"parse_none_equivalents": False}, "null", "None"),
        ({"parse_none_equivalents": False}, "NULL", "None"),
    ],
)
def test_none_converter_validate_value(config, input_value, expected) -> None:
    """Verifies the functionality of the NoneConverter class validate_value() method.

    Evaluates the following scenarios:
        0 - Conversion of a None input to a None output.
        1 - Conversion of a string input to a None output.
        2 - Conversion of a string input to a None output.
        3 - Conversion of a string input to a None output.
        4 - Conversion of a string input to a None output.
        5 - Conversion of a string input to a None output.
        6 - Conversion of a string input to a None output.
        7 - Failure for a string input.
        8 - Failure for an integer input.
        9 - Failure for a float input.
        10 - Failure for a boolean input.
        11 - Failure for a boolean input.
        12 - Failure for a string with when None equivalents disabled.
        13 - Failure for a string with when None equivalents disabled.
        14 - Failure for a string with when None equivalents disabled.
        15 - Failure for a string with when None equivalents disabled.
    """
    converter = NoneConverter(**config)
    output = converter.validate_value(input_value)
    assert output == expected
    assert isinstance(output, type(expected))


def test_none_converter_repr() -> None:
    """Verifies the functionality of the NoneConverter class __repr__() method."""
    converter = NoneConverter()
    representation_string = f"NoneConverter(parse_none_equivalents={True})"
    assert repr(converter) == representation_string


def test_none_converter_init_errors() -> None:
    """Verifies the error-handling behavior of the NoneConverter class initialization method."""

    # Verifies that an invalid parse_none_equivalents type is handled correctly
    invalid_input = "invalid"
    message = (
        f"Unable to initialize NoneConverter class instance. Expected a boolean parse_none_equivalents "
        f"argument value, but encountered {invalid_input} of "
        f"type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        NoneConverter(parse_none_equivalents=invalid_input)


def test_none_converter_properties() -> None:
    """Verifies the functionality NoneConverter class accessor properties."""
    converter = NoneConverter(parse_none_equivalents=True)
    assert converter.parse_none_equivalents


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"parse_none_equivalents": False},
    ],
)
def test_none_converter_config(config) -> None:
    """Verifies that initializing NoneConverter class using **kwargs config works as expected."""
    converter = NoneConverter(**config)
    for key, value in config.items():
        if key == "parse_none_equivalents":
            assert converter.parse_none_equivalents == value


@pytest.mark.parametrize(
    "config,input_value,expected",
    [
        ({"allow_string_conversion": True}, "Spongebob", "Spongebob"),
        ({"allow_string_conversion": True}, 5, "5"),
        ({"allow_string_conversion": True}, 5.5, "5.5"),
        ({"allow_string_conversion": True}, True, "True"),
        ({"allow_string_conversion": True}, False, "False"),
        ({"allow_string_conversion": True}, None, "None"),
        ({"allow_string_conversion": True, "string_options": ["1", "2"]}, 1, "1"),
        ({"allow_string_conversion": True, "string_options": ["1", "2"]}, 2, "2"),
        ({"allow_string_conversion": True, "string_force_lower": True}, "Spongebob", "spongebob"),
        (
            {"allow_string_conversion": True, "string_force_lower": False, "string_options": ["SPONGEBOB"]},
            "SpOnGeBoB",
            "SpOnGeBoB",
        ),
        ({"string_options": ["spongebob", "patrick"]}, "Spongebob", "Spongebob"),
        ({"string_options": ["Spongebob", "PatRick"], "string_force_lower": True}, "Patrick", "patrick"),
        ({"allow_string_conversion": True, "string_options": ["1", "2"]}, 3, None),
        ({"string_options": ["Spongebob", "Patrick"]}, "Squidward", None),
        ({}, 1, None),
        ({}, 1.0, None),
        ({}, True, None),
        ({}, False, None),
        ({}, None, None),
    ],
)
def test_string_converter_validate_method(config, input_value, expected) -> None:
    """Verifies the functionality of the StringConverter class validate_value() method.

    Evaluates the following test cases:
        0 - Conversion of a string input to a string output.
        1 - Conversion of a string-convertible integer input to a string output.
        3 - Conversion of a string-convertible float input to a string output.
        4 - Conversion of a string-convertible boolean True input to a string output.
        5 - Conversion of a string-convertible boolean False input to a string output.
        6 - Conversion of a string-convertible None input to a string output.
        7 - Conversion of an integer input to a string output with a list of string options.
        8 - Conversion of an integer input to a string output with a list of string options.
        9 - Conversion of a string input to a string output with forced lower case conversion.
        10 - Validation of a string input.
        11 - Validation of a string input with a list of valid options, making use of automatic lower-case-conversion.
        12 - Failure for an integer input, when the input is not in the list of valid string options.
        13 - Failure for a string input, when the input is not in the list of valid string options.
        14 - Failure for an integer input.
        15 - Failure for a float input.
        16 - Failure for a boolean input.
        17 - Failure for a boolean input.
        18 - Failure for a None input.
    """
    converter = StringConverter(**config)
    output = converter.validate_value(input_value)
    assert output == expected
    assert isinstance(output, type(expected))


def test_string_converter_repr() -> None:
    """Verifies the functionality of the StringConverter class __repr__() method."""
    converter = StringConverter()
    representation_string = (
        f"StringConverter(allow_string_conversion={False}, " f"string_force_lower={False}, string_options={None},)"
    )
    assert repr(converter) == representation_string


def test_string_converter_init_errors() -> None:
    """Verifies the error-handling behavior of the StringConverter class initialization method."""

    # Tests invalid initialization argument types:
    invalid_input = "invalid"

    # Lower-case conversion
    message = (
        f"Unable to initialize StringConverter class instance. Expected a boolean string_force_lower "
        f"argument value, but encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        StringConverter(string_force_lower=invalid_input)

    # String conversion
    message = (
        f"Unable to initialize StringConverter class instance. Expected a boolean allow_string_conversion "
        f"argument value, but encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        StringConverter(allow_string_conversion=invalid_input)

    # String-Options
    message = (
        f"Unable to initialize StringConverter class instance. Expected a None, tuple, or list string_options "
        f"argument value, but encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        StringConverter(string_options=invalid_input)

    # Invalid string-option element types
    invalid_options = ["1", 2, "3"]
    message = (
        f"Unable to initialize StringConverter class instance. Expected all elements of string-options "
        f"argument to be strings, but encountered {2} of type {type(2).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        StringConverter(string_options=invalid_options)

    # Empty string-option iterable
    invalid_options = tuple()
    message = (
        f"Unable to initialize StringConverter class instance. Expected at least one option inside the "
        f"string-options list or tuple argument, but encountered an empty iterable. To disable limiting "
        f"strings to certain options, set string_options to None."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        StringConverter(string_options=invalid_options)


def test_string_converter_properties() -> None:
    """Verifies the functionality StringConverter class accessor properties."""
    converter = StringConverter(allow_string_conversion=True, string_options=["A", "B"], string_force_lower=True)
    assert converter.allow_string_conversion
    assert converter.string_options == ["a", "b"]  # Ensures automatic lower-case conversion
    assert converter.string_force_lower


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"allow_string_conversion": True},
        {"string_options": ["A", "B"]},
        {"string_force_lower": True},
    ],
)
def test_string_converter_config(config) -> None:
    """Verifies that initializing StringConverter class using **kwargs config works as expected."""
    converter = StringConverter(**config)
    for key, value in config.items():
        if key == "allow_string_conversion":
            assert converter.allow_string_conversion == value
        elif key == "string_options":
            # noinspection PyTypeChecker
            assert converter.string_options == [element.lower() for element in value]  # Ensures elements are lower case
        elif key == "string_force_lower":
            assert converter.string_force_lower == value


@pytest.mark.parametrize(
    "config,input_value,expected",
    [
        # Test with only NumericConverter
        ({"numeric_converter": NumericConverter(allow_float_output=False)}, 5, 5),
        ({"numeric_converter": NumericConverter()}, 5.5, 5.5),
        ({"numeric_converter": NumericConverter()}, "10", 10.0),
        ({"numeric_converter": NumericConverter()}, "not a number", "Validation/ConversionError"),
        # Test with only BooleanConverter
        ({"boolean_converter": BooleanConverter()}, True, True),
        ({"boolean_converter": BooleanConverter()}, "true", True),
        ({"boolean_converter": BooleanConverter()}, "not a boolean", "Validation/ConversionError"),
        # Test with only NoneConverter
        ({"none_converter": NoneConverter()}, None, None),
        ({"none_converter": NoneConverter()}, "null", None),
        ({"none_converter": NoneConverter()}, "not none", "Validation/ConversionError"),
        # Test with only StringConverter
        ({"string_converter": StringConverter()}, "hello", "hello"),
        ({"string_converter": StringConverter(allow_string_conversion=True)}, 123, "123"),
        # Test with multiple converters
        (
            {
                "numeric_converter": NumericConverter(),
                "boolean_converter": BooleanConverter(),
                "none_converter": NoneConverter(),
                "string_converter": StringConverter(),
            },
            5,
            5,
        ),
        (
            {
                "numeric_converter": NumericConverter(),
                "boolean_converter": BooleanConverter(),
                "none_converter": NoneConverter(),
                "string_converter": StringConverter(),
            },
            "true",
            True,
        ),
        (
            {
                "numeric_converter": NumericConverter(),
                "boolean_converter": BooleanConverter(),
                "none_converter": NoneConverter(),
                "string_converter": StringConverter(),
            },
            "null",
            None,
        ),
        (
            {
                "numeric_converter": NumericConverter(),
                "boolean_converter": BooleanConverter(),
                "none_converter": NoneConverter(),
                "string_converter": StringConverter(),
            },
            "hello",
            "hello",
        ),
        # Test with iterable input
        ({"numeric_converter": NumericConverter()}, [1, 2, 3], (1, 2, 3)),
        ({"numeric_converter": NumericConverter(), "iterable_output_type": "list"}, [1, 2, 3], [1, 2, 3]),
        ({"numeric_converter": NumericConverter(), "iterable_output_type": "tuple"}, [1, 2, 3], (1, 2, 3)),
        # Test with filter_failed_elements
        ({"numeric_converter": NumericConverter(), "filter_failed_elements": True}, [1, "two", 3], (1, 3)),
        ({"numeric_converter": NumericConverter(), "filter_failed_elements": True}, ["one", "two", "3"], 3.0),
        (
            {"numeric_converter": NumericConverter(), "filter_failed_elements": True},
            ["one", "two", "three"],
            "Validation/ConversionError",
        ),
        # Test with mixed types in iterable
        (
            {
                "numeric_converter": NumericConverter(),
                "boolean_converter": BooleanConverter(),
                "none_converter": NoneConverter(),
                "string_converter": StringConverter(),
            },
            [1, "true", None, "hello"],
            (1, True, None, "hello"),
        ),
    ],
)
def test_python_data_converter_validate_value(config, input_value, expected) -> None:
    """Verifies the functionality of the PythonDataConverter class validate_value() method.

    Evaluates the following test cases:
        0 - Validation of an integer input with NumericConverter.
        1 - Validation of a float input with NumericConverter.
        2 - Conversion of a string-convertible integer input with NumericConverter.
        3 - Failure for a non-numeric string input with NumericConverter.
        4 - Validation of a boolean True input with BooleanConverter.
        5 - Conversion of a string-convertible boolean input with BooleanConverter.
        6 - Failure for a non-boolean string input with BooleanConverter.
        7 - Validation of a None input with NoneConverter.
        8 - Conversion of a "null" string input to None with NoneConverter.
        9 - Failure for a non-None string input with NoneConverter.
        10 - Validation of a string input with StringConverter.
        11 - Conversion of an integer input to a string with StringConverter.
        12 - Validation of an integer input with multiple converters (NumericConverter prioritized).
        13 - Validation of a boolean input with multiple converters (BooleanConverter prioritized).
        14 - Validation of a None input with multiple converters (NoneConverter prioritized).
        15 - Validation of a string input with multiple converters (StringConverter used).
        16 - Conversion of an iterable input to a tuple with NumericConverter.
        17 - Conversion of an iterable input to a list with NumericConverter and specified output type.
        18 - Filtering of failed elements in an iterable input with NumericConverter.
        19 - Validation of mixed types in an iterable input with multiple converters.
    """
    converter = PythonDataConverter(**config)
    result = converter.validate_value(input_value)
    assert result == expected
    assert isinstance(result, type(expected))


def test_python_data_converter_repr() -> None:
    """Verifies the functionality of PythonDataConverter class __repr__() method."""
    converter = PythonDataConverter(
        numeric_converter=NumericConverter(),
        boolean_converter=BooleanConverter(),
        none_converter=NoneConverter(),
        string_converter=StringConverter(),
        iterable_output_type="list",
        filter_failed_elements=True,
        raise_errors=False,
    )
    representation_string = (
        f"PythonDataConverter(allowed_output_types=('NoneType', 'bool', 'float', 'int', 'str'), "
        f"iterable_output_type=list, filter_failed_elements=True, raise_errors=False)"
    )
    assert repr(converter) == representation_string


def test_python_data_converter_init_errors() -> None:
    """Verifies the error-handling behavior of the PythonDataConverter class initialization method."""

    # Tests invalid initialization argument types:
    invalid_input = "invalid"

    # Numeric converter
    message = (
        f"Unable to initialize PythonDataConverter class instance. Expected a numeric_validator argument "
        f"of type {type(NumericConverter).__name__} or {type(None).__name__}, but "
        f"encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        PythonDataConverter(numeric_converter=invalid_input)

    # Boolean converter
    message = (
        f"Unable to initialize PythonDataConverter class instance. Expected a boolean_validator argument "
        f"of type {type(BooleanConverter).__name__} or {type(None).__name__}, but "
        f"encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        PythonDataConverter(boolean_converter=invalid_input)

    # None converter
    message = (
        f"Unable to initialize PythonDataConverter class instance. Expected a none_validator argument "
        f"of type {type(NoneConverter).__name__} or {type(None).__name__}, but "
        f"encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        PythonDataConverter(none_converter=invalid_input)

    # String converter
    message = (
        f"Unable to initialize PythonDataConverter class instance. Expected a string_validator argument "
        f"of type {type(StringConverter).__name__} or {type(None).__name__}, but "
        f"encountered {invalid_input} of type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        PythonDataConverter(string_converter=invalid_input)

    # Filter failed elements
    message = (
        f"Unable to initialize PythonDataConverter class instance. Expected a boolean filter_failed_elements "
        f"argument value, but encountered {invalid_input} of "
        f"type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        PythonDataConverter(filter_failed_elements=invalid_input)

    # Raise errors
    message = (
        f"Unable to initialize PythonDataConverter class instance. Expected a boolean raise_errors "
        f"argument value, but encountered {invalid_input} of "
        f"type {type(invalid_input).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        PythonDataConverter(raise_errors=invalid_input)

    # Invalid iterable output type
    invalid_iterable_type = "set"
    message = (
        f"Unsupported output iterable type {invalid_iterable_type} requested when initializing "
        f"PythonDataConverter class instance. Select one of the supported options: "
        f"{PythonDataConverter._supported_iterables.keys()}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        PythonDataConverter(iterable_output_type=invalid_iterable_type)

    # All converters set to None
    message = (
        f"Unable to initialize PythonDataConverter class instance. Expected at least one of the class "
        f"converter arguments to be set to a supported converter class, but all are set to None. This class "
        f"requires at least one configured base converter (NumericConverter, BooleanConverter, NoneConverter, "
        f"StringConverter) to operate as intended."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        PythonDataConverter()


def test_python_data_converter_properties() -> None:
    """Verifies the functionality of PythonDataConverter class accessor properties."""
    numeric_converter = NumericConverter()
    boolean_converter = BooleanConverter()
    none_converter = NoneConverter()
    string_converter = StringConverter()

    converter = PythonDataConverter(
        numeric_converter=numeric_converter,
        boolean_converter=boolean_converter,
        none_converter=none_converter,
        string_converter=string_converter,
        iterable_output_type="list",
        filter_failed_elements=True,
        raise_errors=False,
    )

    assert converter.numeric_converter == numeric_converter
    assert converter.boolean_converter == boolean_converter
    assert converter.none_converter == none_converter
    assert converter.string_converter == string_converter
    assert converter.iterable_output_type == "list"
    assert converter.filter_failed is True
    assert converter.raise_errors is False
    assert converter.allowed_output_types == ("NoneType", "bool", "float", "int", "str")


def test_python_data_converter_supported_iterables() -> None:
    """Verifies the functionality of PythonDataConverter class supported_iterables() method."""
    assert PythonDataConverter.supported_iterables() == ("tuple", "list")


@pytest.mark.parametrize(
    "config,expected_types",
    [
        ({"numeric_converter": NumericConverter()}, ("float", "int")),
        ({"boolean_converter": BooleanConverter()}, ("bool",)),
        ({"none_converter": NoneConverter()}, ("NoneType",)),
        ({"string_converter": StringConverter()}, ("str",)),
        (
            {
                "numeric_converter": NumericConverter(),
                "boolean_converter": BooleanConverter(),
                "none_converter": NoneConverter(),
                "string_converter": StringConverter(),
            },
            ("NoneType", "bool", "float", "int", "str"),
        ),
    ],
)
def test_python_data_converter_allowed_output_types(config, expected_types) -> None:
    """Verifies the functionality of PythonDataConverter class allowed_output_types property.

    Tests with different converter configurations.
    """
    converter = PythonDataConverter(**config)
    assert converter.allowed_output_types == expected_types


@pytest.mark.parametrize(
    "config,input_value,expected_error,expected_message",
    [
        # Test case for multi-dimensional iterable input
        (
            {"numeric_converter": NumericConverter()},
            [1, [2, 3], 4],
            ValueError,
            "Unable to validate the input collection of values ([1, [2, 3], 4]). Currently, this method only "
            "supports one-dimensional iterable inputs. Instead, a sub-iterable was discovered when "
            "evaluating element 1 ([2, 3]).",
        ),
        # Test case for raising error on invalid input when raise_errors is True
        (
            {"numeric_converter": NumericConverter(), "raise_errors": True},
            "not a number",
            ValueError,
            "Unable to validate the input value (not a number). The class is configured to conditionally "
            "return the following types: ('float', 'int'). This means that the input value is "
            "conditionally not convertible into any allowed type. Note, this may be due to failing secondary "
            "checks, such as numeric limits or string-option filters. The value was provided as part of this "
            "collection of values: ['not a number'].",
        ),
        # Test case for raising error on invalid input within an iterable when raise_errors is True
        (
            {"numeric_converter": NumericConverter(), "raise_errors": True},
            [1, "not a number", 3],
            ValueError,
            "Unable to validate the input value (not a number). The class is configured to conditionally "
            "return the following types: ('float', 'int'). This means that the input value is "
            "conditionally not convertible into any allowed type. Note, this may be due to failing secondary "
            "checks, such as numeric limits or string-option filters. The value was provided as part of this "
            "collection of values: [1, 'not a number', 3].",
        ),
    ],
)
def test_python_data_converter_validate_value_errors(config, input_value, expected_error, expected_message) -> None:
    """Verifies the error-raising behavior of the PythonDataConverter class validate_value() method.

    Evaluates the following test cases:
        0 - Raising ValueError for multidimensional iterable input.
        1 - Raising ValueError for invalid input when raise_errors is True.
        2 - Raising ValueError for invalid input within iterable when raise_errors is True.
    """
    converter = PythonDataConverter(**config)

    with pytest.raises(expected_error, match=error_format(expected_message)):
        converter.validate_value(input_value)


def test_numpyconverter_init_validation():
    """
    Verifies that NumpyDataConverter initialization method functions as expected and correctly catches invalid inputs.
    """
    # Tests valid initialization
    validator = PythonDataConverter(
        validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True
    )
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert type(converter.python_converter) is PythonDataConverter
    assert converter.output_bit_width == "auto"
    assert converter.signed

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        NumpyDataConverter(python_converter="not a validator", output_bit_width="auto")

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        NumpyDataConverter(python_converter=validator, output_bit_width="not a string")

    validator = PythonDataConverter(validator=StringConverter(), iterable_output_type="list", filter_failed=True)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        NumpyDataConverter(python_converter=validator, output_bit_width="auto")

    validator = PythonDataConverter(
        validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=False
    )
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        NumpyDataConverter(python_converter=validator, output_bit_width="auto")

    validator = PythonDataConverter(validator=NumericConverter(), iterable_output_type="list", filter_failed=True)
    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        NumpyDataConverter(python_converter=validator, output_bit_width=8)


def test_numpyconverter_setters():
    """
    Verifies the functioning of NumpyDataConverter class validator setter method.
    """
    validator = PythonDataConverter(
        validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True
    )
    converter = NumpyDataConverter(validator, output_bit_width="auto")

    converter.set_output_bit_width(8)
    assert converter.output_bit_width == 8

    converter.set_output_bit_width(16)
    assert converter.output_bit_width == 16

    converter.set_output_bit_width(32)
    assert converter.output_bit_width == 32

    converter.set_output_bit_width(64)
    assert converter.output_bit_width == 64

    converter.set_output_bit_width("auto")
    assert converter.output_bit_width == "auto"

    assert not converter.toggle_signed()
    assert not converter.signed
    assert converter.toggle_signed()
    assert converter.signed

    converter.set_python_converter(
        PythonDataConverter(validator=BoolConverter(), iterable_output_type="list", filter_failed=True)
    )
    assert type(converter.python_converter) is PythonDataConverter

    with pytest.raises(ValueError):
        # noinspection PyTypeChecker
        converter.set_output_bit_width("not a string")

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        converter.set_python_converter("not a number")


def test_numpyconverter_success():
    """
    Verifies correct validation behavior for different configurations of NumpyDataConverter class.
    """
    validator = PythonDataConverter(
        validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True
    )
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert np.array_equal(converter.python_to_numpy_converter([5, 5.5, True, False, None, "7.1"]), np.array([5, 1, 0]))
    assert np.array_equal(converter.python_to_numpy_converter(-5), -5)
    converter.set_output_bit_width(8)
    converter.toggle_signed()
    assert np.array_equal(
        converter.python_to_numpy_converter([2**4, 2**5, 2**6, 2**7, 2**8, 2**9]),
        np.array([16, 32, 64, 128, np.inf, np.inf]),
    )
    value = converter.python_to_numpy_converter(2**6)
    assert isinstance(value, np.uint8)

    validator = PythonDataConverter(
        validator=NumericConverter(allow_int=False), iterable_output_type="list", filter_failed=True
    )
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert np.array_equal(converter.python_to_numpy_converter([5.5, 6.0]), np.array([5.5, 6.0]))

    validator = PythonDataConverter(
        validator=NumericConverter(allow_int=False), iterable_output_type="list", filter_failed=True
    )
    converter = NumpyDataConverter(validator, output_bit_width=16)
    assert np.array_equal(
        converter.python_to_numpy_converter([5.5e-30, 6.0e30]), np.array([np.nan, np.inf]), equal_nan=True
    )

    validator = PythonDataConverter(validator=BoolConverter(), iterable_output_type="tuple", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert np.array_equal(
        converter.python_to_numpy_converter([5, 5.5, True, False, None, "7.1"]),
        np.array([np.bool(True), np.bool(False)]),
    )

    validator = PythonDataConverter(validator=NoneConverter(), iterable_output_type="list", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert np.array_equal(
        converter.python_to_numpy_converter([5, 5.5, "None", "Null", None, "7.1"]),
        np.array([np.nan, np.nan, np.nan]),
        equal_nan=True,
    )

    # Numpy to Python conversion
    validator = PythonDataConverter(
        validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True
    )
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert converter.numpy_to_python_converter(np.array([5, 1, 0])) == [5, 1, 0]
    assert converter.numpy_to_python_converter(np.int_(5.0)) == 5
    assert converter.numpy_to_python_converter(np.array([5])) == 5

    validator = PythonDataConverter(validator=BoolConverter(), iterable_output_type="tuple", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert converter.numpy_to_python_converter(np.array([1, 0])) == (True, False)
    assert converter.numpy_to_python_converter(np.int_(1)) == True
    assert converter.numpy_to_python_converter(np.bool_(True)) == True

    validator = PythonDataConverter(validator=NoneConverter(), iterable_output_type="list", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert converter.numpy_to_python_converter(np.array([np.nan, np.nan, np.nan])) == [None, None, None]
    assert converter.numpy_to_python_converter(np.nan) is None
    assert converter.numpy_to_python_converter(np.inf) is None


def test_numpyconverter_failure():
    """
    Verifies correct validation failure behavior for different configurations of NumpyDataConverter class.
    """
    with pytest.raises(OverflowError):
        validator = PythonDataConverter(
            validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True
        )
        converter = NumpyDataConverter(validator, output_bit_width="auto")
        converter.python_to_numpy_converter([2**100, 2**100])

    with pytest.raises(ValueError):
        validator = PythonDataConverter(
            validator=NumericConverter(allow_int=False), iterable_output_type="list", filter_failed=True
        )
        converter = NumpyDataConverter(validator, output_bit_width=8)
        converter.python_to_numpy_converter([0.22, 0.33])


def test_numpyconverter_properties():
    """
    Verifies that accessor properties of NumpyDataConverter class function as expected
    """
    validator = PythonDataConverter(
        validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True
    )
    converter = NumpyDataConverter(validator, output_bit_width="auto")

    assert type(converter.python_converter) is PythonDataConverter
    assert converter.output_bit_width == "auto"
    assert converter.signed
