import pytest
from pydantic import ValidationError
from typing import Union, Optional
from ataraxis_data_structures.data_converters.python_models import NumericConverter, BoolConverter, NoneConverter, StringConverter


@pytest.mark.parametrize("config,input_value,expected", [
    ({}, 5, 5),
    ({}, 5.5, 5.5),
    ({}, True, 1),
    ({"allow_int": False}, True, 1.0),
    ({"allow_int": False}, 5, 5.0),
    ({"allow_float": False}, 5.0, 5),
    ({"parse_number_strings": True}, "5.5", 5.5),
    ({"parse_number_strings": True}, "5", 5),
    ({"number_lower_limit": 0, "number_upper_limit": 10}, 5, 5),
    ({"allow_int": False, "allow_float": True, "number_lower_limit": 0, "number_upper_limit": 10}, 5, 5.0),
    ({"allow_int": True, "allow_float": False, "number_lower_limit": 0, "number_upper_limit": 10}, 5.0, 5),
])
def test_numericconverter_success(config, input_value, expected):
    """Verifies correct validation behavior for different configurations of  NumericConverter class.

    Evaluates:
        0 - Validation of an integer input when integers are allowed.
        1 - Validation of a float input when floats are allowed.
        2 - Conversion of a boolean input to integer output.
        3 - Conversion of a boolean input to float output, when integers are not allowed.
        4 - Conversion of an integer input into a float, when integers are not allowed.
        5 - Conversion of an integer-convertible float input into an integer, when floats are not allowed.
        6 - Conversion of a string into a float.
        7 - Conversion of a string into an integer.
        8 - Validation of a number within the minimum and maximum limits.
        9 - Conversion of an integer into float, when floats are not allowed and limits are enforced.
        10 - Conversion of an integer-convertible float into an integer, when integers are not allowed and limits
            are enforced.

    Args:
       config: The class configuration to be used for the test. Passed to the class via the **kwargs argument.
       input_value: The value passed to the validation function of the configured class instance.
    """
    converter = NumericConverter(**config)
    assert converter.validate_number(input_value) == expected


@pytest.mark.parametrize("config,input_value", [
    ({}, "not a number"),
    ({}, [1, 2, 3]),
    ({"allow_int": False, "allow_float": False}, 5),
    ({"parse_number_strings": False}, "5.5"),
    ({"number_lower_limit": 0}, -5),
    ({"number_upper_limit": 10}, 15),
    ({"allow_float": False}, 5.5),
])
def test_numericconverter_failure(config, input_value):
    """Verifies correct validation failure behavior for different configurations of  NumericConverter class.

    Evaluates:
        0 - Failure for a non-number-convertible string.
        1 - Failure for a non-supported input value (list).
        2 - Failure when both integer and float outputs are disabled.
        3 - Failure for a string input when string parsing is disabled
        4 - Failure for a number below the lower limit.
        5 - Failure for a number above the upper limit
        6 - Failure for a float input when floats are not allowed and the input is not integer-convertible.

    Args:
       config: The class configuration to be used for the test. Passed to the class via the **kwargs argument.
       input_value: The value passed to the validation function of the configured class instance.
    """
    converter = NumericConverter(**config)
    assert converter.validate_number(input_value) is None


def test_numericconverter_init_validation():
    """Verifies that NumericConverter initialization method functions as expected and correctly catches invalid inputs,
    """
    # Tests valid initialization
    converter = NumericConverter(parse_number_strings=True, allow_int=True, number_lower_limit=0)
    assert converter.parse_strings is True
    assert converter.allow_int is True
    assert converter.lower_limit == 0

    # Tests invalid initialization (relies on pydantic to validate the inputs)
    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        NumericConverter(parse_number_strings="not a bool")

    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        NumericConverter(number_lower_limit="not a number")


def test_numericconverter_properties():
    """Verifies that accessor properties of NumericConverter class function as expected"""
    converter = NumericConverter(parse_number_strings=True, allow_int=True, allow_float=True,
                                 number_lower_limit=0, number_upper_limit=10)

    assert converter.parse_strings
    assert converter.allow_int
    assert converter.allow_float
    assert converter.lower_limit == 0
    assert converter.upper_limit == 10


def test_numericconverter_toggle_methods():
    """Verifies the functioning of NumericConverter configuration flag toggling methods."""
    converter = NumericConverter()

    assert not converter.toggle_string_parsing()
    assert not converter.parse_strings
    assert converter.toggle_string_parsing()
    assert converter.parse_strings

    assert not converter.toggle_integer_outputs()
    assert not converter.allow_int
    assert converter.toggle_integer_outputs()
    assert converter.allow_int

    assert not converter.toggle_float_outputs()
    assert not converter.allow_float
    assert converter.toggle_float_outputs()
    assert converter.allow_float


def test_numericconverter_setter_methods() -> None:
    """Verifies the functioning of NumericConverter class limit setter methods."""
    converter = NumericConverter()

    converter.set_lower_limit(5)
    assert converter.lower_limit == 5

    converter.set_lower_limit(3.33)
    assert converter.lower_limit == 3.33

    converter.set_lower_limit(None)
    assert converter.lower_limit is None

    converter.set_upper_limit(15.5)
    assert converter.upper_limit == 15.5

    converter.set_upper_limit(15)
    assert converter.upper_limit == 15

    converter.set_upper_limit(None)
    assert converter.upper_limit is None


def test_numericconverter_setter_method_errors() -> None:
    """Verifies the error handling of NumericConverter class limit setter methods."""
    converter = NumericConverter()
    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        converter.set_lower_limit('Invalid input')

    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        converter.set_upper_limit('Invalid input')


@pytest.mark.parametrize("config", [
    {},
    {"parse_number_strings": False},
    {"allow_int": False, "allow_float": True},
    {"number_lower_limit": -10, "number_upper_limit": 10},
])
def test_numericconverter_config(config):
    """Verifies that initializing NumericConverter class using **kwargs config works as expected."""
    converter = NumericConverter(**config)
    for key, value in config.items():
        if key == "parse_number_strings":
            assert converter.parse_strings == value
        elif key == "allow_int":
            assert converter.allow_int == value
        elif key == "allow_float":
            assert converter.allow_float == value
        elif key == "number_lower_limit":
            assert converter.lower_limit == value
        elif key == "number_upper_limit":
            assert converter.upper_limit == value

# We will now test the BoolConverter class
@pytest.mark.parametrize("config,input_value,expected", [
    ({"parse_bool_equivalents": False}, True, True),
    ({"parse_bool_equivalents": False}, False, False),
    ({"parse_bool_equivalents": True}, "True", True),
    ({"parse_bool_equivalents": True}, "False", False),
    ({"parse_bool_equivalents": True}, "true", True),
    ({"parse_bool_equivalents": True}, "false", False),
    ({"parse_bool_equivalents": True}, 1, True),
    ({"parse_bool_equivalents": True}, 0, False),
    ({"parse_bool_equivalents": True}, "1", True),
    ({"parse_bool_equivalents": True}, "0", False),
    ({"parse_bool_equivalents": True}, 1.0, True),
    ({"parse_bool_equivalents": True}, 0.0, False),

])
def test_boolconverter_success(config, input_value, expected):
    """
    Verifies correct validation behavior for different configurations of BoolConverter class.

    Evaluates:
        0 - Conversion of a boolean input to a boolean output, when boolean equivalents are disabled.
        1 - Conversion of a boolean input to a boolean output, when boolean equivalents are disabled.
        2 - Conversion of a string input to a boolean output, when boolean equivalents are enabled.
        3 - Conversion of a string input to a boolean output, when boolean equivalents are enabled.
        4 - Conversion of a string input to a boolean output, when boolean equivalents are enabled.
        5 - Conversion of a string input to a boolean output, when boolean equivalents are enabled.
        6 - Conversion of an integer input to a boolean output, when boolean equivalents are enabled.
        7 - Conversion of an integer input to a boolean output, when boolean equivalents are enabled.
        8 - Conversion of a string input to a boolean output, when boolean equivalents are enabled.
        9 - Conversion of a string input to a boolean output, when boolean equivalents are enabled.
        10 - Conversion of a float input to a boolean output, when boolean equivalents are enabled.
        11 - Conversion of a float input to a boolean output, when boolean equivalents are enabled.

    Args:
        config: The class configuration to be used for the test. Passed to the class via the **kwargs argument.
        input_value: The value passed to the validation function of the configured class instance.
        expected: The expected output of the validation function.
    """
    converter = BoolConverter(**config)
    assert converter.validate_bool(input_value) == expected

@pytest.mark.parametrize("config,input_value", [
    ({"parse_bool_equivalents": False}, "True", None),
    ({"parse_bool_equivalents": False}, "False", None),
    ({"parse_bool_equivalents": False}, "true", None),
    ({"parse_bool_equivalents": False}, "false", None),
    ({"parse_bool_equivalents": False}, 1, None),
    ({"parse_bool_equivalents": False}, 0, None),
    ({"parse_bool_equivalents": False}, "1", None),
    ({"parse_bool_equivalents": False}, "0", None),
    ({"parse_bool_equivalents": False}, 1.0, None),
    ({"parse_bool_equivalents": False}, 0.0, None),
])
def test_boolconverter_failure(config, input_value):
    """
    Verifies correct validation failure behavior for different configurations of BoolConverter class.

    Evaluates:
        0 - Failure for a string input when boolean equivalents are disabled.
        1 - Failure for a string input when boolean equivalents are disabled.
        2 - Failure for a string input when boolean equivalents are disabled.
        3 - Failure for a string input when boolean equivalents are disabled.
        4 - Failure for an integer input when boolean equivalents are disabled.
        5 - Failure for an integer input when boolean equivalents are disabled.
        6 - Failure for a string input when boolean equivalents are disabled.
        7 - Failure for a string input when boolean equivalents are disabled.
        8 - Failure for a float input when boolean equivalents are disabled.
        9 - Failure for a float input when boolean equivalents are disabled.

    Args:
        config: The class configuration to be used for the test. Passed to the class via the **kwargs argument.
        input_value: The value passed to the validation function of the configured class instance.
    """
    converter = BoolConverter(**config)
    assert converter.validate_bool(input_value) is None