import numpy as np
import pytest
from pydantic import ValidationError

from ataraxis_data_structures.data_converters.python_models import (
    BoolConverter,
    NoneConverter,
    StringConverter,
    NumericConverter,
)
from ataraxis_data_structures.data_converters.numpy_converter import PythonDataConverter
from ataraxis_data_structures.data_converters.numpy_converter import NumpyDataConverter

# from ataraxis_data_structures.data_converters.numpy_converter import PythonDataConverter


@pytest.mark.parametrize(
    "config,input_value,expected",
    [
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
    ],
)
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
    assert converter.validate_value(input_value) == expected


@pytest.mark.parametrize(
    "config,input_value",
    [
        ({}, "not a number"),
        ({}, [1, 2, 3]),
        ({"allow_int": False, "allow_float": False}, 5),
        ({"parse_number_strings": False}, "5.5"),
        ({"number_lower_limit": 0}, -5),
        ({"number_upper_limit": 10}, 15),
        ({"allow_float": False}, 5.5),
    ],
)
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
    assert converter.validate_value(input_value) is None


def test_numericconverter_init_validation():
    """Verifies that NumericConverter initialization method functions as expected and correctly catches invalid inputs."""
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
    converter = NumericConverter(
        parse_number_strings=True, allow_int=True, allow_float=True, number_lower_limit=0, number_upper_limit=10
    )

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
        converter.set_lower_limit("Invalid input")

    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        converter.set_upper_limit("Invalid input")


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"parse_number_strings": False},
        {"allow_int": False, "allow_float": True},
        {"number_lower_limit": -10, "number_upper_limit": 10},
    ],
)
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
    ],
)
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

    Args:s
        config: The class configuration to be used for the test. Passed to the class via the **kwargs argument.
        input_value: The value passed to the validation function of the configured class instance.
        expected: The expected output of the validation function.
    """
    converter = BoolConverter(**config)
    assert converter.validate_value(input_value) == expected


@pytest.mark.parametrize(
    "config,input_value",
    [
        ({"parse_bool_equivalents": False}, "True"),
        ({"parse_bool_equivalents": False}, "False"),
        ({"parse_bool_equivalents": False}, "true"),
        ({"parse_bool_equivalents": False}, "false"),
        ({"parse_bool_equivalents": False}, 1),
        ({"parse_bool_equivalents": False}, 0),
        ({"parse_bool_equivalents": False}, "1"),
        ({"parse_bool_equivalents": False}, "0"),
        ({"parse_bool_equivalents": False}, 1.0),
        ({"parse_bool_equivalents": False}, 0.0),
    ],
)
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
    assert converter.validate_value(input_value) is None


def test_boolconverter_init_validation():
    """
    Verifies that BoolConverter initialization method functions as expected and correctly catches invalid inputs,
    """
    # Tests valid initialization
    converter = BoolConverter(parse_bool_equivalents=True)
    assert converter.parse_bool_equivalents

    # Tests invalid initialization (relies on pydantic to validate the inputs)
    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        BoolConverter(parse_bool_equivalents="not a bool")


def test_boolconverter_properties():
    """
    Verifies that accessor properties of BoolConverter class function as expected
    """
    converter = BoolConverter(parse_bool_equivalents=True)
    assert converter.parse_bool_equivalents


def test_boolconverter_toggle_methods():
    """
    Verifies the functioning of BoolConverter configuration flag toggling methods.
    """
    converter = BoolConverter()

    assert not converter.toggle_bool_equivalents()
    assert not converter.parse_bool_equivalents
    assert converter.toggle_bool_equivalents()
    assert converter.parse_bool_equivalents


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"parse_bool_equivalents": True},
    ],
)
def test_boolconverter_config(config):
    """
    Verifies that initializing BoolConverter class using **kwargs config works as expected.
    """
    converter = BoolConverter(**config)
    for key, value in config.items():
        if key == "parse_bool_equivalents":
            assert converter.parse_bool_equivalents == value


@pytest.mark.parametrize(
    "config,input_value,expected",
    [
        ({}, None, None),
        ({}, "None", None),
        ({}, "none", None),
        ({}, "null", None),
        ({}, "Null", None),
    ],
)
def test_noneconverter_success(config, input_value, expected):
    """
    Verifies correct validation behavior for different configurations of NoneConverter class.

    Evaluates:
        0 - Conversion of a None input to a None output.
        1 - Conversion of a string input to a None output.
        2 - Conversion of a string input to a None output.
        3 - Conversion of a string input to a None output.
        4 - Conversion of a string input to a None output.
        5 - Conversion of a string input to a None output.
        6 - Conversion of a string input to a None output.

    Args:
        config: The class configuration to be used for the test. Passed to the class via the **kwargs argument.
        input_value: The value passed to the validation function of the configured class instance.
        expected: The expected output of the validation function.
    """
    converter = NoneConverter(**config)
    assert converter.validate_value(input_value) == expected


@pytest.mark.parametrize(
    "config,input_value",
    [
        ({}, "nil"),
        ({}, 5),
        ({}, 5.5),
        ({}, True),
        ({}, False),
        ({"parse_none_equivalents": False}, "None"),
        ({"parse_none_equivalents": False}, "none"),
        ({"parse_none_equivalents": False}, "null"),
        ({"parse_none_equivalents": False}, "NULL"),
    ],
)
def test_noneconverter_failure(config, input_value):
    """
    Verifies correct validation failure behavior for different configurations of NoneConverter class.

    Evaluates:
        0 - Failure for a string input.
        1 - Failure for an integer input.
        2 - Failure for a float input.
        3 - Failure for a boolean input.
        4 - Failure for a boolean input.
        5 - Failure for a string input when None equivalents are disabled.
        6 - Failure for a string input when None equivalents are disabled.
        7 - Failure for a string input when None equivalents are disabled.
        8 - Failure for a string input when None equivalents are disabled.

    Args:
        config: The class configuration to be used for the test. Passed to the class via the **kwargs argument.
        input_value: The value passed to the validation function of the configured class instance.
    """
    converter = NoneConverter(**config)
    assert converter.validate_value(input_value) is "None"


def test_noneconverter_init_validation():
    """
    Verifies that NoneConverter initialization method functions as expected and correctly catches invalid inputs.
    """
    # Tests valid initialization
    converter = NoneConverter(parse_none_equivalents=True)
    assert converter.parse_none_equivalents

    # Tests invalid initialization (relies on pydantic to validate the inputs)
    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        NoneConverter(parse_none_equivalents="not a bool")


def test_noneconverter_properties():
    """
    Verifies that accessor properties of NoneConverter class function as expected
    """
    converter = NoneConverter(parse_none_equivalents=True)
    assert converter.parse_none_equivalents


def test_noneconverter_toggle_methods():
    """
    Verifies the functioning of NoneConverter configuration flag toggling methods.
    """
    converter = NoneConverter()

    assert not converter.toggle_none_equivalents()
    assert not converter.parse_none_equivalents
    assert converter.toggle_none_equivalents()
    assert converter.parse_none_equivalents


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"parse_none_equivalents": True},
    ],
)
def test_noneconverter_config(config):
    """
    Verifies that initializing NoneConverter class using **kwargs config works as expected.
    """
    converter = NoneConverter(**config)
    for key, value in config.items():
        if key == "parse_none_equivalents":
            assert converter.parse_none_equivalents == value


@pytest.mark.parametrize(
    "config,input_value,expected",
    [
        # Converter Capabilities
        ({"allow_string_conversion": True}, "Spongebob", "Spongebob"),
        ({"allow_string_conversion": True}, 5, "5"),
        ({"allow_string_conversion": True}, 5.5, "5.5"),
        ({"allow_string_conversion": True}, True, "True"),
        ({"allow_string_conversion": True}, False, "False"),
        ({"allow_string_conversion": True}, None, "None"),
        ({"allow_string_conversion": True, "string_options": ["1", "2"]}, 1, "1"),
        ({"allow_string_conversion": True, "string_options": ["1", "2"]}, 2, "2"),
        ({"allow_string_conversion": True, "string_force_lower": True}, "Spongebob", "spongebob"),
        # Validator Capabilities
        ({}, "Spongebob", "Spongebob"),
        ({"string_options": ["Spongebob", "Patrick"]}, "Spongebob", "Spongebob"),
        ({"string_options": ["Spongebob", "Patrick"]}, "Patrick", "Patrick"),
    ],
)
def test_stringconverter_success(config, input_value, expected):
    """
    Verifies correct validation behavior for different configurations of StringConverter class.

    Evaluates:
        0 - Conversion of a string input to a string output.
        1 - Conversion of a string input to a string output.
        2 - Conversion of an integer input to a string output.
        3 - Conversion of a float input to a string output.
        4 - Conversion of a boolean input to a string output.
        5 - Conversion of a boolean input to a string output.
        6 - Conversion of a None input to a string output.
        7 - Conversion of an integer input to a string output, using a list of string options.
        8 - Conversion of an integer input to a string output, using a list of string options.
        9 - Conversion of a string input to a string output, with forced lower case conversion.
        10 - Validation of a string input into a string output
        11 - Validation of a string input into a string output with a list of valid options
        12 - Validation of a string input into a string output with a list of valid options
    """
    converter = StringConverter(**config)
    assert converter.validate_value(input_value) == expected


@pytest.mark.parametrize(
    "config,input_value",
    [
        ({"allow_string_conversion": True, "string_options": ["1", "2"]}, 3),
        ({"string_options": ["Spongebob", "Patrick"]}, "Squidward"),
        ({}, 1),
        ({}, 1.0),
        ({}, True),
        ({}, False),
        ({}, None),
    ],
)
def test_stringconverter_failure(config, input_value):
    """
    Verifies correct validation failure behavior for different configurations of StringConverter class.

    Evaluates:
        0 - Failure for an integer input, when the input is not in the list of valid string options.
        1 - Failure for a string input, when the input is not in the list of valid string options.
        2 - Failure for an integer input.
        3 - Failure for a float input.
        4 - Failure for a boolean input.
        5 - Failure for a boolean input.
        6 - Failure for a None input.
    """
    converter = StringConverter(**config)
    assert converter.validate_value(input_value) is None


def test_stringconverter_init_validation():
    """
    Verifies that StringConverter initialization method functions as expected and correctly catches invalid inputs.
    """
    # Tests valid initialization
    converter = StringConverter(allow_string_conversion=True, string_options=["A", "B"], string_force_lower=True)
    assert converter.allow_string_conversion
    assert converter.string_options == ["A", "B"]
    assert converter.string_force_lower

    # Tests invalid initialization (relies on pydantic to validate the inputs)
    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        StringConverter(allow_string_conversion="not a bool")

    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        StringConverter(string_options="not a list")

    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        StringConverter(string_force_lower="not a bool")


def test_stringconverter_properties():
    """
    Verifies that accessor properties of StringConverter class function as expected
    """
    converter = StringConverter(allow_string_conversion=True, string_options=["A", "B"], string_force_lower=True)
    assert converter.allow_string_conversion
    assert converter.string_options == ["A", "B"]
    assert converter.string_force_lower


def test_stringconverter_toggle_methods():
    """
    Verifies the functioning of StringConverter configuration flag toggling methods.
    """
    converter = StringConverter()

    assert converter.toggle_string_conversion()
    assert converter.allow_string_conversion
    assert not converter.toggle_string_conversion()
    assert not converter.allow_string_conversion

    assert converter.toggle_string_force_lower()
    assert converter.string_force_lower
    assert not converter.toggle_string_force_lower()
    assert not converter.string_force_lower


def test_stringconverter_setter_methods() -> None:
    """
    Verifies the functioning of StringConverter class string options setter method.
    """
    converter = StringConverter()

    converter.set_string_options(["A", "B"])
    assert converter.string_options == ["A", "B"]

    converter.set_string_options(["C", "D"])
    assert converter.string_options == ["C", "D"]

    converter.set_string_options(None)
    assert converter.string_options is None


def test_stringconverter_setter_method_errors() -> None:
    """
    Verifies the error handling of StringConverter class string options setter method.
    """
    converter = StringConverter()
    with pytest.raises(ValidationError):
        # noinspection PyTypeChecker
        converter.set_string_options("Invalid input")


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"allow_string_conversion": True},
        {"string_options": ["A", "B"]},
        {"string_force_lower": True},
    ],
)
def test_stringconverter_config(config):
    """
    Verifies that initializing StringConverter class using **kwargs config works as expected.
    """
    converter = StringConverter(**config)
    for key, value in config.items():
        if key == "allow_string_conversion":
            assert converter.allow_string_conversion == value
        elif key == "string_options":
            assert converter.string_options == value
        elif key == "string_force_lower":
            assert converter.string_force_lower == value


def test_pythonconverter_init_validation():
    """
    Verifies that PythonDataConverter initialization method functions as expected and correctly catches invalid inputs.
    """
    # Tests valid initialization
    converter = PythonDataConverter(validator=NumericConverter(), iterable_output_type="list", filter_failed=True)
    assert type(converter.validator) is NumericConverter
    assert converter.iterable_output_type is "list"
    assert converter.filter_failed

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        PythonDataConverter(validator="not a validator")

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        PythonDataConverter(iterable_output_type="not a string")

    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        PythonDataConverter(filter_failed="not a bool")


def test_pythonconverter_success():
    """
    Verifies correct validation behavior for different configurations of PythonDataConverter class.
    """
    converter = PythonDataConverter(validator=NumericConverter(), iterable_output_type="list", filter_failed=True)
    assert converter.validate_value([5, 5.5, True, False, None, "7.1"]) == [5, 5.5, 1, 0, 7.1]

    converter = PythonDataConverter(validator=BoolConverter(), iterable_output_type="tuple", filter_failed=True)
    assert converter.validate_value([5, 5.5, True, False, None, "7.1"]) == (True, False)

    converter = PythonDataConverter(validator=StringConverter(), iterable_output_type="tuple", filter_failed=False)
    assert converter.validate_value([5, 5.5, True, False, None, "7.1"]) == (None, None, None, None, None, "7.1")

    converter = PythonDataConverter(validator=NoneConverter(), iterable_output_type="list", filter_failed=False)
    assert converter.validate_value([5, 5.5, "None", "Null", None, "7.1"]) == ["None", "None", None, None, None, "None"]


def test_pythonconverter_properties():
    """
    Verifies that accessor properties of PythonDataConverter class function as expected
    """
    converter = PythonDataConverter(validator=NumericConverter(), iterable_output_type="list", filter_failed=True)

    assert type(converter.validator) is NumericConverter
    assert converter.iterable_output_type == "list"
    assert converter.filter_failed


def test_pythonconverter_toggle_methods():
    """
    Verifies the functioning of PythonDataConverter configuration flag toggling methods.
    """
    converter = PythonDataConverter(validator=NumericConverter(), iterable_output_type="list", filter_failed=True)

    assert not converter.toggle_filter_failed()
    assert not converter.filter_failed
    assert converter.toggle_filter_failed()
    assert converter.filter_failed


def test_pythonconverter_setter_methods():
    """
    Verifies the functioning of PythonDataConverter class validator setter method.
    """
    converter = PythonDataConverter(validator=NumericConverter(), iterable_output_type="list", filter_failed=True)

    converter.set_validator(BoolConverter())
    assert type(converter.validator) == BoolConverter

    converter.set_validator(NoneConverter())
    assert type(converter.validator) == NoneConverter

    converter.set_validator(StringConverter())
    assert type(converter.validator) == StringConverter

    converter.set_validator(NumericConverter())
    assert type(converter.validator) == NumericConverter


def test_numpyconverter_init_validation():
    """
    Verifies that NumpyDataConverter initialization method functions as expected and correctly catches invalid inputs.
    """
    # Tests valid initialization
    validator = PythonDataConverter(validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert type(converter.converter) is PythonDataConverter
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
    
    validator = PythonDataConverter(validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=False)
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
    validator = PythonDataConverter(validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True)
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
    assert type(converter.converter) is PythonDataConverter


def test_numpyconverter_success():
    """
    Verifies correct validation behavior for different configurations of NumpyDataConverter class.
    """
    validator = PythonDataConverter(validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert np.array_equal(
        converter.python_to_numpy_converter([5, 5.5, True, False, None, "7.1"]), np.array([5, 1, 0])
    )
    converter.set_output_bit_width(8)
    converter.toggle_signed()
    assert np.array_equal(
        converter.python_to_numpy_converter([2**4, 2**5, 2**6, 2**7, 2**8, 2**9]), np.array([16, 32, 64, 128, np.inf, np.inf])
    )
    value = converter.python_to_numpy_converter(2**6)
    assert isinstance(value, np.uint8)

    validator = PythonDataConverter(validator=BoolConverter(), iterable_output_type="tuple", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert np.array_equal(converter.python_to_numpy_converter([5, 5.5, True, False, None, "7.1"]), np.array([1, 0]))

    validator = PythonDataConverter(validator=NoneConverter(), iterable_output_type="list", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert np.array_equal(
        converter.python_to_numpy_converter([5, 5.5, "None", "Null", None, "7.1"]),
        np.array([np.nan, np.nan, np.nan]), equal_nan=True
    )

    # Numpy to Python conversion
    validator = PythonDataConverter(validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")
    assert converter.numpy_to_python_converter(np.array([5, 1, 0])) == [5, 1, 0]
    assert converter.numpy_to_python_converter(np.int_(5.0)) == 5

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
    with pytest.raises(ValueError):
        validator = PythonDataConverter(validator=NumericConverter(allow_int=False), iterable_output_type="list", filter_failed=True)
        converter = NumpyDataConverter(validator, output_bit_width=8)
        converter.python_to_numpy_converter([5.5, 6.0])


def test_numpyconverter_properties():
    """
    Verifies that accessor properties of NumpyDataConverter class function as expected
    """
    validator = PythonDataConverter(validator=NumericConverter(allow_float=False), iterable_output_type="list", filter_failed=True)
    converter = NumpyDataConverter(validator, output_bit_width="auto")

    assert type(converter.converter) is PythonDataConverter
    assert converter.output_bit_width == "auto"
    assert converter.signed
