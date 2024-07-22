from types import NoneType
from typing import Any, Union, Literal, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console

from ..standalone_methods.data_manipulation_methods import ensure_list


class NumericConverter:
    """A factory-like class for validating and converting numeric values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    integer, float, string, and boolean inputs to integer or float outputs. After initial configuration the class
    cannot be reconfigured without re-initialization.

    Notes:
        If both integer and float outputs are allowed, the class will always prioritize floats over integers.
        This is because all integers can be converted to floats without data loss, but not all floats can be
        converted to integers without losing data (rounding).

    Args:
        number_lower_limit: Optional. Lower bound for the returned value, if any. Values below this limit will fail
            validation.
        number_upper_limit: Optional. Upper bound for the returned value, if any. Values above this limit will fail
            validation.
        parse_number_strings: Determines whether to attempt parsing input strings as numbers (integers or floats).
        allow_integer_output: Determines whether to validate, convert, and return inputs as integer values.
        allow_float_output: Determines whether to validate, convert, and return inputs as float values.

    Attributes:
        _lower_limit: Optional. An integer or float that specifies the lower limit for numeric value
            verification. Verified integers and floats that are smaller than the limit number will be considered
            invalid. Set to None to disable lower-limit.
        _upper_limit: Optional. An integer or float that specifies the upper limit for numeric value
            verification. Verified integers and floats that are larger than the limit number will be considered invalid.
            Set to None to disable upper-limit.
        _parse_strings: Determines whether to attempt validating strings as number types (with necessary conversions).
        _allow_int: Determines whether the class can validate and convert inputs into integer values.
        _allow_float: Determines whether the class can validate and convert inputs into float values.

    Raises:
        TypeError: If any of the initialization arguments are not of the expected type.
        ValueError: If the number_lower_limit is larger than or equal to the number_upper_limit, when both limits are
            not None. If both integer and float outputs are not allowed.
    """

    def __init__(
        self,
        number_lower_limit: Optional[Union[int, float]] = None,
        number_upper_limit: Optional[Union[int, float]] = None,
        *,
        parse_number_strings: bool = True,
        allow_integer_output: bool = True,
        allow_float_output: bool = True,
    ) -> None:
        # Verifies that initialization arguments are valid:
        if not isinstance(parse_number_strings, bool):
            message = (
                f"Unable to initialize NumericConverter class instance. Expected a boolean parse_number_strings "
                f"argument value, but encountered {parse_number_strings} of type {type(parse_number_strings).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(allow_integer_output, bool):
            message = (
                f"Unable to initialize NumericConverter class instance. Expected a boolean allow_integer_output "
                f"argument value, but encountered {allow_integer_output} of type {type(allow_integer_output).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(allow_float_output, bool):
            message = (
                f"Unable to initialize NumericConverter class instance. Expected a boolean allow_float_output "
                f"argument value, but encountered {allow_float_output} of type {type(allow_float_output).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(number_lower_limit, (int, float, NoneType)):
            message = (
                f"Unable to initialize NumericConverter class instance. Expected an integer, float or NoneType "
                f"number_lower_limit argument value, but encountered {number_lower_limit} of "
                f"type {type(number_lower_limit).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(number_upper_limit, (int, float, NoneType)):
            message = (
                f"Unable to initialize NumericConverter class instance. Expected an integer, float or NoneType "
                f"number_upper_limit argument value, but encountered {number_upper_limit} of "
                f"type {type(number_upper_limit).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Also ensures that if both lower and upper limits are provided, the lower limit is less than the upper limit.
        if (
            number_lower_limit is not None
            and number_upper_limit is not None
            and not number_lower_limit < number_upper_limit
        ):
            message = (
                f"Unable to initialize NumericConverter class instance. Expected a number_lower_limit that is less "
                f"than the number_upper_limit, but encountered a lower limit of {number_lower_limit} and an upper "
                f"limit of {number_upper_limit}."
            )
            console.error(message=message, error=ValueError)

        # Ensures that at least one output type is allowed
        if not allow_integer_output and not allow_float_output:
            message = (
                f"Unable to initialize NumericConverter class instance. Expected allow_integer_output, "
                f"allow_float_output or both to be True, but both are set to False. At least one output type must be "
                f"enabled to instantiate a class."
            )
            console.error(message=message, error=ValueError)

        # Saves configuration parameters to attributes.
        self._parse_strings = parse_number_strings
        self._allow_int = allow_integer_output
        self._allow_float = allow_float_output
        self._lower_limit = number_lower_limit
        self._upper_limit = number_upper_limit

    def __repr__(self) -> str:
        """Returns a string representation of the NumericConverter instance."""
        representation_string = (
            f"NumericConverter(parse_number_strings={self.parse_number_strings}, "
            f"allow_integer_output={self.allow_integer_output}, allow_float_output={self.allow_float_output}, "
            f"number_lower_limit={self.number_lower_limit}, number_upper_limit={self.number_upper_limit})"
        )
        return representation_string

    @property
    def parse_number_strings(self) -> bool:
        """Returns True if the class is configured to parse input strings as numbers."""
        return self._parse_strings

    @property
    def allow_integer_output(self) -> bool:
        """Returns True if the class is configured to output (after validation and / or conversion) Python integers."""
        return self._allow_int

    @property
    def allow_float_output(self) -> bool:
        """Returns True if the class is configured to output (after validation and / or conversion) Python floats."""
        return self._allow_float

    @property
    def number_lower_limit(self) -> int | float | None:
        """Returns the lower bound used to determine valid numbers or None, if minimum limit is not set."""
        return self._lower_limit

    @property
    def number_upper_limit(self) -> int | float | None:
        """Returns the upper bound used to determine valid numbers or None, if minimum limit is not set."""
        return self._upper_limit

    def validate_value(self, value: bool | str | int | float | None) -> float | int | None:
        """Ensures that the input value is a valid number (integer or float), depending on class configuration.

        If the value is not a number, but is number-convertible, converts the value to the valid number type. Optionally
        carries out additional validation steps, such as checking whether the value is within the specified bounds.

        Notes:
            If both integer and float outputs are allowed, the class will always prioritize floats over integers.
            This is because all integers can be converted to floats without data loss, but not all floats can be
            converted to integers without losing data (rounding).

            Boolean inputs are automatically parsed as floats, as they are derivatives from the base integer class.

            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns None instead of raising an error. This allows sequentially using multiple
            'Converter' classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted number, either as a float or integer, if conversion succeeds. None, if
            conversion fails for any reason.
        """
        # Filters out any types that are definitely not integer or float convertible.
        if not isinstance(value, (int, str, bool, float)):
            return None

        # Converts strings to floats if this is allowed.
        if isinstance(value, str) and self._parse_strings:
            try:
                value = float(value)
            except Exception:
                return None

        # Converts booleans to integers (they already are integers, strictly speaking)
        if isinstance(value, bool):
            value = float(value)

        # If the input value is not converted to int or float by this point, then it cannot be validated.
        if not isinstance(value, (int, float)):
            return None

        # Validates the type of the value, making the necessary and allowed conversions, if possible, to pass this step.
        if isinstance(value, int) and not self._allow_int:
            # If the value is an integer, integers are not allowed and floats are not allowed, returns None.
            if not self._allow_float:
                return None

            # If the value is an integer, integers are not allowed, but floats are allowed, converts the value to float.
            # Relies on the fact that any integer is float-convertible.
            value = float(value)

        elif isinstance(value, float) and not self._allow_float:
            # If the value is a float, floats are not allowed, integers are allowed, and value is integer-convertible
            # without data-loss, converts it to an integer.

            if value.is_integer() and self._allow_int:
                value = int(value)
            # If the value is a float, floats are not allowed, and either integers are not allowed or the value is not
            # integer-convertible without data loss, returns None.
            else:
                return None

        # Validates that the value is in the specified range if any is provided.
        if (self._lower_limit is not None and value < self._lower_limit) or (
            self._upper_limit is not None and value > self._upper_limit
        ):
            return None

        # Returns the validated (and, potentially, converted) value.
        return value


class BooleanConverter:
    """A factory-like class for validating and converting boolean values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    boolean and boolean-equivalent inputs to boolean outputs. After initial configuration the class cannot be
    reconfigured without re-initialization.

    Args:
        parse_boolean_equivalents: Determines whether to attempt parsing boolean equivalents other than True or
            False as boolean values.

    Attributes:
        _parse_bool_equivalents: Determines whether to convert boolean-equivalents to boolean values.
        _true_equivalents: Specifies string and numeric values considered equivalent to boolean True values. When
            boolean-equivalent parsing is allowed, these values will be converted to and recognized as valid boolean
            True values.
        _false_equivalents: Same as true_equivalents, but for boolean False equivalents.

    Raises:
        TypeError: If the input parse_boolean_equivalents argument is not a boolean.
    """

    _true_equivalents: set[str | int | float] = {"True", "true", 1, "1", 1.0}
    _false_equivalents: set[str | int | float] = {"False", "false", 0, "0", 0.0}

    def __init__(self, *, parse_boolean_equivalents: bool = True) -> None:
        # Verifies that initialization arguments are valid:
        if not isinstance(parse_boolean_equivalents, bool):
            message = (
                f"Unable to initialize BooleanConverter class instance. Expected a boolean parse_boolean_equivalents "
                f"argument value, but encountered {parse_boolean_equivalents} of "
                f"type {type(parse_boolean_equivalents).__name__}."
            )
            console.error(message=message, error=TypeError)

        self._parse_bool_equivalents = parse_boolean_equivalents

    def __repr__(self) -> str:
        """Returns a string representation of the BooleanConverter instance."""
        representation_string = f"BooleanConverter(parse_boolean_equivalents={self.parse_boolean_equivalents})"
        return representation_string

    @property
    def parse_boolean_equivalents(self) -> bool:
        """Returns True if the class is configured to parse boolean equivalents as boolean values."""
        return self._parse_bool_equivalents

    def validate_value(self, value: bool | str | int | float | None) -> bool | None:
        """Ensures that the input value is a valid boolean.

        If the value is not a boolean, but is boolean-equivalent, converts the value to the valid boolean type, if
        parsing boolean equivalents is allowed.

        Notes:
            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns None instead of raising an error. This allows sequentially using multiple
            'Converter' classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted boolean value, if conversion succeeds. None, if conversion fails for any reason.
        """
        # If the input is a boolean type returns it to caller unchanged
        if isinstance(value, bool):
            return value

        # Otherwise, if the value is a boolean-equivalent string or number and parsing boolean-equivalents is allowed,
        # converts it to boolean True or False and returns it to caller
        if self.parse_boolean_equivalents and isinstance(value, (str, int, float)):
            # If the value is in the set of true equivalents, returns True.
            if value in self._true_equivalents:
                return True
            # If the value is in the set of false equivalents, returns False.
            elif value in self._false_equivalents:
                return False
        # If the value is not in the list of true or false equivalents, returns None.
        return None


class NoneConverter:
    """A factory-like class for validating and converting None values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    NoneType (None) and None-equivalent inputs to None outputs. After initial configuration the class cannot be
    reconfigured without re-initialization.

    Args:
        parse_none_equivalents: Determines whether to attempt parsing None equivalents as NoneType (None) values.

    Attributes:
        _parse_none_equivalents: Determines whether to convert None-equivalent inputs to None values.
        _none_equivalents: Specifies string values considered equivalent to NoneType (None) values. When
            None-equivalent parsing is allowed, these values will be converted to and recognized as valid NoneType
            values.

    Raises:
        TypeError: If the input parse_none_equivalents argument is not a boolean.
    """

    _none_equivalents: set[str] = {"None", "none", "Null", "null"}

    def __init__(self, *, parse_none_equivalents: bool = True) -> None:
        # Verifies that initialization arguments are valid:
        if not isinstance(parse_none_equivalents, bool):
            message = (
                f"Unable to initialize NoneConverter class instance. Expected a boolean parse_none_equivalents "
                f"argument value, but encountered {parse_none_equivalents} of "
                f"type {type(parse_none_equivalents).__name__}."
            )
            console.error(message=message, error=TypeError)

        self._parse_none_equivalents = parse_none_equivalents

    def __repr__(self) -> str:
        """Returns a string representation of the NoneConverter instance."""
        representation_string = f"NoneConverter(parse_none_equivalents={self._parse_none_equivalents})"
        return representation_string

    @property
    def parse_none_equivalents(self) -> bool:
        """Returns True if the class is configured to parse None-equivalent inputs as None values."""
        return self._parse_none_equivalents

    def validate_value(self, value: Any) -> None | str:
        """Ensures that the input value is a valid NoneType (None).

        If the value is not a None, but is None-equivalent, converts the value to the valid None type, if
        parsing None equivalents is allowed.

        Notes:
            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns "None" string instead of raising an error. This allows sequentially using
            multiple 'Converter' classes as part of a major DataConverter class to implement complex conversion
            hierarchies.

            Note the difference above! Since "None" is the desired output from this class, the error-return uses a
            string type and "None" value.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted None value, if conversion succeeds. The string "None", if conversion fails for
            any reason.
        """
        # If the input is pythonic None, returns None
        if value is None:
            return None

        # If the input is a pythonic-None-equivalent string and the validator is configured to parse none-equivalent
        # strings, returns None
        elif value in self._none_equivalents and self.parse_none_equivalents:
            return None

        # If the value is not in the set of None equivalents, returns the string 'None' to indicate validation failure.
        else:
            return "None"


class StringConverter:
    """A factory-like class for validating and converting string values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    most inputs to String outputs. After initial configuration the class cannot be reconfigured without
    re-initialization.

    Notes:
        Almost any Python object can be converted to a string. Therefore, depending on configuration, this class
        can have a lot of power to almost always return valid string outputs.

        When string-options are provided, the class converts them to lower-case regardless of other parameters.
        Validated strings are also converted to lower-case before checking them against the options. This design
        intentionally makes the class case-insensitive.

    Args:
        string_options: Optional. A tuple or list of strings that are considered valid string values. Any input not
            matching the contents of this iterable will be considered invalid, even if it is string-convertible. Set to
            None to disable option-checking.
        allow_string_conversion: Determines whether to allow converting non-string inputs to strings. Defaults to False.
        string_force_lower: Determines whether to force all string values to lowercase.

    Attributes:
        _string_options: Optional. A tuple or list of string-options. If provided, all validated strings will be
            checked against the input iterable and only considered valid if the string matches one of the options.
        _string_force_lower: Determines if validated string values have to be converted to lower-case.
        _allow_string_conversion: Determines whether to convert non-string inputs to strings. Setting this to true is
            fairly dangerous, as almost anything can be converted to a string.

    Raises:
        TypeError: If any input argument is not of the correct type. This includes the elements of iterable
            string-options argument.
        ValueError: If the string_options argument is an empty iterable.
    """

    def __init__(
        self,
        string_options: Optional[Union[list[str], tuple[str]]] = None,
        *,
        allow_string_conversion: bool = False,
        string_force_lower: bool = False,
    ):
        # Verifies that initialization arguments are valid:
        if not isinstance(allow_string_conversion, bool):
            message = (
                f"Unable to initialize StringConverter class instance. Expected a boolean allow_string_conversion "
                f"argument value, but encountered {allow_string_conversion} of type "
                f"{type(allow_string_conversion).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(string_force_lower, bool):
            message = (
                f"Unable to initialize StringConverter class instance. Expected a boolean string_force_lower "
                f"argument value, but encountered {string_force_lower} of type {type(string_force_lower).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(string_options, (tuple, list, NoneType)):
            message = (
                f"Unable to initialize StringConverter class instance. Expected a None, tuple, or list string_options "
                f"argument value, but encountered {string_options} of type {type(string_options).__name__}."
            )
            console.error(message=message, error=TypeError)

        # For string_options, not set to None, carries out some additional checks.
        if string_options is not None:
            # Does not allow string-options to be empty iterables. To disable option-checking, it has to be set to
            # None
            if len(string_options) == 0:
                message = (
                    f"Unable to initialize StringConverter class instance. Expected at least one option inside the "
                    f"string-options list or tuple argument, but encountered an empty iterable. To disable limiting "
                    f"strings to certain options, set string_options to None."
                )
                console.error(message=message, error=ValueError)

            # For non-empty iterables, ensures that all elements are strings
            for element in string_options:
                if not isinstance(element, str):
                    message = (
                        f"Unable to initialize StringConverter class instance. Expected all elements of string-options "
                        f"argument to be strings, but encountered {element} of type {type(element).__name__}."
                    )
                    console.error(message=message, error=TypeError)

            # Converts options to the lower case as an extra compatibility improvement step
            # (potentially avoids user input errors)
            string_options = [option.lower() for option in string_options]

        self._allow_string_conversion = allow_string_conversion
        self._string_force_lower = string_force_lower
        self._string_options = string_options

    def __repr__(self) -> str:
        """Returns a string representation of the StringConverter instance."""
        representation_string = (
            f"StringConverter(allow_string_conversion={self._allow_string_conversion}, "
            f"string_force_lower={self._string_force_lower}, string_options={self._string_options},)"
        )
        return representation_string

    @property
    def allow_string_conversion(self) -> bool:
        """Returns True if the class is configured to convert non-string inputs to strings."""
        return self._allow_string_conversion

    @property
    def string_options(self) -> list[str] | tuple[str] | None:
        """Returns the list of string-options that are considered valid string values.

        If strings are not limited to a collection of options, returns None.
        """
        return self._string_options

    @property
    def string_force_lower(self) -> bool:
        """Returns True if the class is configured to convert validated strings to lower-case."""
        return self._string_force_lower

    def validate_value(self, value: str | bool | int | float | None) -> str | None:
        """Ensures that the input value is a valid String.

        If the value is not a string, but is string-convertible, converts the value to the valid string type if
        string-conversion is allowed.

        Notes:
            If string option-limiting is enabled, the class will only consider the input string valid if it matches one
            of the predefined string options. Before matching the string to option, the class converts BOTH options and
            checked string to lower-case to amke it case-invariant. 'passed' values are still returned using the input
            case.

            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns None instead of raising an error. This allows sequentially using multiple
            'Converter' classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted string value, if conversion succeeds. None, if conversion fails for any reason.
        """

        # Ensures that the input variable is a string, otherwise returns None to indicate check failure. If the variable
        # is originally not a string, but string-conversions are allowed, attempts to convert it to string, but returns
        # None if the conversion fails (unlikely)
        if not isinstance(value, str) and not self.allow_string_conversion:
            return None
        else:
            try:
                value = str(value)
            except Exception:
                return None

        # If needed, converts the checked value to the lower case. This is done if the class is configured to evaluate
        # the string against a list or tuple of options. The value can still be returned as non-lower-converted string,
        # depending on the 'string_force_lower' attribute value.
        value_lower = value.lower() if self._string_force_lower or self._string_options is not None else value

        # If option-limiting is enabled, validates the value against the iterable of options
        if self._string_options is not None and value_lower in self._string_options:
            # If the validator is configured to convert strings to the lower case, returns lower-case string
            if self.string_force_lower:
                return value_lower
            # Otherwise returns the original input string without alteration
            else:
                return value
        elif self._string_options is None:
            # If option-limiting is not enabled, returns the string value. Converts it to the lower case if requested.
            return value_lower
        else:
            # If the value is not in the options' iterable or if the options, returns None to indicate check failure.
            return None


class PythonDataConverter:
    """After initial configuration, allows conditionally validating and / or converting input values to a specified
    pythonic output type.

    Broadly, this class is designed to wrap one or more 'base' converter classes (NumericConverter, BooleanConverter,
    StringConverter, NoneConverter) and extend their value validation methods to work for iterable inputs. Combining
    multiple converters allows the class to apply them hierarchically to process a broad range of input values
    (see the Notes section below for details). This design achieves maximum conversion / validation flexibility, making
    this class generally usable for a wide range of cases.

    Notes:
        When multiple converter options are used, the class always defers to the following hierarchy:
        float > integer > boolean > None > string. This hierarchy is chosen to (roughly) prioritize outputting
        'non-permissive' types first. For example, an integer is always float-convertible, but not vice versa. Since
        almost every input is potentially convertible to a string, the strings are evaluated last.

        The primary application for this class is to help configuration classes (YamlConfig, for example), which store
        data on disk between runtimes and, typically, convert all data into string format. This class can be used to
        convert the strings loaded by configuration classes back into the intended format. Instances of this class can
        be written and loaded from disk, acting as a repository of correct validation / conversion parameters stored
        in non-volatile data. After loading them from disk, they can restore the rest of the data to the originally
        intended datatype.

        Additionally, this class can be used by UI and similarly interactive elements to validate user inputs in cases
        where UI libraries do not provide a desired input validation mechanism.

        The class is designed to be as input-datatype agnostic as possible. In most cases, if a precise input value
        datatype is known, it is more efficient (and easier) to implement a simple in-code conversion. This class is
        best suited for cases when the input value type can vary widely during runtime and/or consists of many possible
        options.

    Args:
        numeric_converter: Optional. The initialized NumericConverter class instance or None to disable validating and
            converting inputs to numeric types.
        boolean_converter: Optional. The initialized BooleanConverter class instance or None to disable validating and
            converting inputs to boolean types.
        none_converter: Optional. The initialized NoneConverter class instance or None to disable validating and
            converting inputs to None types.
        string_converter: Optional. The initialized StringConverter class instance or None to disable validating and
            converting inputs to string types.
        iterable_output_type: Optional. Determines the type input iterables will be cast to before they are returned.
        filter_failed_elements: Determines whether to filter individual iterable elements that fail validation. By
            default, they are returned as None / "None" strings.
        raise_errors: Determines whether to return outputs that failed validation as None or to raise ValueError
            exceptions. Enabling this option allows using this class similarly to how pydantic models are used.

    Attributes:
        _numeric_converter: Optionally stores the NumericConverter to apply to input values.
        _boolean_converter: Optionally stores the BooleanConverter to apply to input values.
        _none_converter: Optionally stores the NoneConverter to apply to input values.
        _string_converter: Optionally stores the StringConverter to apply to input values.
        _iterable_output_type: Optionally stores the type to cast input iterables to before returning them.
        _filter_failed_elements: Determines whether to fileter scalar elements that fail validation from returned
            iterables.
        _raise_errors: Determines whether to return inputs that failed validation as None or to raise ValueError when
            an input fails validation.
        _allowed_outputs: A set that stores all output types that are supported by the current class configuration.
        _supported_iterables: A dictionary that maps supported output iterable keys to callable types.

    Raises:
        TypeError: If any of the input arguments are not of the expected type.
        ValueError: If the requested iterable output type is not supported. If all converter inputs are set to None.
    """

    _supported_iterables = {"tuple": tuple, "list": list}

    def __init__(
        self,
        numeric_converter: Optional[NumericConverter] = None,
        boolean_converter: Optional[BooleanConverter] = None,
        none_converter: Optional[NoneConverter] = None,
        string_converter: Optional[StringConverter] = None,
        iterable_output_type: Optional[Literal["tuple", "list"]] = None,
        *,
        filter_failed_elements: bool = False,
        raise_errors: bool = False,
    ) -> None:
        # Ensures input arguments have valid types
        if not isinstance(numeric_converter, (NumericConverter, NoneType)):
            message = (
                f"Unable to initialize PythonDataConverter class instance. Expected a numeric_validator argument "
                f"of type {type(NumericConverter).__name__} or {type(None).__name__}, but "
                f"encountered {numeric_converter} of type {type(numeric_converter).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(boolean_converter, (BooleanConverter, NoneType)):
            message = (
                f"Unable to initialize PythonDataConverter class instance. Expected a boolean_validator argument "
                f"of type {type(BooleanConverter).__name__} or {type(None).__name__}, but "
                f"encountered {boolean_converter} of type {type(boolean_converter).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(none_converter, (NoneConverter, NoneType)):
            message = (
                f"Unable to initialize PythonDataConverter class instance. Expected a none_validator argument "
                f"of type {type(NoneConverter).__name__} or {type(None).__name__}, but "
                f"encountered {none_converter} of type {type(none_converter).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(string_converter, (StringConverter, NoneType)):
            message = (
                f"Unable to initialize PythonDataConverter class instance. Expected a string_validator argument "
                f"of type {type(StringConverter).__name__} or {type(None).__name__}, but "
                f"encountered {string_converter} of type {type(string_converter).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(filter_failed_elements, bool):
            message = (
                f"Unable to initialize PythonDataConverter class instance. Expected a boolean filter_failed_elements "
                f"argument value, but encountered {filter_failed_elements} of "
                f"type {type(filter_failed_elements).__name__}."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(raise_errors, bool):
            message = (
                f"Unable to initialize PythonDataConverter class instance. Expected a boolean raise_errors "
                f"argument value, but encountered {raise_errors} of "
                f"type {type(raise_errors).__name__}."
            )
            console.error(message=message, error=TypeError)
        if iterable_output_type is not None and iterable_output_type not in self._supported_iterables.keys():
            message = (
                f"Unsupported output iterable type {iterable_output_type} requested when initializing "
                f"PythonDataConverter class instance. Select one of the supported options: "
                f"{self._supported_iterables.keys()}."
            )
            console.error(message=message, error=ValueError)

        # Depending on the converter configuration, builds a set of allowed outputs. This is used in class error
        # messages
        self._allowed_outputs: set = set()
        if numeric_converter is not None:
            if numeric_converter.allow_float_output:
                self._allowed_outputs.add(type(float()).__name__)
            if numeric_converter.allow_integer_output:
                self._allowed_outputs.add(type(int()).__name__)
        if boolean_converter is not None:
            self._allowed_outputs.add(type(bool()).__name__)
        if none_converter is not None:
            self._allowed_outputs.add(type(None).__name__)
        if string_converter is not None:
            self._allowed_outputs.add(type(str()).__name__)

        # The only way for allowed_outputs to be empty is if all converters are set to None.
        if len(self._allowed_outputs) == 0:
            message = (
                f"Unable to initialize PythonDataConverter class instance. Expected at least one of the class "
                f"converter arguments to be set to a supported converter class, but all are set to None. This class "
                f"requires at least one configured base converter (NumericConverter, BooleanConverter, NoneConverter, "
                f"StringConverter) to operate as intended."
            )
            console.error(message=message, error=ValueError)

        # Saves arguments to appropriate class attributes
        self._numeric_converter: Optional[NumericConverter] = numeric_converter
        self._none_converter: Optional[NoneConverter] = none_converter
        self._string_converter: Optional[StringConverter] = string_converter
        self._boolean_converter: Optional[BooleanConverter] = boolean_converter
        self._iterable_output_type: Optional[Literal["tuple", "list"]] = iterable_output_type
        self._filter_failed_elements: bool = filter_failed_elements
        self._raise_errors: bool = raise_errors

    def __repr__(self) -> str:
        """Returns a string representation of the PythonDataConverter class instance."""
        representation_string: str = (
            f"PythonDataConverter(allowed_output_types={self.allowed_output_types}, "
            f"iterable_output_type={self._iterable_output_type}, "
            f"filter_failed_elements={self._filter_failed_elements}, raise_errors={self._raise_errors})"
        )
        return representation_string

    @property
    def numeric_converter(self) -> Optional[NumericConverter]:
        """Returns the NumericConverter instance used by the class to validate and convert inputs into numbers (integers
        or floats).

        If the class does not support numeric conversion, returns None.
        """
        return self._numeric_converter

    @property
    def boolean_converter(self) -> Optional[BooleanConverter]:
        """Returns the BooleanConverter instance used by the class to validate and convert inputs into booleans.

        If the class does not support boolean conversion, returns None.
        """
        return self._boolean_converter

    @property
    def none_converter(self) -> Optional[NoneConverter]:
        """Returns the NoneConverter instance used by the class to validate and convert inputs into NoneTypes (Nones).

        If the class does not support None conversion, returns None.
        """
        return self._none_converter

    @property
    def string_converter(self) -> Optional[StringConverter]:
        """Returns the StringConverter instance used by the class to validate and convert inputs into strings.

        If the class does not support string conversion, returns None.
        """
        return self._string_converter

    @property
    def iterable_output_type(self) -> Optional[Literal["tuple", "list"]]:
        """Returns the name of the type to which processed iterables are cast before they are returned or None, if
        the class is configured to preserve the original iterable type.
        """
        return self._iterable_output_type

    @property
    def filter_failed(self) -> bool:
        """Returns True if the class is configured to remove elements that failed validation from the processed
        iterables before returning them.
        """
        return self._filter_failed_elements

    @property
    def raise_errors(self) -> bool:
        """Returns True if the class is configured to raise ValueError exceptions when an input fails validation."""
        return self._raise_errors

    @classmethod
    def supported_iterables(cls) -> tuple[str, ...]:
        """Returns a tuple that stores string-names of the supported output iterable types.

        These names are valid inputs to class 'iterable_output_type' initialization argument.
        """
        supported_types = tuple([value.__name__ for value in cls._supported_iterables.values()])
        return supported_types

    @property
    def allowed_output_types(self) -> tuple[str, ...]:
        """Returns the string-names of the scalar python types the class is configured to produce."""

        # Sorts the types stored in teh set and returns them to caller as a tuple
        return tuple(sorted(self._allowed_outputs))

    def _apply_converters(self, value: Any) -> tuple[bool, int | float | str | bool | None]:
        """Hierarchically applies each of the converters to the input scalar value.

        This is a minor service method that allows standardizing iterable and non-iterable input processing. This
        method contains the core validation logic, whereas the validate_value() method primarily provides set-up and
        tear-down functionality.

        Notes:
            Follows the following conversion hierarchy if multiple converters are active:
            float > integer > boolean > None > string.

        Args:
            value: The value to be validated and / or converted.

        Returns: A tuple that contains two values. The first is a boolean that indicates if the returned value passed
            or failed validation. The second is either the validated / converted value or a None placeholder.
        """

        # Applies numeric converter. If it works, returns the result without evaluating further options
        if self._numeric_converter is not None:
            result = self._numeric_converter.validate_value(value)
            if result is not None:
                return True, result

        # Applies boolean converter. If it works, returns the result without evaluating further options
        if self._boolean_converter is not None:
            result = self._boolean_converter.validate_value(value)
            if result is not None:
                return True, result

        # Applies none converter. If it works, returns the result without evaluating further options
        if self._none_converter is not None:
            result = self._none_converter.validate_value(value)
            # None converters use string "None" to indicate conversionf failures
            if result != "None":
                return True, result

        # Applies string converter. If it works, returns the result without evaluating further options
        if self._string_converter is not None:
            result = self._string_converter.validate_value(value)
            if result is not None:
                return True, result

        # If the value did not pass validation with any of the enabled converters, returns None placeholder with a
        # failure code
        return False, None

    def validate_value(
        self,
        value_to_validate: Any,
    ) -> int | float | bool | None | str | list[int | float | bool | str | None] | tuple[int | float | str | None, ...]:
        """Validates input values and converts them to the preferred datatype.

        This method validates input values against the validation parameters of the class instance. If the input value
        passes validation, the method converts it to the preferred datatype. If the input value is iterable, the
        method converts it to the preferred iterable type (tuple if not specified).

        The method can conditionally filter out values that fail validation from the output iterable if the
        filter_failed attribute is set to True. Alternatively, it can raise ValueErrors for failed elements if the
        class is configured to do so.

        Note:
            When this class is equipped with multiple base converter classes, the supported types are evaluated in the
            following order: float > integer > boolean > None > string.

        Args
            value_to_validate: The input value to be validated and converted.

        Returns
            The validated and converted value if the method succeeds. For each value that fails validation when
            filtering is disabled, returns "Validation/ConversionError" string to indicate failure.

        Raises:
            ValueError: If the val;ue to validate is iterable with multiple dimensions. If the input scalar value or
                any element of an iterable value cannot be validated, and the raise_errors attribute is set to True.
        """
        # Converts the input value to a list
        list_value: list[int | float | str | bool | None] = ensure_list(value_to_validate)

        for num, element in enumerate(list_value):
            if isinstance(element, Iterable) and not isinstance(element, (bool, NoneType, int, float, str)):
                message = (
                    f"Unable to validate the input collection of values ({list_value}). Currently, this method only "
                    f"supports one-dimensional iterable inputs. Instead, a sub-iterable was discovered when "
                    f"evaluating element {num} ({element})."
                )
                console.error(message=message, error=ValueError)

        # Uses list-comprehension to call _apply_converters method on each of the values in the list. This generates a
        # list of two-element tuples
        output_iterable: list[tuple[bool, int | float | str | bool | None]] = [
            self._apply_converters(value) for value in list_value
        ]

        # Loops over the tuple-elements of the output_iterables and the original input value(s) and handles them
        # according to the class filtering / error configuration
        final_result = []  # This is the list that will be returned to caller
        for (success, output_value), input_value in zip(output_iterable, list_value):
            # If the evaluated element has failed validation and errors are to be raised, raises a ValueError
            if not success and self.raise_errors:
                message = (
                    f"Unable to validate the input value ({input_value}). The class is configured to conditionally "
                    f"return the following types: {self.allowed_output_types}. This means that the input value is "
                    f"conditionally not convertible into any allowed type. Note, this may be due to failing secondary "
                    f"checks, such as numeric limits or string-option filters. The value was provided as part of this "
                    f"collection of values: {list_value}."
                )
                console.error(message=message, error=ValueError)
                # Fallback to appease mypy, should not be reachable
                raise ValueError(message)  # pragma: no cover

            # Otherwise, if the evaluated element has failed validation and failed elements are to be filtered, skips
            # adding it to the result list:
            elif not success and self.filter_failed:
                continue

            # If filtering is disabled, appends "Validation/ConversionError" string to the output list in place of the
            # value that failed validation
            elif not success:
                final_result.append("Validation/ConversionError")

            # if the value passed validation, appends it to the output list as-is
            else:
                final_result.append(output_value)

        # If the length of the result list is 1, pops out the only element and returns it as a scalar
        if len(final_result) == 1:
            return final_result[0]
        elif len(final_result) == 0:
            # If the length of the list is 0, this indicates that all elements failed validation and were filtered out.
            # In this case, returns an error string
            return "Validation/ConversionError"

        # For iterable outputs, determines and returns it either as a tuple or a list, depending on what was requested.
        # Defaults to tuples if the user did not explicitly request a particular output type.
        return (
            tuple(final_result)
            if self.iterable_output_type == "tuple" or self.iterable_output_type is None
            else final_result
        )


class NumpyDataConverter:
    """After initial configuration, allows conditionally converting input python values to a specified numpy output
    type.

    This class is built on top of our general PythonDataConverter hierarchy. Specifically, it uses a PythonDataConverter
    class instance to filter and convert inputs to specific Python types know to be convertible to specific numpy types.
    It then converts the input to numpy, using the requested bit-width and signed/unsigned type for integers.

    Notes:
        The class deliberately only works with Python inputs. Numpy already contains a robust set of tools for converting
        between numpy datatypes. The purpose of this class is to provide a robust way for converting arbitrary inputs
        into a specific numpy datatype.

        The primary use for this class is to convert input values to numpy datatypes. The class is designed
        to be as input-datatype agnostic as possible. In most cases, if a precise input value type is known, it is more
        efficient (and easier) to implement a simple in-code conversion. This class is best suited for cases when the
        input value type can vary widely during runtime and/or includes many possible options.

    Attributes:
        _python_converter: The PythonDataConverter instance to be used for input value validation and conversion.
        _output_bit_width: The bit-width of the output numpy datatype. Must be one of the supported options: 8, 16, 32, 64,
            'auto'. If set to 'auto', the class will attempt to determine the smallest numpy datatype that can
            accommodate the input value.
        _signed: If True, the output numpy datatype will be signed. If False, the output numpy datatype will be unsigned.

    Raises:
        TypeError: If the provided python_converter argument is not an instance of PythonDataConverter.
            If the provided validator argument is an instance of StringConverter.
        ValueError: If the provided output_bit_width argument is not one of the supported options: 8, 16, 32, 64, 'auto'.
            If the provided filter_failed argument is set to False.
            If the provided NumericConverter configuration allows both allow_int and allow_float to be set to True.
    """

    def __init__(
        self,
        python_converter: PythonDataConverter,
        output_bit_width: int | str = "auto",
        signed: bool = True,
    ):
        # Verifies that the input arguments have valid types
        if not isinstance(python_converter, PythonDataConverter):
            message = (
                f"Unable to initialize NumpyDataConverter class instance. Expected a python_converter argument "
                f"of type {type(PythonDataConverter).__name__}, but encountered {python_converter} of type "
                f"{type(python_converter).__name__}."
            )
            console.error(message=message, error=TypeError)
        if output_bit_width is not None and output_bit_width not in [8, 16, 32, 64, "auto"]:
            message = (
                f"Unable to initialize NumpyDataConverter class instance. Encountered an unsupported output_bit_width "
                f"argument value ({output_bit_width}). Use one of the supported options: 8, 16, 32, 64, 'auto'."
            )
            console.error(message=message, error=ValueError)
        if not isinstance(signed, bool):
            message = (
                f"Unable to initialize NumpyDataConverter class instance. Expected a boolean signed argument type, "
                f"but encountered {signed} of type {type(signed).__name__}."
            )
            console.error(message=message, error=TypeError)

        # Carries out additional verification for the provided PythonDataConverter instance
        allowed_output_types = python_converter.allowed_output_types
        # Does not allow using PythonDataConverter if it is configured to allow more than one output type
        if len(allowed_output_types) != 1:
            message = (
                f"Unable to initialize NumpyDataConverter class instance. The PythonDataConverter class instance "
                f"provided as python_converter argument is configured to allow multiple scalar output types: "
                f"{allowed_output_types}. NumpyDataConverter class requires the PythonDataConverter class to only "
                f"allow a single scalar output type to be compatible."
            )
            console.error(message=message, error=ValueError)

        # Specifically, does not support strings at this time
        elif "string" in allowed_output_types:
            message = (
                f"Unable to initialize NumpyDataConverter class instance. The PythonDataConverter class instance "
                f"provided as python_converter argument is configured to validates strings. Currently, "
                f"NumpyDataConverter does not support converting strings to numpy formats."
            )
            console.error(message=message, error=ValueError)

        # Finally, expects PythonDataConverter to act as an error-generating filter and enforces it to be configured
        # to do so.
        if not python_converter.raise_errors:
            message = (
                f"Unable to initialize NumpyDataConverter class instance. The PythonDataConverter class instance "
                f"provided as python_converter argument is configured to not raise errors on validation failure. "
                f"Currently, NumpyDataConverter expects that PythonDataConverter raises errors if it cannot validate"
                f"inputs."
            )
            console.error(message=message, error=ValueError)

        # Stores arguments into class attributes
        self._signed = signed
        self._python_converter = python_converter
        self._output_bit_width = output_bit_width

    @property
    def python_converter(self) -> PythonDataConverter:
        """Returns the PythonDataConverter instance used to selectively control the range supported input and output
        arguments to class methods."""
        return self._python_converter

    @property
    def output_bit_width(self) -> int | str:
        """Returns the bit-width used by the output numpy datatype."""
        return self._output_bit_width

    @property
    def signed(self) -> bool:
        """Returns True if the class is configured to convert integer inputs to signed or unsigned integers."""
        return self._signed

    def min_scalar_type_signed_or_unsigned(self, value: int | float) -> type:
        """
        Returns the minimum scalar type to represent `value` with the desired signedness.

        Parameters
            value: The input value to be checked.
            signed: If True, return a signed dtype. If False, return an unsigned dtype.

        Returns
            A NumPy callable type.
        """
        # Determine the smallest scalar type that can represent the value
        dtype = np.min_scalar_type(value)
        # If the current dtype already matches the signed/unsigned preference, return it
        if (np.issubdtype(dtype, np.signedinteger) and self._signed) or (
            np.issubdtype(dtype, np.unsignedinteger) and not self._signed
        ):
            if isinstance(dtype, type):
                return dtype
        if type(value) == int:
            # Define the hierarchy of integer types
            signed_types: list[tuple[type, int, int]] = [
                (np.int8, -(2**7), 2**7 - 1),
                (np.int16, -(2**15), 2**15 - 1),
                (np.int32, -(2**31), 2**31 - 1),
                (np.int64, -(2**63), 2**63 - 1),
            ]
            unsigned_types: list[tuple[type, int, int]] = [
                (np.uint8, 0, 2**8 - 1),
                (np.uint16, 0, 2**16 - 1),
                (np.uint32, 0, 2**32 - 1),
                (np.uint64, 0, 2**64 - 1),
            ]

            # Choose the appropriate hierarchy
            types = signed_types if self._signed else unsigned_types

            # Find the smallest dtype that can accommodate the value
            for t, min_val, max_val in types:
                if min_val <= value <= max_val:
                    return t
        elif type(value) == float:
            # Define the hierarchy of float types
            float_types: list[tuple[type, np.floating[Any], np.floating[Any]]] = [
                (np.float16, np.finfo(np.float16).min, np.finfo(np.float16).max),
                (np.float32, np.finfo(np.float32).min, np.finfo(np.float32).max),
                (np.float64, np.finfo(np.float64).min, np.finfo(np.float64).max),
            ]

            # Find the smallest dtype that can accommodate the value
            for t, min_val_f, max_val_f in float_types:
                if min_val_f <= value <= max_val_f:
                    return t
        # To appease tox
        message = f"Value {value} is too large to be represented by a NumPy integer type."
        console.error(message=message, error=OverflowError)
        raise OverflowError(message)

    def python_to_numpy_converter(
        self,
        value_to_convert: int | float | bool | None | str | list[Any] | tuple[Any],
    ) -> np.integer[Any] | np.unsignedinteger[Any] | np.bool | NDArray[Any]:
        """
        Converts input values to numpy datatypes.

        The function converts input values to numpy datatypes based on the configuration of the class instance. The
        function supports conversion of scalar values, lists, and tuples. The function also supports conversion of
        iterable values, such as lists and tuples, to numpy arrays.

        Args
            value_to_convert: The input value to be converted to a numpy datatype.

        Returns
            The converted value as a numpy datatype.

        Raises
            ValueError: If the output_bit_width is set to 8 and the input value is a float. Numpy does not support 8-bit floats.
        """
        signed: list[type] = [np.int8, np.int16, np.int32, np.int64]
        unsigned: list[type] = [np.uint8, np.uint16, np.uint32, np.uint64]
        float_sign: list[type] = [np.float16, np.float16, np.float32, np.float64]

        validated_value: Any = self.python_converter.validate_value(value_to_convert)
        validated_list: list[Any] = PythonDataConverter.ensure_list(validated_value)
        temp: list[Any] = []

        for value in validated_list:
            if self._output_bit_width == "auto" and isinstance(value, (int, float)) and value not in (True, False):
                min_dtype = self.min_scalar_type_signed_or_unsigned(value)
                numpy_value: np.integer[Any] | np.unsignedinteger[Any] | np.bool | NDArray[Any] | float = np.array(
                    value, dtype=min_dtype
                )

            elif isinstance(value, (int, float)) and value not in (True, False):
                if self._output_bit_width == 8 and type(value) == float:
                    message = f"Unable to convert input value to a numpy datatype. Numpy does not support 8-bit floats."
                    console.error(message=message, error=ValueError)

                if type(value) == float:
                    width_list: list[type] = float_sign
                else:
                    width_list = signed if self._signed else unsigned

                index_map: dict[int, int] = {8: 0, 16: 1, 32: 2, 64: 3}
                if type(self._output_bit_width) == int:
                    datatype: type = width_list[index_map[self._output_bit_width]]

                try:
                    numpy_value = datatype(value)
                    if numpy_value == 0.0 and value != 0.0:
                        numpy_value = np.nan
                except OverflowError:
                    numpy_value = np.inf

            # Handle bools and None
            elif type(value) == bool:
                numpy_value = np.bool(value)
            elif value is None:
                numpy_value = np.nan
            temp.append(numpy_value)
        return np.array(temp) if len(temp) > 1 else temp[0]

    def numpy_to_python_converter(
        self,
        value_to_convert: Union[
            np.integer[Any],
            np.unsignedinteger[Any],
            np.bool,
            float,
            NDArray[Any],
        ],
    ) -> int | float | bool | None | list[Any] | tuple[Any, ...]:
        """
        Converts numpy datatypes to Python datatypes.

        The function converts numpy datatypes to Python datatypes based on the configuration of the class instance. The
        function supports conversion of scalar values, lists, and tuples. The function also supports conversion of
        iterable values, such as lists and tuples, to numpy arrays.

        Args
            value_to_convert: The input value to be converted to a Python datatype.

        Returns
            The converted value as a Python datatype.
        """

        if isinstance(value_to_convert, np.ndarray):
            temp: list[Any] = []
            for i in range(len(value_to_convert)):
                if np.isnan(value_to_convert[i]) or np.isinf(value_to_convert[i]):
                    temp.append(None)
                else:
                    temp.append(value_to_convert[i].item())

            validated: int | float | bool | str | None | list[Any] | tuple[Any, ...] = (
                self.python_converter.validate_value(temp)
            )
            if not isinstance(validated, str):
                if type(validated) == list and len(validated) == 1:
                    # To satisfy tox
                    out = validated.pop()
                    if isinstance(out, (int, float, bool, type(None), list, tuple)):
                        return out
                    else:
                        message = f"Unable to convert input value to a Python datatype."
                        console.error(message=message, error=ValueError)
                else:
                    return validated
            else:
                # To satify tox
                message = f"Unable to convert input value to a Python datatype."
                console.error(message=message, error=ValueError)
        if np.isnan(value_to_convert) or np.isinf(value_to_convert):
            return None
        output = self.python_converter.validate_value(np.array(value_to_convert).item(0))
        if not isinstance(output, str):
            return output
        else:
            message = f"Unable to convert input value to a Python datatype."
            console.error(message=message, error=ValueError)
            raise ValueError("Fallback to appease mypi")
