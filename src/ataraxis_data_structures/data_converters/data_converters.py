from typing import Any, Union, Literal, Iterable, Optional
from types import NoneType

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console


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
        parse_number_strings: Determines whether to attempt parsing input strings as numbers (integers or floats).
        allow_integer_output: Determines whether to validate, convert, and return inputs as integer values.
        allow_float_output: Determines whether to validate, convert, and return inputs as float values.
        number_lower_limit: Optional. Lower bound for the returned value, if any. Values below this limit will fail
            validation.
        number_upper_limit: Optional. Upper bound for the returned value, if any. Values above this limit will fail
            validation.

    Attributes:
        _parse_strings: Determines whether to attempt validating strings as number types (with necessary conversions).
        _allow_int: Determines whether the class can validate and convert inputs into integer values.
        _allow_float: Determines whether the class can validate and convert inputs into float values.
        _lower_limit: Optional. An integer or float that specifies the lower limit for numeric value
            verification. Verified integers and floats that are smaller than the limit number will be considered
            invalid. Set to None to disable lower-limit.
        _upper_limit: Optional. An integer or float that specifies the upper limit for numeric value
            verification. Verified integers and floats that are larger than the limit number will be considered invalid.
            Set to None to disable upper-limit.

    Raises:
        TypeError: If any of the initialization arguments are not of the expected type.
        ValueError: If the number_lower_limit is larger than or equal to the number_upper_limit, when both limits are
            not None. If both integer and float outputs are not allowed.
    """

    def __init__(
        self,
        parse_number_strings: bool = True,
        allow_integer_output: bool = True,
        allow_float_output: bool = True,
        number_lower_limit: Optional[Union[int, float]] = None,
        number_upper_limit: Optional[Union[int, float]] = None,
        **kwargs: Any,
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

        # Sets any additional attributes from kwargs. Primarily, this functionality is necessary to support testing,
        # but may also be beneficial for certain API solutions.
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        """Returns a string representation of the NumericConverter instance."""
        representation_string = (
            f"NumericConverter(parse_strings={self.parse_number_strings}, allow_int={self.allow_integer_output}, "
            f"allow_float={self.allow_float_output}, lower_limit={self.number_lower_limit}, "
            f"upper_limit={self.number_upper_limit})"
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

            Since this class is intended to be used together with other validator classes, when conversion fails for
            any reason, it returns None instead of raising an error. This allows sequentially using multiple 'Converter'
            classes as part of a major DataConverter class to implement complex conversion hierarchies.

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
            except ValueError:
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


class BoolConverter:
    """A factory-like class for validating and converting boolean values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    boolean and boolean-equivalent inputs to boolean outputs. After initial configuration the class cannot be
    reconfigured without re-initialization.

    Args:
        parse_boolean_equivalents: Determines whether to attempt parsing boolean equivalents other than True or
            False as boolean values.

    Attributes:
        _parse_bool_equivalents: Determines whether to convert boolean-equivalent strings to boolean values.
        _true_equivalents: Specifies string and numeric values considered equivalent to boolean True values. When
            boolean-equivalent parsing is allowed, these values will be converted to and recognized as valid boolean
            True values.
        _false_equivalents: Same as true_equivalents, but for boolean False equivalents.

    Raises:
        TypeError: If the input parse_boolean_equivalents argument is not a boolean.
    """

    _true_equivalents: set[str | int | float] = {"True", "true", 1, "1", 1.0}
    _false_equivalents: set[str | int | float] = {"False", "false", 0, "0", 0.0}

    def __init__(self, parse_boolean_equivalents: bool = True, **kwargs: Any) -> None:
        # Verifies that initialization arguments are valid:
        if not isinstance(parse_boolean_equivalents, bool):
            message = (
                f"Unable to initialize BooleanConverter class instance. Expected a boolean parse_boolean_equivalents "
                f"argument value, but encountered {parse_boolean_equivalents} of "
                f"type {type(parse_boolean_equivalents).__name__}."
            )
            console.error(message=message, error=TypeError)

        self._parse_bool_equivalents = parse_boolean_equivalents

        # Sets any additional attributes from kwargs. Primarily, this functionality is necessary to support testing,
        # but may also be beneficial for certain API solutions.
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        """Returns a string representation of the BoolConverter instance."""
        representation_string = f"BoolConverter(parse_boolean_equivalents={self.parse_boolean_equivalents})"
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
            Since this class is intended to be used together with other validator classes, when conversion fails for
            any reason, it returns None instead of raising an error. This allows sequentially using multiple 'Model'
            classes as part of a major DataConverter class to implement complex conversion hierarchies.

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
    """
    A factory-like class for validating and converting None values based on a predefined configuration.

    This class can be configured once and then used to validate and, if needed, flexibly between None and None equivalents.
    Specific configuration parameters can be altered through setter methods to dynamically adjust the behavior of the
    instantiated class.

    Args:
        parse_none_equivalents: Determines whether to attempt parsing None equivalents other than None as None values.
            Defaults to True.

    Attributes:
        _parse_none_equivalents: Use this flag to enable converting supported none-equivalent strings to NoneType (None)
            datatype and verifying such strings as None values.
        _none_equivalents: Internal use only. This set specifies None string equivalents. If non-equivalent parsing is
            allowed, these values will be converted to and recognized as None
    """

    _none_equivalents: set[str] = {"None", "none", "Null", "null"}

    def __init__(self, parse_none_equivalents: bool = True, **kwargs: Any) -> None:
        self._parse_none_equivalents = parse_none_equivalents

        # Sets any additional attributes from kwargs. Primarily, this functionality is necessary to support testing,
        # but may also be beneficial for certain API solutions.
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        representation_string = f"NoneConverter(parse_none_equivalents={self._parse_none_equivalents})"
        return representation_string

    @property
    def parse_none_equivalents(self) -> bool:
        """Returns True if the class is configured to attempt parsing None equivalents other than None as None values."""
        return self._parse_none_equivalents

    def toggle_none_equivalents(self) -> bool:
        """
        Flips the value of the attribute that determines if parsing None equivalents is allowed and returns the
        resultant value.
        """
        self._parse_none_equivalents = not self.parse_none_equivalents
        return self.parse_none_equivalents

    def validate_value(self, value: Any) -> None | str:
        """
        Validates and converts the input value into Python None type, based on the configuration.

        Notes:
            If parsing None equivalents is allowed, the class will attempt to convert any input that matches the
            predefined equivalents to a None value.

            Since this class is intended to be used together with other validator classes, when conversion fails for
            any reason, it returns the string 'None' instead of raising an error, since the None type is meaningful in this
            class. This allows sequentially using multiple 'Model' classes as part of a major DataConverter class to
            implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted None value, if conversion succeeds. The string 'None', if conversion fails for
            any reason.
        """
        # If the input is pythonic None, returns None
        if value is None:
            return None
        # If the input is a pythonic-None-equivalent string and the validator is configured to parse none-equivalent
        # strings, returns None
        elif value in self._none_equivalents and self.parse_none_equivalents:
            return None
        # If the value is not in the list of None equivalents, returns the string 'None'.
        else:
            return "None"


class StringConverter:
    """
    A factory-like class for validating and converting string values based on a predefined configuration.

    This class can be configured once and then used to validate and, if needed, flexibly to strings. Specific configuration
    parameters can be altered through setter methods to dynamically adjust the behavior of the instantiated class.

    Args:
        allow_string_conversion: Determines whether to allow converting non-string inputs to strings. Defaults to False.
        string_options: Optional. A list of strings that are considered valid string values.
        string_force_lower: Determines whether to force all string values to lowercase.

    Attributes:
        _allow_string_conversion: Use this flag to enable converting non-string inputs to strings. Since all supported
            input values can be converted to strings, this is a dangerous option that has the potential of overriding
            all verification parameters. It is generally advised to not enable this flag for most use cases.
            Defaults to False because this class is too flexible if this flag is raised.
        _string_options: Optional. A tuple or list of string-options. If provided, all validated strings will be
            checked against the input iterable and only considered valid if the string matches one of the options.
            Set to None to disable string option-limiting. Defaults to None.
        _string_force_lower: Use this flag to force validated string values to be converted to lower-case.
            Only used if allow_string is True and only applies to strings. Defaults to False.
    """

    def __init__(
        self,
        allow_string_conversion: bool = False,
        string_options: Optional[Union[list[str], tuple[str]]] = None,
        string_force_lower: bool = False,
        **kwargs: Any,
    ):
        self._allow_string_conversion = allow_string_conversion
        self._string_options = string_options
        self._string_force_lower = string_force_lower

        # Sets any additional attributes from kwargs. Primarily, this functionality is necessary to support testing,
        # but may also be beneficial for certain API solutions.
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        representation_string = (
            f"StringConverter(allow_string_conversion={self.allow_string_conversion}, string_options={self.string_options}, "
            f"string_force_lower={self.string_force_lower})"
        )
        return representation_string

    @property
    def allow_string_conversion(self) -> bool:
        """Returns True if the class is configured to allow converting non-string inputs to strings."""
        return self._allow_string_conversion

    def toggle_string_conversion(self) -> bool:
        """
        Flips the value of the attribute that determines if converting non-string inputs to strings is allowed and returns
        the resultant value.
        """
        self._allow_string_conversion = not self.allow_string_conversion
        return self.allow_string_conversion

    @property
    def string_options(self) -> list[str] | tuple[str] | None:
        """
        Returns the list of string-options that are considered valid string values.
        """
        return self._string_options

    def set_string_options(self, value: list[str] | tuple[str] | None) -> None:
        """
        Sets the list of string-options that are considered valid string values to the input value.
        """
        self._string_options = value

    @property
    def string_force_lower(self) -> bool:
        """
        Returns True if the class is configured to force validated string values to be converted to lower-case.
        """
        return self._string_force_lower

    def toggle_string_force_lower(self) -> bool:
        """
        Flips the value of the attribute that determines if forcing validated string values to be converted to lower-case is
        allowed and returns the resultant value.
        """
        self._string_force_lower = not self.string_force_lower
        return self.string_force_lower

    def validate_value(self, value: str | bool | int | float | None) -> str | None:
        """
        Validates and converts the input value into Python string type, based on the configuration.

        Notes:
            If string option-limiting is enabled, the class will only consider the input string valid if it matches one
            of the predefined string options. If string force-lower is enabled, the class will convert all validated
            strings to lowercase.

            Since this class is intended to be used together with other validator classes, when conversion fails for
            any reason, it returns None instead of raising an error. This allows sequentially using multiple 'Model'
            classes as part of a major DataConverter class to implement complex conversion hierarchies.

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
            value = str(value)

        # If needed, converts the checked value to lower case. This is done either if the validator is configured to
        # convert strings to lower case or if it is configured to evaluate the string against an iterable of options.
        # In the latter case, the value can still be returned as non-lower-converted string, depending on the
        # 'string_force_lower' attribute setting.
        value_lower = value.lower() if self.string_force_lower or self.string_options else value

        # If option-limiting is enabled, validates the value against the iterable of options
        if self.string_options:
            # Converts options to lower case as an extra compatibility improvement step (potentially avoids user=input
            # errors)
            option_list_lower = [option.lower() for option in self.string_options]

            # Checks if value is in the options list
            if value_lower in option_list_lower:
                # If the validator is configured to convert strings to lower case, returns lower-case string
                if self.string_force_lower:
                    return value_lower
                # Otherwise returns the original input string without alteration
                else:
                    return value
            else:
                # If the value is not in the options list or if the options list is empty, returns None to indicate
                # check failure
                return None

        # If option-limiting is not enabled, returns the string value
        return value_lower


class PythonDataConverter:
    """
    After initial configuration, allows conditionally validating and converting input values to a specified output type.

    The primary use for this class is to convert input values to the datatype(s) defined during class configuration.
    During conversion, input values are checked against the validation parameters of the class prior to being converted
    to the requested datatype, which allows to flexibly and precisely define the range of 'accepted' input and output
    values. This allows the class to serve as both a value converter and validator.

    The primary application for this class is to assist configuration classes, which store data on disk between runtimes
    and, typically, convert all data into string format. This class can be used to convert the strings loaded by
    configuration classes back into the intended format. The class itself can be written and loaded from disk, acting
    as a repository of correct validation / conversion parameters that can be stored in non-volatile memory and used to
    restore the rest of the data to the originally intended datatype.

    Additionally, this class can be used by UI elements to validate user inputs in cases where UI libraries do not
    provide a reliable input validation mechanism.

    Note, the class is designed to be as input-datatype agnostic as possible. In most cases, if a precise input value
    datatype is known, it is more efficient (and easier) to implement a simple in-code conversion. This class is best
    suited for cases when the input value type can vary widely during runtime and/or includes many possible options.

    Attributes:
        _validator: The validator class to be used for value validation. Must be one of the supported validator classes
            (BoolConverter, NoneConverter, NumericConverter, StringConverter).
        _iterable_output_type: Optional. A string-option that allows to convert input iterable values to a particular
            iterable type prior to returning them. Only used when input values are iterables. Valid options
            are 'set', 'tuple' and 'list'. Alternatively, set to None to force the algorithm to use the same iterable
            type for output value as for the input value. Defaults to None.
        _filter_failed: Optional. If set to True, filters out failed values from the output iterable. Defaults to False.

    Raises:
        ValueError: If the input string_options argument is not a tuple, list, or None.
            Also, if the input string_options argument is an empty tuple or  list.
            If the input iterable_output_type argument is not one of the supported iterable output types.
    """

    def __init__(
        self,
        validator: BoolConverter | NoneConverter | NumericConverter | StringConverter,
        iterable_output_type: Optional[Literal["tuple", "list"]] = None,
        filter_failed: bool = False,
    ) -> None:
        self.supported_iterables = {"tuple": tuple, "list": list}

        if not isinstance(validator, (BoolConverter, NoneConverter, NumericConverter, StringConverter)):
            message = (
                f"Unsupported validator class {type(validator).__name__} provided when initializing ValueConverter "
                f"class instance. Must be one of the supported validator classes: "
                f"BoolConverter, NoneConverter, NumericConverter, StringConverter."
            )
            console.error(message=message, error=TypeError)
        if not isinstance(filter_failed, bool):
            message = (
                f"Unsupported filter_failed argument {filter_failed} provided when initializing ValueConverter "
                f"class instance. Must be a boolean value."
            )
            console.error(message=message, error=TypeError)

        # Similarly, checks iterable_output_type for validity
        if iterable_output_type is not None and iterable_output_type not in self.supported_iterables.keys():
            message = (
                f"Unsupported output iterable string-option {iterable_output_type} requested when initializing "
                f"ValueConverter class instance. Select one fo the supported options: "
                f"{self.supported_iterables.keys()}."
            )
            console.error(message=message, error=ValueError)

        # Sets conversion / validation attributes
        self._validator = validator

        self._iterable_output_type = iterable_output_type
        self._filter_failed = filter_failed

    @property
    def validator(self) -> BoolConverter | NoneConverter | NumericConverter | StringConverter:
        return self._validator

    @property
    def iterable_output_type(self) -> Optional[Literal["tuple", "list"]]:
        return self._iterable_output_type

    @property
    def filter_failed(self) -> bool:
        return self._filter_failed

    def toggle_filter_failed(self) -> bool:
        self._filter_failed = not self._filter_failed
        return self._filter_failed

    def set_validator(self, new_validator: BoolConverter | NoneConverter | NumericConverter | StringConverter) -> None:
        if not isinstance(new_validator, (BoolConverter, NoneConverter, NumericConverter, StringConverter)):
            message = (
                f"Unsupported validator class {type(new_validator).__name__} provided when setting ValueConverter "
                f"validator. Must be one of the supported validator classes: "
                f"BoolConverter, NoneConverter, NumericConverter, StringConverter."
            )
            console.error(message=message, error=TypeError)
        self._validator = new_validator
        return

    def validate_value(
        self,
        value_to_validate: int | float | str | bool | None | list[Any] | tuple[Any],
    ) -> (
        int
        | float
        | bool
        | None
        | str
        | list[Union[int, float, bool, str, None]]
        | tuple[int | float | str | None, ...]
    ):
        """
        Validates input values and converts them to the preferred datatype.

        The function validates input values against the validation parameters of the class instance. If the input value
        passes validation, the function converts it to the preferred datatype. If the input value is an iterable, the
        function converts it to the preferred iterable type. The function also filters out failed values from the output
        iterable if the filter_failed attribute is set to True.

        Args
            value_to_validate: The input value to be validated and converted.

        Returns
            The validated and converted value.
        """
        try:
            list_value: list[int | float | str | bool | None] = PythonDataConverter.ensure_list(value_to_validate)
            output_iterable: list[int | float | str | bool | None] = []
            for value in list_value:
                value = self._validator.validate_value(value)
                if self.filter_failed:
                    if type(self.validator) == NoneConverter and value == "None":
                        continue
                    elif value is None and type(self.validator) != NoneConverter:
                        continue
                output_iterable.append(value)

            if len(output_iterable) <= 1:
                return output_iterable[0]

            return tuple(output_iterable) if self.iterable_output_type == "tuple" else output_iterable

        except TypeError as e:
            message = f"Unable to convert input value to a Python list: {e}"
            console.error(message=message, error=TypeError)
            raise TypeError("Fallback to appease mypi")

    @staticmethod
    def ensure_list(
        input_item: str
        | int
        | float
        | bool
        | None
        | np.generic
        | NDArray[Any]
        | tuple[Any, ...]
        | list[Any]
        | set[Any],
    ) -> list[Any]:
        """Ensures that the input object is returned as a list.

        If the object is not already a list, attempts to convert it into a list. If the object is a list, returns the
        object unchanged.

        Notes:
            This function makes no attempt to further validate object data or structure outside of making sure it is
            returned as a list. This means that objects like multidimensional numpy arrays will be returned as nested
            lists and returned lists may contain non-list objects.

            Numpy arrays are fully converted into python types when passing through this function. That is, individual
            data-values will be converted from numpy-scalar to the nearest python-scalar types before being written to a
            list.

        Args:
            input_item: The object to be converted into / preserved as a Python list.

        Returns:
            A Python list that contains input_item data. If the input_item was a scalar, it is wrapped into a list object.
            If the input_item was an iterable, it is converted into list.

        Raises:
            TypeError: If the input object cannot be converted or wrapped into a list.
        """
        # Scalars are added to a list and returned as a one-item lists. Scalars are handled first to avoid clashing with
        # iterable types.
        if np.isscalar(input_item) or input_item is None:  # Covers Python scalars and NumPy scalars
            return [input_item]
        # Numpy arrays are processed based on their dimensionality. This has to dow tih the fact that zero-dimensional
        # numpy arrays are interpreted as scalars by some numpy methods and as array by others.
        if isinstance(input_item, np.ndarray):
            # 1+-dimensional arrays are processed via tolist(), which correctly casts them to Python list format.
            if input_item.ndim > 0:
                output_item: list[Any] = input_item.tolist()
                return output_item
            else:
                # 0-dimensional arrays are essentially scalars, so the data is popped out via item() and is wrapped
                # into a list.
                output_item = [input_item.item()]
                return output_item
        # Lists are returned as-is, without any further modification.
        if isinstance(input_item, list):
            return input_item
        # Iterable types are converted via list() method.
        if isinstance(input_item, Iterable):
            return list(input_item)

        # Catch-all type error to execute if the input is
        message = (
            f"Unable to convert input item to a Python list, as items of type {type(input_item).__name__} "
            f"are not supported."
        )
        console.error(message=message, error=TypeError)
        raise TypeError("Fallback to appease mypi")


class NumpyDataConverter(PythonDataConverter):
    """
    Extends the PythonDataConverter class to allow for conversion of input values to numpy datatypes.

    The primary use for this class is to convert input values to numpy datatypes. The class is designed
    to be as input-datatype agnostic as possible. In most cases, if a precise input value type is known, it is more
    efficient (and easier) to implement a simple in-code conversion. This class is best suited for cases when the input
    value type can vary widely during runtime and/or includes many possible options.

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
        if not isinstance(python_converter, PythonDataConverter):
            message = (
                f"Unsupported python_converter class {type(python_converter).__name__} provided when initializing "
                f"NumpyDataConverter class instance. Must be an instance of PythonDataConverter."
            )
            console.error(message=message, error=TypeError)
        if output_bit_width is not None and output_bit_width not in [8, 16, 32, 64, "auto"]:
            message = (
                f"Unsupported output_bit_width {output_bit_width} provided when initializing NumpyDataConverter "
                f"class instance. Must be one of the supported options: 8, 16, 32, 64, 'auto'."
            )
            console.error(message=message, error=ValueError)
        if type(python_converter.validator) == StringConverter:
            message = (
                f"Unsupported validator class {type(python_converter.validator).__name__} provided when initializing "
                f"NumpyDataConverter class instance. Must be one of the supported validator classes: "
                f"BoolConverter, NoneConverter, NumericConverter."
            )
            console.error(message=message, error=TypeError)
        if not python_converter.filter_failed:
            message = (
                f"Unsupported filter_failed argument {python_converter.filter_failed} provided when initializing "
                f"NumpyDataConverter class instance. Must be set to True."
            )
            console.error(message=message, error=ValueError)
        if type(python_converter.validator) == NumericConverter:
            if python_converter.validator.allow_integer_output and python_converter.validator.allow_float_output:
                message = (
                    f"Unsupported NumericConverter configuration provided when initializing NumpyDataConverter "
                    f"class instance. Both allow_int and allow_float cannot be set to True."
                )
                console.error(message=message, error=ValueError)

        self._signed = signed
        self._python_converter = python_converter
        self._output_bit_width = output_bit_width

    @property
    def python_converter(self) -> PythonDataConverter:
        """Returns the python_converter attribute of the class instance."""
        return self._python_converter

    @property
    def output_bit_width(self) -> int | str:
        """Returns the bit-width of the output numpy datatype."""
        return self._output_bit_width

    @property
    def signed(self) -> bool:
        """Returns the signed attribute of the class instance."""
        return self._signed

    def toggle_signed(self) -> bool:
        """Toggles the signed attribute between True and False and returns the new value."""
        self._signed = not self._signed
        return self._signed

    def set_output_bit_width(self, new_output_bit_width: int | str) -> None:
        if new_output_bit_width not in [8, 16, 32, 64, "auto"]:
            message = (
                f"Unsupported output_bit_width {new_output_bit_width} provided when setting NumpyDataConverter "
                f"output_bit_width. Must be one of the supported options: 8, 16, 32, 64, 'auto'."
            )
            console.error(message=message, error=ValueError)

        self._output_bit_width = new_output_bit_width

    def set_python_converter(self, new_python_converter: PythonDataConverter) -> None:
        """Sets the python_converter attribute of the class instance to a new PythonDataConverter instance."""
        if not isinstance(new_python_converter, PythonDataConverter):
            message = (
                f"Unsupported python_converter class {type(new_python_converter).__name__} provided when setting "
                f"NumpyDataConverter python_converter. Must be an instance of PythonDataConverter."
            )
            console.error(message=message, error=TypeError)
        self._python_converter = new_python_converter

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
