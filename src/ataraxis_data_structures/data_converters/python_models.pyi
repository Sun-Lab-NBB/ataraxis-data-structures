from _typeshed import Incomplete
from typing import Any

class NumericConverter:
    """A factory-like class for validating and converting numeric values based on a predefined configuration.

    This class can be configured once and then used to validate and, if needed, flexibly convert int, float, str,
    and bool inputs to int or float. Specific configuration parameters can be altered through setter methods to
    dynamically adjust the behavior of the instantiated class.

    Args:
        parse_number_strings: Determines whether to attempt parsing input strings as numbers.
        allow_int: Determines whether to allow returning integer values.
        allow_float: Determines whether to allow returning float values.
        number_lower_limit: Optional. Lower bound for the returned value, if any.
        number_upper_limit: Optional. Upper bound for the returned value, if any.

    Attributes:
        _parse_strings: Use this flag to enable converting all integer- and float-equivalent strings to the
            appropriate numeric datatype and verifying such strings as integers and/or floats.
        _allow_int: Use this flag to allow the validated value to be an integer or an integer-convertible string or
            float. Integer-convertible strings are only allowed if parse_number_strings flag is True. When enabled
            together with allow_float, the algorithm always tries to convert floats into integers where possible.
            The range of allowed values can be constricted with number_lower_limit and number_upper_limit attributes.
        _allow_float: Use this flag to allow the validated value to be a float or a float-convertible string or integer.
            Float-convertible strings are only allowed if parse_number_strings flag is True. When enabled together
            with allow_int, the algorithm always tries to convert floats into integers where possible. The range of
            allowed values can be constricted with number_lower_limit and number_upper_limit attributes.
        _lower_limit: Optional. An integer or float that specifies the lower limit for numeric value
            verification. Verified integers and floats that are smaller than the limit number will be considered
            invalid. Set to None to disable lower-limit. Defaults to None.
        _upper_limit: Optional. An integer or float that specifies the upper limit for numeric value
            verification. Verified integers and floats that are larger than the limit number will be considered invalid.
            Set to None to disable upper-limit. Defaults to None.
    """
    _parse_strings: Incomplete
    _allow_int: Incomplete
    _allow_float: Incomplete
    _lower_limit: Incomplete
    _upper_limit: Incomplete
    def __init__(self, parse_number_strings: bool = True, allow_int: bool = True, allow_float: bool = True, number_lower_limit: int | float | None = None, number_upper_limit: int | float | None = None, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def parse_strings(self) -> bool:
        """Returns True if the class is configured to attempt parsing input strings as numbers."""
    def toggle_string_parsing(self) -> bool:
        """Flips the value of the attribute that determines if parsing strings into numbers is allowed and returns the
        resultant value.
        """
    @property
    def allow_int(self) -> bool:
        """Returns True if the class is configured to convert inputs into Python integers."""
    def toggle_integer_outputs(self) -> bool:
        """
        Flips the value of the attribute that determines if returning integer values is allowed and returns the
        resultant value.
        """
    @property
    def allow_float(self) -> bool:
        """Returns True if the class is configured to convert inputs into Python floats."""
    def toggle_float_outputs(self) -> bool:
        """
        Flips the value of the attribute that determines if returning float values is allowed and returns the
        resultant value.
        """
    @property
    def lower_limit(self) -> int | float | None:
        """Returns the lower bound used to determine valid numbers or None, if minimum limit is not set."""
    def set_lower_limit(self, value: int | float | None) -> None:
        """Sets the lower bound used to determine valid numbers to the input value."""
    @property
    def upper_limit(self) -> int | float | None:
        """Returns the upper bound used to determine valid numbers or None, if minimum limit is not set."""
    def set_upper_limit(self, value: int | float | None) -> None:
        """Sets the upper bound used to determine valid numbers to the input value."""
    def validate_value(self, value: bool | str | int | float | None) -> float | int | None:
        """
        Validates and converts the input value into Python float or integer type, based on the configuration.

        Notes:
            If both integer and float outputs are allowed, the class will always prioritize floats over integers.
            This is because all integers can be converted to floats without data loss, but not all floats can be
            converted to integers without losing data.

            Boolean inputs are automatically parsed as integers, as they are derivatives from the base integer class.

            Since this class is intended to be used together with other validator classes, when conversion fails for
            any reason, it returns None instead of raising an error. This allows sequentially using multiple 'Model'
            classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted number, either as a float or integer, if conversion succeeds. None, if
            conversion fails for any reason.
        """

class BoolConverter:
    """
    A factory-like class for validating and converting boolean values based on a predefined configuration.

    This class can be configured once and then used to validate and, if needed, flexibly between bool and bool equivalents.
    Specific configuration parameters can be altered through setter methods to dynamically adjust the behavior of the
    instantiated class.

    Args:
        parse_bool_equivalents: Determines whether to attempt parsing boolean equivalents other than True or False as
            boolean values. Defaults to True.

    Attributes:
        _parse_bool_equivalents: Use this flag to enable converting supported boolean-equivalent strings to boolean
            datatype and verifying such strings as boolean values.
        _true_equivalents: Internal use only. This set specifies boolean True string and integer equivalents. If
            boolean-equivalent parsing is allowed, these values will be converted to and recognized as valid boolean
            True values.
        _false_equivalents: Internal use only. Same as true_equivalents, but for boolean False equivalents.
    """
    _true_equivalents: set[str | int | float]
    _false_equivalents: set[str | int | float]
    _parse_bool_equivalents: Incomplete
    def __init__(self, parse_bool_equivalents: bool = True, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def parse_bool_equivalents(self) -> bool:
        """
        Returns True if the class is configured to attempt parsing boolean equivalents other than True or False as boolean
        values.
        """
    def toggle_bool_equivalents(self) -> bool:
        """
        Flips the value of the attribute that determines if parsing boolean equivalents is allowed and returns the
        resultant value.
        """
    def validate_value(self, value: bool | str | int | float | None) -> bool | None:
        """
        Validates and converts the input value into Python boolean type, based on the configuration.

        Notes:
            If parsing boolean equivalents is allowed, the class will attempt to convert any input that matches the
            predefined equivalents to a boolean value.

            Since this class is intended to be used together with other validator classes, when conversion fails for
            any reason, it returns None instead of raising an error. This allows sequentially using multiple 'Model'
            classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted boolean value, if conversion succeeds. None, if conversion fails for any reason.
        """

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
    _none_equivalents: set[str]
    _parse_none_equivalents: Incomplete
    def __init__(self, parse_none_equivalents: bool = True, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def parse_none_equivalents(self) -> bool:
        """Returns True if the class is configured to attempt parsing None equivalents other than None as None values."""
    def toggle_none_equivalents(self) -> bool:
        """
        Flips the value of the attribute that determines if parsing None equivalents is allowed and returns the
        resultant value.
        """
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
    _allow_string_conversion: Incomplete
    _string_options: Incomplete
    _string_force_lower: Incomplete
    def __init__(self, allow_string_conversion: bool = False, string_options: list[str] | tuple[str] | None = None, string_force_lower: bool = False, **kwargs: Any) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def allow_string_conversion(self) -> bool:
        """Returns True if the class is configured to allow converting non-string inputs to strings."""
    def toggle_string_conversion(self) -> bool:
        """
        Flips the value of the attribute that determines if converting non-string inputs to strings is allowed and returns
        the resultant value.
        """
    @property
    def string_options(self) -> list[str] | tuple[str] | None:
        """
        Returns the list of string-options that are considered valid string values.
        """
    def set_string_options(self, value: list[str] | tuple[str] | None) -> None:
        """
        Sets the list of string-options that are considered valid string values to the input value.
        """
    @property
    def string_force_lower(self) -> bool:
        """
        Returns True if the class is configured to force validated string values to be converted to lower-case.
        """
    def toggle_string_force_lower(self) -> bool:
        """
        Flips the value of the attribute that determines if forcing validated string values to be converted to lower-case is
        allowed and returns the resultant value.
        """
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
