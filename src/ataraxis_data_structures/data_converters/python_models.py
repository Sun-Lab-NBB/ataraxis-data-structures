from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, field_validator, ValidationError, validate_call

# TODO
"""
1) Refactor the remaining models to match the final factory-like class architecture:
    a) Replace errors with None returns to indicate validation failures. For None validator, use string 'None', since 
        None returns are actually meaningful for that class.
    b) Add properties and setter methods to allow dynamically altering class configuration post-initialization.
    c) Make all attributes protected to preserve them from accidental end-user modification.
    d) Add docstrings to class descriptions and all methods. Init arguments should be documented inside main class 
        docstring.
    e) Add **kwarg initialization support to init (very useful for testing).
2) Make sure all aspects of each class are well tested. This includes success and correct error-handling (failure)
    cases for all conceivable class configurations. This also includes pydantic-assisted init validation via 
    'validate_call' method. See tests/data_converters_python_models_test.py for examples. Use the same test architecture
    as this allows running them in-parallel.
    
* Most models had to be commented-out as importing the file for testing was running into errors. Go through them one at
a time and convert + test each until all modules are complete

-- I
"""


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

    @validate_call()
    def __init__(
            self,
            parse_number_strings: bool = True,
            allow_int: bool = True,
            allow_float: bool = True,
            number_lower_limit: Optional[Union[int, float]] = None,
            number_upper_limit: Optional[Union[int, float]] = None,
            **kwargs
    ):
        self._parse_strings = parse_number_strings
        self._allow_int = allow_int
        self._allow_float = allow_float
        self._lower_limit = number_lower_limit
        self._upper_limit = number_upper_limit

        # Sets any additional attributes from kwargs. Primarily, this functionality is necessary to support testing,
        # but may also be beneficial for certain API solutions.
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        representation_string = (
            f"NumericConverter(parse_strings={self.parse_strings}, allow_int={self.allow_int}, "
            f"allow_float={self.allow_float}, lower_limit={self.lower_limit}, upper_limit={self.upper_limit})"
        )
        return representation_string

    @property
    def parse_strings(self) -> bool:
        """Returns True if the class is configured to attempt parsing input strings as numbers."""
        return self._parse_strings

    def toggle_string_parsing(self) -> bool:
        """Flips the value of the attribute that determines if parsing strings into numbers is allowed and returns the
        resultant value.
        """
        self._parse_strings = not self.parse_strings
        return self.parse_strings

    @property
    def allow_int(self) -> bool:
        """Returns True if the class is configured to convert inputs into Python integers."""
        return self._allow_int

    def toggle_integer_outputs(self) -> bool:
        """
        Flips the value of the attribute that determines if returning integer values is allowed and returns the
        resultant value.
        """
        self._allow_int = not self.allow_int
        return self._allow_int

    @property
    def allow_float(self) -> bool:
        """Returns True if the class is configured to convert inputs into Python floats."""
        return self._allow_float

    def toggle_float_outputs(self) -> bool:
        """
        Flips the value of the attribute that determines if returning float values is allowed and returns the
        resultant value.
        """
        self._allow_float = not self.allow_float
        return self._allow_float

    @property
    def lower_limit(self) -> int | float | None:
        """Returns the lower bound used to determine valid numbers or None, if minimum limit is not set."""
        return self._lower_limit

    @validate_call()
    def set_lower_limit(self, value: int | float | None) -> None:
        """Sets the lower bound used to determine valid numbers to the input value."""
        self._lower_limit = value

    @property
    def upper_limit(self) -> int | float | None:
        """Returns the upper bound used to determine valid numbers or None, if minimum limit is not set."""
        return self._upper_limit

    @validate_call()
    def set_upper_limit(self, value: int | float | None) -> None:
        """Sets the upper bound used to determine valid numbers to the input value."""
        self._upper_limit = value

    def validate_number(self, value: Any) -> str | int | None:
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
        # Filters out any types that are definitely not integer or float convertible.
        if not isinstance(value, (int, str, bool, float)):
            return None

        # Converts strings to floats, if this is allowed.
        if isinstance(value, str) and self._parse_strings:
            try:
                value = float(value)
            except ValueError:
                return None

        # If the input values is not converted to int or float by this point, then it cannot be validated.
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
            # If the value is a float, floats are not allowed, integers are allowed and value is integer-convertible
            # without data-loss, converts it to an integer.
            if value.is_integer() and self._allow_int:
                value = int(value)
            # If the value is a float, floats are not allowed and either integers are not allowed or the value is not
            # integer-convertible without data loss, returns None.
            else:
                return None

        # Validates that the value is in the specified range, if any is provided.
        if (self._lower_limit is not None and value < self._lower_limit) or (
                self._upper_limit is not None and value > self._upper_limit):
            return None

        # Returns the validated (and, potentially, converted) value.
        return value


class BoolConverter():
    """
    A factory-like class for validating and converting boolean values based on a predefined configuration.

    This class can be configured once and then used to validate and, if needed, flexibly between bool and bool equivalents.
    Specific configuration parameters can be altered through setter methods to dynamically adjust the behavior of the
    instantiated class.

    Args:
        parse_bool_equivalents: Determines whether to attempt parsing boolean equivalents other than True or False as 
            boolean values.

    Attributes:
        _parse_bool_equivalents: Use this flag to enable converting supported boolean-equivalent strings to boolean
            datatype and verifying such strings as boolean values.
        _true_equivalents: Internal use only. This set specifies boolean True string and integer equivalents. If
            boolean-equivalent parsing is allowed, these values will be converted to and recognized as valid boolean
            True values.
        _false_equivalents: Internal use only. Same as true_equivalents, but for boolean False equivalents.
    """   
    
    _true_equivalents: set[str | int | float] = {"True", "true", 1, "1", 1.0}    
    _false_equivalents: set[str | int | float] = {"False", "false", 0, "0", 0.0}

    @validate_call()
    def __init__(
            self,
            parse_bool_equivalents: bool,
            **kwargs
    ):
        self._parse_bool_equivalents = parse_bool_equivalents

        # Sets any additional attributes from kwargs. Primarily, this functionality is necessary to support testing,
        # but may also be beneficial for certain API solutions.
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        representation_string = f"BoolConverter(parse_bool_equivalents={self.parse_bool_equivalents})"
        return representation_string
    
    @property 
    def parse_bool_equivalents(self) -> bool:
        """
        Returns True if the class is configured to attempt parsing boolean equivalents other than True or False as boolean 
        values.
        """
        return self._parse_bool_equivalents
    
    def toggle_bool_equivalents(self) -> bool:
        """
        Flips the value of the attribute that determines if parsing boolean equivalents is allowed and returns the
        resultant value.
        """
        self._parse_bool_equivalents = not self.parse_bool_equivalents
        return self.parse_bool_equivalents

    def validate_bool(self, value: Any) -> bool | None:
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
        # If the input is a boolean type returns it back to caller unchanged
        if isinstance(value, bool):
            return value

        # Otherwise, if the value is a boolean-equivalent string or number and parsing boolean-equivalents is allowed
        # converts it to boolean True or False and returns it to caller
        if self.parse_bool_equivalents:
            raise ValueError('FFFFFFFFFFFFFFFFFFFFFFFFFFF')

        if self.parse_bool_equivalents and isinstance(value, (str, int, float)):
            # If the value is in the list of true equivalents, returns True.
            if value in self._true_equivalents:
                return True
            # If the value is in the list of false equivalents, returns False.
            elif value in self._false_equivalents:
                return False
        # If the value is not in the list of true or false equivalents, returns None.
        return None


class NoneConverter(BaseModel):
    """
    A factory-like class for validating and converting None values based on a predefined configuration.

    This class can be configured once and then used to validate and, if needed, flexibly between None and None equivalents.
    Specific configuration parameters can be altered through setter methods to dynamically adjust the behavior of the
    instantiated class.

    Args:
        parse_none_equivalents: Determines whether to attempt parsing None equivalents other than None as None values.

    Attributes:
        _parse_none_equivalents: Use this flag to enable converting supported none-equivalent strings to NoneType (None)
            datatype and verifying such strings as None values.
        _none_equivalents: Internal use only. This set specifies None string equivalents. If non-equivalent parsing is
            allowed, these values will be converted to and recognized as None
    """

    _none_equivalents: dict[str] = {"None", "none", "Null", "null"}

    @validate_call()
    def __init__(
        self, 
        parse_none_equivalents: bool,
        **kwargs
    ):
        self._parse_none_equivalents = parse_none_equivalents

        # Sets any additional attributes from kwargs. Primarily, this functionality is necessary to support testing,
        # but may also be beneficial for certain API solutions.
        for key, value in kwargs.items():
            setattr(self, f"_{key}", value)

    def __repr__(self) -> str:
        representation_string = f"NoneConverter(parse_none_equivalents={self.parse_none_equivalents})"
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

    def validate_none(self, value: Any) -> None | str:
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
            value = None
        # If the value is not in the list of None equivalents, returns the string 'None'.
        else:
            return 'None'


class StringConverter(BaseModel):
    """
    A factory-like class for validating and converting string values based on a predefined configuration.

    This class can be configured once and then used to validate and, if needed, flexibly to strings. Specific configuration
    parameters can be altered through setter methods to dynamically adjust the behavior of the instantiated class.

    Args:
        allow_string_conversion: Determines whether to allow converting non-string inputs to strings.
        string_options: Optional. A list of strings that are considered valid string values.
        string_force_lower: Determines whether to force all string values to lowercase.

    Attributes:
        allow_string_conversion: Use this flag to enable converting non-string inputs to strings. Since all supported
            input values can be converted to strings, this is a dangerous option that has the potential of overriding
            all verification parameters. It is generally advised to not enable this flag for most use cases.
            Defaults to False.
        string_options: Optional. A tuple or list of string-options. If provided, all validated strings will be
            checked against the input iterable and only considered valid if the string matches one of the options.
            Set to None to disable string option-limiting. Defaults to None.
        string_force_lower: Use this flag to force validated string values to be converted to lower-case.
            Only used if allow_string is True and only applies to strings. Defaults to False.
    """

    @validate_call()
    def __init__(
            self,
            allow_string_conversion: bool,
            string_options: Optional[Union[list[str], tuple[str]]] = None,
            string_force_lower: bool = False,
            **kwargs
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
    
    @validate_call()
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

    def validate_string(self, value: Any) -> str | None:
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
            try:
                value = str(value)
            except Exception:
                return None

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
