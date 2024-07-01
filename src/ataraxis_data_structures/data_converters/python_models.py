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
        _parse_strings: Determines whether to attempt parsing strings as numbers.
        _allow_int: Determines whether to allow returning integer values.
        _allow_float: Determines whether to allow returning float values.
        _lower_limit: Optional. Lower bound for the returned value, if any.
        _upper_limit: Optional. Upper bound for the returned value, if any.
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
        """Flips the value of the attribute that determines if returning integer values is allowed and returns the
        resultant value.
        """
        self._allow_int = not self.allow_int
        return self._allow_int

    @property
    def allow_float(self) -> bool:
        """Returns True if the class is configured to convert inputs into Python floats."""
        return self._allow_float

    def toggle_float_outputs(self) -> bool:
        """Flips the value of the attribute that determines if returning float values is allowed and returns the
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
        """Validates and converts the input value into Python float or integer type, based on the configuration.

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

# class BoolModel(BaseModel):
#     value: bool
#     parse_bool_equivalents: bool
#
#     true_equivalents: dict[str | int | float] = {"True", "true", 1, "1", 1.0}
#     false_equivalents: dict[str | int | float] = {"False", "false", 0, "0", 0.0}
#
#     @field_validator("value")
#     @classmethod
#     def validate_bool_type(cls, value: Any) -> bool:
#         if cls.parse_bool_equivalents and isinstance(value, (str, int)):
#             if value in cls.true_equivalents:
#                 value = True
#             elif value in cls.false_equivalents:
#                 return False
#             else:
#                 custom_error_message = f"Unable to parse value {value} as a boolean."
#                 raise ValueError(custom_error_message)
#
#     def get_bool(self) -> bool:
#         return self.value
#
#
# class NoneModel(BaseModel):
#     value: None
#     parse_none_equivalents: bool
#
#     none_equivalents: dict[str] = {"None", "none", "Null", "null"}
#
#     @field_validator("value")
#     @classmethod
#     def validate_none_type(cls, value: Any) -> None:
#         if value in cls.none_equivalents and cls.parse_none_equivalents:
#             value = None
#         else:
#             custom_error_message = f"Unable to parse value {value} as None."
#             raise ValueError(custom_error_message)
#
#     def get_none(self) -> None:
#         return self.value
#
#
# class StringModel(BaseModel):
#     value: str
#     allow_string_conversion: bool
#
#     string_options: Optional[list[str] | tuple[str]] = None
#     string_force_lower: bool = False
#
#     @field_validator("value")
#     @classmethod
#     def validate_string_type(cls, value: Any) -> str:
#         if not cls.allow_string_conversion:
#             custom_error_message = f"Unable to parse value {value} as a string."
#             raise ValueError(custom_error_message)
#
#         value_lower = value.lower() if cls.string_force_lower or cls.string_options else value
#
#         if cls.string_options:
#             option_list_lower = [option.lower() for option in cls.string_options]
#
#             if value_lower in option_list_lower:
#                 if cls.string_force_lower:
#                     cls.value = value_lower
#                 else:
#                     cls.value = value
#             else:
#                 custom_error_message = f"Unable to parse value {value} as a string."
#                 raise ValueError(custom_error_message)
#
#     def get_string(self) -> str:
#         return self.value
