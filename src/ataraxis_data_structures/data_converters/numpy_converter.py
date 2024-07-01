from typing import Any, Literal, Optional
from pydantic import validate_call
from python_models import BoolConverter, NoneConverter, NumericConverter, StringConverter


class PythonDataConverter:
    """
    After inital configuration, allows to conditionally validate and convert input values to a specified output type.

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
        allow_bool: Use this flag to allow the validated value to be a boolean or a boolean-equivalent string. Only
            allows boolean-equivalents if parse_bool_equivalents flag is True. Note, if enabled, string 'True'/'False'
            are always parsed as boolean values, but string and integer '1'/'0' values will be preferentially converted
            to integers or floats if allow_int / allow_float and parse_number_strings are also enabled
            (due to datatype preference order).
        allow_string: Use this flag to allow the validated value to be a string. By itself, the flag unconditionally
            validates all string inputs, but string_options (see below) attribute can be used to limit the allowed
            strings to a set of options.
        allow_int: Use this flag to allow the validated value to be an integer or an integer-convertible string or
            float. Integer-convertible strings are only allowed if parse_number_strings flag is True. When enabled
            together with allow_float, the algorithm always tries to convert floats into integers where possible.
            The range of allowed values can be constricted with number_lower_limit and number_upper_limit attributes.
        allow_float: Use this flag to allow the validated value to be a float or a float-convertible string or integer.
            Float-convertible strings are only allowed if parse_number_strings flag is True. When enabled together
            with allow_int, the algorithm always tries to convert floats into integers where possible. The range of
            allowed values can be constricted with number_lower_limit and number_upper_limit attributes.
        allow_none: Use this flag to allow the validated value to be a None (NoteType) or a None-equivalent string.
            Only allows none-equivalents if parse_none_equivalents flag is True.
        allow_iterables: Use this flag to allow the validated value to be a shallow tuple, list or set. Only supports
            shallow iterables as the algorithm calls the verification function onto each value in the iterable,
            which will fail for any value that is not a string, float, integer, boolean or NoneType.
        allow_string_conversion: 
        string_options: Optional. A tuple, list or set of string-options. If provided, all validated strings will be
            checked against the input iterable and only considered valid if the string matches one of the options.
            Set to None to disable string option-limiting. Defaults to None.
        string_force_lower: Use this flag to force validated string values to be converted to lower-case.
            Only used if allow_string is True and only applies to strings. Defaults to False.
        number_lower_limit: Optional. An integer or float that specifies the lower limit for numeric value
            verification. Verified integers and floats that are smaller than the limit number will be considered
            invalid. Set to None to disable lower-limit. Defaults to None.
        number_upper_limit: Optional. An integer or float that specifies the upper limit for numeric value
            verification. Verified integers and floats that are larger than the limit number will be considered invalid.
            Set to None to disable upper-limit. Defaults to None.
        parse_bool_equivalents: Use this flag to enable converting supported boolean-equivalent strings to boolean
            datatype and verifying such strings as boolean values.
        parse_none_equivalents: Use this flag to enable converting supported none-equivalent strings to NoneType (None)
            datatype and verifying such strings as None values.
        parse_number_strings: Use this flag to enable converting all integer- and float-equivalent strings to the
            appropriate numeric datatype and verifying such strings as integers and/or floats.
        iterable_output_type: Optional. A string-option that allows to convert input iterable values to a particular
            iterable type prior to returning them. Only used when input values are iterables. Valid options
            are 'set', 'tuple' and 'list'. Alternatively, set to None to force the algorithm to use the same iterable
            type for output value as for the input value. Defaults to None.
        supported_iterables: Internal use only. This dictionary maps supported iterable classes to their string-names.
            The dictionary is used internally to support output iterable type setting and for checks and error messages.
    
    Raises:
        ValueError: If the input string_options argument is not a tuple, list or None.
            Also, if the input string_options argument is an empty tuple or  list.
            If the input iterable_output_type argument is not one of the supported iterable output types.
    
    Methods:
        convert_value: The master function of the class. Sets-up the validation and conversion procedure for all input
            value types (iterables and non-iterables) and returns the converted value to caller. This is the only method
            that should be called externally, the rest of the clas methods are designed for internal class use only.
        validate_value: The central validation function that calls the rest of the class validation functions to
            determine whether the input value can be parsed as any of the supported (and allowed) datatypes. Also
            contains the logic that select the most preferred datatype to convert the value to if it can represent
            multiple allowed datatypes.
    """
    def __init__(
        self,
        allow_bool: bool,
        allow_string: bool,
        allow_int: bool,
        allow_float: bool,
        allow_none: bool,
        allow_iterables: bool,
        allow_string_conversion: bool = False,
        string_options: Optional[list[str] | tuple[str]] = None,
        string_force_lower: bool = False,
        number_lower_limit: Optional[int | float] = None,
        number_upper_limit: Optional[int | float] = None,
        parse_bool_equivalents: bool = False,
        parse_none_equivalents: bool = True,
        parse_number_strings: bool = True,
        iterable_output_type: Optional[Literal["tuple", "list", "tuple"]] = None,
    ) -> None:
        self.supported_iterables = {"tuple": tuple, "list": list}

        # Checks some inputs, mostly for cases that can lead to implicit failures (incorrect class behavior, due to
        # logically well-formed, but semantically incorrect input).
        # If string-option-limiting is enabled checks the input for validity
        if string_options is not None:
            # If the options list is empty, this is treated as an unintentional equivalent of setting the variable to
            # None and an exception is raised.
            if len(string_options) < 1:
                custom_error_message = (
                    f"An empty string-option iterable detected when initializing ValueConverter class instance. "
                    f"Provide a tuple, list or set of string-datatype options or set string_options argument to None to"
                    f"disable limiting string-options."
                )
                raise ValueError(custom_error_message)
            
            # If any of the options is not a string, this will implicitly fail the validation of the string, which is
            # considered undesirable. Raises an exception to ask for the options to be strings
            for option in string_options:
                if not isinstance(option, str):
                    custom_error_message = (
                        f"Unsupported string-option {option} detected when initializing ValueConverter class instance. "
                        f"Provide a tuple, list or set of string-datatype options or set string_options argument to "
                        f"None to disable limiting string-options."
                    )
                    raise ValueError(custom_error_message)

        # Similarly, checks iterable_output_type for validity
        if iterable_output_type is not None and iterable_output_type not in self.supported_iterables.keys():
            custom_error_message = (
                f"Unsupported output iterable string-option {iterable_output_type} requested when initializing "
                f"ValueConverter class instance. Select one fo the supported options: "
                f"{self.supported_iterables.keys()}."
            )
            raise ValueError(custom_error_message)
        
        # Sets conversion / validation attributes
        self.allow_bool = allow_bool
        self.allow_string = allow_string
        self.allow_int = allow_int
        self.allow_float = allow_float
        self.allow_none = allow_none
        self.allow_iterables = allow_iterables
        self.allow_string_conversion = allow_string_conversion
        self.string_options = string_options
        self.string_force_lower = string_force_lower
        self.number_lower_limit = number_lower_limit
        self.number_upper_limit = number_upper_limit
        self.parse_bool_equivalents = parse_bool_equivalents
        self.parse_none_equivalents = parse_none_equivalents
        self.parse_number_strings = parse_number_strings
        self.iterable_output_type = iterable_output_type

    @validate_call
    def convert_value(
        self, value: str | int | float | bool | None | list | tuple
    ) -> str | int | float | bool | None | list | tuple:
        """
        Validates the input value and, if needed, converts it to the correct datatype before returning to caller.

        This is the main function of the class that calls other class functions as necessary. It handles setting-up the
        validation procedure for both single variable inputs and iterables (iterates over iterables and calls validation
        subroutines on each iterable member value) and multiprocessing-related error wrapping.

        For each single value that passes validation, returns the value converted to the most preferred valid datatype
        in the order:  int > float > bool > none > string. For iterables that pass validation, first processes each
        value in the iterable using the procedure outlined above. Then, either converts the iterable to the preferred
        output type specified by iterable_output_type class attribute or to the same type as the input iterable if no
        preference is provided (class attribute is set to None).

        Args:
            value: The value to be validated. Currently, accepts strings, integers, floats, booleans and NoneTypes under
                all circumstances and shallow lists, tuples and sets if class allow_iterables attribute is True.
           
        Returns:
            The verified value or an iterable of verified values for iterable inputs converted to the most preferred
            valid datatype. For example, a string input value '14.15' will be returned as a float, if floats are
            allowed, as floats are preferred over strings.

        Raises:
            TypeError: If the type of the value to be verified is not one of the supported types, including cases when
                it is a list, tuple or set when allow_iterables class attributes is set to False.
            ValueError: If validation fails for the input value or any value in the input iterable.
            RuntimeError: If the input value is a set, the class is configured to return a set as an output and the
                number of items in the output set does not match the number of items in the input set.
        """
        in_type = type(value).__name__  # Records the incoming value type. One used fr iterable processing

        # Unconditionally allows standard single python types to pass to the verification step. Also
        # allows most standard python iterables, if class attribute allow_iterables is True.
        if not isinstance(value, (str, int, float, bool, type(None))) and not (
            isinstance(value, (tuple, list, set)) and self.allow_iterables
        ):  
            # Produces slightly different error messages for the case where iterables are allowed and when they are
            # not, but generates an error message whenever the input type is not an acceptable type given class
            # configuration (iterables allowed or not)
            if not self.allow_iterables:
                custom_error_message = (
                    f"Expected a single integer, boolean, float, string or None input when converting value, but "
                    f"encountered a value of '{type(value).__name__}' type instead. To enable processing a "
                    f"tuple, list or set inputs, set allow_list class attribute to True."
                )
            else:
                custom_error_message = (
                    f"Expected a single integer, boolean, float, string or None input or a tuple, list, "
                    f"or set iterable of such inputs when converting value, but encountered a value of "
                    f"'{type(value).__name__}' type instead."
                )
            raise TypeError(custom_error_message)

        # Converts all inputs to iterable (tuple) format to reuse the same processing code for all supported
        # input types (also for efficiency)
        value = tuple(PythonDataConverter.ensure_list(value))

        # Uses list comprehension to run the validation function on each value in the tuple generated above
        try:
            out_value = [self.validate_value(val) for val in value]

        except ValueError as e:
            # If validation fails, augments the message with a slightly different head-string to indicate if the
            # failed value was a single-value input or part of an iterable
            if in_type in self.supported_iterables.keys():
                custom_error_message = (
                    f"Unable to validate at least one value in the input iterable {value} when converting value."
                )
            else:
                custom_error_message = f"Unable to validate the input value when converting value."
            raise TypeError(custom_error_message)

        # Has a slightly different return procedure for iterable and non-iterable input type
        if in_type in self.supported_iterables.keys():
            # If the class is configured to return specific iterable types for iterables, converts the output
                # list to the requested datatype
            if self.iterable_output_type is not None:
                value_to_return = self.supported_iterables[self.iterable_output_type](out_value)
            # Otherwise, converts the output list to the same datatype as the input iterable
            else:
                value_to_return = self.supported_iterables[in_type](out_value)
            return value_to_return
        else:
            return out_value.pop(0)


    def validate_value(self, value_to_validate: Any) -> int | float | bool | None | str:
        """
        Checks the input value against all value types supported by the validator class.

        Evaluates the input value against class validation conditions to determine whether it conforms to the
        range of datatypes and values supported by the validator class. If validation fails, raises a ValueError
        exception. Otherwise, returns the most preferred successfully validated value type (see below).

        When the validator class is configured to allow multiple output value types, it uses the following order of
        preference when returning the variable: int > float > bool > none > string. This sequence is generally optimal
        for most intended use cases, but can be implicitly overriden with various class attributes modifying each
        verification step (for example, it is possible to disable numeric strings from being recognized as integers and
        floats, ensuring strings are returned as strings even if they are number-convertible).

        Note, this function is intended to be called by the convert_value() function and, as such, it does not contain
        multiprocessing error handling capabilities. It expects the caller function to handle the necessary error
        message wrapping to support error messaging during multiprocessing execution.

        Args:
            value_to_validate: The value to be validated.

        Returns:
            The value converted to the most preferred datatype that passed validation.

        Raises:
            ValueError: If the value cannot be validated as any of the supported datatypes.
        """
        # If allowed, parses the variable as a number
        if self.allow_int | self.allow_float:
            num_var = NumericConverter(
                value=value_to_validate,
                allow_int=self.allow_int,
                allow_float=self.allow_float,
                parse_number_strings=self.parse_number_strings,
                number_lower_limit=self.number_lower_limit,
                number_upper_limit=self.number_upper_limit,
            )
        else:
            num_var = None

        # If allowed, parses the variable as a boolean
        if self.allow_bool:
            bool_var = BoolConverter(
                value=value_to_validate,
                parse_bool_equivalents=self.parse_bool_equivalents,
            )
        else:
            bool_var = None

        # If allowed, parses the variable as a NoneType
        if self.allow_none:
            none_var = NoneConverter(
                value=value_to_validate,
                parse_none_equivalents=self.parse_none_equivalents,
            )
        else:
            none_var = 'None'

        # If allowed, parses the variable as a string
        if self.allow_string:
            string_model = StringConverter(
                value=value_to_validate,
                allow_string_conversion=self.allow_string_conversion,
                string_options=self.string_options,
                string_force_lower=self.string_force_lower,
            )
        else:
            string_var = None

        # If at least one validation step returns a passing value, selects the first valid value in the order
        # num (int > float) > bool > none > string and returns it to caller
        # The primary reason for numeric processing being first is that
        # booleans can be represented with integer 0 and 1, so processing numbers first ensures that integers are not
        # converted to booleans when both booleans and integers are valid value types.
        if not isinstance(num_var, type(None)):
            return num_var
        elif not isinstance(bool_var, type(None)):
            return bool_var
        elif not none_var == 'None':
            return none_var
        elif not isinstance(string_var, type(None)):
            return string_var
        # Otherwise, raises an exception to communicate the failed value, type and validator parameters that jointly
        # led to validation failure
        else:
            custom_error_message = (
                f"Failed to validate the value '{value_to_validate}' of type '{type(value_to_validate).__name__}'. "
                f"Verification parameters: allow_int={self.allow_int}, allow_float={self.allow_float}, "
                f"allow_bool={self.allow_bool}, allow_none={self.allow_none}, allow_string={self.allow_string}, "
                f"allow_string_conversion={self.allow_string_conversion}, string_options={self.string_options}, "
                f"number_lower_limit={self.number_lower_limit}, number_upper_limit={self.number_upper_limit}, "
                f"parse-bool_equivalents={self.parse_bool_equivalents}, "
                f"parse_none_equivalents={self.parse_none_equivalents}, "
                f"parse_number_strings={self.parse_number_strings}."
            )
            raise ValueError(custom_error_message)

    @validate_call
    @staticmethod
    def ensure_list(input_item: str | int | float | tuple | list) -> list:
        """Checks whether input item is a python list and, if not, converts it to list.

        If the item is a list, returns the item unchanged.

        Args:
            input_item: The variable to be made into / preserved as a python list.

        Returns:
            A python list that contains all items inside the input_item variable.

        Raises:
            TypeError: If the input item is not of a supported type.
            Exception: If an unexpected error is encountered.

        """
        try:
            if isinstance(input_item, list):
                return input_item
            elif isinstance(input_item, (tuple)):
                return list(input_item)
            elif isinstance(input_item, (str, int, float, bool, type(None))):
                return [input_item]
            else:
                raise TypeError(
                    f"Unable to convert input item to a python list, as items of type {type(input_item).__name__} are not "
                    f"supported."
                )
        except Exception as e:
            raise TypeError(f"Unable to convert the input item {input_item} to a python list.")
