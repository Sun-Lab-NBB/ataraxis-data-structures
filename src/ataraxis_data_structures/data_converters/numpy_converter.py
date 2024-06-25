from typing import Literal, Optional


class NumpyDataConverter:
    """
    """

    def __init__(
        self,
        allow_bool: bool,
        allow_string: bool,
        allow_int8: bool,
        allow_int16: bool,
        allow_int32: bool,
        allow_int64: bool,
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

        if string_options is not None:
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

        self.allow_bool = allow_bool
        self.allow_string = allow_string
        self.allow_int8 = allow_int8
        self.allow_int16 = allow_int16
        self.allow_int32 = allow_int32
        self.allow_int64 = allow_int64
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

        self.true_equivalents = {"True", "true", 1, "1", 1.0}
        self.false_equivalents = {"False", "false", 0, "0", 0.0}
        self.none_equivalents = {"None", "none", "Null", "null"}

    def convert_value(
            self, value: str | int | float | bool | None | list | tuple
    ) -> str | int | float | bool | None | list | tuple:
        handled = False
        try:
            in_type = type(value).__name__  # Records the incoming value type. Only used for iterable processing

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
                handled = True
                raise TypeError(custom_error_message)
        except Exception as e:
            if not handled:
                custom_error_message = f"Unexpected error when converting value."
                augment_exception_message(e=e, additional_message=custom_error_message)
            if not multiprocessing:
                raise
            else:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                error = AtaraxisError(
                    exception_type=exc_type,
                    exception_value=exc_value,
                    exception_traceback=traceback.format_tb(exc_traceback),
                )
                return error