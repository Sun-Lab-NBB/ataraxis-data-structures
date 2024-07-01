from typing import Any, Literal, Optional

from python_models import BoolModel, NoneModel, NumericConverter, StringModel


class PythonDataConverter:
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

        if string_options is not None:
            if len(string_options) < 1:
                custom_error_message = (
                    f"An empty string-option iterable detected when initializing ValueConverter class instance. "
                    f"Provide a tuple, list or set of string-datatype options or set string_options argument to None to"
                    f"disable limiting string-options."
                )
                raise ValueError(custom_error_message)

            for option in string_options:
                if not isinstance(option, str):
                    custom_error_message = (
                        f"Unsupported string-option {option} detected when initializing ValueConverter class instance. "
                        f"Provide a tuple, list or set of string-datatype options or set string_options argument to "
                        f"None to disable limiting string-options."
                    )
                    raise ValueError(custom_error_message)

        if iterable_output_type is not None and iterable_output_type not in self.supported_iterables.keys():
            custom_error_message = (
                f"Unsupported output iterable string-option {iterable_output_type} requested when initializing "
                f"ValueConverter class instance. Select one fo the supported options: "
                f"{self.supported_iterables.keys()}."
            )
            raise ValueError(custom_error_message)

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

    def convert_value(
        self, value: str | int | float | bool | None | list | tuple
    ) -> str | int | float | bool | None | list | tuple:
        in_type = type(value).__name__

        if not isinstance(value, (str, int, float, bool, type(None))) and not (
            isinstance(value, (tuple, list, set)) and self.allow_iterables
        ):
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

        value = tuple(ensure_list(value))

        try:
            out_value = [self.validate_value(val) for val in value]

        except ValueError as e:
            if in_type in self.supported_iterables.keys():
                custom_error_message = (
                    f"Unable to validate at least one value in the input iterable {value} when converting value."
                )
            else:
                custom_error_message = f"Unable to validate the input value when converting value."
            raise TypeError(custom_error_message)

        if in_type in self.supported_iterables.keys():
            if self.iterable_output_type is not None:
                value_to_return = self.supported_iterables[self.iterable_output_type](out_value)
            else:
                value_to_return = self.supported_iterables[in_type](out_value)
            if in_type == "set" and len(value_to_return) != len(value):
                custom_error_message = (
                    f"The number of values ({len(value_to_return)}) in the output set does not match the number of "
                    f"values in the input set ({len(value)}), when converting value. This is likely due to the "
                    f"validation process producing one or more duplicated values."
                )
                raise RuntimeError(custom_error_message)
            else:
                return value_to_return
        else:
            return out_value.pop(0)

    def validate_value(self, value_to_validate: Any) -> int | float | bool | None | str:
        if self.allow_int | self.allow_float:
            try:
                num_model = NumericConverter(
                    value=value_to_validate,
                    allow_int=self.allow_int,
                    allow_float=self.allow_float,
                    parse_number_strings=self.parse_number_strings,
                    number_lower_limit=self.number_lower_limit,
                    number_upper_limit=self.number_upper_limit,
                )
                num_var = num_model.get_num()
            except ValueError:
                num_var = None
        else:
            num_var = None

        if self.allow_bool:
            try:
                bool_model = BoolModel(
                    value=value_to_validate,
                    parse_bool_equivalents=self.parse_bool_equivalents,
                )
                bool_var = bool_model.get_bool()
            except ValueError:
                bool_var = None
        else:
            bool_var = None

        if self.allow_none:
            try:
                none_model = NoneModel(
                    value=value_to_validate,
                    parse_none_equivalents=self.parse_none_equivalents,
                )
                none_var = none_model.get_none()
            except ValueError:
                none_var = False
        else:
            none_var = False

        if self.allow_string:
            try:
                string_model = StringModel(
                    value=value_to_validate,
                    allow_string_conversion=self.allow_string_conversion,
                    string_options=self.string_options,
                    string_force_lower=self.string_force_lower,
                )
                string_var = string_model.get_string()
            except ValueError:
                string_var = None
        else:
            string_var = None

        if not isinstance(num_var, type(None)):
            return num_var
        elif not isinstance(bool_var, type(None)):
            return bool_var
        elif not isinstance(none_var, type(None)):
            return none_var
        elif not isinstance(string_var, type(None)):
            return string_var
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

    @staticmethod
    def ensure_list(input_item: str | int | float | tuple | list) -> list:
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
