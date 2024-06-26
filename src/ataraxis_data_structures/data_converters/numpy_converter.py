from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class NumpyDataConverter(BaseModel):
    allow_bool: bool
    allow_string: bool
    allow_int: bool
    allow_float: bool
    allow_none: bool
    allow_iterable: bool
    allow_string_conversion: bool = False
    string_options: Optional[list[str] | tuple[str] | set[str]] = Field(default=None, min_length=1)
    string_force_lower: bool = False
    number_lower_limit: Optional[int | float] = None
    number_upper_limit: Optional[int | float] = None
    parse_bool_equivalents: bool = False
    parse_none_equivalents: bool = True
    parse_number_strings: bool = True
    iterable_output_type: Optional[Literal["tuple", "list"]] = None

    _supported_iterables: dict[str, type] = {"tuple": tuple, "list": list}
    _true_equivalents: set[str | int | float] = {"True", "true", 1, "1", 1.0}
    _false_equivalents: set[str | int | float] = {"False", "false", 0, "0", 0.0}
    _none_equivalents: set[str] = {"None", "none", "Null", "null"}

    @field_validator("string_options")
    @classmethod
    def check_string_options(cls, v: Optional[list[str] | tuple[str] | set[str]]) -> None:
        if v is not None:
            if len(v) < 1:
                custom_error_message = (
                    f"An empty string-option iterable detected when initializing ValueConverter class instance. "
                    f"Provide a tuple, list or set of string-datatype options or set string_options argument to None to"
                    f"disable limiting string-options."
                )
                raise ValueError(custom_error_message)

            for option in v:
                if not isinstance(option, str):
                    custom_error_message = (
                        f"Unsupported string-option {option} detected when initializing ValueConverter class instance. "
                        f"Provide a tuple, list or set of string-datatype options or set string_options argument to "
                        f"None to disable limiting string-options."
                    )
                    raise ValueError(custom_error_message)

    @field_validator("iterable_output_type")
    @classmethod
    def check_iterable_output_type(cls, v: Optional[Literal["tuple", "list"]]) -> None:
        if v is not None and v not in cls._supported_iterables.keys():
            custom_error_message = (
                f"Unsupported output iterable string-option {cls.iterable_output_type} requested when initializing "
                f"ValueConverter class instance. Select one fo the supported options: "
                f"{cls.supported_iterables.keys()}."
            )
            raise ValueError(custom_error_message)
