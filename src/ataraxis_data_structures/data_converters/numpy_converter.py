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
        string_options: Optional[list[str] | tuple[str] | set[str]] = None,
        string_force_lower: bool = False,
        number_lower_limit: Optional[int | float] = None,
        number_upper_limit: Optional[int | float] = None,
        parse_bool_equivalents: bool = False,
        parse_none_equivalents: bool = True,
        parse_number_strings: bool = True,
        iterable_output_type: Optional[Literal["tuple", "list", "tuple"]] = None,
    ) -> None:
        pass

