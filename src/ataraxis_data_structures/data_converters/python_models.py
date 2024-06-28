from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator


class NumModel(BaseModel):
    value: int | float
    parse_number_strings: bool
    allow_int: bool
    allow_float: bool

    number_lower_limit: Optional[int] = None
    number_upper_limit: Optional[int] = None

    @field_validator("value")
    @classmethod
    def validate_number_type(cls, value: Any) -> int | float:
        if isinstance(value, str) and cls.parse_number_strings:
            try:
                value = float(value)
            except ValueError:
                return None

        if isinstance(value, (int, float)):
            if value.is_integer() and cls.allow_int:
                cls.value = int(value)
            elif cls.allow_float:
                cls.value = float(value)
            else:
                custom_error_message = f"Unable to parse value {value} as a number."
                raise ValueError(custom_error_message)

        if (cls.number_lower_limit is not None and cls.value < cls.number_lower_limit) or (
            cls.number_upper_limit is not None and cls.value > cls.number_upper_limit
        ):
            custom_error_message = (
                f"Value {cls.value} is outside the allowed range of "
                f"[{cls.number_lower_limit}, {cls.number_upper_limit}]"
            )
            raise ValueError(custom_error_message)

    def get_num(self) -> int | float:
        return self.value


class BoolModel(BaseModel):
    value: bool
    parse_bool_equivalents: bool

    true_equivalents: dict[str | int | float] = {"True", "true", 1, "1", 1.0}
    false_equivalents: dict[str | int | float] = {"False", "false", 0, "0", 0.0}

    @field_validator("value")
    @classmethod
    def validate_bool_type(cls, value: Any) -> bool:
        if cls.parse_bool_equivalents and isinstance(value, (str, int)):
            if value in cls.true_equivalents:
                value = True
            elif value in cls.false_equivalents:
                return False
            else:
                custom_error_message = f"Unable to parse value {value} as a boolean."
                raise ValueError(custom_error_message)

    def get_bool(self) -> bool:
        return self.value


class NoneModel(BaseModel):
    value: None
    parse_none_equivalents: bool

    none_equivalents: dict[str] = {"None", "none", "Null", "null"}

    @field_validator("value")
    @classmethod
    def validate_none_type(cls, value: Any) -> None:
        if value in cls.none_equivalents and cls.parse_none_equivalents:
            value = None
        else:
            custom_error_message = f"Unable to parse value {value} as None."
            raise ValueError(custom_error_message)

    def get_none(self) -> None:
        return self.value


class StringModel(BaseModel):
    value: str
    allow_string_conversion: bool

    string_options: Optional[list[str] | tuple[str]] = None
    string_force_lower: bool = False

    @field_validator("value")
    @classmethod
    def validate_string_type(cls, value: Any) -> str:
        if not cls.allow_string_conversion:
            custom_error_message = f"Unable to parse value {value} as a string."
            raise ValueError(custom_error_message)

        value_lower = value.lower() if cls.string_force_lower or cls.string_options else value

        if cls.string_options:
            option_list_lower = [option.lower() for option in cls.string_options]

            if value_lower in option_list_lower:
                if cls.string_force_lower:
                    cls.value = value_lower
                else:
                    cls.value = value
            else:
                custom_error_message = f"Unable to parse value {value} as a string."
                raise ValueError(custom_error_message)

    def get_string(self) -> str:
        return self.value
