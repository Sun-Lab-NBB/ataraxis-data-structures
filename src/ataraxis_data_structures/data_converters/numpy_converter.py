from typing import Any, Union, Literal, Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console

from .python_models import BoolConverter, NoneConverter, StringConverter, NumericConverter


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
        _validator: The validator class to be used for value validation. Must be one of the supported validator classes
            (BoolConverter, NoneConverter, NumericConverter, StringConverter).
        _iterable_output_type: Optional. A string-option that allows to convert input iterable values to a particular
            iterable type prior to returning them. Only used when input values are iterables. Valid options
            are 'set', 'tuple' and 'list'. Alternatively, set to None to force the algorithm to use the same iterable
            type for output value as for the input value. Defaults to None.
        _filter_failed: Optional. If set to True, filters out failed values from the output iterable. Defaults to False.

    Raises:
        ValueError: If the input string_options argument is not a tuple, list or None.
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
            raise TypeError(
                f"Unsupported validator class {type(validator).__name__} provided when initializing ValueConverter "
                f"class instance. Must be one of the supported validator classes: "
                f"BoolConverter, NoneConverter, NumericConverter, StringConverter."
            )
        if not isinstance(filter_failed, bool):
            raise TypeError(
                f"Unsupported filter_failed argument {filter_failed} provided when initializing ValueConverter "
                f"class instance. Must be a boolean value."
            )

        # Similarly, checks iterable_output_type for validity
        if iterable_output_type is not None and iterable_output_type not in self.supported_iterables.keys():
            custom_error_message = (
                f"Unsupported output iterable string-option {iterable_output_type} requested when initializing "
                f"ValueConverter class instance. Select one fo the supported options: "
                f"{self.supported_iterables.keys()}."
            )
            raise ValueError(custom_error_message)

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
            raise TypeError(
                f"Unsupported validator class {type(new_validator).__name__} provided when setting ValueConverter "
                f"validator. Must be one of the supported validator classes: "
                f"BoolConverter, NoneConverter, NumericConverter, StringConverter."
            )
        self._validator = new_validator

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
            raise TypeError(f"Unable to convert input value to a python list: {e}")

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
        raise TypeError(
            f"Unable to convert input item to a Python list, as items of type {type(input_item).__name__} "
            f"are not supported."
        )


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
            raise TypeError(
                f"Unsupported python_converter class {type(python_converter).__name__} provided when initializing "
                f"NumpyDataConverter class instance. Must be an instance of PythonDataConverter."
            )
        if output_bit_width is not None and output_bit_width not in [8, 16, 32, 64, "auto"]:
            raise ValueError(
                f"Unsupported output_bit_width {output_bit_width} provided when initializing NumpyDataConverter "
                f"class instance. Must be one of the supported options: 8, 16, 32, 64, 'auto'."
            )
        if type(python_converter.validator) == StringConverter:
            raise TypeError(
                f"Unsupported validator class {type(python_converter.validator).__name__} provided when initializing "
                f"NumpyDataConverter class instance. Must be one of the supported validator classes: "
                f"BoolConverter, NoneConverter, NumericConverter."
            )
        if not python_converter.filter_failed:
            raise ValueError(
                f"Unsupported filter_failed argument {python_converter.filter_failed} provided when "
                f"initializing NumpyDataConverter class instance. Must be set to True."
            )
        if type(python_converter.validator) == NumericConverter:
            if python_converter.validator.allow_int and python_converter.validator.allow_float:
                raise ValueError(
                    f"Unsupported NumericConverter configuration provided when initializing NumpyDataConverter "
                    f"class instance. Both allow_int and allow_float cannot be set to True."
                )

        self._signed = signed
        self._python_converter = python_converter
        self._output_bit_width = output_bit_width

    @property
    def python_converter(self) -> PythonDataConverter:
        """Returns the python_converter attribute of the class instance."""
        return self._python_converter

    @property
    def converter(self) -> PythonDataConverter:
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
            raise ValueError(
                f"Unsupported output_bit_width {new_output_bit_width} provided when setting NumpyDataConverter "
                f"output_bit_width. Must be one of the supported options: 8, 16, 32, 64, 'auto'."
            )
        self._output_bit_width = new_output_bit_width

    def set_python_converter(self, new_python_converter: PythonDataConverter) -> None:
        """Sets the python_converter attribute of the class instance to a new PythonDataConverter instance."""
        if not isinstance(new_python_converter, PythonDataConverter):
            raise TypeError(
                f"Unsupported python_converter class {type(new_python_converter).__name__} provided when setting "
                f"NumpyDataConverter python_converter. Must be an instance of PythonDataConverter."
            )
        self._python_converter = new_python_converter

    def python_to_numpy_converter(
        self,
        value_to_convert: int | float | bool | None | str | list[Any] | tuple[Any],
    ) -> np.integer[Any] | np.unsignedinteger[Any] | np.bool | NDArray[Any]:
        """
        Converts input values to numpy datatypes.

        The function converts input values to numpy datatypes based on the configuration of the class instance. The
        function supports conversion of scalar values, lists and tuples. The function also supports conversion of
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
            if self._output_bit_width == "auto" and isinstance(value, (int)):
                min_dtype = self.min_scalar_type_signed_or_unsigned(value)
                numpy_value: np.integer[Any] | np.unsignedinteger[Any] | np.bool | NDArray[Any] | float = np.array(
                    value, dtype=min_dtype
                )

            elif isinstance(value, (int, float)):
                if self._output_bit_width == 8 and type(value) == float:
                    raise ValueError(f"Unable to convert input value to a numpy datatype.")

                if type(value) == float:
                    width_list: list[type] = float_sign
                else:
                    width_list = signed if self._signed else unsigned

                index_map: dict[int, int] = {8: 0, 16: 1, 32: 2, 64: 3}
                if type(self._output_bit_width) == int:
                    datatype: type = width_list[index_map[self._output_bit_width]]

                try:
                    numpy_value = datatype(value)
                except Exception as e:
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
        function supports conversion of scalar values, lists and tuples. The function also supports conversion of
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
                    out = validated.pop()
                    if isinstance(out, (int, float, bool, type(None), list, tuple)):
                        return out
                    else:
                        raise ValueError(f"Unable to convert input value to a Python datatype.")
                else:
                    return validated
            else:
                raise ValueError(f"Unable to convert input value to a Python datatype.")
        if np.isnan(value_to_convert) or np.isinf(value_to_convert):
            return None
        output = self.python_converter.validate_value(np.array(value_to_convert).item(0))
        if not isinstance(output, str):
            return output
        else:
            raise ValueError(f"Unable to convert input value to a Python datatype.")

    def min_scalar_type_signed_or_unsigned(self, value: int) -> type:
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
        if (isinstance(dtype, np.signedinteger) and self._signed) or (
            isinstance(dtype, np.unsignedinteger) and not self._signed
        ):
            # if (np.issubdtype(dtype, np.signedinteger) and self._signed) or (
            #     np.issubdtype(dtype, np.unsignedinteger) and not self._signed
            # ):
            return dtype

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

        raise OverflowError(f"Value {value} is too large to be represented by a NumPy integer type.")
