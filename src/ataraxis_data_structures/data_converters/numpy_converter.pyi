import numpy as np
from .python_models import BoolConverter as BoolConverter, NoneConverter as NoneConverter, NumericConverter as NumericConverter, StringConverter as StringConverter
from _typeshed import Incomplete
from numpy.typing import NDArray
from typing import Any, Literal

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
    supported_iterables: Incomplete
    _validator: Incomplete
    _iterable_output_type: Incomplete
    _filter_failed: Incomplete
    def __init__(self, validator: BoolConverter | NoneConverter | NumericConverter | StringConverter, iterable_output_type: Literal['tuple', 'list'] | None = None, filter_failed: bool = False) -> None: ...
    @property
    def validator(self) -> BoolConverter | NoneConverter | NumericConverter | StringConverter: ...
    @property
    def iterable_output_type(self) -> Literal['tuple', 'list'] | None: ...
    @property
    def filter_failed(self) -> bool: ...
    def toggle_filter_failed(self) -> bool: ...
    def set_validator(self, new_validator: BoolConverter | NoneConverter | NumericConverter | StringConverter) -> None: ...
    def validate_value(self, value_to_validate: int | float | str | bool | None | list[Any] | tuple[Any]) -> int | float | bool | None | str | list[int | float | bool | str | None] | tuple[int | float | str | None, ...]:
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
    @staticmethod
    def ensure_list(input_item: str | int | float | bool | None | np.generic | NDArray[Any] | tuple[Any, ...] | list[Any] | set[Any]) -> list[Any]:
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
    _signed: Incomplete
    _python_converter: Incomplete
    _output_bit_width: Incomplete
    def __init__(self, python_converter: PythonDataConverter, output_bit_width: int | str = 'auto', signed: bool = True) -> None: ...
    @property
    def converter(self) -> PythonDataConverter:
        """Returns the python_converter attribute of the class instance."""
    @property
    def output_bit_width(self) -> int | str:
        """Returns the bit-width of the output numpy datatype."""
    @property
    def signed(self) -> bool:
        """Returns the signed attribute of the class instance."""
    def toggle_signed(self) -> bool:
        """Toggles the signed attribute between True and False and returns the new value."""
    def set_output_bit_width(self, new_output_bit_width: int | str) -> None: ...
    def set_python_converter(self, new_python_converter: PythonDataConverter) -> None:
        """Sets the python_converter attribute of the class instance to a new PythonDataConverter instance."""
    def python_to_numpy_converter(self, value_to_convert: int | float | bool | None | str | list[Any] | tuple[Any]) -> np.integer[Any] | np.unsignedinteger[Any] | np.bool | NDArray[Any]:
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
    def numpy_to_python_converter(self, value_to_convert: np.integer[Any] | np.unsignedinteger[Any] | np.bool | float | NDArray[Any]) -> int | float | bool | None | list[Any] | tuple[Any, ...]:
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
    def min_scalar_type_signed_or_unsigned(self, value: int) -> type:
        """
        Returns the minimum scalar type to represent `value` with the desired signedness.

        Parameters
            value: The input value to be checked.
            signed: If True, return a signed dtype. If False, return an unsigned dtype.

        Returns
            A NumPy callable type.
        """
