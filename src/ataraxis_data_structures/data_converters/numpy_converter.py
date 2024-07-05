from typing import Any, Literal, Optional
import numpy as np
from .python_models import BoolConverter, NoneConverter, NumericConverter, StringConverter


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
        validator: The validator class to be used for value validation. Must be one of the supported validator classes
            (BoolConverter, NoneConverter, NumericConverter, StringConverter).
        iterable_output_type: Optional. A string-option that allows to convert input iterable values to a particular
            iterable type prior to returning them. Only used when input values are iterables. Valid options
            are 'set', 'tuple' and 'list'. Alternatively, set to None to force the algorithm to use the same iterable
            type for output value as for the input value. Defaults to None.
        filter_failed: Optional. If set to True, filters out failed values from the output iterable. Defaults to False.

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

    def validate_value(self, value_to_validate: Any) -> int | float | bool | None | str | list | tuple:
        try:
            list_value = PythonDataConverter.ensure_list(value_to_validate)
            output_iterable = []
            for value in list_value:
                value = self._validator.validate_value(value)
                if self.filter_failed:
                    if type(self.validator) == NoneConverter and value is 'None':
                        continue
                    elif value is None:
                        continue
                output_iterable.append(value)
            
            if len(output_iterable) <= 1:
                return output_iterable[0]
            
            return tuple(output_iterable) if self.iterable_output_type == 'tuple' else output_iterable
            
        except TypeError as e:
            raise TypeError(f"Unable to convert input value to a python list: {e}")


    @staticmethod
    def ensure_list(input_item: str | int | float | tuple | list | np.ndarray) -> list:
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
        if input_item is not None and not isinstance(input_item, (str, int, float, bool, tuple, list, np.ndarray)):
            raise TypeError(
                f"Unsupported input item type {type(input_item).__name__} provided to ensure_list function. "
                f"Supported types are: str, int, float, bool, tuple, list, np.ndarray."
            )

        try:
            if isinstance(input_item, list):
                return input_item
            elif isinstance(input_item, (tuple, set, np.ndarray)):
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


class NumpyDataConverter(PythonDataConverter):
    """
    Extends the PythonDataConverter class to allow for conversion of input values to numpy datatypes.

    The class extends the PythonDataConverter class to allow for conversion of input values to numpy datatypes. The
    class supports all numpy datatypes, including numpy arrays and numpy scalars. The class is designed
    """
    def __init__(
            self, 
            python_converter: PythonDataConverter, 
            numpy_output_type: (
                np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 |np.uint64 | 
                np.float8 | np.float16 | np.float32 | np.float64 | np.float128 | 
                np.bool | np.str | np.nan | np.ndarray
            ),
            output_bit_width: Literal[8, 16, 32, 64, 'auto'],
            python_output_type: int | float | bool | None | str | list | tuple
    ):
        if not isinstance(python_converter, PythonDataConverter):
            raise TypeError(
                f"Unsupported python_converter class {type(python_converter).__name__} provided when initializing "
                f"NumpyDataConverter class instance. Must be an instance of PythonDataConverter."
            )
        if not isinstance(numpy_output_type, (np.int, np.uint, np.float, np.bool, np.str, np.nan, np.ndarray)):
            raise TypeError(
                f"Unsupported output_data_type {numpy_output_type} provided when initializing NumpyDataConverter "
                f"class instance. Must be a numpy datatype."
            )
        if output_bit_width is not None and output_bit_width not in [8, 16, 32, 64, 'auto']:
            raise ValueError(
                f"Unsupported output_bit_width {output_bit_width} provided when initializing NumpyDataConverter "
                f"class instance. Must be one of the supported options: 8, 16, 32, 64, 'auto'."
            )
        if not isinstance(python_output_type, (int, float, bool, type(None), str, list, tuple)):
            raise TypeError(
                f"Unsupported python_output_type {type(python_output_type).__name__} provided when initializing "
                f"NumpyDataConverter class instance. Must be one of the supported types: int, float, bool, None, str, "
                f"list, tuple."
            )

        self._python_converter = python_converter
        self._output_data_type = numpy_output_type
        self._output_bit_width = output_bit_width
        self._python_output_type = python_output_type

    @property
    def python_converter(self) -> PythonDataConverter:
        return self._python_converter
    
    @property
    def output_data_type(self) -> (
        np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 |np.uint64 | 
        np.float8 | np.float16 | np.float32 | np.float64 | np.float128 | 
        np.bool | np.str | np.nan | np.ndarray
    ):
        return self._output_data_type
    
    @property
    def output_bit_width(self) -> Literal[8, 16, 32, 64, 'auto']:
        return self._output_bit_width
    
    @property
    def python_output_type(self) -> int | float | bool | None | str | list | tuple:
        return self._python_output_type
    
    def convert_value():
        pass

    def python_to_numpy_converter():
        pass

    def numpy_to_python_converter():
        pass
