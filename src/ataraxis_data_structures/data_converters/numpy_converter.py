from typing import Any, Literal, Optional, Union
import numpy as np
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
    @validate_call
    def __init__(
        self,
        validator: BoolConverter | NoneConverter | NumericConverter | StringConverter,
        
        iterable_output_type: Optional[Literal["tuple", "list"]] = None,
        filter_failed: bool = False,
    ) -> None:
        self.supported_iterables = {"tuple": tuple, "list": list}

       

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



    def validate_value(self, value_to_validate: Any) -> int | float | bool | None | str | list | tuple:
        try:
            list_value = PythonDataConverter.ensure_list(value_to_validate)
            output_iterable = []
            for value in list_value:
                output_iterable.append(self._validator.validate_value(value))
            
            if len(output_iterable) <= 1:
                return output_iterable[0]
            
        except TypeError as e:
            raise TypeError(f"Unable to convert input value to a python list: {e}")

    @validate_call
    @staticmethod
    def ensure_list(input_item: str | int | float | tuple | list | np.array) -> list:
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
