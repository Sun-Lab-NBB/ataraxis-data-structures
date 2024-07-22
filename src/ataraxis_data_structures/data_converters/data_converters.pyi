from typing import Any, Literal

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

class NumericConverter:
    """A factory-like class for validating and converting numeric values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    integer, float, string, and boolean inputs to integer or float outputs. After initial configuration the class
    cannot be reconfigured without re-initialization.

    Notes:
        If both integer and float outputs are allowed, the class will always prioritize floats over integers.
        This is because all integers can be converted to floats without data loss, but not all floats can be
        converted to integers without losing data (rounding).

    Args:
        number_lower_limit: Optional. Lower bound for the returned value, if any. Values below this limit will fail
            validation.
        number_upper_limit: Optional. Upper bound for the returned value, if any. Values above this limit will fail
            validation.
        parse_number_strings: Determines whether to attempt parsing input strings as numbers (integers or floats).
        allow_integer_output: Determines whether to validate, convert, and return inputs as integer values.
        allow_float_output: Determines whether to validate, convert, and return inputs as float values.

    Attributes:
        _lower_limit: Optional. An integer or float that specifies the lower limit for numeric value
            verification. Verified integers and floats that are smaller than the limit number will be considered
            invalid. Set to None to disable lower-limit.
        _upper_limit: Optional. An integer or float that specifies the upper limit for numeric value
            verification. Verified integers and floats that are larger than the limit number will be considered invalid.
            Set to None to disable upper-limit.
        _parse_strings: Determines whether to attempt validating strings as number types (with necessary conversions).
        _allow_int: Determines whether the class can validate and convert inputs into integer values.
        _allow_float: Determines whether the class can validate and convert inputs into float values.

    Raises:
        TypeError: If any of the initialization arguments are not of the expected type.
        ValueError: If the number_lower_limit is larger than or equal to the number_upper_limit, when both limits are
            not None. If both integer and float outputs are not allowed.
    """

    _parse_strings: Incomplete
    _allow_int: Incomplete
    _allow_float: Incomplete
    _lower_limit: Incomplete
    _upper_limit: Incomplete
    def __init__(
        self,
        number_lower_limit: int | float | None = None,
        number_upper_limit: int | float | None = None,
        *,
        parse_number_strings: bool = True,
        allow_integer_output: bool = True,
        allow_float_output: bool = True,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the NumericConverter instance."""
    @property
    def parse_number_strings(self) -> bool:
        """Returns True if the class is configured to parse input strings as numbers."""
    @property
    def allow_integer_output(self) -> bool:
        """Returns True if the class is configured to output (after validation and / or conversion) Python integers."""
    @property
    def allow_float_output(self) -> bool:
        """Returns True if the class is configured to output (after validation and / or conversion) Python floats."""
    @property
    def number_lower_limit(self) -> int | float | None:
        """Returns the lower bound used to determine valid numbers or None, if minimum limit is not set."""
    @property
    def number_upper_limit(self) -> int | float | None:
        """Returns the upper bound used to determine valid numbers or None, if minimum limit is not set."""
    def validate_value(self, value: bool | str | int | float | None) -> float | int | None:
        """Ensures that the input value is a valid number (integer or float), depending on class configuration.

        If the value is not a number, but is number-convertible, converts the value to the valid number type. Optionally
        carries out additional validation steps, such as checking whether the value is within the specified bounds.

        Notes:
            If both integer and float outputs are allowed, the class will always prioritize floats over integers.
            This is because all integers can be converted to floats without data loss, but not all floats can be
            converted to integers without losing data (rounding).

            Boolean inputs are automatically parsed as floats, as they are derivatives from the base integer class.

            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns None instead of raising an error. This allows sequentially using multiple
            'Converter' classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted number, either as a float or integer, if conversion succeeds. None, if
            conversion fails for any reason.
        """

class BooleanConverter:
    """A factory-like class for validating and converting boolean values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    boolean and boolean-equivalent inputs to boolean outputs. After initial configuration the class cannot be
    reconfigured without re-initialization.

    Args:
        parse_boolean_equivalents: Determines whether to attempt parsing boolean equivalents other than True or
            False as boolean values.

    Attributes:
        _parse_bool_equivalents: Determines whether to convert boolean-equivalents to boolean values.
        _true_equivalents: Specifies string and numeric values considered equivalent to boolean True values. When
            boolean-equivalent parsing is allowed, these values will be converted to and recognized as valid boolean
            True values.
        _false_equivalents: Same as true_equivalents, but for boolean False equivalents.

    Raises:
        TypeError: If the input parse_boolean_equivalents argument is not a boolean.
    """

    _true_equivalents: set[str | int | float]
    _false_equivalents: set[str | int | float]
    _parse_bool_equivalents: Incomplete
    def __init__(self, *, parse_boolean_equivalents: bool = True) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the BooleanConverter instance."""
    @property
    def parse_boolean_equivalents(self) -> bool:
        """Returns True if the class is configured to parse boolean equivalents as boolean values."""
    def validate_value(self, value: bool | str | int | float | None) -> bool | None:
        """Ensures that the input value is a valid boolean.

        If the value is not a boolean, but is boolean-equivalent, converts the value to the valid boolean type, if
        parsing boolean equivalents is allowed.

        Notes:
            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns None instead of raising an error. This allows sequentially using multiple
            'Converter' classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted boolean value, if conversion succeeds. None, if conversion fails for any reason.
        """

class NoneConverter:
    """A factory-like class for validating and converting None values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    NoneType (None) and None-equivalent inputs to None outputs. After initial configuration the class cannot be
    reconfigured without re-initialization.

    Args:
        parse_none_equivalents: Determines whether to attempt parsing None equivalents as NoneType (None) values.

    Attributes:
        _parse_none_equivalents: Determines whether to convert None-equivalent inputs to None values.
        _none_equivalents: Specifies string values considered equivalent to NoneType (None) values. When
            None-equivalent parsing is allowed, these values will be converted to and recognized as valid NoneType
            values.

    Raises:
        TypeError: If the input parse_none_equivalents argument is not a boolean.
    """

    _none_equivalents: set[str]
    _parse_none_equivalents: Incomplete
    def __init__(self, *, parse_none_equivalents: bool = True) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the NoneConverter instance."""
    @property
    def parse_none_equivalents(self) -> bool:
        """Returns True if the class is configured to parse None-equivalent inputs as None values."""
    def validate_value(self, value: Any) -> None | str:
        """Ensures that the input value is a valid NoneType (None).

        If the value is not a None, but is None-equivalent, converts the value to the valid None type, if
        parsing None equivalents is allowed.

        Notes:
            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns "None" string instead of raising an error. This allows sequentially using
            multiple \'Converter\' classes as part of a major DataConverter class to implement complex conversion
            hierarchies.

            Note the difference above! Since "None" is the desired output from this class, the error-return uses a
            string type and "None" value.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted None value, if conversion succeeds. The string "None", if conversion fails for
            any reason.
        """

class StringConverter:
    """A factory-like class for validating and converting string values based on a predefined configuration.

    After initial configuration, an instance of this class can be used to validate and, if needed, flexibly convert
    most inputs to String outputs. After initial configuration the class cannot be reconfigured without
    re-initialization.

    Notes:
        Almost any Python object can be converted to a string. Therefore, depending on configuration, this class
        can have a lot of power to almost always return valid string outputs.

        When string-options are provided, the class converts them to lower-case regardless of other parameters.
        Validated strings are also converted to lower-case before checking them against the options. This design
        intentionally makes the class case-insensitive.

    Args:
        string_options: Optional. A tuple or list of strings that are considered valid string values. Any input not
            matching the contents of this iterable will be considered invalid, even if it is string-convertible. Set to
            None to disable option-checking.
        allow_string_conversion: Determines whether to allow converting non-string inputs to strings. Defaults to False.
        string_force_lower: Determines whether to force all string values to lowercase.

    Attributes:
        _string_options: Optional. A tuple or list of string-options. If provided, all validated strings will be
            checked against the input iterable and only considered valid if the string matches one of the options.
        _string_force_lower: Determines if validated string values have to be converted to lower-case.
        _allow_string_conversion: Determines whether to convert non-string inputs to strings. Setting this to true is
            fairly dangerous, as almost anything can be converted to a string.

    Raises:
        TypeError: If any input argument is not of the correct type. This includes the elements of iterable
            string-options argument.
        ValueError: If the string_options argument is empty iterable.
    """

    _allow_string_conversion: Incomplete
    _string_force_lower: Incomplete
    _string_options: Incomplete
    def __init__(
        self,
        string_options: list[str] | tuple[str] | None = None,
        *,
        allow_string_conversion: bool = False,
        string_force_lower: bool = False,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the StringConverter instance."""
    @property
    def allow_string_conversion(self) -> bool:
        """Returns True if the class is configured to convert non-string inputs to strings."""
    @property
    def string_options(self) -> list[str] | tuple[str] | None:
        """Returns the list of string-options that are considered valid string values.

        If strings are not limited to a collection of options, returns None.
        """
    @property
    def string_force_lower(self) -> bool:
        """Returns True if the class is configured to convert validated strings to lower-case."""
    def validate_value(self, value: str | bool | int | float | None) -> str | None:
        """Ensures that the input value is a valid String.

        If the value is not a string, but is string-convertible, converts the value to the valid string type if
        string-conversion is allowed.

        Notes:
            If string option-limiting is enabled, the class will only consider the input string valid if it matches one
            of the predefined string options. Before matching the string to option, the class converts BOTH options and
            checked string to lower-case to amke it case-invariant. 'passed' values are still returned using the input
            case.

            Since this class is intended to be used together with other converter / validator classes, when conversion
            fails for any reason, it returns None instead of raising an error. This allows sequentially using multiple
            'Converter' classes as part of a major DataConverter class to implement complex conversion hierarchies.

        Args:
            value: The value to validate and potentially convert.

        Returns:
            The validated and converted string value, if conversion succeeds. None, if conversion fails for any reason.
        """

class PythonDataConverter:
    """After initial configuration, allows conditionally validating and / or converting input values to a specified
    pythonic output type.

    Broadly, this class is designed to wrap one or more \'base\' converter classes (NumericConverter, BooleanConverter,
    StringConverter, NoneConverter) and extend their value validation methods to work for iterable inputs. Combining
    multiple converters allows the class to apply them hierarchically to process a broad range of input values
    (see the Notes section below for details). This design achieves maximum conversion / validation flexibility, making
    this class generally usable for a wide range of cases.

    Notes:
        When multiple converter options are used, the class always defers to the following hierarchy:
        float > integer > boolean > None > string. This hierarchy is chosen to (roughly) prioritize outputting
        \'non-permissive\' types first. For example, an integer is always float-convertible, but not vice versa. Since
        almost every input is potentially convertible to a string, the strings are evaluated last.

        The primary application for this class is to help configuration classes (YamlConfig, for example), which store
        data on disk between runtimes and, typically, convert all data into string format. This class can be used to
        convert the strings loaded by configuration classes back into the intended format. Instances of this class can
        be written and loaded from disk, acting as a repository of correct validation / conversion parameters stored
        in non-volatile data. After loading them from disk, they can restore the rest of the data to the originally
        intended datatype.

        Additionally, this class can be used by UI and similarly interactive elements to validate user inputs in cases
        where UI libraries do not provide a desired input validation mechanism.

        The class is designed to be as input-datatype agnostic as possible. In most cases, if a precise input value
        datatype is known, it is more efficient (and easier) to implement a simple in-code conversion. This class is
        best suited for cases when the input value type can vary widely during runtime and/or consists of many possible
        options.

    Args:
        numeric_converter: Optional. The initialized NumericConverter class instance or None to disable validating and
            converting inputs to numeric types.
        boolean_converter: Optional. The initialized BooleanConverter class instance or None to disable validating and
            converting inputs to boolean types.
        none_converter: Optional. The initialized NoneConverter class instance or None to disable validating and
            converting inputs to None types.
        string_converter: Optional. The initialized StringConverter class instance or None to disable validating and
            converting inputs to string types.
        iterable_output_type: Optional. Determines the type input iterables will be cast to before they are returned.
        filter_failed_elements: Determines whether to filter individual iterable elements that fail validation. By
            default, they are returned as None / "None" strings.
        raise_errors: Determines whether to return outputs that failed validation as None or to raise ValueError
            exceptions. Enabling this option allows using this class similarly to how pydantic models are used.

    Attributes:
        _numeric_converter: Optionally stores the NumericConverter to apply to input values.
        _boolean_converter: Optionally stores the BooleanConverter to apply to input values.
        _none_converter: Optionally stores the NoneConverter to apply to input values.
        _string_converter: Optionally stores the StringConverter to apply to input values.
        _iterable_output_type: Optionally stores the type to cast input iterables to before returning them.
        _filter_failed_elements: Determines whether to fileter scalar elements that fail validation from returned
            iterables.
        _raise_errors: Determines whether to return inputs that failed validation as None or to raise ValueError when
            an input fails validation.
        _allowed_outputs: A set that stores all output types that are supported by the current class configuration.
        _supported_iterables: A dictionary that maps supported output iterable keys to callable types.

    Raises:
        TypeError: If any of the input arguments are not of the expected type.
        ValueError: If the requested iterable output type is not supported. If all converter inputs are set to None.
    """

    _supported_iterables: Incomplete
    _allowed_outputs: Incomplete
    _numeric_converter: Incomplete
    _none_converter: Incomplete
    _string_converter: Incomplete
    _boolean_converter: Incomplete
    _iterable_output_type: Incomplete
    _filter_failed_elements: Incomplete
    _raise_errors: Incomplete
    def __init__(
        self,
        numeric_converter: NumericConverter | None = None,
        boolean_converter: BooleanConverter | None = None,
        none_converter: NoneConverter | None = None,
        string_converter: StringConverter | None = None,
        iterable_output_type: Literal["tuple", "list"] | None = None,
        *,
        filter_failed_elements: bool = False,
        raise_errors: bool = False,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the PythonDataConverter class instance."""
    @property
    def numeric_converter(self) -> NumericConverter | None:
        """Returns the NumericConverter instance used by the class to validate and convert inputs into numbers (integers
        or floats).

        If the class does not support numeric conversion, returns None.
        """
    @property
    def boolean_converter(self) -> BooleanConverter | None:
        """Returns the BooleanConverter instance used by the class to validate and convert inputs into booleans.

        If the class does not support boolean conversion, returns None.
        """
    @property
    def none_converter(self) -> NoneConverter | None:
        """Returns the NoneConverter instance used by the class to validate and convert inputs into NoneTypes (Nones).

        If the class does not support None conversion, returns None.
        """
    @property
    def string_converter(self) -> StringConverter | None:
        """Returns the StringConverter instance used by the class to validate and convert inputs into strings.

        If the class does not support string conversion, returns None.
        """
    @property
    def iterable_output_type(self) -> Literal["tuple", "list"] | None:
        """Returns the name of the type to which processed iterables are cast before they are returned or None, if
        the class is configured to preserve the original iterable type.
        """
    @property
    def filter_failed(self) -> bool:
        """Returns True if the class is configured to remove elements that failed validation from the processed
        iterables before returning them.
        """
    @property
    def raise_errors(self) -> bool:
        """Returns True if the class is configured to raise ValueError exceptions when an input fails validation."""
    @classmethod
    def supported_iterables(cls) -> tuple[str, ...]:
        """Returns a tuple that stores string-names of the supported output iterable types.

        These names are valid inputs to class 'iterable_output_type' initialization argument.
        """
    @property
    def allowed_output_types(self) -> tuple[str, ...]:
        """Returns the string-names of the scalar python types the class is configured to produce."""
    def _apply_converters(self, value: Any) -> tuple[bool, int | float | str | bool | None]:
        """Hierarchically applies each of the converters to the input scalar value.

        This is a minor service method that allows standardizing iterable and non-iterable input processing. This
        method contains the core validation logic, whereas the validate_value() method primarily provides set-up and
        tear-down functionality.

        Notes:
            Follows the following conversion hierarchy if multiple converters are active:
            float > integer > boolean > None > string.

        Args:
            value: The value to be validated and / or converted.

        Returns: A tuple that contains two values. The first is a boolean that indicates if the returned value passed
            or failed validation. The second is either the validated / converted value or a None placeholder.
        """
    def validate_value(
        self, value_to_validate: Any
    ) -> int | float | bool | None | str | list[int | float | bool | str | None] | tuple[int | float | str | None, ...]:
        """Validates input values and converts them to the preferred datatype.

        This method validates input values against the validation parameters of the class instance. If the input value
        passes validation, the method converts it to the preferred datatype. If the input value is iterable, the
        method converts it to the preferred iterable type (tuple if not specified).

        The method can conditionally filter out values that fail validation from the output iterable if the
        filter_failed attribute is set to True. Alternatively, it can raise ValueErrors for failed elements if the
        class is configured to do so.

        Note:
            When this class is equipped with multiple base converter classes, the supported types are evaluated in the
            following order: float > integer > boolean > None > string.

        Args
            value_to_validate: The input value to be validated and converted.

        Returns
            The validated and converted value if the method succeeds. For each value that fails validation when
            filtering is disabled, returns "Validation/ConversionError" string to indicate failure.

        Raises:
            ValueError: If the val;ue to validate is iterable with multiple dimensions. If the input scalar value or
                any element of an iterable value cannot be validated, and the raise_errors attribute is set to True.
        """

class NumpyDataConverter:
    """After initial configuration, allows conditionally converting input python values to a specified numpy output
    format.

    This class is built on top of our general PythonDataConverter hierarchy. Specifically, it uses a PythonDataConverter
    class instance to filter and convert inputs to specific Python types know to be convertible to specific numpy types.
    It then converts the input to numpy, using the requested bit-width and signed/unsigned type for integers.

    Notes:
        The class deliberately only works with Python inputs. Numpy already contains a robust set of tools for
        converting between numpy datatypes. The purpose of this class is to provide a robust way for converting
        arbitrary inputs into a specific numpy datatypes.

        The class is designed to be as input-datatype agnostic as possible. In most cases, if a precise input value type
        is known, it is more efficient (and easier) to implement a simple in-code conversion. This class is best suited
        for cases when the input value type can vary widely during runtime and/or includes many possible options.

        At this time, the class does not support converting strings from python to numpy.

    Attributes:
        _python_converter: The initialized PythonDataConverter instance used to validate and convert the inputs into one
            datatype known to be convertible to the requested numpy datatype. While the class can use multiple 'base'
            converters to potentially output multiple datatypes, the instance used here has to be configured to produce
            exactly one output datatype. The converter should not be configured to output strings, as, at this time,
            strings are not supported.
        _output_bit_width: The bit-width of the output numpy datatype. Must be one of the supported options:
            8, 16, 32, 64, 'auto'. If set to 'auto', the class will determine the smallest numpy datatype that can
            accommodate the input value. Generally, unless you have specific bit-width requirements, 'auto' is
            recommended for memory efficiency.
        _signed: Determines the type Python integers are converted to, since numpy distinguishes signed and unsigned
            integers.
        _signed_types: A list of tuples that stores supported signed integer numpy datatypes alongside the maximum and
            minimum values that 'fit' into each of the datatypes. This is used to optimize internal processing and
            should not be modified externally.
        _unsigned_types: Same as _signed_types, but stores supported unsigned integer numpy datatypes.
        _float_types: Same as _signed_types, but stores supported floating point numpy datatypes.
        _integer_index_map: Maps supported integer numpy datatype bit-widths to the indices that can be used to slice
            the 'types' lists to retrieve the callable datatype. This is used to access the callable datatypes used for
            python to numpy conversion based on teh class configuration.
        _floating_index_map: Same as _integer_index_map, but for floating point numpy datatypes.
        _supported_output_bit_widths: Stores supported output_bit_width argument values as a set.

    Raises:
        TypeError: If any class arguments are not of the expected types.
        ValueError: If the provided output_bit_width argument is not one of the supported options.
            If the provided python_converter is configured to output multiple datatypes or to output strings. Also, if
            it is not configured to raise errors for failed validations attempts.
    """

    _signed_types: list[tuple[type, int, int]]
    _unsigned_types: list[tuple[type, int, int]]
    _float_types: list[tuple[type, np.floating[Any], np.floating[Any]]]
    _integer_index_map: dict[int, int]
    _floating_index_map: dict[int, int]
    _supported_output_bit_widths: set[int | str]
    _signed: Incomplete
    _python_converter: Incomplete
    _output_bit_width: Incomplete
    def __init__(
        self, python_converter: PythonDataConverter, output_bit_width: int | str = "auto", signed: bool = True
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the NumpyDataConverter instance."""
    @property
    def python_converter(self) -> PythonDataConverter:
        """Returns the PythonDataConverter instance used to selectively control the range of supported input and
        output values that can be processed by this class."""
    @property
    def output_bit_width(self) -> int | str:
        """Returns the bit-width used by the output (converted) numpy values."""
    @property
    def signed(self) -> bool:
        """Returns True if the class is configured to convert integer inputs to signed integers.

        Returns False if the class is configured to convert integer inputs to unsigned integers."""
    @property
    def supported_output_bit_widths(self) -> tuple[int | str, ...]:
        """Returns the supported output_bit_width initialization argument values that can be used with this class."""
    def _resolve_scalar_type(self, value: int | float) -> type:
        """Returns the scalar numpy type with the minimum appropriate bit-width required to represent the input value,
        taking into consideration the intended output numpy datatype.

        Currently, this method supports datatype-discovery for 8, 16, 32, and 64-bit signed and unsigned integers, and
        16, 32, and 64-bit floating numbers.

        Notes:
            This returns the type instead of converting the value to enable using this method to aggregate conversion
            into the same method. This method is only used to handle 'auto' bit-width conversions.

        Parameters
            value: The value for which to resolve (determine) the scalar datatype with the minimum possible bit-width.

        Returns
            A callable NumPy scalar type that can be used to convert the value into that type.

        Raises:
            TypeError: If the value is not an integer or float.
            OverflowError: If the value is too large to be represented by the supported integer or floating numpy
                datatype bit-widths.
        """
    def convert_value_to_numpy(
        self, value: int | float | bool | None | str | list[Any] | tuple[Any]
    ) -> np.integer[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool | NDArray[Any]:
        """Converts input python values to numpy scalars and arrays of the predetermined datatype.

        The method converts input values to numpy datatypes based on the configuration of the class instance, which
        includes the bit-width and whether to return signed and unsigned integer types. The method works by first
        'funneling' the input values through the PythonDataConverter instance to ensure they use the same scalar or
        iterable Python type. The values are then converted to appropriate numpy datatypes, generating a list of
        numpy scalars. Finally, the list is either returned as a numpy array which automatically aligns datatype
        bit-widths if needed or as a numpy scalar if the input was scalar.

        Args
            value: The value to be converted to a numpy datatype. Note, string inputs are currently not
                supported, unless they are directly convertible to one of the supported types: floating, integer,
                boolean, or None.

        Returns
            The converted value as a numpy scalar or array.

        Raises
            OverflowError: If the input value cannot be converted to the requested numpy datatype due to being
                outside the representation limits for that type.
            ValueError: If the output_bit_width is set to 8, and the input value is a float. Currently, 8-bit floats
                are not supported.
        """
    def convert_value_from_numpy(
        self, value: np.integer[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool | NDArray[Any]
    ) -> int | float | bool | None | list[Any] | tuple[Any, ...]:
        """Converts numpy values to Python datatypes.

        The method converts input numpy datatypes to Python datatypes based on the configuration of the class instance.
        Specifically, it first converts numpy values to python values using item() (for scalars) and tolist()
        (for arrays) method and then passes the resultant values through PythonDataConverter. This ensures the values
        are converted to the same Python datatype regardless of automated NumPy conversion.

        Args
            value: The value to be converted to a Python datatype. Has to be a numpy scalar or array value.
                Currently, this method only supports one-dimensional numpy arrays.

        Returns
            The value converted to a Python datatype.
        """
