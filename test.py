from ataraxis_data_structures.data_converters import (
    NumericConverter,
    BooleanConverter,
    StringConverter,
    PythonDataConverter,
)

# Configured converters to be combined through PythonDataConverter
numeric_converter = NumericConverter(allow_integer_output=True, allow_float_output=False, parse_number_strings=True)
bool_converter = BooleanConverter(parse_boolean_equivalents=False)
string_converter = StringConverter(allow_string_conversion=True)

# When provided with multiple converters, they are applied in this order: Numeric > Boolean > None > String
python_converter = PythonDataConverter(
    numeric_converter=numeric_converter, boolean_converter=bool_converter, string_converter=string_converter
)

# Output depends on the application hierarchy and the configuration of each 'Base' converter. If at least one converter
# 'validates' the value successfully, the 'highest' success value is returned.
assert python_converter.validate_value('33') == 33  # Parses number-convertible integer as integer

# Defaults to tuple outputs. Unlike 'Base' Converters, the class uses a long 'Validation/ConversionError' string to
# denote outputs that failed to be converted
assert python_converter.validate_value(["33", 11, 14.0, 3.32]) == (33, 11, 14, "Validation/ConversionError")

# Optionally, the class can be configured to filter 'failed' iterable elements out and return a list instead of a tuple
python_converter = PythonDataConverter(
    numeric_converter=numeric_converter, filter_failed_elements=True, iterable_output_type="list"
)
assert python_converter.validate_value(["33", 11, 14.0, 3.32]) == [33, 11, 14]
