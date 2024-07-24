"""This pacakge provides multiple Converter classes that can be used to convert a wide range of inputs into specific
Python and NumPy datatypes.

Currently, it exposes the following classes:
    - NumericConverter: A class that converts scalar Python inputs into floats and/or integers.
    - BooleanConverter: A class that converts scalar Python inputs into boolean values.
    - StringConverter: A class that converts scalar Python inputs into strings.
    - NoneConverter: A class that converts scalar Python inputs into None values.
    - PythonDataConverter: A class that wraps one or more of the base Python Converter class instances and uses them to
    convert scalar and iterable Python inputs into scalar and iterable Python outputs based on the class configuration.
    The class is designed to be as flexible as possible to support a wide range of conversions and fine-tuning.
    - NumpyDataConverter: A class that converts scalar and iterable Python inputs into scalar amd iterable NumPy outputs
    and vice versa.

See data_converters.py for more details on each of the exposed classes.
"""

from .data_converters import (
    NoneConverter,
    StringConverter,
    BooleanConverter,
    NumericConverter,
    NumpyDataConverter,
    PythonDataConverter,
)

__all__ = [
    "NumericConverter",
    "NoneConverter",
    "BooleanConverter",
    "StringConverter",
    "PythonDataConverter",
    "NumpyDataConverter",
]
