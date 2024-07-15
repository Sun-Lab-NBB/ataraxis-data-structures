import numpy as np
from numpy.typing import NDArray
from typing import Any, Generator, Literal

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
        If the input_item was iterable, it is converted into a list.

    Raises:
        TypeError: If the input object cannot be converted or wrapped into a list.
    """
def chunk_iterable(iterable: NDArray[Any] | tuple[Any] | list[Any], chunk_size: int) -> Generator[tuple[Any, ...] | NDArray[Any], None, None]:
    """Yields successive chunk_size-sized chunks from the input iterable or NumPy array.

    This function supports lists, tuples, and NumPy arrays, including multidimensional arrays. For NumPy arrays, it
    maintains the original data type and dimensionality, returning NumPy array chunks. For other iterables, it
    returns tuple chunks.

    The last yielded chunk will contain any leftover elements if the iterable's length is not evenly divisible by
    chunk_size. This last chunk may be smaller than chunk_size.

    Args:
        iterable: The iterable or NumPy array to split into chunks.
        chunk_size: The size of the chunks to split the iterable into.

    Raises:
        TypeError: If 'iterable' is not of a correct type.
        ValueError: If 'chunk_size' value is below 1.

    Returns:
        Chunks of the input iterable (as a tuple) or NumPy array, containing at most chunk_size elements.
    """
def check_condition(checked_value: int | float | str | bool | tuple[Any] | list[Any] | NDArray[Any] | np.number[Any], condition_value: int | float | str | bool | np.number[Any] | np.bool_, condition_operator: Literal['>', '<', '>=', '<=', '==', '!=']) -> bool | np.bool_ | NDArray[np.bool_] | tuple[bool, ...]:
    """Checks the input value against the condition value, using requested condition operator.

    Can take tuples, lists, and numpy arrays as checked_value, in which case the condition_value is applied
    element-wise, and the result is an array (for numpy inputs) or tuple (for Python iterables) of boolean values that
    communicates the result of the operation.

    Currently, only supports simple mathematical operators, but this may be extended in the future.

    Args:
        checked_value: The value, iterable, or numpy array to be checked against the condition.
        condition_value: The condition value that, in combination with comparison operator, determines whether each
            checked_value is matched to a True or False boolean output value.
        condition_operator: An operator symbol. Currently, only supports simple mathematical operators
            of '>','<','>=','<=','==','!='.

    Returns:
        A boolean value for Python scalar inputs. A numpy.bool_ value for NumPy scalar inputs. A boolean numpy array for
        NumPy array inputs. A tuple of boolean values for Python iterable inputs.

    Raises:
        KeyError: If an unsupported operator symbol is provided.
        TypeError: If checked_value is not of a supported type.
    """
def compare_nested_tuples(x: tuple[Any, ...], y: tuple[Any, ...]) -> bool:
    '''Compares two input one-level nested tuples and returns True if all elements in one tuple are equal to the other.

    This function is primarily designed to be used for assertion testing, in place of the numpy array_equal function
    whenever the two compared tuples are not immediately convertible to a numpy 2D array. This is true for tuples that
    use mixed datatype elements (1 and "1") and elements with irregular shapes (tuple of tuple with inner tuples having
    different number of elements).

    Notes:
        This function only works for 2-dimensional (1 nesting level) tuples. It will also work for 1-dimensional tuples,
        but it is more efficient to use the equivalence operator or numpy.equal() on those tuples if possible.
        The function will NOT work for tuples with more than 2 dimensions.

    Args:
        x: The first tuple to be compared.
        y: The second tuple to be compared.

    Returns:
        True, if all elements in each sub-tuple of the two tuples are equal. If either the number of
        sub-tuples, their shapes or the element values in each sub-tuple differ for the two tuples, returns False.

    Raises:
        TypeError: If x or y is not a tuple
    '''
