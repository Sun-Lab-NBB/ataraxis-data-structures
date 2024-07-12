"""This module contains miscellaneous data manipulation methods that either abstract away common operations to reduce
boilerplate code or provide functionality not commonly available from popular Python libraries.

Methods from this module largely belong to two groups. The first group is for 'convenience' methods. These methods
typically abstract away template code that is readily available from Python or common libraries (a good example is the
ensure_list() method). Another set of methods, such as compare_nested_tuples() method, provides novel functionality
not readily available from other libraries. Both groups of methods are useful for a wide range of data-related
applications.

See the API documentation for the description of the methods offered through this module.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal, Generator, Iterable
import operator
from ataraxis_base_utilities import console


def ensure_list(
    input_item: str | int | float | bool | None | np.generic | NDArray[Any] | tuple[Any, ...] | list[Any] | set[Any],
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
        If the input_item was iterable, it is converted into a list.

    Raises:
        TypeError: If the input object cannot be converted or wrapped into a list.
    """
    # Scalars are added to a list and returned as a one-item list. Scalars are handled first to avoid clashing with
    # iterable types.
    if np.isscalar(input_item) or input_item is None:  # Covers Python scalars and NumPy scalars
        return [input_item]
    # Numpy arrays are processed based on their dimensionality. This has to dow tih the fact that zero-dimensional
    # numpy arrays are interpreted as scalars by some numpy methods and as array by others.
    if isinstance(input_item, np.ndarray):
        # 1+-dimensional arrays are processed via tolist(), which correctly casts them to Python list format.
        if input_item.ndim > 0:
            return input_item.tolist()
        else:
            # 0-dimensional arrays are essentially scalars, so the data is popped out via item() and is wrapped
            # into a list.
            return [input_item.item()]
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


def chunk_iterable(
    iterable: NDArray[Any] | tuple[Any] | list[Any], chunk_size: int
) -> Generator[tuple[Any, ...] | NDArray[Any], None, None]:
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
    if not isinstance(iterable, (np.ndarray, list, tuple)):
        message: str = (
            f"Unsupported 'iterable' type encountered when chunking iterable. Expected a list, tuple or numpy array, "
            f"but encountered {iterable} of type {type(iterable).__name__}."
        )
        console.error(message=message, error=TypeError)
        raise TypeError(message)  # Fallback, should not be reachable

    if chunk_size < 1:
        message = (
            f"Unsupported 'chunk_size' value encountered when chunking iterable. Expected a positive non-zero value, "
            f"but encountered {chunk_size}."
        )
        console.error(message=message, error=ValueError)
        raise ValueError(message)  # Fallback, should not be reachable

    # Chunking is performed along the first dimension for both NumPy arrays and Python sequences.
    # This preserves array dimensionality within chunks for NumPy arrays.
    for chunk in range(0, len(iterable), chunk_size):
        chunk_slice = iterable[chunk : chunk + chunk_size]
        yield np.array(chunk_slice) if isinstance(iterable, np.ndarray) else tuple(chunk_slice)


def check_condition(
    checked_value: int | float | str | bool | tuple[Any] | list[Any] | NDArray[Any] | np.number[Any] | np.bool_,
    condition_value: int | float | str | bool | np.number[Any] | np.bool_,
    condition_operator: Literal[">", "<", ">=", "<=", "==", "!="],
) -> bool | np.bool_ | NDArray[np.bool_] | tuple[bool, ...]:
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
    operators = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if condition_operator not in operators:
        message: str = (
            f"Unsupported operator symbol ({condition_operator}) encountered when checking condition, use one of the "
            f"supported operators {', '.join(operators.keys())}."
        )
        console.error(message=message, error=KeyError)
        raise KeyError(message)  # Fallback, should not be reachable

    op = operators[condition_operator]

    # Python scalars
    if isinstance(checked_value, (int, float, str, bool)):
        return bool(op(checked_value, condition_value))
    # Numpy arrays
    elif isinstance(checked_value, np.ndarray):
        return np.array(op(checked_value, condition_value), dtype=np.bool_)
    # Numpy scalars
    elif np.isscalar(checked_value) and isinstance(checked_value, np.generic):
        return np.bool_(op(checked_value, condition_value))
    # Python iterables
    elif isinstance(checked_value, Iterable):
        return tuple(op(v, condition_value) for v in checked_value)
    else:
        message = (
            f"Unsupported checked_value ({checked_value}) type ({type(checked_value).__name__}) encountered when "
            f"checking condition. See API documentation / function signature for supported types."
        )
        console.error(message=message, error=TypeError)
        raise TypeError(message)  # Fallback, should not be reachable


def compare_nested_tuples(x: tuple[Any, ...], y: tuple[Any, ...]) -> bool:
    """Compares two input one-level nested tuples and returns True if all elements in one tuple are equal to the other.

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
    """
    if not isinstance(x, tuple) or not isinstance(y, tuple):
        message = (
            f"Unsupported type encountered when comparing tuples. Either x ({type(x).__name__}) or y "
            f"({type(y).__name__}) is not a tuple."
        )
        console.error(message=message, error=TypeError)
        raise TypeError(message)  # Fallback, should not be reachable

    # Optimized check to short-fail on length mismatch and also as soon as any mismatched element is found to
    # speed up failure case return times
    return len(x) == len(y) and all(subtuple1 == subtuple2 for subtuple1, subtuple2 in zip(x, y))
