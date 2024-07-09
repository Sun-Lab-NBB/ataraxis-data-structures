import numpy as np
from numpy.typing import NDArray
from typing import Any, Literal, Generator, Iterable
import operator
from scipy.ndimage import find_objects, label  # type: ignore
from ataraxis_base_utilities import console


def ensure_list(
    input_item: str | int | float | bool | None | np.generic | NDArray[Any] | tuple[Any, ...] | list[Any] | set[Any],
) -> list[Any]:
    """Checks whether input item is a Python list and, if not, converts it to list.

    If the item is a list, returns the item unchanged.

    Args:
        input_item: The variable to be made into / preserved as a Python list.

    Returns:
        A Python list that contains all items inside the input_item variable.

    Raises:
        TypeError: If the input item is not of a supported type.
    """
    if isinstance(input_item, list):
        return input_item
    if isinstance(input_item, (tuple, set, np.ndarray)):
        return list(input_item)
    if np.isscalar(input_item) or input_item is None:  # Covers Python scalars and NumPy scalars
        return [input_item]
    raise TypeError(
        f"Unable to convert input item to a Python list, as items of type {type(input_item).__name__} "
        f"are not supported."
    )


def chunk_iterable(
    iterable: NDArray[Any] | tuple[Any] | list[Any], chunk_size: int
) -> Generator[tuple[Any, ...] | NDArray[Any], None, None]:
    """Yields successive chunk_size-sized chunks from the input iterable or NumPy array.

    This function supports lists, tuples and NumPy arrays, including multidimensional arrays. For NumPy arrays, it
    maintains the original data type and dimensionality, returning NumPy array chunks. For other iterables, it
    returns tuple chunks.

    The last yielded chunk will contain any leftover elements if the iterable's length is not evenly divisible by
    chunk_size. This last chunk may be smaller than chunk_size.

    Args:
        iterable: The iterable or NumPy array to split into chunks.
        chunk_size: The size of the chunks to split the iterable into.

    Raises:
        TypeError: If 'iterable' is not of a correct type.

    Returns:
        Chunks of the input iterable (as a tuple) or NumPy array, containing at most chunk_size elements.
    """
    if not isinstance(iterable, (np.ndarray, list, tuple)):
        message: str = (
            f"Unsupported iterable type encountered when chunking iterable. Expected a list, tuple or numpy array, "
            f"but encountered {iterable} of type {type(iterable).__name__}."
        )
        console.error(message=message, error=TypeError)
        raise TypeError(message)  # Fallback, should not be reachable

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

    Can take tuples, lists and numpy arrays as checked_value, in which case the condition_value is applied element-wise
    and the result is an array (for numpy inputs) or tuple (for Python iterables) of boolean values that communicates
    the result of the operation.

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
            f"Unsupported operator symbol '{condition_operator}' encountered when checking condition, use one of the "
            f"supported operators {','.join(operators.keys())}."
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


def find_closest_indices(
    target_array: NDArray[Any] | list[Any] | tuple[Any, ...], source_array: NDArray[Any] | list[Any] | tuple[Any, ...]
) -> NDArray[Any] | tuple[Any, ...]:
    """For every value inside target_array, finds the closest values (based on magnitude) in source_array and returns
    their indices.

    This effectively maps target_array onto source_array, in a manner analogous to numpy.where() method, except it
    works for arrays rather than single conditions and without list comprehension (so, in pure numpy).

    Args:
        target_array: The numpy array, list or tuple that contains the values to be mapped to source_array.
        source_array: The numpy array, list or tuple to which target array values are mapped.

    Returns:
        A numpy array that stores the positional indices of the mapped array-values in the source_array, if both inputs
        were numpy arrays. Otherwise, converts the array into a tuple before returning it to caller.
    """
    # Converts inputs to numpy arrays if they aren't already
    target = np.asarray(target_array)
    source = np.asarray(source_array)

    # Reshapes target_array for broadcasting (turns it into a column vector)
    target = target.reshape(-1, 1)

    # Computes the differences between each target value and all source values
    differences: NDArray[Any] = np.abs(source - target)
    index_mappings: NDArray[Any] = np.argmin(differences, axis=1)

    # Returns the indices that resulted in minimum differences for each array. If at least one of the inputs was not a
    # numpy array, returns the result as a tuple. Otherwise, returns it as a numpy array
    if not isinstance(target_array, np.ndarray) or not isinstance(source_array, np.ndarray):
        return tuple(index_mappings.tolist())
    else:
        return index_mappings


def find_event_boundaries(
    trace: NDArray[Any] | list[int | float] | tuple[int | float, ...],
    *,
    make_offsets_exclusive: bool = True,
    allow_no_events: bool = True,
) -> tuple[tuple[int, int], ...]:
    """Finds onset and offset indices for each event in the input trace.

    Assumes the input trace has been binarized prior to calling this function. Uses scipy.signal.label internally,
    so any value above 0 will be labeled as signal and processed accordingly.

    This function is a domain-converter that turns continuous traces into tuples of event coordinates along the trace
    axis.

    Args:
        trace: The numpy array, list, or tuple to search through.
        make_offsets_exclusive: A toggle that determines whether offset indices will be set to the last HIGH (1) value
            of each event or to the first LOW (0) value after the event.
        allow_no_events: Determines whether to raise an error or to return an empty tuple if no events are discovered.
            This flag allows to flexibly handles use cases where discovering no events indicates a critical problem and
            cases where this is a valid outcome.

    Returns:
        A tuple of two-integer tuples that contain the onset (first integer) and offset (second integer) indices for
        each event that satisfies the conditions.

    Raises:
        RuntimeError: If the function does not find any event boundaries in the input trace and this result is not
            allowed.
        ValueError: If the input trace is not 1-dimensional.
    """
    # Converts input to numpy array if it's not already
    if not isinstance(trace, np.ndarray):
        trace = np.asarray(trace)

    if trace.ndim != 1:
        message: str = (
            f"Unsupported NumPy array 'trace' input detected when finding event boundaries. Currently, only "
            f"1-dimensional numpy arrays are supported. Instead, encountered an array with shape '{trace.shape}' and"
            f"dimensionality '{trace.ndim}'"
        )
        console.error(message=message, error=ValueError)
        raise ValueError(message)  # Fallback, should not be reachable

    # Binarizes the trace
    binary_trace = (trace > 0).astype(int)

    # Labels the events
    labeled_trace, num_events = label(binary_trace)

    # If no events are discovered and this is allowed, returns an empty tuple.
    if num_events == 0 and allow_no_events:
        return tuple()
    elif num_events == 0:
        message = (
            f"Unable to find any event boundaries in the input trace. Either the input trace has not been binarized "
            f"or it contains no discoverable events."
        )
        console.error(message=message, error=RuntimeError)
        raise RuntimeError(message)  # Fallback, should not be reachable

    # Finds objects (events)
    events = find_objects(labeled_trace)

    # Extracts boundaries as a list of tuples.
    boundaries = [(int(event[0].start), int(event[0].stop + int(make_offsets_exclusive) - 1)) for event in events]

    # Returns extracted event boundaries as a tuple of tuples
    return tuple(boundaries)


def compare_nested_tuples(x: tuple[Any, ...], y: tuple[Any, ...]) -> bool:
    """Compares two input one-level nested tuples and returns True if all elements in one tuple are equal to the other.

    This function is primarily designed to be used for assertion testing, in place of the numpy array_equal function
    whenever the two compared tuples are not immediately convertible to numpy 2D array. This is true for tuples that use
    mixed datatype elements (1 and "1") and elements with irregular shapes (tuple of tuple with inner tuples having
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
            f"({type(x).__name__}) is not a tuple."
        )
        console.error(message=message, error=TypeError)
        raise TypeError(message)  # Fallback, should not be reachable

    # Optimized check to short-fail on length mismatch and also as soon as any mismatched element is found to
    # speed-up failure case return times
    return len(x) == len(y) and all(subtuple1 == subtuple2 for subtuple1, subtuple2 in zip(x, y))
