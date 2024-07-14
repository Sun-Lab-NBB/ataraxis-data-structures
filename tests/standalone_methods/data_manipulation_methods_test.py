"""Contains tests for functions stored in the standalone_methods package."""

import re
from typing import Any
import textwrap

import numpy as np
import pytest

from ataraxis_data_structures.standalone_methods import (
    ensure_list,
    chunk_iterable,
    check_condition,
    compare_nested_tuples,
)


def error_format(message: str) -> str:
    """Formats the input message to match the default Console format and escapes it using re, so that it can be used to
    verify raised exceptions.

    This method is used to set up pytest 'match' clauses to verify raised exceptions.

    Args:
        message: The message to format and escape, according to standard Ataraxis testing parameters.

    Returns:
        Formatted and escape message that can be used as the 'match' argument of pytest.raises() method.
    """
    return re.escape(textwrap.fill(message, width=120, break_long_words=False, break_on_hyphens=False))


# noinspection PyRedundantParentheses
@pytest.mark.parametrize(
    "input_item, expected",
    [
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        ({1, 2, 3}, [1, 2, 3]),
        ([1], [1]),
        ((1), [1]),
        ({1}, [1]),
        (np.array([1, 2, 3]), [1, 2, 3]),
        (np.array([[1, 2, 3], [4, 5, 6]]), [[1, 2, 3], [4, 5, 6]]),
        (np.array([1]), [1]),
        (1, [1]),
        (1.0, [1.0]),
        ("a", ["a"]),
        (True, [True]),
        (None, [None]),
        (np.int32(1), [1]),
    ],
)
def test_ensure_list(input_item: Any, expected: list) -> None:
    """Verifies the functioning of the ensure_list() method for all supported scenarios.

    Tests the following inputs:
        - 0 multi-item lists
        - 1 multi-item tuples
        - 2 multi-item sets
        - 3 one-item lists
        - 4 one-item tuples
        - 5 one-item sets
        - 6 one-dimensional numpy arrays
        - 7 multidimensional numpy arrays
        - 9 zero-dimensional numpy arrays
        - 7 ints
        - 8 floats
        - 9 strings
        - 10 bools
        - 11 Nones
        - 12 Numpy scalars
    """
    output = ensure_list(input_item)
    # Checks output value
    assert output == expected
    # Checks output type
    assert type(output) is type(expected)


def test_ensure_list_error() -> None:
    """Verifies that ensure_list() correctly handles unsupported input types."""
    message = (
        f"Unable to convert input item to a Python list, as items of type {type(object()).__name__} "
        f"are not supported."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        ensure_list(object())


# noinspection PyRedundantParentheses
@pytest.mark.parametrize(
    "input_iterable, chunk_size, expected_chunks",
    [
        ([1, 2, 3, 4, 5], 2, [(1, 2), (3, 4), (5,)]),
        (np.array([1, 2, 3, 4, 5]), 2, [np.array([1, 2]), np.array([3, 4]), np.array([5])]),
        ((1, 2, 3, 4, 5), 3, [(1, 2, 3), (4, 5)]),
    ],
)
def test_chunk_iterable(input_iterable, chunk_size: int, expected_chunks) -> None:
    """Verifies the functioning of the chunk_iterable() method for various input types and chunk sizes.

    Tests the following scenarios:
        - 0 List input with even chunks and a remainder
        - 1 NumPy array input with even chunks and a remainder
        - 2 Tuple input with uneven chunks
    """
    # Returns a generator that can be iterated to get successive chunks
    result = list(chunk_iterable(input_iterable, chunk_size))

    # Verifies that the obtained number of chunks matches expectation
    assert len(result) == len(expected_chunks)

    # Verifies that the individual chunks match expected chunks
    for r, e in zip(result, expected_chunks):
        if isinstance(r, np.ndarray):
            assert np.array_equal(r, e)
        else:
            assert r == e


def test_chunk_iterable_error() -> None:
    """Verifies that chunk_iterable() correctly handles unsupported iterables types and chunk_size values."""
    message: str = (
        f"Unsupported 'iterable' type encountered when chunking iterable. Expected a list, tuple or numpy array, "
        f"but encountered {1} of type {type(1).__name__}."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        list(chunk_iterable(iterable=1, chunk_size=2))

    message = (
        f"Unsupported 'chunk_size' value encountered when chunking iterable. Expected a positive non-zero value, "
        f"but encountered {-4}."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        list(chunk_iterable(iterable=[1, 2, 3], chunk_size=-4))


@pytest.mark.parametrize(
    "checked_value, condition_value, condition_operator, expected",
    [
        (5, 3, ">", True),
        (5, 3, "<", False),
        (5, 5, ">=", True),
        (5, 5, "<=", True),
        (5, 5, "==", True),
        (5, 3, "!=", True),
        ([1, 2, 3], 2, ">", (False, False, True)),
        ([1, 2, 3], 2, "<", (True, False, False)),
        ([1, 2, 3], 2, ">=", (False, True, True)),
        ([1, 2, 3], 2, "<=", (True, True, False)),
        ([1, 2, 3], 2, "==", (False, True, False)),
        ([1, 2, 3], 2, "!=", (True, False, True)),
        (np.array([1, 2, 3]), 2, ">", np.array([False, False, True])),
        (np.array([1, 2, 3]), 2, "<", np.array([True, False, False])),
        (np.array([1, 2, 3]), 2, ">=", np.array([False, True, True])),
        (np.array([1, 2, 3]), 2, "<=", np.array([True, True, False])),
        (np.array([1, 2, 3]), 2, "==", np.array([False, True, False])),
        (np.array([1, 2, 3]), 2, "!=", np.array([True, False, True])),
        (np.int32(5), 3, ">", np.bool_(True)),
        (np.int32(5), 3, "<", np.bool_(False)),
        (np.int32(5), 5, ">=", np.bool_(True)),
        (np.int32(5), 5, "<=", np.bool_(True)),
        (np.int32(5), 5, "==", np.bool_(True)),
        (np.int32(5), 3, "!=", np.bool_(True)),
    ],
)
def test_check_condition(checked_value: Any, condition_value: Any, condition_operator: str, expected: Any) -> None:
    """Verifies the functioning of the check_condition() method for all supported operators and various input types.

    Tests the following scenarios:
        - 0-5: Python scalar comparisons with all operators (>, <, >=, <=, ==, !=)
        - 6-11: List comparisons with all operators
        - 12-17: NumPy array comparisons with all operators
        - 18-23: NumPy scalar comparisons with all operators

    For each input type (Python scalar, list, NumPy array, NumPy scalar), all six supported operators are tested:
    '>', '<', '>=', '<=', '==', '!='.
    """
    # noinspection PyTypeChecker
    result = check_condition(checked_value, condition_value, condition_operator)
    if isinstance(result, np.ndarray):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


def test_check_condition_error() -> None:
    """Verifies that check_condition() correctly handles invalid unsupported input types."""
    message = (
        f"Unsupported checked_value ({object()}) type ({type(object()).__name__}) encountered when "
        f"checking condition. See API documentation / function signature for supported types."
    )

    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        check_condition(checked_value=object(), condition_value=1, condition_operator=">")

    # Also verifies the handling of unsupported operator inputs
    message = f"Unsupported operator symbol ({'!>'}) encountered when checking condition "
    with pytest.raises(KeyError, match=error_format(message)):
        # noinspection PyTypeChecker
        check_condition(checked_value=11, condition_value=11, condition_operator="!>")


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (((1, 2), (3, 4)), ((1, 2), (3, 4)), True),
        (((1, 2), (3, 4)), ((1, 2), (3, 5)), False),
        ((("a", "b"), ("c",)), (("a", "b"), ("c",)), True),
        ((("a", "b"), ("c",)), (("a", "b"), ("d",)), False),
    ],
)
def test_compare_nested_tuples(x: tuple, y: tuple, expected: bool) -> None:
    """Verifies the functioning of the compare_nested_tuples() method for various nested tuple scenarios.

    Tests the following scenarios:
        - 0 Identical nested tuples with numbers
        - 1 Different nested tuples with numbers
        - 2 Identical nested tuples with strings and different inner tuple lengths
        - 3 Different nested tuples with strings and different inner tuple lengths
    """
    assert compare_nested_tuples(x, y) == expected


def test_compare_nested_tuples_error() -> None:
    """Verifies that compare_nested_tuples() correctly handles non-tuple inputs."""

    message = (
        f"Unsupported type encountered when comparing tuples. Either x ({type([1, 2]).__name__}) or y "
        f"({type((1, 2)).__name__}) is not a tuple."
    )

    with pytest.raises(TypeError, match=error_format(message)):
        # noinspection PyTypeChecker
        compare_nested_tuples(x=[1, 2], y=(1, 2))
