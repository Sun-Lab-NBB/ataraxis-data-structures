import pytest
import numpy as np
from typing import Any
from src.ataraxis_data_structures.standalone_methods.data_manipulation_methods import (
    ensure_list, chunk_iterable, check_condition, find_closest_indices,
    find_event_boundaries, compare_nested_tuples
)


@pytest.mark.parametrize("input_item, expected", [
    ([1, 2, 3], [1, 2, 3]),
    ((1, 2, 3), [1, 2, 3]),
    ({1, 2, 3}, [1, 2, 3]),
    (np.array([1, 2, 3]), [1, 2, 3]),
    (1, [1]),
    (1.0, [1.0]),
    ("a", ["a"]),
    (True, [True]),
    (None, [None]),
    (np.int32(1), [1]),
])
def test_ensure_list(input_item: Any, expected: list) -> None:
    assert ensure_list(input_item) == expected


def test_ensure_list_error() -> None:
    with pytest.raises(TypeError, match="Unable to convert input item to a Python list"):
        # noinspection PyTypeChecker
        ensure_list(object())


@pytest.mark.parametrize("iterable, chunk_size, expected", [
    ([1, 2, 3, 4, 5], 2, [(1, 2), (3, 4), (5,)]),
    (np.array([1, 2, 3, 4, 5]), 2, [np.array([1, 2]), np.array([3, 4]), np.array([5])]),
    ((1, 2, 3, 4, 5), 3, [(1, 2, 3), (4, 5)]),
])
def test_chunk_iterable(iterable: Any, chunk_size: int, expected: list) -> None:
    result = list(chunk_iterable(iterable, chunk_size))
    assert len(result) == len(expected)
    for r, e in zip(result, expected):
        if isinstance(r, np.ndarray):
            assert np.array_equal(r, e)
        else:
            assert r == e


def test_chunk_iterable_error() -> None:
    with pytest.raises(TypeError, match="Unsupported iterable type encountered when chunking iterable"):
        list(chunk_iterable(1, 2))


@pytest.mark.parametrize("checked_value, condition_value, condition_operator, expected", [
    (5, 3, ">", True),
    ([1, 2, 3], 2, "<=", (True, True, False)),
    (np.array([1, 2, 3]), 2, "<", np.array([True, False, False])),
    (np.int32(5), 3, ">=", np.bool_(True)),
])
def test_check_condition(checked_value: Any, condition_value: Any, condition_operator: str, expected: Any) -> None:
    # noinspection PyTypeChecker
    result = check_condition(checked_value, condition_value, condition_operator)
    if isinstance(result, np.ndarray):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


def test_check_condition_error() -> None:
    with pytest.raises(KeyError, match="Unsupported operator symbol"):
        # noinspection PyTypeChecker
        check_condition(1, 1, "invalid")

    with pytest.raises(TypeError, match="Unsupported checked_value"):
        # noinspection PyTypeChecker
        check_condition(object(), 1, ">")


@pytest.mark.parametrize("target_array, source_array, expected", [
    ([1, 5, 10], [2, 4, 6, 8], (0, 1, 3)),
    (np.array([1, 5, 10]), np.array([2, 4, 6, 8]), np.array([0, 1, 3])),
])
def test_find_closest_indices(target_array: Any, source_array: Any, expected: Any) -> None:
    result = find_closest_indices(target_array, source_array)
    if isinstance(result, np.ndarray):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize("trace, make_offsets_exclusive, allow_no_events, expected", [
    ([0, 1, 1, 0, 1, 1, 1, 0], True, True, ((1, 3), (4, 7))),
    ([0, 1, 1, 0, 1, 1, 1, 0], False, True, ((1, 2), (4, 6))),
    ([0, 0, 0], True, True, ()),
])
def test_find_event_boundaries(trace: Any, make_offsets_exclusive: bool, allow_no_events: bool,
                               expected: tuple) -> None:
    result = find_event_boundaries(trace, make_offsets_exclusive=make_offsets_exclusive,
                                   allow_no_events=allow_no_events)
    assert result == expected


def test_find_event_boundaries_error() -> None:
    with pytest.raises(ValueError, match="Unsupported NumPy array 'trace' input detected"):
        find_event_boundaries(np.array([[1, 2], [3, 4]]))

    with pytest.raises(RuntimeError, match="Unable to find any event boundaries"):
        find_event_boundaries([0, 0, 0], allow_no_events=False)


@pytest.mark.parametrize("x, y, expected", [
    (((1, 2), (3, 4)), ((1, 2), (3, 4)), True),
    (((1, 2), (3, 4)), ((1, 2), (3, 5)), False),
    ((('a', 'b'), ('c',)), (('a', 'b'), ('c',)), True),
    ((('a', 'b'), ('c',)), (('a', 'b'), ('d',)), False),
])
def test_compare_nested_tuples(x: tuple, y: tuple, expected: bool) -> None:
    assert compare_nested_tuples(x, y) == expected


def test_compare_nested_tuples_error() -> None:
    with pytest.raises(TypeError, match="Unsupported type encountered when comparing tuples"):
        # noinspection PyTypeChecker
        compare_nested_tuples([1, 2], (1, 2))
