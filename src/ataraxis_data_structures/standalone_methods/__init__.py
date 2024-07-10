"""This package provides standalone data manipulation methods that are both used by other packages of the library
and other Sun Lab projects.

Methods from this package largely belong to two groups. The first group is for 'convenience' methods. These methods
typically abstract away template code that is readily available from Python or common libraries (a good example is the
ensure_list() method). Another set of methods, such as find_event_boundaries() method, provides novel functionality
not readily available from other libraries. Both groups of methods are useful for a wide range of data-related
applications.
"""

from .data_manipulation_methods import (
    ensure_list,
    chunk_iterable,
    check_condition,
    find_closest_indices,
    find_event_boundaries,
    compare_nested_tuples,
)

__all__ = [
    "ensure_list",
    "chunk_iterable",
    "check_condition",
    "find_closest_indices",
    "find_event_boundaries",
    "compare_nested_tuples",
]
