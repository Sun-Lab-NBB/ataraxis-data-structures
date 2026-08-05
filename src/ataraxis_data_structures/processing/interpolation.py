"""Provides data interpolation utilities for time-series alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def interpolate_data(
    source_coordinates: NDArray[np.number[Any]],
    source_values: NDArray[np.number[Any]],
    target_coordinates: NDArray[np.number[Any]],
    *,
    is_discrete: bool,
) -> NDArray[np.number[Any]]:
    """Interpolates the data values at the requested coordinates using the source coordinate-value distribution.

    Notes:
        This function expects ``source_coordinates`` and ``target_coordinates`` arrays to be one-dimensional and
        monotonically increasing.

        Discrete interpolated data is returned as an array with the same datatype as the input data. Continuous
        interpolated data is returned as a float64 datatype array.

        Continuous data is interpolated using the linear interpolation method. Discrete data is interpolated to the
        last known value at or to the left of each target coordinate. Target coordinates below the source range are
        clamped to the first source value, and those above the source range are clamped to the last source value.

    Args:
        source_coordinates: The source coordinate values.
        source_values: The data values at each source coordinate.
        target_coordinates: The target coordinates for which to interpolate the data values.
        is_discrete: Determines whether the interpolated data is discrete or continuous.

    Returns:
        A one-dimensional NumPy array with the same length as the ``target_coordinates`` array that stores the
        interpolated data values.
    """
    if is_discrete:
        # Locates the last source coordinate at or to the left of each target coordinate. The subtraction lands a
        # target above the source range on the final index, and a target below it on -1, which the lower clip raises
        # to the first index. The upper clip binds only when the value array is shorter than the coordinate array.
        indices = np.searchsorted(a=source_coordinates, v=target_coordinates, side="right")
        indices -= 1
        np.clip(a=indices, a_min=0, a_max=source_values.size - 1, out=indices)

        return source_values[indices]

    # Casts all inputs to float64 because linear interpolation always produces float64 outputs. np.asarray returns the
    # input object itself when it already carries that dtype, so a float64 caller pays no copy.
    return np.interp(
        x=np.asarray(a=target_coordinates, dtype=np.float64),
        xp=np.asarray(a=source_coordinates, dtype=np.float64),
        fp=np.asarray(a=source_values, dtype=np.float64),
    )
