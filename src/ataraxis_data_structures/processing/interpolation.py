"""Provides data interpolation utilities for time-series alignment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ataraxis_base_utilities import console

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
        Expects the ``source_coordinates`` and ``target_coordinates`` arrays to be one-dimensional and monotonically
        increasing.

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
        The interpolated data value at each target coordinate, in the order the target coordinates were supplied.

    Raises:
        ValueError: If the source coordinate array or the source value array holds no element.
    """
    # A source distribution holding no point defines no value to interpolate to. Refusing here keeps that case from
    # reaching the two backends below, which fail it differently: the discrete path inverts its clip bounds and indexes
    # an empty array, while the continuous path raises out of np.interp.
    if source_coordinates.size == 0 or source_values.size == 0:
        message = (
            f"Unable to interpolate the data values at the requested coordinates. The 'source_coordinates' and "
            f"'source_values' arguments must each hold at least one element, but got {source_coordinates.size} "
            f"coordinate(s) and {source_values.size} value(s)."
        )
        console.error(message=message, error=ValueError)

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
