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
        This function expects 'source_coordinates' and 'target_coordinates' arrays to be monotonically increasing.

        Discrete interpolated data is returned as an array with the same datatype as the input data. Continuous
        interpolated data is returned as a float64 datatype array.

        Continuous data is interpolated using the linear interpolation method. Discrete data is interpolated to the
        last known value to the left of each interpolated coordinate.

    Args:
        source_coordinates: The source coordinate values.
        source_values: The data values at each source coordinate.
        target_coordinates: The target coordinates for which to interpolate the data values.
        is_discrete: Determines whether the interpolated data is discrete or continuous.

    Returns:
        A one-dimensional NumPy array with the same length as the 'target_coordinates' array that stores the
        interpolated data values.
    """
    if is_discrete:
        interpolated_data = np.empty(target_coordinates.shape, dtype=source_values.dtype)

        # Handles boundary conditions in bulk using boolean masks. Clamps all target coordinates below the first source
        # coordinate to the first source coordinate's value. Clamps all target coordinates above the last source
        # coordinate to the last source coordinate's value.
        below_minimum = target_coordinates < source_coordinates[0]
        above_maximum = target_coordinates > source_coordinates[-1]

        within_bounds = ~(below_minimum | above_maximum)

        interpolated_data[below_minimum] = source_values[0]
        interpolated_data[above_maximum] = source_values[-1]

        # Processes within-boundary coordinates by finding the last known certain value to the left of each target
        # coordinate and setting each to that value.
        if np.any(within_bounds):
            indices = np.searchsorted(source_coordinates, target_coordinates[within_bounds], side="right") - 1
            interpolated_data[within_bounds] = source_values[indices]

        return interpolated_data

    # Casts all inputs to float64 because linear interpolation always produces float64 outputs.
    return np.interp(
        x=target_coordinates.astype(np.float64),
        xp=source_coordinates.astype(np.float64),
        fp=source_values.astype(np.float64),
    )
