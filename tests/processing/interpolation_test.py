"""Contains tests for the interpolate_data function provided by the processing module."""

import numpy as np

from ataraxis_data_structures import interpolate_data


class TestInterpolateData:
    """Contains tests for the interpolate_data function."""

    def test_discrete_interpolation_within_bounds(self) -> None:
        """Verifies discrete interpolation returns last known value for coordinates within source bounds."""
        source_coordinates = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        source_values = np.array([10, 20, 30, 40, 50])
        target_coordinates = np.array([0.5, 1.5, 2.5, 3.5])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        # Each target coordinate should get the value of the last source coordinate to its left.
        expected = np.array([10, 20, 30, 40])
        np.testing.assert_array_equal(result, expected)

    def test_discrete_interpolation_below_minimum(self) -> None:
        """Verifies discrete interpolation clamps coordinates below minimum to first value."""
        source_coordinates = np.array([1.0, 2.0, 3.0])
        source_values = np.array([100, 200, 300])
        target_coordinates = np.array([-1.0, 0.0, 0.5])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        # All target coordinates below the first source coordinate should get the first value.
        expected = np.array([100, 100, 100])
        np.testing.assert_array_equal(result, expected)

    def test_discrete_interpolation_above_maximum(self) -> None:
        """Verifies discrete interpolation clamps coordinates above maximum to last value."""
        source_coordinates = np.array([1.0, 2.0, 3.0])
        source_values = np.array([100, 200, 300])
        target_coordinates = np.array([3.5, 4.0, 10.0])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        # All target coordinates above the last source coordinate should get the last value.
        expected = np.array([300, 300, 300])
        np.testing.assert_array_equal(result, expected)

    def test_discrete_interpolation_exact_match(self) -> None:
        """Verifies discrete interpolation handles exact coordinate matches correctly."""
        source_coordinates = np.array([0.0, 1.0, 2.0, 3.0])
        source_values = np.array([10, 20, 30, 40])
        target_coordinates = np.array([0.0, 1.0, 2.0, 3.0])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        np.testing.assert_array_equal(result, source_values)

    def test_discrete_interpolation_preserves_dtype(self) -> None:
        """Verifies discrete interpolation preserves the source values dtype."""
        source_coordinates = np.array([0.0, 1.0, 2.0])
        source_values = np.array([10, 20, 30], dtype=np.int32)
        target_coordinates = np.array([0.5, 1.5])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        assert result.dtype == np.int32

    def test_discrete_interpolation_mixed_boundaries(self) -> None:
        """Verifies discrete interpolation handles mixed boundary conditions."""
        source_coordinates = np.array([1.0, 2.0, 3.0, 4.0])
        source_values = np.array([10, 20, 30, 40])
        target_coordinates = np.array([0.0, 1.5, 2.5, 5.0])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        # 0.0 -> below min (10), 1.5 -> last known is 1.0 (10), 2.5 -> last known is 2.0 (20), 5.0 -> above max (40)
        expected = np.array([10, 10, 20, 40])
        np.testing.assert_array_equal(result, expected)

    def test_continuous_interpolation_linear(self) -> None:
        """Verifies continuous interpolation performs linear interpolation."""
        source_coordinates = np.array([0.0, 1.0, 2.0])
        source_values = np.array([0.0, 10.0, 20.0])
        target_coordinates = np.array([0.5, 1.5])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=False,
        )

        expected = np.array([5.0, 15.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_continuous_interpolation_returns_float64(self) -> None:
        """Verifies continuous interpolation always returns float64 dtype."""
        source_coordinates = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        source_values = np.array([0, 10, 20], dtype=np.int32)
        target_coordinates = np.array([0.5, 1.5], dtype=np.float32)

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=False,
        )

        assert result.dtype == np.float64

    def test_continuous_interpolation_boundary_clamping(self) -> None:
        """Verifies continuous interpolation clamps out-of-bounds coordinates."""
        source_coordinates = np.array([1.0, 2.0, 3.0])
        source_values = np.array([10.0, 20.0, 30.0])
        target_coordinates = np.array([0.0, 4.0])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=False,
        )

        # np.interp clamps to boundary values.
        expected = np.array([10.0, 30.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_continuous_interpolation_exact_match(self) -> None:
        """Verifies continuous interpolation handles exact coordinate matches correctly."""
        source_coordinates = np.array([0.0, 1.0, 2.0, 3.0])
        source_values = np.array([10.0, 20.0, 30.0, 40.0])
        target_coordinates = np.array([0.0, 1.0, 2.0, 3.0])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=False,
        )

        np.testing.assert_array_almost_equal(result, source_values)

    def test_single_source_value_discrete(self) -> None:
        """Verifies discrete interpolation handles single source value."""
        source_coordinates = np.array([1.0])
        source_values = np.array([42])
        target_coordinates = np.array([0.0, 1.0, 2.0])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        # All targets should map to the single source value.
        expected = np.array([42, 42, 42])
        np.testing.assert_array_equal(result, expected)

    def test_single_source_value_continuous(self) -> None:
        """Verifies continuous interpolation handles single source value."""
        source_coordinates = np.array([1.0])
        source_values = np.array([42.0])
        target_coordinates = np.array([0.0, 1.0, 2.0])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=False,
        )

        # All targets should map to the single source value.
        expected = np.array([42.0, 42.0, 42.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_target_coordinates(self) -> None:
        """Verifies interpolation handles empty target coordinates array."""
        source_coordinates = np.array([0.0, 1.0, 2.0])
        source_values = np.array([10.0, 20.0, 30.0])
        target_coordinates = np.array([])

        result_discrete = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        result_continuous = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=False,
        )

        assert len(result_discrete) == 0
        assert len(result_continuous) == 0

    def test_discrete_interpolation_uint8_values(self) -> None:
        """Verifies discrete interpolation works with uint8 dtype values."""
        source_coordinates = np.array([0.0, 1.0, 2.0, 3.0])
        source_values = np.array([0, 127, 200, 255], dtype=np.uint8)
        target_coordinates = np.array([0.5, 1.5, 2.5])

        result = interpolate_data(
            source_coordinates=source_coordinates,
            source_values=source_values,
            target_coordinates=target_coordinates,
            is_discrete=True,
        )

        assert result.dtype == np.uint8
        expected = np.array([0, 127, 200], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
