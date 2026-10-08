"""Regression tests for refinement behavior."""

import numpy as np

from adapt.refine import refine_1D, refine_scalar_field


def test_1d_handles_unsorted_points():
    points = np.array([2.0, 0.0, 1.0])
    values = np.array([4.0, 0.0, 1.0])

    result = refine_1D(
        points,
        values,
        criterion="difference",
        threshold="non_zero",
    )

    assert result is not None
    np.testing.assert_allclose(result, [0.5, 1.5])


def test_constant_values_need_no_refinement():
    points = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, 1.0, 1.0])

    result = refine_1D(
        points,
        values,
        criterion="difference",
        threshold="non_zero",
    )

    assert result is None


def test_2d_returns_original_coordinates():
    points = np.array([
        [10.0, 20.0],
        [12.0, 20.0],
        [10.0, 24.0],
    ])
    values = np.array([0.0, 1.0, 2.0])

    result = refine_scalar_field(
        points,
        values,
        criterion="difference",
        threshold="non_zero",
    )

    assert result is not None
    assert set(map(tuple, result)) == {
        (11.0, 20.0),
        (10.0, 22.0),
        (11.0, 22.0),
    }
