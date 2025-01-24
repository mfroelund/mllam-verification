"""Unit tests for the statistics module."""

import pytest
import xarray as xr
from mllam_verification.operations.statistics import (
    calculate_error_per_gridpoint,
    calculate_global_error,
)


@pytest.fixture(name="expected_dims_1d")
def fixture_expected_dims_1d(ds_prediction_1d: xr.Dataset) -> dict:
    """Fixture that returns dict of expected dimensions for 1D error variable."""
    # The analysis_time and grid_index dimensions are expected to have been
    # averaged out.
    expected_dims = dict(ds_prediction_1d.sizes)
    for key in ["analysis_time", "grid_index"]:
        expected_dims.pop(key)

    return expected_dims


@pytest.fixture(name="expected_dims_2d")
def fixture_expected_dims_2d(ds_prediction_2d: xr.Dataset) -> dict:
    """Fixture that returns dict of expected dimensions for 2D error variable."""
    # The analysis_time dimension is expected to have been averaged out.
    expected_dims = dict(ds_prediction_2d.sizes)
    expected_dims.pop("analysis_time")

    return expected_dims


class TestCalculateGlobalError:
    """Tests for the calculate_global_error function."""

    def test_without_persistence(
        self,
        ds_reference_1d_relevant_times_and_aligned: xr.Dataset,
        ds_prediction_1d: xr.Dataset,
        expected_dims_1d: dict,
    ):
        """Test the calculation of global error without persistence."""
        ds_error = calculate_global_error(
            ds_reference_1d_relevant_times_and_aligned, ds_prediction_1d
        )
        assert "error" in ds_error
        assert ds_error.sizes == expected_dims_1d
        assert "cell_methods" in ds_error["error"].attrs

    def test_with_persistence(
        self,
        ds_reference_1d_relevant_times_and_aligned: xr.Dataset,
        ds_prediction_1d: xr.Dataset,
        expected_dims_1d: dict,
    ):
        """Test the calculation of global error with persistence."""
        ds_error = calculate_global_error(
            ds_reference_1d_relevant_times_and_aligned,
            ds_prediction_1d,
            include_persistence=True,
        )
        assert "error" in ds_error
        assert ds_error.sizes == expected_dims_1d
        assert "cell_methods" in ds_error["error"].attrs

        assert "persistence_error" in ds_error
        assert "cell_methods" in ds_error["persistence_error"].attrs


class TestCalculateErrorPerGridpoint:
    """Tests for the calculate_error_per_gridpoint function"""

    def test_without_persistence(
        self,
        ds_reference_2d_relevant_times_and_aligned: xr.Dataset,
        ds_prediction_2d: xr.Dataset,
        expected_dims_2d: dict,
    ):
        """Test the calculation of error per gridpoint without persistence."""
        ds_error = calculate_error_per_gridpoint(
            ds_reference_2d_relevant_times_and_aligned, ds_prediction_2d
        )
        assert "error" in ds_error
        assert ds_error.sizes == expected_dims_2d

    def test_with_persistence(
        self,
        ds_reference_2d_relevant_times_and_aligned: xr.Dataset,
        ds_prediction_2d: xr.Dataset,
        expected_dims_2d: dict,
    ):
        """Test the calculation of error per gridpoint with persistence."""
        ds_error = calculate_error_per_gridpoint(
            ds_reference_2d_relevant_times_and_aligned,
            ds_prediction_2d,
            include_persistence=True,
        )
        assert "error" in ds_error
        # Assert the shape of the error dataset
        assert ds_error.sizes == expected_dims_2d

        assert "persistence_error" in ds_error
        assert "cell_methods" in ds_error["persistence_error"].attrs
