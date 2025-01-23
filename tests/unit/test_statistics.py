"""Unit tests for the statistics module."""

import xarray as xr
from mllam_verification.operations.statistics import (
    calculate_error_per_gridpoint,
    calculate_global_error,
)


class TestCalculateGlobalError:
    """Tests for the calculate_global_error function."""

    def test_without_persistence(
        self,
        ds_reference_1d: xr.Dataset,
        ds_prediction_1d: xr.Dataset,
    ):
        """Test the calculation of global error without persistence."""
        expected_dims = tuple(
            ds_prediction_1d.dims[dim]
            for dim in ds_prediction_1d.dims
            if dim != "grid_index"
        )

        ds_error = calculate_global_error(ds_reference_1d, ds_prediction_1d)
        assert "error" in ds_error
        assert ds_error["error"].shape == expected_dims
        assert "cell_methods" in ds_error["error"].attrs

    def test_with_persistence(
        self,
        ds_reference_1d: xr.Dataset,
        ds_prediction_1d: xr.Dataset,
    ):
        """Test the calculation of global error with persistence."""
        expected_dims = tuple(
            ds_prediction_1d.dims[dim]
            for dim in ds_prediction_1d.dims
            if dim != "grid_index"
        )

        ds_error = calculate_global_error(
            ds_reference_1d, ds_prediction_1d, include_persistence=True
        )
        assert "error" in ds_error
        assert ds_error["error"].shape == expected_dims
        assert "cell_methods" in ds_error["error"].attrs

        assert "persistence_error" in ds_error
        assert ds_error["persistence_error"].shape == expected_dims
        assert "cell_methods" in ds_error["persistence_error"].attrs


class TestCalculateErrorPerGridpoint:
    """Tests for the calculate_error_per_gridpoint function"""

    def test_without_persistence(
        self,
        ds_reference_2d_relevant_times_and_aligned: xr.Dataset,
        ds_prediction_2d: xr.Dataset,
    ):
        """Test the calculation of error per gridpoint without persistence."""
        ds_error = calculate_error_per_gridpoint(
            ds_reference_2d_relevant_times_and_aligned, ds_prediction_2d
        )
        assert "error" in ds_error
        ds_prediction_2d_sizes = dict(ds_prediction_2d.sizes)
        ds_prediction_2d_sizes.pop("analysis_time")
        assert ds_error.sizes == ds_prediction_2d_sizes

    def test_with_persistence(
        self,
        ds_reference_2d_relevant_times_and_aligned: xr.Dataset,
        ds_prediction_2d: xr.Dataset,
    ):
        """Test the calculation of error per gridpoint with persistence."""
        ds_error = calculate_error_per_gridpoint(
            ds_reference_2d_relevant_times_and_aligned,
            ds_prediction_2d,
            include_persistence=True,
        )
        assert "error" in ds_error
        # Assert the shape of the error variable - note, that the analysis_time
        # dimension has been averaged out
        assert (
            ds_error["error"].shape
            == ds_reference_2d_relevant_times_and_aligned["state"].shape[1:]
        )

        assert "persistence_error" in ds_error
        # Assert the shape of the persistence_error variable - note, that the
        # analysis_time dimension has been averaged out
        assert (
            ds_error["persistence_error"].shape
            == ds_reference_2d_relevant_times_and_aligned["state"].shape[1:]
        )
        assert "cell_methods" in ds_error["persistence_error"].attrs
