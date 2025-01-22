"""Unit tests for the statistics module."""

import xarray as xr
from mllam_verification.operations.statistics import (
    calculate_error_per_gridpoint,
    calculate_global_error,
)
from numpy._typing._array_like import NDArray


class TestCalculateGlobalError:
    """Tests for the calculate_global_error function."""

    def test_without_persistence(
        self,
        ds_reference_1d: xr.Dataset,
        ds_prediction_1d: xr.Dataset,
        elapsed_forecast_duration: NDArray,
    ):
        """Test the calculation of global error without persistence."""
        ds_error = calculate_global_error(ds_reference_1d, ds_prediction_1d)
        assert "error" in ds_error
        assert ds_error["error"].shape == (1, elapsed_forecast_duration.size - 1)
        assert "cell_methods" in ds_error["error"].attrs

    def test_with_persistence(
        self,
        ds_reference_1d: xr.Dataset,
        ds_prediction_1d: xr.Dataset,
        elapsed_forecast_duration: NDArray,
    ):
        """Test the calculation of global error with persistence."""
        ds_error = calculate_global_error(
            ds_reference_1d, ds_prediction_1d, include_persistence=True
        )
        assert "error" in ds_error
        assert ds_error["error"].shape == (1, elapsed_forecast_duration.size - 1)
        assert "cell_methods" in ds_error["error"].attrs

        assert "persistence_error" in ds_error
        assert ds_error["persistence_error"].shape == (
            1,
            elapsed_forecast_duration.size - 1,
        )
        assert "cell_methods" in ds_error["persistence_error"].attrs


class TestCalculateErrorPerGridpoint:
    """Tests for the calculate_error_per_gridpoint function"""

    def test_without_persistence(
        self,
        ds_reference_2d: xr.Dataset,
        ds_prediction_2d: xr.Dataset,
    ):
        """Test the calculation of error per gridpoint without persistence."""
        ds_error = calculate_error_per_gridpoint(ds_reference_2d, ds_prediction_2d)
        assert "error" in ds_error
        assert ds_error["error"].shape == ds_reference_2d["state"].shape

    def test_with_persistence(
        self,
        ds_reference_2d: xr.Dataset,
        ds_prediction_2d: xr.Dataset,
    ):
        """Test the calculation of error per gridpoint with persistence."""
        ds_error = calculate_error_per_gridpoint(
            ds_reference_2d, ds_prediction_2d, include_persistence=True
        )
        assert "error" in ds_error
        assert ds_error["error"].shape == ds_reference_2d["state"].shape

        assert "persistence_error" in ds_error
        assert ds_error["persistence_error"].shape == ds_reference_2d["state"].shape
        assert "cell_methods" in ds_error["persistence_error"].attrs
