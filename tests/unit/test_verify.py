import xarray as xr
from numpy._typing._array_like import NDArray

from mllam_verification.verify import (
    calculate_error_per_gridpoint,
    calculate_global_error,
)


class TestCalculateGlobalError:
    def test_without_persistence(
        self,
        ds_reference_1D: xr.Dataset,
        ds_prediction_1D: xr.Dataset,
        time: NDArray,
    ):
        ds_error = calculate_global_error(ds_reference_1D, ds_prediction_1D)
        assert "error" in ds_error
        assert ds_error["error"].shape == (time.size,)
        assert "cell_methods" in ds_error["error"].attrs

    def test_with_persistence(
        self,
        ds_reference_1D: xr.Dataset,
        ds_prediction_1D: xr.Dataset,
        time: NDArray,
    ):
        ds_error = calculate_global_error(
            ds_reference_1D, ds_prediction_1D, include_persistence=True
        )
        assert "error" in ds_error
        assert ds_error["error"].shape == (time.size,)
        assert "cell_methods" in ds_error["error"].attrs

        assert "persistence_error" in ds_error
        assert ds_error["persistence_error"].shape == (time.size,)
        assert "cell_methods" in ds_error["persistence_error"].attrs


class TestCalculateErrorPerGridpoint:
    def test_without_persistence(
        self,
        ds_reference_2D: xr.Dataset,
        ds_prediction_2D: xr.Dataset,
    ):
        ds_error = calculate_error_per_gridpoint(ds_reference_2D, ds_prediction_2D)
        assert "error" in ds_error
        assert ds_error["error"].shape == ds_reference_2D["state"].shape
        assert "cell_methods" in ds_error["error"].attrs

    def test_with_persistence(
        self,
        ds_reference_2D: xr.Dataset,
        ds_prediction_2D: xr.Dataset,
    ):
        ds_error = calculate_error_per_gridpoint(
            ds_reference_2D, ds_prediction_2D, include_persistence=True
        )
        assert "error" in ds_error
        assert ds_error["error"].shape == ds_reference_2D["state"].shape
        assert "cell_methods" in ds_error["error"].attrs

        assert "persistence_error" in ds_error
        assert ds_error["persistence_error"].shape == ds_reference_2D["state"].shape
        assert "cell_methods" in ds_error["persistence_error"].attrs
