"""Unit tests for the saving module."""

import tempfile
from pathlib import Path

import pytest
import xarray as xr
from mllam_verification.operations.saving import save_xarray_dataset


class TestSaveXarrayDataset:
    """Unit tests for the save_xarray_dataset function."""

    @pytest.mark.parametrize(
        ["format_", "suffix"], [("netcdf", "nc"), ("zarr", "zarr")]
    )
    def test_save_xarray_dataset(
        self, ds_prediction_2d: xr.Dataset, format_: str, suffix: str
    ):
        """Test saving an xarray dataset to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"test_dataset.{suffix}"
            save_xarray_dataset(ds_prediction_2d, tmp_path, format_)
            assert tmp_path.exists()
