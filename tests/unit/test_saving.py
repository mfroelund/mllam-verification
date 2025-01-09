import tempfile
from pathlib import Path

import pytest
import xarray as xr

from mllam_verification.operations.saving import save_xarray_dataset


class TestSaveXarrayDataset:
    @pytest.mark.parametrize("format_", ["netcdf", "zarr"])
    def test_save_xarray_dataset(self, ds_prediction_2D: xr.Dataset, format_: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / f"test_dataset.{format_}"
            save_xarray_dataset(ds_prediction_2D, tmp_path, format_)
            assert tmp_path.exists()

    # @pytest.mark.parametrize("format_", ["netcdf", "zarr"])
    # def test_save_xarray_dataset(self, ds_prediction_2D: xr.Dataset, format_: str):
    #     # with tempfile.TemporaryDirectory() as tmpdir:
    #     # tmpdir = tempfile.mkdtemp(dir=".")
    #     tmpdir = Path("./test_datasets")
    #     tmpdir.mkdir(exist_ok=True)
    #     # print("tmpdir", tmpdir)
    #     # tmp_path = Path(tmpdir) / f"test_dataset.{format_}"
    #     tmp_path = tmpdir / f"ds_prediction_2D.{format_}"
    #     save_xarray_dataset(ds_prediction_2D, tmp_path, format_)
    #     assert tmp_path.exists()
