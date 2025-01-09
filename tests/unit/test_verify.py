import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import xarray as xr
from numpy._typing._array_like import NDArray

from mllam_verification.config import (
    Config,
    CoordRanges,
    Dataset,
    Datasets,
    Inputs,
    Method,
    Output,
    TimeRange,
)
from mllam_verification.verify import (
    calculate_error_per_gridpoint,
    calculate_global_error,
    verify,
)


@pytest.fixture(name="config", scope="module")
def fixture_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(
            schema_version="v0.1.0",
            inputs=Inputs(
                datasets=Datasets(
                    reference=Dataset(path=Path(tmpdir) / "reference.zarr"),
                    predictions=[
                        Dataset(path=Path(tmpdir) / "prediction1.zarr"),
                    ],
                ),
                variables=["2t", "10u"],
                coord_ranges=CoordRanges(
                    time=TimeRange(
                        start=datetime(1990, 9, 3, 0, 0),
                        end=datetime(1990, 9, 9, 0, 0),
                        step=timedelta(hours=3),
                    )
                ),
            ),
            methods=[
                Method(
                    object="mllam_verification.verify.calculate_global_error",
                    name="global_error",
                    include_persistence=True,
                ),
                Method(
                    object="mllam_verification.verify.calculate_error_per_gridpoint",
                    name="error_per_gridpoint",
                    include_persistence=True,
                ),
            ],
            output=Output(path=Path(tmpdir)),
        )
        yield config


@pytest.fixture(name="ds_reference_2D_temp")
def fixture_ds_reference_2D_temp(ds_reference_2D: xr.Dataset, config: Config):
    ds_reference_2D.to_zarr(config.inputs.datasets.reference.path)


@pytest.fixture(name="ds_prediction_2D_temp")
def fixture_ds_prediction_2D_temp(ds_prediction_2D: xr.Dataset, config: Config):
    ds_prediction_2D.to_zarr(config.inputs.datasets.predictions[0].path)


class TestVerify:

    @pytest.mark.usefixtures("ds_reference_2D_temp")
    @pytest.mark.usefixtures("ds_prediction_2D_temp")
    def test_verify(
        self,
        config: Config,
    ):
        verify(config)

        output_content = list(config.output.path.iterdir())
        # Assert on the content of the output directory
        assert (config.output.path / "global_error.zarr").exists(), (
            "global_error.zarr not found in output directory. "
            f"Content of {config.output.path}: {output_content}"
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
