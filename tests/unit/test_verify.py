"""Unit tests for the verify module."""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import xarray as xr
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
from mllam_verification.verify import verify_prediction


@pytest.fixture(name="config", scope="module")
def fixture_config():
    """Fixture for the configuration.

    The configuration is created with a temporary directory for input and
    output data.
    """
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
                    object="mllam_verification.operations.statistics.calculate_global_error",
                    name="global_error",
                    include_persistence=True,
                ),
                Method(
                    object="mllam_verification.operations.statistics.calculate_error_per_gridpoint",
                    name="error_per_gridpoint",
                    include_persistence=True,
                ),
            ],
            output=Output(path=Path(tmpdir)),
        )
        yield config


@pytest.fixture(name="ds_reference_2d_temp")
def fixture_ds_reference_2d_temp(ds_reference_2d: xr.Dataset, config: Config):
    """Fixture that saves the reference dataset saved to temp dir on disk."""
    ds_reference_2d.to_zarr(config.inputs.datasets.reference.path)


@pytest.fixture(name="ds_prediction_2d_temp")
def fixture_ds_prediction_2d_temp(ds_prediction_2d: xr.Dataset, config: Config):
    """Fixture that saves the prediction dataset saved to temp dir on disk."""
    ds_prediction_2d.to_zarr(config.inputs.datasets.predictions[0].path)


class TestVerify:
    """Tests of the verify function"""

    @pytest.mark.usefixtures("ds_reference_2d_temp")
    @pytest.mark.usefixtures("ds_prediction_2d_temp")
    def test_verify(
        self,
        config: Config,
    ):
        """Test the verification of predictions against reference data."""
        verify_prediction(config)

        output_content = list(config.output.path.iterdir())
        # Assert on the content of the output directory
        assert (config.output.path / "global_error.zarr").exists(), (
            "global_error.zarr not found in output directory. "
            f"Content of {config.output.path}: {output_content}"
        )
