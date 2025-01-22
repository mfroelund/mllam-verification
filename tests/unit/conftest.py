"""
This module contains pytest fixtures for generating test data for unit tests.
The fixtures create various datasets and arrays used in the tests.
"""

import collections
from typing import Tuple

import numpy as np
import pytest
import xarray as xr
from numpy._typing._array_like import NDArray

DOMAIN_WIDTH = 100
NFEATURES = 5
MeshGrid = collections.namedtuple("MeshGrid", "x y")


@pytest.fixture(name="elapsed_forecast_duration")
def fixture_elapsed_forecast_duration():
    """Fixture that returns a numpy array representing elapsed_forecast_duration steps."""
    return np.arange(0, 10)


@pytest.fixture(name="meshgrid")
def fixture_meshgrid() -> Tuple[NDArray, NDArray]:
    """Fixture that returns a meshgrid of shape (DOMAIN_WIDTH, DOMAIN_WIDTH)."""
    index = np.arange(0, DOMAIN_WIDTH)
    return MeshGrid(*np.meshgrid(index, index))


@pytest.fixture(name="gaussian_blob")
def fixture_gaussian_blob(meshgrid: MeshGrid):
    """Fixture that returns a 2D Gaussian blob.

    The blob is centered at (50, 50) with a standard deviation of 10.
    """
    return np.exp(-((meshgrid.x - 50) ** 2 + (meshgrid.y - 50) ** 2) / (2 * 10**2))


@pytest.fixture(name="moving_gaussian_blob")
def fixture_moving_gaussian_blob(
    elapsed_forecast_duration: NDArray, gaussian_blob: NDArray
):
    """Fixture of a 3D array representing a moving Gaussian blob."""
    step_size = DOMAIN_WIDTH // len(elapsed_forecast_duration)
    return np.array(
        [
            np.roll(gaussian_blob, shift * step_size, axis=0)
            for shift in elapsed_forecast_duration[
                1:
            ]  # Only produce data for forecast steps
        ]
    )


@pytest.fixture(name="moving_gaussian_blobs")
def fixture_moving_gaussian_blobs(moving_gaussian_blob: NDArray):
    """Fixture of a 4D array representing multiple moving Gaussian blobs.

    Each blob is shifted by a random dx, dy and signifies an individual
    state_feature.
    """
    return np.array(
        [
            np.roll(moving_gaussian_blob, (dx, dy), axis=(1, 2))
            for dx, dy in np.random.randint(
                -DOMAIN_WIDTH // 2, DOMAIN_WIDTH // 2, size=(NFEATURES, 2)
            )
        ]
    ).transpose((1, 2, 3, 0))


@pytest.fixture(name="ds_reference_1d")
def fixture_ds_reference_1d(
    elapsed_forecast_duration: NDArray, moving_gaussian_blobs: NDArray
) -> xr.Dataset:
    """Fixture that returns Dataset with 1d moving gaussian blob reference data."""
    data = moving_gaussian_blobs.reshape(
        (len(elapsed_forecast_duration) - 1, -1, NFEATURES)
    )
    grid_index = np.arange(data.shape[1])

    return xr.Dataset(
        {
            "state": (
                [
                    "analysis_time",
                    "elapsed_forecast_duration",
                    "grid_index",
                    "state_feature",
                ],
                data[np.newaxis, ...],
            )
        },
        coords={
            "analysis_time": [elapsed_forecast_duration[0]],
            "elapsed_forecast_duration": elapsed_forecast_duration[1:],
            "grid_index": grid_index,
            "state_feature": [f"feature{i}" for i in np.arange(NFEATURES)],
        },
    )


@pytest.fixture(name="ds_prediction_1d")
def fixture_ds_prediction_1d(ds_reference_1d: xr.Dataset) -> xr.Dataset:
    """Fixture that returns Dataset with 1d moving gaussian blob prediction data.

    The prediction data is the reference data with added noise and bias.
    """
    noise = np.random.normal(0, 0.1, ds_reference_1d["state"].shape)
    # bias = np.linspace(-0.2, 0.2, ds_reference_1d["state"].shape[1])
    # bias = np.tile(bias, (ds_reference_1d["state"].shape[0], 1))

    bias = 0
    data = ds_reference_1d["state"].values + noise + bias
    return xr.Dataset(
        {
            "state": (
                [
                    "analysis_time",
                    "elapsed_forecast_duration",
                    "grid_index",
                    "state_feature",
                ],
                data,
            )
        },
        coords=ds_reference_1d.coords,
    )


@pytest.fixture(name="ds_reference_2d")
def fixture_ds_reference_2d(
    elapsed_forecast_duration: NDArray,
    moving_gaussian_blobs: NDArray,
    meshgrid: MeshGrid,
) -> xr.Dataset:
    """Fixture that returns Dataset 2d moving gaussian blob reference data."""
    return xr.Dataset(
        {
            "state": (
                [
                    "analysis_time",
                    "elapsed_forecast_duration",
                    "x",
                    "y",
                    "state_feature",
                ],
                moving_gaussian_blobs[np.newaxis, ...],
            )
        },
        coords={
            "analysis_time": [elapsed_forecast_duration[0]],
            "elapsed_forecast_duration": elapsed_forecast_duration[1:],
            "x": meshgrid.x[0, :],
            "y": meshgrid.y[:, 0],
            "state_feature": [f"feature{i}" for i in np.arange(NFEATURES)],
        },
    )


@pytest.fixture(name="ds_prediction_2d")
def fixture_ds_prediction_2d(ds_reference_2d: xr.Dataset) -> xr.Dataset:
    """Fixture that returns Dataset 2d moving gaussian blob prediction data.

    The prediction data is the reference data with added noise and bias.
    """
    noise = np.random.normal(0, 0.1, ds_reference_2d["state"].shape)
    # bias = np.random.normal(0, 0.1, ds_reference_2d["state"].shape)

    data = ds_reference_2d["state"].values + noise  # + bias
    return xr.Dataset(
        {
            "state": (
                [
                    "analysis_time",
                    "elapsed_forecast_duration",
                    "x",
                    "y",
                    "state_feature",
                ],
                data,
            )
        },
        coords=ds_reference_2d.coords,
    )
