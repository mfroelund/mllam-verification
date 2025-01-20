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
MeshGrid = collections.namedtuple("MeshGrid", "x y")


@pytest.fixture(name="time")
def fixture_time():
    """Fixture that returns a numpy array representing time steps."""
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
def fixture_moving_gaussian_blob(time: NDArray, gaussian_blob: NDArray):
    """Fixture of a 3D array representing a moving Gaussian blob."""
    step_size = DOMAIN_WIDTH // len(time)
    return np.array(
        [np.roll(gaussian_blob, shift * step_size, axis=0) for shift in time]
    )


@pytest.fixture(name="ds_reference_1d")
def fixture_ds_reference_1d(time: NDArray, moving_gaussian_blob: NDArray) -> xr.Dataset:
    """Fixture that returns Dataset with 1d moving gaussian blob reference data."""
    data = moving_gaussian_blob.reshape((len(time), -1))
    grid_index = np.arange(data.shape[1])

    return xr.Dataset(
        {"state": (["time", "grid_index"], data)},
        coords={"time": time, "grid_index": grid_index},
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
        {"state": (["time", "grid_index"], data)},
        coords=ds_reference_1d.coords,
    )


@pytest.fixture(name="ds_reference_2d")
def fixture_ds_reference_2d(
    time: NDArray, moving_gaussian_blob: NDArray, meshgrid: MeshGrid
) -> xr.Dataset:
    """Fixture that returns Dataset 2d moving gaussian blob reference data."""
    return xr.Dataset(
        {"state": (["time", "x", "y"], moving_gaussian_blob)},
        coords={"time": time, "x": meshgrid.x[0, :], "y": meshgrid.y[:, 0]},
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
        {"state": (["time", "x", "y"], data)},
        coords=ds_reference_2d.coords,
    )
