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
    return np.arange(0, 10)


@pytest.fixture(name="meshgrid")
def fixture_meshgrid() -> Tuple[NDArray, NDArray]:
    index = np.arange(0, DOMAIN_WIDTH)
    return MeshGrid(*np.meshgrid(index, index))


@pytest.fixture(name="gaussian_blob")
def fixture_gaussian_blob(meshgrid: MeshGrid):
    return np.exp(-((meshgrid.x - 50) ** 2 + (meshgrid.y - 50) ** 2) / (2 * 10**2))


@pytest.fixture(name="moving_gaussian_blob")
def fixture_moving_gaussian_blob(time: NDArray, gaussian_blob: NDArray):
    step_size = DOMAIN_WIDTH // len(time)
    return np.array(
        [np.roll(gaussian_blob, shift * step_size, axis=0) for shift in time]
    )


@pytest.fixture(name="ds_reference_1D")
def fixture_ds_reference_1D(time: NDArray, moving_gaussian_blob: NDArray):
    data = moving_gaussian_blob.reshape((len(time), -1))
    grid_index = np.arange(data.shape[1])

    return xr.Dataset(
        {"state": (["time", "grid_index"], data)},
        coords={"time": time, "grid_index": grid_index},
    )


@pytest.fixture(name="ds_prediction_1D")
def fixture_ds_prediction_1D(ds_reference_1D: xr.Dataset):
    noise = np.random.normal(0, 0.1, ds_reference_1D["state"].shape)
    # bias = np.linspace(-0.2, 0.2, ds_reference_1D["state"].shape[1])
    # bias = np.tile(bias, (ds_reference_1D["state"].shape[0], 1))
    bias = 0
    data = ds_reference_1D["state"].values + noise + bias
    return xr.Dataset(
        {"state": (["time", "grid_index"], data)},
        coords=ds_reference_1D.coords,
    )


@pytest.fixture(name="ds_reference_2D")
def fixture_ds_reference_2D(
    time: NDArray, moving_gaussian_blob: NDArray, meshgrid: MeshGrid
):
    return xr.Dataset(
        {"state": (["time", "x", "y"], moving_gaussian_blob)},
        coords={"time": time, "x": meshgrid.x[0, :], "y": meshgrid.y[:, 0]},
    )


@pytest.fixture(name="ds_prediction_2D")
def fixture_ds_prediction_2D(ds_reference_2D: xr.Dataset):
    noise = np.random.normal(0, 0.1, ds_reference_2D["state"].shape)
    # bias = np.random.normal(0, 0.1, ds_reference_2D["state"].shape)
    data = ds_reference_2D["state"].values + noise  # + bias
    return xr.Dataset(
        {"state": (["time", "x", "y"], data)},
        coords=ds_reference_2D.coords,
    )
