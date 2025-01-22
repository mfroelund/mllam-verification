import pytest
import xarray as xr
from mllam_verification.operations.plot import plot_error_map, plot_error_timeline


@pytest.fixture(name="global_error_with_persistence")
def fixture_global_error_with_persistence() -> xr.Dataset:
    """Fixture that returns test global error data with persistence."""

    ds = xr.Dataset(
        {
            "error": (
                ["elapsed_forecast_duration", "state_feature"],
                [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
            ),
            "persistence_error": (
                ["elapsed_forecast_duration", "state_feature"],
                [[1.1, 1.1], [1.2, 1.2], [1.3, 1.3]],
            ),
        },
        coords={
            "elapsed_forecast_duration": [0, 1, 2],
            "state_feature": ["feature1", "feature2"],
        },
    )
    # Add cell_methods attribute to data variables
    ds["error"].attrs["cell_methods"] = "grid_index: method1"
    ds["persistence_error"].attrs["cell_methods"] = "grid_index: method2"

    return ds


@pytest.fixture(name="error_per_gridpoint")
def fixture_error_per_gridpoint() -> xr.Dataset:
    """Fixture that returns test error per gridpoint data."""

    ds = xr.Dataset(
        {
            "error": (
                ["time", "x", "y"],
                [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            ),
            "persistence_error": (
                ["time", "x", "y"],
                [[[1.1, 1.2], [1.3, 1.4]], [[1.5, 1.6], [1.7, 1.8]]],
            ),
        },
        coords={"time": [0, 1], "x": [0, 1], "y": [0, 1]},
    )
    # Add cell_methods attribute to data variables
    ds["error"].attrs["cell_methods"] = "grid_index: method1"
    ds["persistence_error"].attrs["cell_methods"] = "grid_index: method2"

    return ds


def test_plot_error_timeline(global_error_with_persistence):
    """Test producing a timeline plot of the global error."""

    plot_error_timeline(global_error_with_persistence)


def test_plot_error_per_gridpoint(error_per_gridpoint):
    """Test producing a 2D plot of the error per gridpoint."""
    plot_error_map(error_per_gridpoint)
