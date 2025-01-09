import xarray as xr
from loguru import logger

from .config import Config
from .operations.loading import load_xarray_dataset
from .operations.saving import save_xarray_dataset
from .operations.statistics import (
    diff_mean_per_time,
    diff_per_time_and_gridpoint,
    rmse_per_time,
)


def verify(config: Config):
    """Verify the prediction against the reference dataset.

    Parameters:
    -----------
    config: Config
        The configuration object
    """
    logger.info(config)
    ds_reference = load_xarray_dataset(config.inputs.datasets.reference.path)

    for ds_prediction in config.inputs.datasets.predictions:
        ds_prediction = load_xarray_dataset(ds_prediction.path)
        for method in config.methods:
            error = method.object(
                ds_reference, ds_prediction, method.include_persistence
            )

            save_xarray_dataset(error, config.output.path / method.name, format_="zarr")


def calculate_global_error(
    ds_reference: xr.Dataset, ds_prediction: xr.Dataset, include_persistence=False
) -> xr.Dataset:
    """Calculate global error between prediction and reference datasets.

    If specified, calculate the error relative to persistence too.

    The error is returned as a dataset with the following specification:

    Data variables:
    - error [elapsed_forecast_duration]:
        the error between the prediction and reference datasets
    Coordinates:
    - elapsed_forecast_duration:
        the elapsed forecast duration as a timedelta object

    Parameters:
    -----------
    ds_reference: xr.Dataset
        The reference dataset to calculate global error against. Assumed to have
        coordinates [time, grid_index] or [time, x, y].
    ds_prediction: xr.Dataset
        The prediction dataset to calculate global error off. Assumed to have
        coordinates [analysis_time, elapsed_forecast_duration, grid_index] or
        [analysis_time, elapsed_forecast_duration, x, y].
    include_persistence: bool
        Whether to calculate the error relative to persistence
    """
    # Stack the x and y dimensions into a single grid index if necessary
    if "x" in ds_prediction.dims and "y" in ds_prediction.dims:
        ds_prediction = ds_prediction.stack(grid_index=["x", "y"])
    if "x" in ds_reference.dims and "y" in ds_reference.dims:
        ds_reference = ds_reference.stack(grid_index=["x", "y"])

    error = rmse_per_time(ds_prediction - ds_reference)
    # Rename the variable to "error"
    error = error.rename({"state": "error"})

    if include_persistence:
        persistence = diff_mean_per_time(ds_reference)
        persistence_error = rmse_per_time(ds_prediction - persistence)
        persistence_error = persistence_error.rename({"state": "persistence_error"})
        error = xr.merge([error, persistence_error])
    return error


def calculate_error_per_gridpoint(
    ds_reference: xr.Dataset, ds_prediction: xr.Dataset, include_persistence=True
) -> xr.Dataset:
    """Calculate error per gridpoint between prediction and reference datasets.

    If specified, calculate the error relative to persistence too.

    The error is returned as a dataset with the following specification:

    Data variables:
    - error [elapsed_forecast_duration, x, y]:
        the error between the prediction and reference datasets
    Coordinates:
    - elapsed_forecast_duration:
        the elapsed forecast duration as a timedelta object
    - x:
        the x coordinate of the error
    - y:
        the y coordinate of the error

    Parameters:
    -----------
    ds_reference: xr.Dataset
        The reference dataset to calculate error against. Assumed to have
        coordinates [time, x, y]
    ds_prediction: xr.Dataset
        The prediction dataset to calculate error off. Assumed to have
        coordinates [analysis_time, elapsed_forecast_duration, x, y]
    include_persistence: bool
        Whether to calculate the error relative to persistence
    """
    error = ds_prediction - ds_reference
    # Rename the variable to "error"
    error = error.rename({"state": "error"})

    if include_persistence:
        persistence_error = diff_per_time_and_gridpoint(ds_reference)
        persistence_error = persistence_error.rename({"state": "persistence_error"})
        error = xr.merge([error, persistence_error])
    return error
