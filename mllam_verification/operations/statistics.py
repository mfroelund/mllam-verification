"""Module to calculate statistics for a given Dataset."""

from typing import List, Optional

import xarray as xr

xr.set_options(keep_attrs=True)


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

    # Calculate the error and rename the variable
    error = rmse_per_time(ds_prediction - ds_reference)
    error = error.rename({"state": "error"})

    # Calculate the persistence error and merge with the error dataset
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
    # Calculate the error and rename the variable
    error = ds_prediction - ds_reference
    error = error.rename({"state": "error"})

    # Calculate the persistence error
    if include_persistence:
        # diff_per_time_and_gridpoint takes the difference between the next and
        # the current time step (for n_diff_steps=1). Since the persistence is
        # a measure of the error one does, if the current step is taken as the
        # reference, we need to take the difference between the current and the
        # next time step (the actual reference) to get the persistence error,
        # hence the minus sign.
        persistence_error = -diff_per_time_and_gridpoint(ds_reference)
        persistence_error = persistence_error.rename({"state": "persistence_error"})
        error = xr.merge([error, persistence_error])

    return error


def compute_pipeline_statistic(
    ds: xr.Dataset,
    stats_op: Optional[str] = None,
    stats_dims: Optional[str | List[str]] = None,
    diff_dim: Optional[str] = None,
    n_diff_steps: Optional[int] = 1,
    groupby: Optional[str] = None,
):
    """
    Apply a series of oprations to compute a specific compound statistic
    The operations applied in order are:
    1. (If diff_dim != None) Apply diff over the `diff_dim` dimension (default to 1 step diff)
    2. (If groupby != None) Apply grouping of dataset according to the `groupby` index
    3. Apply the stats_op to the dataarray over the `stats_dims` dimensions.
       If no stats_dims are provided, apply operator accross time and grid_index.
    Parameters
    ----------
    ds : xr.Dataset
        Dataset to compute the statistic on
    stats_op : str
        Statistic operation to apply, must be a valid xarray operation
    diff_dim : str, optional
        Dimension to apply diff over, by default None
    n_diff_steps : int, optional
        Number of steps to compute the diff over, by default 1
    groupby : str, optional
        Index to group over, by default None
    """
    # Convert string to list to unify type of stats_dims
    if isinstance(stats_dims, str):
        stats_dims = [stats_dims]
    # Build up CF compliant cell-method attribute so that people know what
    # operations were applied
    cell_methods = []

    if diff_dim:
        # Only keep variables that have the diff_dim as a dimension
        vars_to_keep = [v for v in ds.data_vars if diff_dim in ds[v].dims]
        if not vars_to_keep:
            raise ValueError(f"No variables found with dimension {diff_dim}")

        # Apply the diff operation
        ds = ds[vars_to_keep].diff(dim=diff_dim, n=n_diff_steps)
        # Get unit of the diff'ed array
        diff_unit_array: xr.DataArray = ds[diff_dim][1] - ds[diff_dim][0]
        diff_unit = diff_unit_array.values
        if diff_dim == "elapsed_forecast_duration":
            # Convert the diff unit to hours
            diff_unit = diff_unit.astype("timedelta64[h]")
        else:
            raise NotImplementedError(
                f"diff_dim of type {type(diff_dim)} not supported"
            )

        # Update the cell_methods with the operation applied
        cell_methods.append(f"{diff_dim}: diff (interval: {diff_unit})")

    if groupby:
        # Apply the groupby operation
        ds = ds.groupby(groupby)
        # Update the cell_methods with the operation applied
        cell_methods.append(f"{groupby}: groupby")

    if stats_op:
        if isinstance(stats_op, str):
            # If no stats_dims are provided, apply operator accross time and grid_index
            if not stats_dims:
                stats_dims = ["grid_index", "time"]

            # Build up the cell_methods attribute
            methods = [f"{dim}:" for dim in stats_dims]
            methods.append(stats_op)

            ds = getattr(ds, stats_op)(dim=stats_dims)

            # Update the cell_methods with the operation applied
            cell_methods.extend(methods)
        else:
            raise NotImplementedError(
                f"stats_op of type {type(stats_op)} not supported"
            )

    cell_methods_str = " ".join(cell_methods)
    # Add cell_methods attribute to all variables
    for var in ds.data_vars:
        ds[var].attrs["cell_methods"] = cell_methods_str

    return ds


def mean(ds: xr.Dataset):
    """Compute the mean across grid_index and time for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="mean")


def mean_per_gridpoint(ds: xr.Dataset):
    """Compute the mean across time for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="mean", stats_dims="time")


def mean_per_time(ds: xr.Dataset):
    """Compute the mean across grid_index for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="mean", stats_dims="grid_index")


def std(ds: xr.Dataset):
    """Compute the standard deviation across grid_index and time for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="std")


def std_per_gridpoint(ds: xr.Dataset):
    """Compute the standard deviation across time for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(ds, stats_op="std", stats_dims="time")


def rmse_per_time(ds: xr.Dataset) -> xr.Dataset:
    """Compute the root mean squared error across grid_index for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    ds_rmse_per_time = mean_per_time((ds) ** 2) ** 2
    # Update cell_methods attributes
    for _, da_var in ds_rmse_per_time.items():
        da_var.attrs["cell_methods"] = "grid_index: root_mean_square"

    return ds_rmse_per_time


def diff_mean(ds: xr.Dataset):
    """Compute the mean across grid_point and time of the difference in time
    for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="mean", diff_dim="time", n_diff_steps=1
    )


def diff_mean_per_gridpoint(ds: xr.Dataset):
    """Compute the mean across time of the difference in time for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="mean", stats_dims="time", diff_dim="time", n_diff_steps=1
    )


def diff_mean_per_time(ds: xr.Dataset):
    """Compute the mean across grid_point of the difference in time for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds,
        stats_op="mean",
        stats_dims="grid_index",
        diff_dim="elapsed_forecast_duration",
        n_diff_steps=1,
    )


def diff_per_time_and_gridpoint(ds: xr.Dataset) -> xr.Dataset:
    """Compute the difference in time per grid_point for all variables.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, diff_dim="elapsed_forecast_duration", n_diff_steps=1
    )


def diff_std(ds: xr.Dataset):
    """Compute the std across grid_point and time of the difference in time
    for all variables.
    The difference is computed over 1 time step.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="std", diff_dim="time", n_diff_steps=1
    )


def diff_std_per_gridpoint(ds: xr.Dataset):
    """Compute the std across time of the difference in time for all variables.
    The difference is computed over 1 time step.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, stats_op="std", stats_dims="time", diff_dim="time", n_diff_steps=1
    )


def diurnal_diff_mean(ds: xr.Dataset):
    """Compute the diurnal mean across grid_index and time of the difference in
    time for all variables.
    The data is grouped by time.hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, groupby="time.hour", stats_op="mean", diff_dim="time", n_diff_steps=1
    )


def diurnal_diff_mean_per_gridpoint(ds: xr.Dataset):
    """Compute the diurnal mean across time of the difference in time for all
    variables.
    The data is grouped by time.hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds,
        groupby="time.hour",
        stats_op="mean",
        stats_dims="time",
        diff_dim="time",
        n_diff_steps=1,
    )


def diurnal_diff_std(ds: xr.Dataset):
    """Compute the diurnal std across grid_index and time of the difference in
    time for all variables.
    The data is grouped by the hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds, groupby="time.hour", stats_op="std", diff_dim="time", n_diff_steps=1
    )


def diurnal_diff_std_per_gridpoint(ds: xr.Dataset):
    """Compute the diurnal std across time of the difference in time for all
    variables.
    The data is grouped by time.hour to make the operator be applied accross
    diurnal cycles.
    The difference in time is computed over 1 time step.
    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    return compute_pipeline_statistic(
        ds,
        stats_dims="time",
        groupby="time.hour",
        stats_op="std",
        diff_dim="time",
        n_diff_steps=1,
    )
