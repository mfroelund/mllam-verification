import xarray as xr


def elapsed_to_utc(ds: xr.Dataset) -> xr.Dataset:
    """
    Change the `elapsed_forecast_duration` coordinate to a `time` coordinate.
    
    Assumes that the dataset as coordinates `start_time` and
    `elapsed_forecast_duration`.

    NB: if the `start_time` is not aligned or if the
    `elapsed_forecast_duration` is not the same length this will fill with NaNs
    for times where the forecast is not available.
    
    from:
        start_time       elapsed_forecast_duration ("lead time")
                         00:00  03:00  06:00  09:00
        1/1/2025  00:00    x      x      x      x
        1/1/2025  03:00    x      x      x      x
        
    to:
        start_time       time
                         00:00  03:00  06:00  09:00  12:00
        1/1/2025  00:00    x      x      x      x     nan
        1/1/2025  03:00   nan     x      x      x      x
        
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with `elapsed_forecast_duration` and `start_time` coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with `time` and `start_time` coordinates.
    
    """
    slices = []
    for t_start in ds.start_time.values:
        ds_slice = ds.sel(start_time=t_start)
        ds_slice["time"] = t_start + ds_slice.elapsed_forecast_duration
        ds_slice = ds_slice.swap_dims({"elapsed_forecast_duration": "time"})
        slices.append(ds_slice)
    ds_by_utc = xr.concat(slices, dim="start_time")
    return ds_by_utc

def utc_to_elapsed(ds: xr.Dataset) -> xr.Dataset:
    """
    Change the `time` coordinate to a `elapsed_forecast_duration` coordinate.
    
    Assumes that the dataset as coordinates `start_time` and `time`.
    
    OBS: any times that have nans will be dropped, this is to ensure that we
    don't have NaNs values at the start and end of the `elapsed_forecast_duration`
    coordinate.
    
    from:
        start_time       time
                         00:00  03:00  06:00  09:00  12:00
        1/1/2025  00:00    x      x      x     nan    nan
        1/1/2025  06:00   nan    nan     x      x      x
        
    to:
        start_time       elapsed_forecast_duration ("lead time")
                         00:00  03:00  06:00
        1/1/2025  00:00    x      x      x
        1/1/2025  06:00    x      x      x
        
    Parameters
    ----------
    ds : xr.Dataset
        Dataset with `time` and `start_time` coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with `elapsed_forecast_duration` and `start_time` coordinates.
    """
    slices = []
    for t_start in ds.start_time.values:
        # need to drop NaNs to remove any NaNs steps that were added as padding
        # when converting from elapsed to UTC, this will removes any NaNs that
        # were there originally too
        ds_slice = ds.sel(start_time=t_start).dropna(dim="time", how="all")
        da_time_delta = ds_slice.time - t_start
        ds_slice["elapsed_forecast_duration"] = da_time_delta
        ds_slice = ds_slice.swap_dims({"time": "elapsed_forecast_duration"})
        slices.append(ds_slice)
    ds_by_elapsed = xr.concat(slices, dim="start_time")
    return ds_by_elapsed