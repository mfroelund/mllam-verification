import numpy as np
import pandas as pd
import xarray as xr


def get_relevant_reference_data(
    ds_reference: xr.Dataset, ds_prediction: xr.Dataset
) -> xr.Dataset:
    # Determine relevant reference times based on the prediction analysis times
    # and elapsed forecast durations
    reference_times = (
        ds_prediction["analysis_time"]
        .values[..., np.newaxis]
        .repeat(len(ds_prediction["elapsed_forecast_duration"]), axis=1)
    )
    reference_times = (
        reference_times + ds_prediction["elapsed_forecast_duration"].values
    )

    # Select the relevant reference data
    ds_reference = ds_reference.sel(time=reference_times.flatten())
    # Unstack the reference dataset based on the temporary multi-index to get

    return ds_reference


def align_shapes(ds_reference: xr.Dataset, ds_prediction: xr.Dataset) -> xr.Dataset:
    # Convert reference dataset to a format compatible with the prediction dataset
    # by merging all data variables into a single state variable
    ds_reference = (
        ds_reference.to_array(dim="state_feature")
        .to_dataset(name="state")
        .transpose("time", "x", "y", "state_feature")
        .assign_coords(
            analysis_time=np.zeros(len(ds_prediction["analysis_time"]), dtype=int)
        )
    )

    # Determine relevant reference times based on the prediction analysis times
    # and elapsed forecast durations
    # reference_times = (
    #     ds_prediction["analysis_time"]
    #     .values[..., np.newaxis]
    #     .repeat(len(ds_prediction["elapsed_forecast_duration"]), axis=1)
    # )
    # reference_times = (
    #     reference_times + ds_prediction["elapsed_forecast_duration"].values
    # )

    # Create a MultiIndex based on the analysis times and the elapsed forecast
    # durations to be used to re-index the reference dataset.
    index = pd.MultiIndex.from_product(
        [
            ds_prediction["analysis_time"].values,
            ds_prediction["elapsed_forecast_duration"].values,
        ],
        names=["analysis_time", "elapsed_forecast_duration"],
    )
    # Select the relevant reference data
    # ds_reference = ds_reference.sel(time=reference_times.flatten())
    # Unstack the reference dataset based on the temporary multi-index to get
    # two new coordinates: analysis_time and elapsed_forecast_duration
    ds_reference = ds_reference.assign(tmp_time=index).unstack("tmp_time")

    # Reshape the "state" variable of the reference dataset to go from shape
    # (time, x, y, state_feature) to (analysis_time, elapsed_forecast_duration,
    # x, y, state_feature)
    data = ds_reference["state"].values.reshape(
        (
            ds_reference.sizes["analysis_time"],
            ds_reference.sizes["elapsed_forecast_duration"],
            ds_reference.sizes["x"],
            ds_reference.sizes["y"],
            ds_reference.sizes["state_feature"],
        )
    )
    # Assign the reshaped data to the reference dataset and drop the old time
    # coordinate
    ds_reference = ds_reference.assign(
        state=(
            (
                "analysis_time",
                "elapsed_forecast_duration",
                "x",
                "y",
                "state_feature",
            ),
            data,
        )
    ).drop_vars("time")

    return ds_reference
