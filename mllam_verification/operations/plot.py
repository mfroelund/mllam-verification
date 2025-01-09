import math
from typing import Hashable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import xarray as xr


def plot_error_timeline(ds_error: xr.Dataset):
    """Plot the error vs time from the error dataset.

    Parameters:
    -----------
    error_ds: xr.Dataset
        The dataset containing the error with coordinates [time]
    """
    plt.figure(figsize=(10, 5), constrained_layout=True)
    for var in ds_error.data_vars:
        ds_error[var].plot.line(
            x="time",
            label=f"{var} [cell_methods: {ds_error[var].attrs['cell_methods']}]",
        )

    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Error vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig("error_timeline.png")


def plot_error_map(ds_error: xr.Dataset):
    """Plot the error map from the error dataset.

    Parameters:
    -----------
    error_ds: xr.Dataset
        The dataset containing the error with coordinates [x, y]
    """

    num_vars = len(ds_error.data_vars)
    nrows = math.floor(math.sqrt(num_vars))  # Number of columns for the grid
    ncols = math.ceil(num_vars / nrows)  # Calculate the number of rows needed

    # Get max and min values for the colorbars
    ds_max = ds_error.max()
    ds_min = ds_error.min()

    var: Hashable
    time: xr.DataArray
    for time in ds_error["time"]:
        fig = plt.figure(figsize=(10, 5 * nrows), constrained_layout=True)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig)
        plt.suptitle("Error Map")
        for i, var in enumerate(ds_error.data_vars):
            row = i // ncols
            col = i % ncols
            axes = fig.add_subplot(gs[row, col])
            ds_error[var].isel(time=time).plot.pcolormesh(
                x="x",
                y="y",
                cmap="bwr",
                ax=axes,
                vmin=ds_min[var],
                vmax=ds_max[var],
            )
            axes.set_title(f"{var} at time {time.values}")
        plt.savefig(f"error_map_time{time.values}.png")
